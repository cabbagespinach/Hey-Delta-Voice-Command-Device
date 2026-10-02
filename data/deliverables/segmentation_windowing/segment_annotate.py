#!/usr/bin/env python3
"""
Long-recording segmentation, wakeword annotation and fixed-window generation.

Stages (each reads the previous stage's CSV, so they can be rerun independently):

  inventory   manifest.csv + frozen dataset_split -> outputs/recording_inventory.csv
              (one row per audio file; inherits split/source_group; extends the
              split to recordings the split deliverable does not know about)
  annotate    real recordings   -> Silero VAD + per-chunk Faster-Whisper word
                                   timestamps -> ASR *candidate* annotations
              synthetic GT      -> streaming_eval insertion timestamps and TTS
                                   whole-clip spans -> ground-truth annotations
              -> outputs/candidate_annotations.csv, outputs/asr_crosscheck_synthetic.csv
  resolve     candidate_annotations + review/review_overrides.csv
              -> outputs/resolved_annotations.csv
  windows     resolved annotations + window_config.json
              -> outputs/windows.csv, outputs/rejected_windows.csv
  review-tool -> review/review_tool.html, review/review_queue.csv
  materialize (optional) write window WAVs for a subset
  all         inventory, annotate, resolve, windows, review-tool

Usage:
  python segment_annotate.py all
  python segment_annotate.py resolve && python segment_annotate.py windows   # after review
  python validate_windows.py
"""
from pathlib import Path
import argparse, hashlib, json, math, re, sys, datetime
import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "window_config.json"

CATEGORY_MAP = {  # identical to dataset_split/split_dataset.py
    "positives": "positive_wakeword",
    "negatives_general": "negative_general_speech",
    "negatives_media": "negative_media",
    "negatives_confusable": "negative_confusable",
    "negatives_partial": "negative_partial_wakeword",
    "negatives_silence": "negative_silence_noise",
    "streaming_eval": "streaming_eval",
}
SPLITS = ["train", "validation", "test"]

# Annotation vocabulary -------------------------------------------------------
ORIGIN_GT = "synthetic_ground_truth"
ORIGIN_ASR = "asr_candidate"
ORIGIN_HUMAN = "human"
MATCH_COMPLETE = "complete"
MATCH_FUZZY = "fuzzy_complete"
MATCH_PARTIAL = "partial_fragment"
MATCH_UNMATCHED = "unmatched_speech"
MATCH_NOT_WW = "non_wakeword"
MATCH_TYPES = {MATCH_COMPLETE, MATCH_FUZZY, MATCH_PARTIAL, MATCH_UNMATCHED, MATCH_NOT_WW}
ST_GT = "ground_truth"
ST_CANDIDATE = "candidate"
ST_NEEDS_REVIEW = "needs_review"
ST_REVIEWED = "reviewed"
ST_REMOVED = "removed"

ANNOTATION_COLUMNS = [
    "annotation_id", "file_id", "filepath", "recording_id", "source_id", "source_group",
    "split", "origin", "gt_source", "match_type", "status", "review_reasons",
    "start_sec", "end_sec", "raw_start_sec", "raw_end_sec", "boundary_method",
    "vad_chunk_start_sec", "vad_chunk_end_sec", "asr_chunk_text", "asr_words",
    "asr_min_word_prob", "edit_ratio",
]


# ----------------------------------------------------------------------------
# Config / IO helpers
# ----------------------------------------------------------------------------
def load_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    base = path.parent
    p = cfg["paths"]
    root = (base / p["project_root"]).resolve()
    cfg["_root"] = root
    cfg["_base"] = base
    cfg["_out"] = (base / p["output_dir"]).resolve()
    cfg["_review"] = (base / p["review_dir"]).resolve()
    for k in ["manifest", "normalized_manifest", "split_assignments", "streaming_eval_manifest"]:
        cfg["_" + k] = root / p[k]
    cfg["_out"].mkdir(parents=True, exist_ok=True)
    cfg["_review"].mkdir(parents=True, exist_ok=True)
    return cfg


def config_hash(cfg, section=None):
    obj = {k: v for k, v in cfg.items() if not k.startswith("_")}
    if section:
        obj = obj[section]
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def stable_hash(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def is_blank(v):
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def read_mono(path):
    y, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return y.mean(axis=1), sr


def to_16k(y, sr, target=16000):
    if sr == target:
        return y.astype(np.float32)
    from scipy.signal import resample_poly
    g = math.gcd(int(sr), int(target))
    return resample_poly(y, target // g, int(sr) // g).astype(np.float32)


# ----------------------------------------------------------------------------
# Text matching
# ----------------------------------------------------------------------------
def norm_token(w):
    return re.sub(r"[^a-z0-9]+", "", str(w).lower())


def edit_ratio(a, b):
    a, b = a.strip(), b.strip()
    if not a and not b:
        return 0.0
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[-1] / max(len(a), len(b), 1)


def detect_in_chunk(chunk, cfg, recording_is_positive):
    """Classify one VAD chunk's ASR words into candidate annotations.

    Returns a list of dicts (match_type, start, end, words, reasons, edit_ratio).
    Only an exact adjacent 'hey' + 'delta' pair can become MATCH_COMPLETE; every
    softer match is FUZZY/PARTIAL and can never be positive without review.
    """
    ww = cfg["wakeword"]
    gates = cfg["asr"]["quality_gates"]
    t1, t2 = ww["first_token"], ww["second_token"]
    target = f"{t1} {t2}"
    words = [w for w in chunk["words"] if norm_token(w["word"])]
    toks = [norm_token(w["word"]) for w in words]
    c0, c1 = chunk["chunk_start"], chunk["chunk_end"]
    out = []

    def span(ws):
        s = max(c0, min(w["start"] for w in ws))
        e = min(c1, max(w["end"] for w in ws))
        return s, e, min(w["start"] for w in ws), max(w["end"] for w in ws)

    i = 0
    while i < len(toks):
        a = toks[i]
        b = toks[i + 1] if i + 1 < len(toks) else None
        if b is not None and a == t1 and b == t2:
            ws = words[i:i + 2]
            s, e, rs, re_ = span(ws)
            reasons = []
            if min(w["probability"] for w in ws) < gates["min_word_probability"]:
                reasons.append("low_word_probability")
            if ws[1]["start"] - ws[0]["end"] > gates["max_inter_word_gap_sec"]:
                reasons.append("large_inter_word_gap")
            if any(w["end"] - w["start"] < gates["min_word_duration_sec"] for w in ws):
                reasons.append("collapsed_word_timestamp")
            if not (ww["min_occurrence_duration_sec"] <= e - s <= ww["max_occurrence_duration_sec"]):
                reasons.append("implausible_duration")
            if not recording_is_positive:
                reasons.append("wakeword_detected_in_negative_recording")
            out.append(dict(match_type=MATCH_COMPLETE, s=s, e=e, rs=rs, re=re_, words=ws,
                            reasons=reasons, edit_ratio=0.0))
            i += 2
            continue
        if b is not None and edit_ratio(f"{a} {b}", target) <= ww["fuzzy_pair_max_edit_ratio"]:
            ws = words[i:i + 2]
            s, e, rs, re_ = span(ws)
            out.append(dict(match_type=MATCH_FUZZY, s=s, e=e, rs=rs, re=re_, words=ws,
                            reasons=["fuzzy_text_match"], edit_ratio=edit_ratio(f"{a} {b}", target)))
            i += 2
            continue
        if a == t2 or edit_ratio(a, t2) <= ww["fuzzy_word_max_edit_ratio"]:
            ws = words[i:i + 1]
            s, e, rs, re_ = span(ws)
            out.append(dict(match_type=MATCH_PARTIAL, s=s, e=e, rs=rs, re=re_, words=ws,
                            reasons=["isolated_second_token"], edit_ratio=edit_ratio(a, t2)))
        elif a == t1:
            ws = words[i:i + 1]
            s, e, rs, re_ = span(ws)
            out.append(dict(match_type=MATCH_PARTIAL, s=s, e=e, rs=rs, re=re_, words=ws,
                            reasons=["isolated_first_token"], edit_ratio=edit_ratio(a, t1)))
        i += 1

    # Whole-chunk fuzzy fallback (e.g. "Heydelta", "hey-delta") when nothing matched.
    text = " ".join(toks)
    if not out and text and edit_ratio(text, target) <= ww["fuzzy_pair_max_edit_ratio"]:
        out.append(dict(match_type=MATCH_FUZZY, s=c0, e=c1, rs=c0, re=c1, words=words,
                        reasons=["fuzzy_chunk_text_match"], edit_ratio=edit_ratio(text, target)))
    # Speech in a positive recording that yields no wakeword-like text may be a
    # missed wakeword: exclude it from negatives and send it to review.
    if (not any(d["match_type"] in (MATCH_COMPLETE, MATCH_FUZZY) for d in out)
            and recording_is_positive and cfg["annotation_policy"]["flag_unmatched_speech_in_positive_recordings"]):
        out.append(dict(match_type=MATCH_UNMATCHED, s=c0, e=c1, rs=c0, re=c1, words=words,
                        reasons=["no_wakeword_text_in_positive_recording_speech"],
                        edit_ratio=edit_ratio(text, target) if text else 1.0))
    return out


# ----------------------------------------------------------------------------
# Stage 1: inventory + split inheritance/extension
# ----------------------------------------------------------------------------
def extend_split(new_rows, existing, cfg, prior=None):
    """Assign splits to source_groups the frozen split does not contain.

    Append-only: groups already in `prior` (the previous split_extension.csv) keep
    their split and count towards their stratum's totals; only unseen groups are
    assigned. Unseen groups are stratified by canonical_category x device (parsed
    from the filename; a stratification key only, never used as identity) and
    assigned largest duration first to the split with the largest absolute
    duration deficit against the ratios.
    """
    ext = cfg["split_extension"]
    ratios, seed = ext["ratios"], ext["seed"]
    rx = re.compile(ext["device_regex"])
    prior = prior if prior is not None else pd.DataFrame(columns=["source_group", "split", "assignment_method"])
    prior_split = dict(zip(prior.source_group, prior.split))
    prior_method = dict(zip(prior.source_group, prior.assignment_method))
    assign, records = {}, []
    rows = new_rows.copy()
    rows["device"] = rows.filepath.map(lambda p: (rx.search(Path(p).name) or [None, "unknown"])[1])
    groups = (rows.groupby("source_group")
              .agg(duration=("duration_sec", "sum"), cat=("canonical_category", "first"), device=("device", "first"))
              .reset_index())
    groups = groups[~groups.source_group.isin(existing)]
    for (cat, dev), g in groups.groupby(["cat", "device"]):
        total = g.duration.sum()
        have = {s: 0.0 for s in ratios}
        known = g[g.source_group.isin(prior_split)]
        for r in known.itertuples(index=False):              # frozen: keep split, count its duration
            s = prior_split[r.source_group]
            assign[r.source_group] = s
            have[s] += r.duration
            records.append(dict(source_group=r.source_group, canonical_category=cat, stratum_device=dev,
                                duration_sec=round(r.duration, 4), split=s,
                                assignment_method=prior_method[r.source_group]))
        g = g[~g.source_group.isin(prior_split)]
        g = g.assign(h=g.source_group.map(lambda x: stable_hash(seed, x)))
        for _, r in g.sort_values(["duration", "h"], ascending=[False, True]).iterrows():
            s = max(ratios, key=lambda s: (ratios[s] * total - have[s], -stable_hash(seed, r.source_group, s)))
            assign[r.source_group] = s
            have[s] += r.duration
            records.append(dict(source_group=r.source_group, canonical_category=cat, stratum_device=dev,
                                duration_sec=round(r.duration, 4), split=s,
                                assignment_method=("split_extension_append_abs_deficit" if len(prior)
                                                   else "split_extension_greedy_abs_deficit")))
    return assign, pd.DataFrame(records)


def slice_offset(filepath, source, cfg):
    if source not in cfg["sources"]["sliced_long_synthetic_sources"]:
        return np.nan
    m = re.search(r"_(\d{3})\.wav$", filepath)
    return int(m.group(1)) * cfg["sources"]["sliced_clip_len_sec"] if m else np.nan


LABEL_OVERRIDE_COLUMNS = ["file_id", "new_label", "new_category", "reviewer", "reviewed_at", "note"]


def apply_label_overrides(inv, cfg):
    """Recording-level label corrections from review/recording_label_overrides.csv.

    Applied after split assignment (which uses manifest metadata), so a relabel
    never moves a recording or anything else between splits. The manifest values
    are kept in manifest_label/manifest_category. A relabelled recording is
    windowed from its human label (annotation_mode=human_label), not from ASR.
    """
    p = cfg["_review"] / "recording_label_overrides.csv"
    if not p.exists():
        pd.DataFrame(columns=LABEL_OVERRIDE_COLUMNS).to_csv(p, index=False)
    ov = pd.read_csv(p, dtype=str).fillna("")
    inv = inv.copy()
    inv["manifest_label"], inv["manifest_category"] = inv["label"], inv["category"]
    inv["label_origin"], inv["label_note"] = "manifest", ""
    idx = inv.set_index("file_id").index
    for o in ov.itertuples(index=False):
        if o.file_id not in idx:
            raise ValueError(f"recording_label_overrides: unknown file_id {o.file_id}")
        if o.new_category not in CATEGORY_MAP or o.new_label not in ("positive", "negative") or \
                (o.new_label == "positive") != (o.new_category == "positives"):
            raise ValueError(f"recording_label_overrides: inconsistent label/category for {o.file_id}")
        m = inv.file_id == o.file_id
        inv.loc[m, ["label", "category", "canonical_category"]] = [o.new_label, o.new_category,
                                                                   CATEGORY_MAP[o.new_category]]
        inv.loc[m, ["annotation_mode", "label_origin"]] = ["human_label", "human_relabel"]
        inv.loc[m, "label_note"] = f"{o.reviewer} {o.reviewed_at}: {o.note}"
    return inv


def stage_inventory(cfg):
    root = cfg["_root"]
    man = pd.read_csv(cfg["_manifest"])
    norm = pd.read_csv(cfg["_normalized_manifest"])
    frozen = pd.read_csv(cfg["_split_assignments"])
    frozen_map = dict(zip(frozen.source_group, frozen.split))

    parent_rec = man.set_index("filepath")["recording_id"].to_dict()
    man["source_group"] = man["parent_filepath"].map(parent_rec).fillna(man["recording_id"])
    man["canonical_category"] = man["category"].map(CATEGORY_MAP)
    if man.canonical_category.isna().any():
        raise ValueError(f"Unmapped categories: {man[man.canonical_category.isna()].category.unique()}")

    # Inherit split for rows known to the frozen split deliverable.
    known = norm.set_index("filepath")[["split", "source_group"]]
    man = man.join(known, on="filepath", rsuffix="_frozen")
    mismatch = man.source_group_frozen.notna() & (man.source_group_frozen != man.source_group)
    if mismatch.any():
        raise AssertionError(f"source_group disagrees with dataset_split for {mismatch.sum()} rows")
    man["split_origin"] = np.where(man.split.notna(), "dataset_split", "")
    # Rows unknown to the split but whose group is known inherit the group's split.
    inherit = man.split.isna() & man.source_group.isin(frozen_map)
    man.loc[inherit, "split"] = man.loc[inherit, "source_group"].map(frozen_map)
    man.loc[inherit, "split_origin"] = "dataset_split_group"
    new = man[man.split.isna()]
    ext_df = pd.DataFrame(columns=["source_group", "canonical_category", "stratum_device", "duration_sec", "split", "assignment_method"])
    ext_path = cfg["_out"] / "split_extension.csv"
    prior = pd.read_csv(ext_path) if ext_path.exists() else None
    if len(new):
        ext_assign, ext_df = extend_split(new, set(frozen_map), cfg, prior)
        if prior is not None:                                # append-only: no earlier group may move
            moved = {g: (s, ext_assign.get(g)) for g, s in zip(prior.source_group, prior.split)
                     if g in ext_assign and ext_assign[g] != s}
            if moved:
                raise AssertionError(f"split extension would move frozen groups: {moved}")
        idx = man.split.isna()
        man.loc[idx, "split"] = man.loc[idx, "source_group"].map(ext_assign)
        man.loc[idx, "split_origin"] = "split_extension"
    ext_df.to_csv(ext_path, index=False)
    man["is_eval_only"] = False

    rows = [man]
    if cfg["sources"]["include_streaming_eval"]:
        se = pd.read_csv(cfg["_streaming_eval_manifest"])
        se = se.rename(columns={"wakeword_metadata": "gt_insertions"})
        se["source_group"] = se["recording_id"]
        se["canonical_category"] = "streaming_eval"
        se["split"] = cfg["sources"]["streaming_eval_split"]
        se["split_origin"] = "streaming_eval_manifest"
        se["is_eval_only"] = True
        rows.append(se)
    inv = pd.concat(rows, ignore_index=True, sort=False)

    src = cfg["sources"]
    def mode(r):
        if r.canonical_category == "streaming_eval":
            return "gt_insertions"
        if r.source in src["real_recording_sources"]:
            return "asr"
        if r.source in src["tts_positive_sources"] and r.label == "positive":
            return "gt_whole_clip"
        return "generation_label"
    inv["annotation_mode"] = inv.apply(mode, axis=1)
    if not src["include_generation_label_clips"]:
        inv = inv[inv.annotation_mode != "generation_label"]
    inv["file_id"] = inv.filepath.map(lambda p: Path(p).stem)
    if inv.file_id.duplicated().any():
        raise AssertionError(f"Non-unique file stems: {inv[inv.file_id.duplicated()].file_id.tolist()[:5]}")
    inv = apply_label_overrides(inv, cfg)
    inv["recording_is_positive"] = inv.label.eq("positive")
    inv["source_recording_offset_sec"] = [slice_offset(f, s, cfg) for f, s in zip(inv.filepath, inv.source)]

    durs, srs = [], []
    for fp in inv.filepath:
        info = sf.info(str(root / fp))
        durs.append(info.frames / info.samplerate)
        srs.append(info.samplerate)
    inv["manifest_duration_sec"] = inv["duration_sec"]
    inv["manifest_sample_rate"] = inv["sample_rate"]
    inv["duration_sec"] = durs
    inv["sample_rate"] = srs

    cols = ["file_id", "filepath", "recording_id", "source_id", "source_group", "speaker_id", "split",
            "split_origin", "is_eval_only", "label", "category", "canonical_category", "source",
            "label_origin", "manifest_label", "manifest_category", "label_note",
            "annotation_mode", "recording_is_positive", "duration_sec", "manifest_duration_sec",
            "sample_rate", "manifest_sample_rate", "parent_filepath", "source_recording_offset_sec", "gt_insertions"]
    inv = inv[cols].sort_values("file_id").reset_index(drop=True)
    inv.to_csv(cfg["_out"] / "recording_inventory.csv", index=False)
    print(f"[inventory] {len(inv)} files; modes: {inv.annotation_mode.value_counts().to_dict()}")
    print(f"[inventory] split extension assigned {len(ext_df)} new source groups")
    return inv


# ----------------------------------------------------------------------------
# Stage 2: annotation
# ----------------------------------------------------------------------------
class Asr:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None

    def _load(self):
        if self.model is not None:
            return
        import os, subprocess
        a = self.cfg["asr"]
        device = a["device"]
        if device == "auto":
            device = "cpu"
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                                               "--format=csv,noheader,nounits"]).decode()
                best = max((tuple(map(int, l.split(","))) for l in out.strip().splitlines()), key=lambda r: r[1])
                if "CUDA_VISIBLE_DEVICES" not in os.environ:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(best[0])
                device = "cuda"
            except Exception:
                pass
        from faster_whisper import WhisperModel
        ct = a["compute_type_cuda"] if device == "cuda" else a["compute_type_cpu"]
        self.model = WhisperModel(a["model"], device=device, compute_type=ct)
        print(f"[asr] faster-whisper {a['model']} on {device} ({ct})")

    def run(self, path):
        """VAD-chunk the file and transcribe each chunk; cached by config hash."""
        a = self.cfg["asr"]
        cache_dir = self.cfg["_out"] / "asr_raw"
        cache_dir.mkdir(exist_ok=True)
        key = config_hash(self.cfg, "asr")
        cache = cache_dir / f"{Path(path).stem}.json"
        if cache.exists():
            d = json.loads(cache.read_text())
            if d.get("asr_config_hash") == key:
                return d
        self._load()
        from faster_whisper.vad import get_speech_timestamps, VadOptions
        y, sr = read_mono(path)
        y16 = to_16k(y, sr, a["input_sample_rate"])
        peak = float(np.abs(y16).max()) if len(y16) else 0.0
        if a["peak_normalize_to"] and peak > 0:
            y16 = y16 / peak * a["peak_normalize_to"]
        fs = a["input_sample_rate"]
        if len(y16) / fs <= a["whole_file_max_sec"]:
            spans = [dict(start=0, end=len(y16))]
        else:
            spans = get_speech_timestamps(y16, VadOptions(**a["vad"]))
        chunks = []
        for sp in spans:
            seg = y16[sp["start"]:sp["end"]]
            c0 = sp["start"] / fs
            segs, _ = self.model.transcribe(
                seg, language=a["language"], beam_size=a["beam_size"],
                word_timestamps=a["word_timestamps"], vad_filter=False,
                condition_on_previous_text=a["condition_on_previous_text"],
                initial_prompt=a["initial_prompt"], hotwords=a["hotwords"])
            words, texts = [], []
            for s in segs:
                texts.append(s.text.strip())
                for w in (s.words or []):
                    words.append(dict(word=w.word.strip(), start=round(c0 + w.start, 3),
                                      end=round(c0 + w.end, 3), probability=round(float(w.probability), 4)))
            chunks.append(dict(chunk_start=round(c0, 3), chunk_end=round(sp["end"] / fs, 3),
                               text=" ".join(texts).strip(), words=words))
        d = dict(filepath=str(path), asr_config_hash=key, created_at=now_iso(),
                 duration_sec=len(y) / sr, input_peak=peak, chunks=chunks)
        cache.write_text(json.dumps(d, indent=1))
        return d


def energy_trim(path, frame_sec, top_db):
    y, sr = read_mono(path)
    n = max(1, int(round(frame_sec * sr)))
    frames = len(y) // n
    if frames == 0:
        return 0.0, len(y) / sr
    rms = np.sqrt((y[:frames * n].reshape(frames, n) ** 2).mean(axis=1) + 1e-20)
    db = 20 * np.log10(rms / rms.max())
    idx = np.where(db > -top_db)[0]
    s = idx[0] * n / sr
    e = min(len(y), (idx[-1] + 1) * n) / sr
    return round(s, 4), round(e, 4)


def frame_db(y, sr, frame_sec):
    n = max(1, int(round(frame_sec * sr)))
    k = len(y) // n
    rms = np.sqrt((y[:k * n].reshape(k, n) ** 2).mean(axis=1) + 1e-20)
    return 20 * np.log10(rms)


def refine_bounds(db, s, e, R, hi_sec):
    """Snap ASR word bounds to the acoustic onset/offset nearby.

    Active frames exceed max(span_peak - rel_db, local_floor + floor_margin_db),
    where local_floor is a low percentile of frames within local_context_sec of
    the span (a global floor underestimates room background during speech).
    End: from the last active frame inside [s, e], walk forward until
    min_silence_sec of inactivity (at most max_end_extension_sec past e).
    Start: first frame in [s - 0.05, min(e, s + max_start_shift_sec)] that begins
    a run of >= min_onset_run_sec active frames (skips recording-start clicks).
    Returns None when no frame inside the span is active (caller keeps raw ASR).
    """
    fs = R["frame_sec"]
    i0, i1 = max(0, int(s / fs)), min(len(db), int(math.ceil(e / fs)))
    if i1 <= i0:
        return None
    c0 = max(0, int((s - R["local_context_sec"]) / fs))
    c1 = min(len(db), int((e + R["local_context_sec"]) / fs))
    floor = np.percentile(db[c0:c1], R["local_floor_percentile"])
    thr = max(db[i0:i1].max() - R["rel_db"], floor + R["floor_margin_db"])
    act = db > thr
    inside = np.where(act[i0:i1])[0]
    if not len(inside):
        return None
    j = last = i0 + inside[-1]
    sil, lim = 0, min(len(db), int((e + R["max_end_extension_sec"]) / fs), int(hi_sec / fs))
    while j + 1 < lim:
        j += 1
        if act[j]:
            last, sil = j, 0
        else:
            sil += 1
            if sil * fs >= R["min_silence_sec"]:
                break
    a, b = max(0, int((s - 0.05) / fs)), min(i1, int((s + R["max_start_shift_sec"]) / fs))
    run = max(1, int(round(R["min_onset_run_sec"] / fs)))
    onsets = [i for i in range(a, b) if act[i:i + run].all()]
    st = onsets[0] * fs if onsets else s
    return round(st, 3), round((last + 1) * fs, 3)


def _ann_row(r, **kw):
    base = dict(file_id=r.file_id, filepath=r.filepath, recording_id=r.recording_id,
                source_id=r.source_id, source_group=r.source_group, split=r.split)
    base.update(kw)
    return base


def annotate_gt(r, cfg):
    rows = []
    if r.annotation_mode == "gt_insertions":
        ins = json.loads(r.gt_insertions) if isinstance(r.gt_insertions, str) else []
        for k, m in enumerate(sorted(ins, key=lambda m: m["start_sec"])):
            rows.append(_ann_row(
                r, annotation_id=f"{r.file_id}#gt{k:03d}", origin=ORIGIN_GT,
                gt_source=f"streaming_eval_manifest.wakeword_metadata[{k}] voice={m.get('voice', '')}",
                match_type=MATCH_COMPLETE, status=ST_GT, review_reasons="",
                start_sec=m["start_sec"], end_sec=m["end_sec"],
                raw_start_sec=m["start_sec"], raw_end_sec=m["end_sec"], boundary_method="synthetic_insertion"))
    elif r.annotation_mode == "gt_whole_clip":
        t = cfg["synthetic_ground_truth"]["tts_positive_whole_clip"]
        s, e = energy_trim(cfg["_root"] / r.filepath, t["frame_sec"], t["top_db"])
        rows.append(_ann_row(
            r, annotation_id=f"{r.file_id}#gt000", origin=ORIGIN_GT,
            gt_source=f"tts_positive_whole_clip energy_trim(top_db={t['top_db']}) clip=[0,{r.duration_sec:.4f}]",
            match_type=MATCH_COMPLETE, status=ST_GT, review_reasons="",
            start_sec=s, end_sec=e, raw_start_sec=0.0, raw_end_sec=r.duration_sec,
            boundary_method="synthetic_whole_clip_energy_trim"))
    return rows


def refine_detection(d, ch, db, cfg):
    if d["match_type"] == MATCH_UNMATCHED:
        return d["s"], d["e"], "vad_chunk"
    ref = refine_bounds(db, d["s"], d["e"], cfg["asr"]["boundary_refinement"], ch["chunk_end"])
    if ref is None or ref[1] <= ref[0]:
        return d["s"], d["e"], "asr_raw"
    return ref[0], ref[1], "asr_energy_refined"


def annotate_asr(r, asr_result, cfg):
    rows = []
    k = 0
    y, sr = read_mono(cfg["_root"] / r.filepath)
    db = frame_db(y, sr, cfg["asr"]["boundary_refinement"]["frame_sec"])
    for ch in asr_result["chunks"]:
        for d in detect_in_chunk(ch, cfg, bool(r.recording_is_positive)):
            s, e, method = refine_detection(d, ch, db, cfg)
            if d["match_type"] == MATCH_COMPLETE:
                status = ST_NEEDS_REVIEW if d["reasons"] else ST_CANDIDATE
            elif d["match_type"] == MATCH_PARTIAL:
                status = ST_NEEDS_REVIEW if r.recording_is_positive else ST_CANDIDATE
            else:
                status = ST_NEEDS_REVIEW
            probs = [w["probability"] for w in d["words"]]
            rows.append(_ann_row(
                r, annotation_id=f"{r.file_id}#asr{k:03d}", origin=ORIGIN_ASR, gt_source="",
                match_type=d["match_type"], status=status, review_reasons=";".join(d["reasons"]),
                start_sec=round(s, 3), end_sec=round(e, 3),
                raw_start_sec=d["rs"], raw_end_sec=d["re"], boundary_method=method,
                vad_chunk_start_sec=ch["chunk_start"], vad_chunk_end_sec=ch["chunk_end"],
                asr_chunk_text=ch["text"], asr_words=json.dumps(d["words"]),
                asr_min_word_prob=min(probs) if probs else np.nan,
                edit_ratio=round(d["edit_ratio"], 4)))
            k += 1
    return rows


def crosscheck(r, gt_rows, asr_result, cfg):
    """ASR-vs-GT agreement for synthetic recordings. Never used as labels."""
    y, sr = read_mono(cfg["_root"] / r.filepath)
    db = frame_db(y, sr, cfg["asr"]["boundary_refinement"]["frame_sec"])
    dets = []
    for ch in asr_result["chunks"]:
        for d in detect_in_chunk(ch, cfg, True):
            if d["match_type"] in (MATCH_COMPLETE, MATCH_FUZZY):
                d["rs2"], d["re2"], d["method"] = refine_detection(d, ch, db, cfg)
                dets.append(d)
    out = []
    for g in gt_rows:
        best, best_iou = None, 0.0
        for d in dets:
            inter = max(0.0, min(g["end_sec"], d["e"]) - max(g["start_sec"], d["s"]))
            union = max(g["end_sec"], d["e"]) - min(g["start_sec"], d["s"])
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best, best_iou = d, iou
        out.append(dict(annotation_id=g["annotation_id"], file_id=r.file_id, annotation_mode=r.annotation_mode,
                        gt_start_sec=g["start_sec"], gt_end_sec=g["end_sec"],
                        asr_match_type=best["match_type"] if best else "none",
                        asr_start_sec=best["s"] if best else np.nan, asr_end_sec=best["e"] if best else np.nan,
                        iou=round(best_iou, 4),
                        start_error_sec=round(best["s"] - g["start_sec"], 4) if best else np.nan,
                        end_error_sec=round(best["e"] - g["end_sec"], 4) if best else np.nan,
                        refined_method=best["method"] if best else "",
                        refined_start_error_sec=round(best["rs2"] - g["start_sec"], 4) if best else np.nan,
                        refined_end_error_sec=round(best["re2"] - g["end_sec"], 4) if best else np.nan,
                        used_for_labels=False))
    unmatched = sum(1 for d in dets
                    if not any(overlaps(d["s"], d["e"], g["start_sec"], g["end_sec"]) for g in gt_rows))
    return out, unmatched


def stage_annotate(cfg):
    inv = pd.read_csv(cfg["_out"] / "recording_inventory.csv")
    asr = Asr(cfg)
    ann, xc, rec_flags = [], [], []
    for r in inv.itertuples(index=False):
        if r.annotation_mode in ("gt_insertions", "gt_whole_clip"):
            g = annotate_gt(r, cfg)
            ann += g
            if cfg["asr"]["run_crosscheck_on_synthetic_ground_truth"]:
                res = asr.run(cfg["_root"] / r.filepath)
                rows, extra = crosscheck(r, g, res, cfg)
                xc += rows
                if extra:
                    xc.append(dict(annotation_id="", file_id=r.file_id, annotation_mode=r.annotation_mode,
                                   asr_match_type=f"{extra}_asr_detection(s)_without_gt", used_for_labels=False))
        elif r.annotation_mode == "asr":
            res = asr.run(cfg["_root"] / r.filepath)
            a = annotate_asr(r, res, cfg)
            ann += a
            n_complete = sum(1 for x in a if x["match_type"] == MATCH_COMPLETE)
            flag = ""
            if (r.recording_is_positive and n_complete == 0
                    and cfg["annotation_policy"]["flag_positive_recordings_without_complete_detection"]):
                flag = "positive_recording_without_complete_detection"
            rec_flags.append(dict(file_id=r.file_id, n_vad_chunks=len(res["chunks"]),
                                  n_complete=n_complete,
                                  n_needs_review=sum(1 for x in a if x["status"] == ST_NEEDS_REVIEW),
                                  recording_flag=flag))
    df = pd.DataFrame(ann).reindex(columns=ANNOTATION_COLUMNS)
    df.to_csv(cfg["_out"] / "candidate_annotations.csv", index=False)
    xcd = pd.DataFrame(xc)
    xcd.to_csv(cfg["_out"] / "asr_crosscheck_synthetic.csv", index=False)
    write_calibration(xcd, cfg)
    pd.DataFrame(rec_flags).to_csv(cfg["_out"] / "recording_flags.csv", index=False)
    print(f"[annotate] {len(df)} annotations: {df.groupby(['origin', 'match_type', 'status']).size().to_dict()}")
    return df


def write_calibration(xcd, cfg):
    """Boundary-error summary on clean, energy-trimmed TTS positives (the only
    synthetic GT whose span is speech-tight). Containment needs
    pre >= max(asr_start - gt_start) and post >= max(gt_end - asr_end)."""
    if xcd.empty:
        return
    c = xcd[(xcd.annotation_mode == "gt_whole_clip") & (xcd.asr_match_type == MATCH_COMPLETE)]
    # Speech-tight means the energy trim found silence at BOTH ends. Some Piper voices added 2026-09-29
    # (VCTK in particular) carry a constant noise floor, so the trim keeps the whole clip: such a span is a
    # safe label (a window must contain all of it) but not a timing reference, and one of them would inflate
    # the required tolerance to over 1 s.
    dur = pd.read_csv(cfg["_out"] / "recording_inventory.csv", usecols=["file_id", "duration_sec"],
                      low_memory=False).set_index("file_id").duration_sec
    tight = (c.gt_start_sec > 0) & (c.gt_end_sec < c.file_id.map(dur) - 0.005)
    # The 2026-09-29 voices (tts2_*) are training data only, not a calibration reference: 329 of their 598
    # ASR-matched clips cannot be trimmed at all, so even their trimmed spans are not known to be tight.
    # Calibration stays on the phase-1 TTS positives, whose spans were checked when this rule was set.
    tight &= ~c.file_id.str.startswith("tts2_")
    n_loose = int((~tight).sum())
    c = c[tight]
    s_all = xcd[xcd.annotation_id.notna() & (xcd.annotation_id != "")]
    out = dict(reference="gt_whole_clip (energy-trimmed TTS positives)", n=int(len(c)),
               excluded_not_speech_tight=n_loose,
               asr_recall_exact={m: f"{int((g.asr_match_type == MATCH_COMPLETE).sum())}/{len(g)}"
                                 for m, g in s_all.groupby("annotation_mode")})
    for name, cs, ce, sub in [("asr_raw", "start_error_sec", "end_error_sec", c),
                              ("asr_energy_refined", "refined_start_error_sec", "refined_end_error_sec",
                               c[c.refined_method == "asr_energy_refined"])]:
        if len(sub):
            out[name] = dict(n=int(len(sub)),
                             required_pre_sec=round(float(max(0, sub[cs].max())), 4),
                             required_post_sec=round(float(max(0, -sub[ce].min())), 4),
                             start_error_quantiles={q: round(float(sub[cs].quantile(q)), 4) for q in (0.05, 0.5, 0.95)},
                             end_error_quantiles={q: round(float(sub[ce].quantile(q)), 4) for q in (0.05, 0.5, 0.95)})
    (cfg["_out"] / "asr_boundary_calibration.json").write_text(json.dumps(out, indent=2))


# ----------------------------------------------------------------------------
# Stage 3: apply review overrides
# ----------------------------------------------------------------------------
OVERRIDE_COLUMNS = ["override_id", "action", "annotation_id", "file_id", "new_start_sec", "new_end_sec",
                    "new_match_type", "reviewer", "reviewed_at", "note"]
ACTIONS = {"confirm_complete", "correct_bounds", "mark_partial", "mark_not_wakeword", "remove",
           "add", "clip_level_positive", "clear_recording_flag"}


def overrides_path(cfg):
    p = cfg["_review"] / "review_overrides.csv"
    if not p.exists():
        pd.DataFrame(columns=OVERRIDE_COLUMNS).to_csv(p, index=False)
    return p


def finalize_annotations(ann, cfg):
    """label_source, positive eligibility and boundary tolerance for every annotation."""
    ann = ann.copy()
    def label_source(a):
        if a.origin == ORIGIN_GT:
            return "synthetic_ground_truth"
        if a.origin == ORIGIN_HUMAN:
            return "human"
        return "asr_candidate_reviewed" if a.status in (ST_REVIEWED, ST_REMOVED) else "asr_candidate_unreviewed"
    ann["label_source"] = ann.apply(label_source, axis=1)
    policy = cfg["annotation_policy"]["asr_unreviewed_positive_policy"]
    ok_status = ann.status.isin([ST_GT, ST_REVIEWED, ST_CANDIDATE])
    ok_origin = ann.label_source.ne("asr_candidate_unreviewed") | (policy == "allow_flagged")
    ann["positive_eligible"] = ann.match_type.eq(MATCH_COMPLETE) & ok_status & ok_origin
    tol = cfg["annotation_policy"]["boundary_tolerance_sec"]
    ann["bounds_verified"] = ann.bounds_verified.astype(bool)
    key = np.where(ann.bounds_verified, "verified_bounds",
                   np.where(ann.boundary_method.eq("asr_energy_refined"), "asr_energy_refined", "asr_raw"))
    ann["tol_pre_sec"] = [tol[k]["pre"] for k in key]
    ann["tol_post_sec"] = [tol[k]["post"] for k in key]
    return ann


def stage_resolve(cfg):
    ann = pd.read_csv(cfg["_out"] / "candidate_annotations.csv", keep_default_na=True)
    inv = pd.read_csv(cfg["_out"] / "recording_inventory.csv").set_index("file_id")
    flags = pd.read_csv(cfg["_out"] / "recording_flags.csv").set_index("file_id")
    ov = pd.read_csv(overrides_path(cfg), dtype=str).fillna("")
    for c in ["reviewer", "reviewed_at", "review_note", "review_action"]:
        ann[c] = ""
    # Only GT, human-drawn and human-corrected bounds are "verified"; ASR bounds
    # keep the calibrated asymmetric tolerance even after a confirm_complete.
    ann["bounds_verified"] = ann.origin.eq(ORIGIN_GT)
    ann["review_reasons"] = ann["review_reasons"].fillna("")
    ann = ann.set_index("annotation_id", drop=False)
    errors, applied, cleared_flags = [], 0, set()
    for o in ov.itertuples(index=False):
        act = o.action.strip()
        tag = f"override {o.override_id or '?'} ({act})"
        if act not in ACTIONS:
            errors.append(f"{tag}: unknown action"); continue
        if act == "clear_recording_flag":
            if o.file_id not in inv.index:
                errors.append(f"{tag}: unknown file_id {o.file_id}"); continue
            cleared_flags.add(o.file_id); applied += 1; continue
        if act in ("add", "clip_level_positive"):
            if o.file_id not in inv.index:
                errors.append(f"{tag}: unknown file_id {o.file_id}"); continue
            r = inv.loc[o.file_id]
            if act == "clip_level_positive":
                # Reviewer asserts the whole file holds exactly one complete wakeword.
                # [0, duration] is a containing span, not a tight one (see partial_negative).
                s, e, method = 0.0, float(r.duration_sec), "human_clip_extent"
            else:
                try:
                    s, e = float(o.new_start_sec), float(o.new_end_sec)
                except ValueError:
                    errors.append(f"{tag}: add requires numeric new_start_sec/new_end_sec"); continue
                method = "human"
            mt = MATCH_COMPLETE if act == "clip_level_positive" else (o.new_match_type or MATCH_COMPLETE)
            if mt not in MATCH_TYPES:
                errors.append(f"{tag}: bad new_match_type {mt}"); continue
            aid = f"{o.file_id}#human{o.override_id or len(ann)}"
            ann.loc[aid] = pd.Series(dict(
                annotation_id=aid, file_id=o.file_id, filepath=r.filepath, recording_id=r.recording_id,
                source_id=r.source_id, source_group=r.source_group, split=r.split, origin=ORIGIN_HUMAN,
                gt_source="", match_type=mt, status=ST_REVIEWED, review_reasons="",
                start_sec=s, end_sec=e, raw_start_sec=s, raw_end_sec=e, boundary_method=method,
                bounds_verified=True, reviewer=o.reviewer,
                reviewed_at=o.reviewed_at, review_note=o.note, review_action=act))
            applied += 1
            continue
        if o.annotation_id not in ann.index:
            errors.append(f"{tag}: annotation_id {o.annotation_id!r} not found (orphaned override)"); continue
        a = ann.loc[o.annotation_id]
        if a.origin == ORIGIN_GT and act != "remove":
            errors.append(f"{tag}: synthetic ground truth can only be removed, not '{act}'"); continue
        if act in ("confirm_complete", "correct_bounds") and (o.new_start_sec or o.new_end_sec):
            try:
                ann.at[o.annotation_id, "start_sec"] = float(o.new_start_sec or a.start_sec)
                ann.at[o.annotation_id, "end_sec"] = float(o.new_end_sec or a.end_sec)
                ann.at[o.annotation_id, "bounds_verified"] = True
                ann.at[o.annotation_id, "boundary_method"] = "human_corrected"
            except ValueError:
                errors.append(f"{tag}: non-numeric bounds"); continue
        if act == "confirm_complete":
            ann.at[o.annotation_id, "match_type"] = MATCH_COMPLETE
        elif act == "mark_partial":
            ann.at[o.annotation_id, "match_type"] = MATCH_PARTIAL
        elif act == "mark_not_wakeword":
            ann.at[o.annotation_id, "match_type"] = MATCH_NOT_WW
        ann.at[o.annotation_id, "status"] = ST_REMOVED if act == "remove" else ST_REVIEWED
        ann.at[o.annotation_id, "reviewer"] = o.reviewer
        ann.at[o.annotation_id, "reviewed_at"] = o.reviewed_at
        ann.at[o.annotation_id, "review_note"] = o.note
        ann.at[o.annotation_id, "review_action"] = act
        applied += 1

    ann = finalize_annotations(ann.reset_index(drop=True), cfg)

    rf = flags["recording_flag"].fillna("") if len(flags) else pd.Series(dtype=str)
    rec = pd.DataFrame({"file_id": inv.index})
    rec["recording_flag"] = rec.file_id.map(rf).fillna("")
    rec["flag_cleared_by_review"] = rec.file_id.isin(cleared_flags)
    rec.to_csv(cfg["_out"] / "recording_review_state.csv", index=False)

    ann.to_csv(cfg["_out"] / "resolved_annotations.csv", index=False)
    (cfg["_out"] / "override_errors.txt").write_text("\n".join(errors) + ("\n" if errors else ""))
    print(f"[resolve] applied {applied} overrides, {len(errors)} errors; "
          f"positive_eligible={int(ann.positive_eligible.sum())}")
    if errors:
        print("  " + "\n  ".join(errors[:10]))
    return ann


# ----------------------------------------------------------------------------
# Stage 4: window generation
# ----------------------------------------------------------------------------
def overlaps(a0, a1, b0, b1):
    return min(a1, b1) - max(a0, b0) > 1e-9


def subtract(intervals, a, b):
    """[a,b] minus union(intervals) -> list of free intervals."""
    free, cur = [], a
    for s, e in sorted(intervals):
        if e <= cur or s >= b:
            continue
        if s > cur:
            free.append((cur, min(s, b)))
        cur = max(cur, e)
    if cur < b:
        free.append((cur, b))
    return free


def window_record(r, start, W, label, wtype, label_source, cfg, ann=None, coverage=np.nan, note=""):
    D = r.duration_sec
    start = round(start, 4)
    end = round(start + W, 4)
    rec = dict(
        window_id=f"{r.file_id}@{int(round(start * 1000))}ms_{label[0]}",
        split=r.split, is_eval_only=bool(r.is_eval_only), recording_id=r.recording_id,
        source_id=r.source_id, source_group=r.source_group, speaker_id=r.speaker_id,
        file_id=r.file_id, filepath=r.filepath, parent_category=r.category, parent_source=r.source,
        recording_duration_sec=round(D, 4), start_sec=start, end_sec=end,
        src_start_sec=round(max(0.0, start), 4), src_end_sec=round(min(D, end), 4),
        pad_left_sec=round(max(0.0, -start), 4), pad_right_sec=round(max(0.0, end - D), 4),
        source_recording_offset_sec=r.source_recording_offset_sec,
        label=label, window_type=wtype, label_source=label_source,
        annotation_id=ann.annotation_id if ann is not None else "",
        wakeword_start_in_window_sec=round(ann.start_sec - start, 4) if ann is not None else np.nan,
        wakeword_end_in_window_sec=round(ann.end_sec - start, 4) if ann is not None else np.nan,
        wakeword_coverage=coverage, note=note)
    return rec


def positive_windows(r, a, others, cfg, rejected):
    P, W = cfg["positive"], cfg["window"]["duration_sec"]
    maxpad = cfg["window"]["max_pad_fraction"] * W
    tol = f"pre={a.tol_pre_sec},post={a.tol_post_sec}"
    s, e = a.start_sec - a.tol_pre_sec, a.end_sec + a.tol_post_sec
    if (e - s) + P["min_pre_context_sec"] + P["min_post_context_sec"] > W + 1e-9:
        rejected.append(dict(file_id=r.file_id, annotation_id=a.annotation_id, reason="occurrence_too_long_for_window",
                             detail=f"span+tol={e - s:.3f}s"))
        return []
    out, starts = [], []
    for post in P["post_roll_offsets_sec"]:
        if len(out) >= P["max_windows_per_occurrence"]:
            break
        if post < P["min_post_context_sec"]:
            continue
        start = e + post - W
        why = None
        if s - start < P["min_pre_context_sec"] - 1e-9:
            why = "insufficient_pre_context"
        elif max(0, -start) + max(0, start + W - r.duration_sec) > maxpad + 1e-9:
            why = "padding_exceeds_max_pad_fraction"
        elif any(abs(start - x) < P["min_shift_between_windows_sec"] - 1e-9 for x in starts):
            why = "near_duplicate"
        elif any(overlaps(start, start + W, o.start_sec - o.tol_pre_sec, o.end_sec + o.tol_post_sec)
                 for o in others):
            why = "overlaps_other_wakeword_like_annotation"
        if why:
            rejected.append(dict(file_id=r.file_id, annotation_id=a.annotation_id, reason=why,
                                 detail=f"post_roll={post}", window_start_sec=round(start, 4)))
            continue
        starts.append(start)
        out.append(window_record(r, start, W, "positive", "positive_wakeword", a.label_source, cfg, ann=a,
                                 coverage=1.0, note=f"end_anchored post_roll={post} tol={tol}"))
    if not out and P.get("fallback_alignment") == "center":
        # Span fits but no fixed post-roll leaves both margins: centre it once.
        start = s - (W - (e - s)) / 2
        pad = max(0, -start) + max(0, start + W - r.duration_sec)
        clash = any(overlaps(start, start + W, o.start_sec - o.tol_pre_sec, o.end_sec + o.tol_post_sec) for o in others)
        if pad <= maxpad + 1e-9 and not clash:
            out.append(window_record(r, start, W, "positive", "positive_wakeword", a.label_source, cfg, ann=a,
                                     coverage=1.0, note=f"centered_fallback tol={tol}"))
        else:
            rejected.append(dict(file_id=r.file_id, annotation_id=a.annotation_id,
                                 reason="centered_fallback_failed", window_start_sec=round(start, 4)))
    return out


def partial_windows(r, a, others, cfg, rejected):
    P, W = cfg["partial_negative"], cfg["window"]["duration_sec"]
    if not P["enabled"] or a.label_source not in P["allowed_label_sources"] or \
            (P["require_verified_bounds"] and not a.bounds_verified) or \
            a.boundary_method in P["exclude_boundary_methods"]:
        return []
    maxpad = cfg["window"]["max_pad_fraction"] * W
    L = a.end_sec - a.start_sec
    f = P["fragment_coverage"]
    out = []
    for pos in P["positions"]:
        start = (a.start_sec + f * L - W) if pos == "head" else (a.end_sec - f * L)
        end = start + W
        cov = max(0.0, min(end, a.end_sec) - max(start, a.start_sec)) / L
        why = None
        if cov > P["max_coverage_fraction"] + 1e-9 or cov <= 0:
            why = "fragment_coverage_out_of_range"
        elif max(0, -start) + max(0, end - r.duration_sec) > maxpad + 1e-9:
            why = "padding_exceeds_max_pad_fraction"
        elif any(overlaps(start, end, o.start_sec - o.tol_pre_sec, o.end_sec + o.tol_post_sec)
                 for o in others):
            why = "overlaps_other_wakeword_like_annotation"
        if why:
            rejected.append(dict(file_id=r.file_id, annotation_id=a.annotation_id, reason=f"partial_{pos}:{why}",
                                 window_start_sec=round(start, 4)))
            continue
        out.append(window_record(r, start, W, "negative", "negative_partial_wakeword",
                                 f"{a.label_source}_fragment", cfg, ann=a, coverage=round(cov, 4),
                                 note=f"{pos} fragment"))
    return out


def negative_windows(r, anns, chunks, cfg, blocked, rejected):
    N, W = cfg["negative"], cfg["window"]["duration_sec"]
    D = r.duration_sec
    if blocked:
        rejected.append(dict(file_id=r.file_id, reason="negatives_blocked_unresolved_recording_flag"))
        return []
    forbid = []
    for a in anns.itertuples(index=False):
        if a.status == ST_REMOVED or a.match_type == MATCH_NOT_WW:
            continue
        forbid.append((a.start_sec - a.tol_pre_sec - N["guard_sec"],
                       a.end_sec + a.tol_post_sec + N["guard_sec"]))
    if r.annotation_mode == "asr" and r.recording_is_positive and N["speech_policy_in_positive_recordings"] == "non_speech_only":
        forbid += [(c0 - N["vad_chunk_guard_sec"], c1 + N["vad_chunk_guard_sec"]) for c0, c1 in chunks]

    if r.annotation_mode == "asr":
        wtype = "negative_non_speech_from_positive_recording" if r.recording_is_positive else "negative_real_background"
        lsrc = "asr_no_detection_unreviewed"
    elif r.annotation_mode == "gt_insertions":
        wtype, lsrc = "negative_stream_background", "synthetic_ground_truth_absence"
    elif r.annotation_mode == "human_label":
        wtype, lsrc = r.canonical_category, "human_recording_label"
    else:
        wtype, lsrc = r.canonical_category, "synthetic_generation_label"
    if wtype == "positive_wakeword":
        return []

    starts = []
    free = subtract(forbid, 0.0, D)
    if not forbid and D < W:
        if N["short_clip_policy"] == "single_centered_window":
            starts.append(-(W - D) / 2)
    else:
        for a0, a1 in free:
            # Round inward so 0.1 ms window rounding can never enter a guard zone.
            a0, a1 = math.ceil(a0 * 1e4 - 1e-6) / 1e4, math.floor(a1 * 1e4 + 1e-6) / 1e4
            R = a1 - a0
            if R < W:
                rejected.append(dict(file_id=r.file_id, reason="negative_region_shorter_than_window",
                                     detail=f"[{a0:.3f},{a1:.3f}]"))
                continue
            st = a0
            while st + W <= a1 + 1e-9:
                starts.append(st)
                st += N["hop_sec"]
            last_end = starts[-1] + W
            if a1 - last_end >= N["min_tail_fraction"] * W - 1e-9:
                starts.append(a1 - W)  # right-aligned tail inside the region, no padding
    if len(starts) > N["max_windows_per_recording"]:
        idx = np.linspace(0, len(starts) - 1, N["max_windows_per_recording"]).round().astype(int)
        rejected.append(dict(file_id=r.file_id, reason="negative_cap_subsampled",
                             detail=f"{len(starts)}->{len(set(idx))}"))
        starts = [starts[i] for i in sorted(set(idx))]
    note = "short_clip_centered" if (not forbid and D < W) else ""
    return [window_record(r, st, W, "negative", wtype, lsrc, cfg, note=note) for st in starts]


def stage_windows(cfg):
    inv = pd.read_csv(cfg["_out"] / "recording_inventory.csv")
    ann = pd.read_csv(cfg["_out"] / "resolved_annotations.csv")
    ann["review_reasons"] = ann["review_reasons"].fillna("")
    state = pd.read_csv(cfg["_out"] / "recording_review_state.csv").fillna("").set_index("file_id")
    by_file = {k: g for k, g in ann.groupby("file_id")}
    chunks_by_file = {}
    for fid in inv.loc[inv.annotation_mode == "asr", "file_id"]:
        d = json.loads((cfg["_out"] / "asr_raw" / f"{fid}.json").read_text())
        chunks_by_file[fid] = [(c["chunk_start"], c["chunk_end"]) for c in d["chunks"]]

    hold = cfg.get("hold_recordings", {})
    windows, rejected = [], []
    for r in inv.itertuples(index=False):
        if r.file_id in hold:
            # Held recordings yield no windows at all (neither positives nor negatives) until released.
            rejected.append(dict(file_id=r.file_id, reason="held_recording", detail=hold[r.file_id]))
            continue
        anns = by_file.get(r.file_id, ann.iloc[0:0])
        live = anns[(anns.status != ST_REMOVED) & (anns.match_type != MATCH_NOT_WW)]
        for a in live[live.positive_eligible].itertuples(index=False):
            others = [o for o in live.itertuples(index=False) if o.annotation_id != a.annotation_id]
            windows += positive_windows(r, a, others, cfg, rejected)
            windows += partial_windows(r, a, others, cfg, rejected)
        for a in live[~live.positive_eligible & live.match_type.eq(MATCH_COMPLETE)].itertuples(index=False):
            rejected.append(dict(file_id=r.file_id, annotation_id=a.annotation_id,
                                 reason=f"not_positive_eligible:{a.status}:{a.label_source}",
                                 detail=a.review_reasons))
        st = state.loc[r.file_id] if r.file_id in state.index else None
        blocked = bool(cfg["negative"]["block_recordings_with_unresolved_recording_flag"] and st is not None
                       and st.recording_flag and not str(st.flag_cleared_by_review) == "True")
        windows += negative_windows(r, anns, chunks_by_file.get(r.file_id, []), cfg, blocked, rejected)

    wdf = pd.DataFrame(windows)
    wdf["config_hash"] = config_hash(cfg)
    wdf.to_csv(cfg["_out"] / "windows.csv", index=False)
    pd.DataFrame(rejected).to_csv(cfg["_out"] / "rejected_windows.csv", index=False)
    summary = (wdf.groupby(["split", "label", "window_type", "label_source"]).size()
               .rename("windows").reset_index())
    summary.to_csv(cfg["_out"] / "window_summary.csv", index=False)
    print(f"[windows] {len(wdf)} windows, {len(rejected)} rejection records")
    print(wdf.groupby(["split", "label"]).size().to_string())
    return wdf


# ----------------------------------------------------------------------------
# Stage 5: review tool (static HTML, opened locally next to the audio)
# ----------------------------------------------------------------------------
def envelope(path, rate=100):
    y, sr = read_mono(path)
    n = max(1, sr // rate)
    k = len(y) // n
    if k == 0:
        return []
    env = np.abs(y[:k * n]).reshape(k, n).max(axis=1)
    env = env / (env.max() + 1e-9)
    return [round(float(v), 3) for v in env]


def stage_review_tool(cfg):
    import os
    inv = pd.read_csv(cfg["_out"] / "recording_inventory.csv")
    ann = pd.read_csv(cfg["_out"] / "resolved_annotations.csv").fillna("")
    state = pd.read_csv(cfg["_out"] / "recording_review_state.csv").fillna("").set_index("file_id")
    ov = pd.read_csv(overrides_path(cfg), dtype=str).fillna("")
    queue = ann[(ann.status != ST_REMOVED) &
                ((ann.status == ST_NEEDS_REVIEW) | (ann.label_source == "asr_candidate_unreviewed"))]
    queue = (queue.assign(_p=queue.status.ne(ST_NEEDS_REVIEW)).sort_values(["_p", "file_id", "start_sec"])
             .drop(columns="_p"))
    queue.to_csv(cfg["_review"] / "review_queue.csv", index=False)

    recs = []
    for r in inv[inv.annotation_mode == "asr"].itertuples(index=False):
        d = json.loads((cfg["_out"] / "asr_raw" / f"{r.file_id}.json").read_text())
        a = ann[ann.file_id == r.file_id]
        recs.append(dict(
            file_id=r.file_id, split=r.split, label=r.label, duration=round(r.duration_sec, 3),
            audio=os.path.relpath(cfg["_root"] / r.filepath, cfg["_review"]).replace(os.sep, "/"),
            flag=state.loc[r.file_id].recording_flag if r.file_id in state.index else "",
            env=envelope(cfg["_root"] / r.filepath),
            chunks=[dict(s=c["chunk_start"], e=c["chunk_end"], t=c["text"]) for c in d["chunks"]],
            anns=[dict(id=x.annotation_id, s=float(x.start_sec), e=float(x.end_sec), mt=x.match_type,
                       st=x.status, src=x.label_source, why=x.review_reasons, txt=x.asr_chunk_text)
                  for x in a.itertuples(index=False)]))
    recs.sort(key=lambda x: (-sum(1 for a in x["anns"] if a["st"] == ST_NEEDS_REVIEW) - (5 if x["flag"] else 0),
                             x["file_id"]))
    data = json.dumps(dict(recordings=recs, overrides=ov.to_dict("records"), generated_at=now_iso(),
                           config_hash=config_hash(cfg)))
    tpl = (HERE / "review_tool_template.html").read_text()
    (cfg["_review"] / "review_tool.html").write_text(tpl.replace("/*__DATA__*/null", data))
    print(f"[review-tool] {len(recs)} recordings, {len(queue)} queue items -> {cfg['_review'] / 'review_tool.html'}")


# ----------------------------------------------------------------------------
# Optional: materialize window audio
# ----------------------------------------------------------------------------
def stage_materialize(cfg, split=None, limit=None, out_dir=None):
    wdf = pd.read_csv(cfg["_out"] / "windows.csv")
    if split:
        wdf = wdf[wdf.split == split]
    if limit:
        wdf = wdf.groupby("label", group_keys=False).head(int(limit))
    out_dir = Path(out_dir) if out_dir else cfg["_out"] / "window_audio"
    target_sr = cfg["window"]["output_sample_rate"]
    cache = {}
    for w in wdf.itertuples(index=False):
        if w.filepath not in cache:
            cache = {w.filepath: read_mono(cfg["_root"] / w.filepath)}  # windows.csv is grouped by file
        y, sr = cache[w.filepath]
        a, b = int(round(w.src_start_sec * sr)), int(round(w.src_end_sec * sr))
        seg = np.concatenate([np.zeros(int(round(w.pad_left_sec * sr)), np.float32), y[a:b],
                              np.zeros(int(round(w.pad_right_sec * sr)), np.float32)])
        n = int(round(cfg["window"]["duration_sec"] * sr))
        seg = np.pad(seg, (0, max(0, n - len(seg))))[:n]
        if target_sr and target_sr != sr:
            seg, sr_out = to_16k(seg, sr, target_sr), target_sr  # generic polyphase resample
        else:
            sr_out = sr
        p = out_dir / w.split / w.label / f"{w.window_id}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), seg, sr_out)
    print(f"[materialize] wrote {len(wdf)} windows under {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["inventory", "annotate", "resolve", "windows", "review-tool",
                                      "materialize", "all"])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--split")
    ap.add_argument("--limit")
    ap.add_argument("--out-dir")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.stage in ("inventory", "all"):
        stage_inventory(cfg)
    if args.stage in ("annotate", "all"):
        stage_annotate(cfg)
    if args.stage in ("resolve", "all"):
        stage_resolve(cfg)
    if args.stage in ("windows", "all"):
        stage_windows(cfg)
    if args.stage in ("review-tool", "all"):
        stage_review_tool(cfg)
    if args.stage == "materialize":
        stage_materialize(cfg, args.split, args.limit, args.out_dir)


if __name__ == "__main__":
    main()
