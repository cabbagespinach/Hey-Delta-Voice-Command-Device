#!/usr/bin/env python3
"""
Source-confound diagnostic: does the model score *source* (real device vs synthetic)
or *loudness* instead of the wakeword?

Validation windows only (the test split and training data are not read).

1. Observational: real vs synthetic negatives, before and after matching on speech-band
   loudness (peak_db) and background floor (floor_db). "Source AUC" = how well the score
   separates real from synthetic negatives (0.5 = no source information).
2. Interventional: re-score the same windows after changing only their level:
   - synthetic windows brought to device level (peak U-free target = RPI median -39.8 dB),
     with and without a device-like noise floor (procedural pink, -54.2 dB floor);
   - real RPI windows raised to synthetic level (TTS median peak -10.9 dB).
   If scores follow the level change, the model uses level as a cue.

Usage: python diagnostics/source_confound.py [--model-class module:Class --checkpoint path]
Writes diagnostics/source_confound_report.md and source_confound.json.
"""
from pathlib import Path
import argparse, datetime, json, sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL))
import detection as det               # noqa: E402
import isolated_eval as iso           # noqa: E402
import run_evaluation as re_          # noqa: E402
import augmentation as aug            # noqa: E402
import reproducibility as rep         # noqa: E402

AUGCFG = json.loads((EVAL.parent / "deployment_driven_augmentation_strategy/augmentation_config.json").read_text())
MV = AUGCFG["evidence_scope"]["measured_values"]
DEVICE_PEAK = MV["rpi_positive_peak_db"]["median"]                      # -39.8
DEVICE_FLOOR = MV["rpi_positive_floor_db"]["median"]                    # -54.2
SYNTH_PEAK = MV["tts_positive_peak_db_median"]                          # -10.9
NOISE_BANK = EVAL.parent / "deployment_driven_augmentation_strategy/real_noise_bank.csv"
# Floor-spectrum check (2026-09-29): train-split background recordings whose quiet frames contain a motor-hum
# series (~56.5 Hz with harmonics at ~226/282 Hz, plus 120/240 Hz mains lines) vs older ones without it.
HUM_RECORDINGS = ["Manual-BG-RPI18", "Manual-BG-RPI19"]
NO_HUM_RECORDINGS = ["Manual-BG-RPI10", "Manual-BG-RPI11"]
SR = 16000
SEED = 20260930


def noise_pool(recordings, cfg, per_recording=30):
    """Train-split real background windows (from the noise bank) of the given recordings, at 16 kHz."""
    bank = pd.read_csv(NOISE_BANK)
    assert (bank.split == "train").all()
    pool = []
    for rec in recordings:
        rows = bank[bank.file_id == rec]
        rows = rows.iloc[np.linspace(0, len(rows) - 1, min(per_recording, len(rows))).round().astype(int)]
        wave = det.wp.load_waveform(det.ROOT / rows.filepath.iloc[0], cfg).numpy().astype(np.float64)
        for r in rows.itertuples(index=False):
            a = int(round(r.src_start_sec * SR))
            pool.append(wave[a:a + cfg.num_samples])
    return pool


def add_real_noise(x, floor_db, key, pool):
    """Add a real background window (chosen deterministically by key) scaled to floor_db."""
    n = pool[rep.stable_seed(SEED, "pool", key) % len(pool)]
    n = np.tile(n, int(np.ceil(len(x) / len(n))))[:len(x)]
    return x + n * 10 ** ((floor_db - aug.floor_db(n, SR)) / 20)


def body(d, i):
    """Real-audio excerpt of window i at 16 kHz, plus its start offset inside the window."""
    m = d.meta[i]
    wave = d._wave(m["filepath"])
    start = int(round(m["start_sec"] * SR))
    a, b = max(0, start), min(wave.numel(), start + d.cfg.num_samples)
    return wave[a:b].numpy().astype(np.float64), start - a


def to_peak(x, target):
    pk = aug.peak_db(x, SR)
    if pk <= -90:
        return x
    g = 10 ** ((target - pk) / 20)
    m = np.abs(x).max()
    return x * min(g, 0.999 / m if m > 0 else g)


def add_floor(x, floor_db, key):
    n = aug.synthetic_noise("synth_pink", len(x), SR, np.random.default_rng(rep.stable_seed(SEED, key)))
    return x + n * 10 ** ((floor_db - aug.floor_db(n, SR)) / 20)


def score(d, scorer, items):
    """items: list of (index, excerpt, offset) -> scores using the unchanged front end."""
    frames = [det.wp.frame_at(torch.from_numpy(x.astype(np.float32)), off, d.cfg, pad_key=d.meta[i]["window_id"])
              for i, x, off in items]
    out = []
    with rep.single_thread():
        for k in range(0, len(frames), 256):
            out.append(scorer(d.pre(torch.stack(frames[k:k + 256]))))
    return np.concatenate(out)


def cluster_boot_mean(diff: np.ndarray, groups: np.ndarray, reps=2000):
    """Mean paired difference with a source-group cluster bootstrap 95% CI."""
    rng = np.random.default_rng(SEED)
    g = pd.Series(diff).groupby(groups)
    sums, counts = g.sum().to_numpy(), g.size().to_numpy()
    k = len(sums)
    idx = rng.integers(0, k, size=(reps, k))
    boots = sums[idx].sum(1) / counts[idx].sum(1)
    return float(diff.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(EVAL / "baseline/baseline_cnn.pt"))
    ap.add_argument("--model-class", default=None)
    ap.add_argument("--results", default=str(EVAL / "results"), help="run_evaluation output dir (threshold source)")
    ap.add_argument("--out", default=str(HERE), help="where to write this diagnostic's outputs")
    args = ap.parse_args()
    scorer = det.TorchScorer(re_.load_model(args.checkpoint, args.model_class))
    thr = json.loads((Path(args.results) / "results.json").read_text())["isolated"]["threshold"]
    OUTD = Path(args.out)
    OUTD.mkdir(parents=True, exist_ok=True)
    d, manifest = iso.load_isolated_set("validation")

    rows, excerpts = [], {}
    for i, m in enumerate(d.meta):
        x, off = body(d, i)
        excerpts[i] = (x, off)
        rows.append(dict(i=i, window_id=m["window_id"], label=m["label"], window_type=m["window_type"],
                         # "real" = our own device/microphone recordings. Public-corpus audio (real_external)
                         # is neither device audio nor synthetic, so it stays out of both comparison groups.
                         source={"synthetic": "synthetic", "real_external": "external"}.get(m["eval_subset"], "real"),
                         source_group=m["source_group"], peak_db=aug.peak_db(x, SR), floor_db=aug.floor_db(x, SR),
                         silent=bool(np.abs(x).max() == 0)))
    w = pd.DataFrame(rows)
    w["score"] = score(d, scorer, [(i, *excerpts[i]) for i in w.i])
    out = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               model=args.model_class or "pipeline-check baseline", threshold=thr,
               targets=dict(device_peak_db=DEVICE_PEAK, device_floor_db=DEVICE_FLOOR, synthetic_peak_db=SYNTH_PEAK))

    # ---- 1. observational ------------------------------------------------------------------
    neg = w[(w.label == "negative") & ~w.silent]
    real, syn = neg[neg.source == "real"], neg[neg.source == "synthetic"]
    lvl = neg.groupby("source")[["peak_db", "floor_db"]].median().round(1).to_dict("index")
    matched = []
    for r in real.itertuples(index=False):                      # 5 nearest synthetic negatives in (peak, floor)
        dist = np.hypot(syn.peak_db - r.peak_db, syn.floor_db - r.floor_db)
        nn = syn.assign(dist=dist).nsmallest(5, "dist")
        nn = nn[nn.dist <= 6.0]
        matched += [dict(real_window=r.window_id, real_score=r.score, real_type=r.window_type, syn_score=s.score,
                         syn_type=s.window_type, dist=s.dist) for s in nn.itertuples(index=False)]
    mt = pd.DataFrame(matched)
    y = np.r_[np.ones(len(real)), np.zeros(len(syn))]
    out["observational"] = dict(
        n_real_negatives=len(real), real_negative_recordings=int(real.source_group.nunique()),
        n_synthetic_negatives=len(syn), median_levels=lvl,
        mean_score=dict(real=float(real.score.mean()), synthetic=float(syn.score.mean())),
        fpr_at_threshold=dict(real=float((real.score >= thr).mean()), synthetic=float((syn.score >= thr).mean())),
        source_auc_unmatched=det.roc_auc(y, np.r_[real.score, syn.score]),
        matched_pairs=len(mt), matched_real_windows=int(mt.real_window.nunique()) if len(mt) else 0,
        source_auc_level_matched=(det.roc_auc(np.r_[np.ones(mt.real_window.nunique()), np.zeros(len(mt))],
                                              np.r_[mt.groupby("real_window").real_score.first(), mt.syn_score])
                                  if len(mt) else None),
        mean_score_level_matched=dict(real=float(mt.groupby("real_window").real_score.first().mean()) if len(mt) else None,
                                      synthetic=float(mt.syn_score.mean()) if len(mt) else None),
        real_negatives=real.sort_values("score", ascending=False)[["window_id", "window_type", "peak_db", "floor_db", "score"]]
        .round(3).to_dict("records"))

    # ---- 2. interventional -----------------------------------------------------------------
    def intervene(sub, name, fn):
        items = [(r.i, fn(excerpts[r.i][0], r.window_id), excerpts[r.i][1]) for r in sub.itertuples(index=False)]
        s_new = score(d, scorer, items)
        diff = s_new - sub.score.to_numpy()
        mean, lo, hi = cluster_boot_mean(diff, sub.source_group.to_numpy())
        new_peak = np.array([aug.peak_db(x, SR) for _, x, _ in items])
        return dict(condition=name, n=len(sub), groups=int(sub.source_group.nunique()),
                    peak_db_before=float(sub.peak_db.median()), peak_db_after=float(np.median(new_peak)),
                    mean_score_before=float(sub.score.mean()), mean_score_after=float(s_new.mean()),
                    mean_change=mean, mean_change_ci95=[lo, hi],
                    above_threshold_before=int((sub.score >= thr).sum()), above_threshold_after=int((s_new >= thr).sum()))

    neg_floor = round(float(real.floor_db.median()), 1)          # measured on this validation set's real negatives
    hum_pool, quiet_pool = noise_pool(HUM_RECORDINGS, d.cfg), noise_pool(NO_HUM_RECORDINGS, d.cfg)
    out["floor_check"] = dict(real_negative_median_floor_db=neg_floor, positive_floor_db=DEVICE_FLOOR,
                              hum_recordings=HUM_RECORDINGS, no_hum_recordings=NO_HUM_RECORDINGS)
    speechy = ["negative_general_speech", "negative_media", "negative_confusable", "negative_partial_wakeword"]
    syn_neg = w[(w.source == "synthetic") & (w.label == "negative") & w.window_type.isin(speechy)]
    syn_noise = w[(w.source == "synthetic") & (w.window_type == "negative_silence_noise")]
    syn_pos = w[(w.source == "synthetic") & (w.label == "positive")]
    real_pos = w[(w.source == "real") & (w.label == "positive")]
    real_neg = w[(w.source == "real") & (w.label == "negative") & ~w.silent]
    tests = [
        (syn_neg, "synthetic speech-like negatives -> device level", lambda x, k: to_peak(x, DEVICE_PEAK)),
        (syn_neg, "synthetic speech-like negatives -> device level + device floor",
         lambda x, k: add_floor(to_peak(x, DEVICE_PEAK), DEVICE_FLOOR, k)),
        (syn_neg, "synthetic speech-like negatives -> device floor only (level unchanged)",
         lambda x, k: add_floor(x, DEVICE_FLOOR, k)),
        # Floor LEVEL: same pink floor, at the real validation negatives' median floor instead of the positives'.
        (syn_neg, f"synthetic speech-like negatives -> device level + pink floor at real-negative level ({neg_floor} dB)",
         lambda x, k: add_floor(to_peak(x, DEVICE_PEAK), neg_floor, k)),
        # Floor SPECTRUM: real train background at one fixed level, with vs without the motor hum.
        (syn_neg, f"synthetic speech-like negatives -> device level + real floor WITH hum ({DEVICE_FLOOR} dB)",
         lambda x, k: add_real_noise(to_peak(x, DEVICE_PEAK), DEVICE_FLOOR, k, hum_pool)),
        (syn_neg, f"synthetic speech-like negatives -> device level + real floor WITHOUT hum ({DEVICE_FLOOR} dB)",
         lambda x, k: add_real_noise(to_peak(x, DEVICE_PEAK), DEVICE_FLOOR, k, quiet_pool)),
        (syn_noise, "synthetic noise clips -> device level", lambda x, k: to_peak(x, DEVICE_PEAK)),
        (syn_pos, "synthetic positives -> device level + device floor",
         lambda x, k: add_floor(to_peak(x, DEVICE_PEAK), DEVICE_FLOOR, k)),
        (real_pos, "real RPI positives -> synthetic level", lambda x, k: to_peak(x, SYNTH_PEAK)),
        (real_neg, "real RPI negatives -> synthetic level", lambda x, k: to_peak(x, SYNTH_PEAK)),
    ]
    for t in syn_neg.window_type.unique():
        tests.append((syn_neg[syn_neg.window_type == t], f"  {t} -> device level + device floor",
                      lambda x, k: add_floor(to_peak(x, DEVICE_PEAK), DEVICE_FLOOR, k)))
    out["interventional"] = [intervene(sub, name, fn) for sub, name, fn in tests if len(sub)]
    out["synthetic_positive_groups"] = syn_pos.source_group.value_counts().to_dict()
    out["flank_check"] = flank_check(d, scorer, w, thr)

    (OUTD / "source_confound.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    write_report(out, OUTD)
    print((OUTD / "source_confound_report.md").read_text())


def flank_check(d, scorer, w, thr, snrs=(20, 10), per_group=150):
    """Positives with background over the WHOLE 1.5 s window, padding included (added 2026-09-29).
    Section 2 adds changes to the real-audio excerpt only; a model that keys on quiet padded flanks
    (short TTS clips are ~1 s inside a 1.5 s window) passes that test but misses wakewords inside
    continuous background. Background = validation public-corpus speech / music / noise windows."""
    bg = [i for i, m in enumerate(d.meta) if m["source"] in ("ext_fleurs", "ext_musan_music", "ext_musan_noise")]
    if not bg:
        return None
    rng = np.random.default_rng(SEED)
    N = d.cfg.num_samples
    res = {}
    for name, sub in (("synthetic positives", w[(w.source == "synthetic") & (w.label == "positive")]),
                      ("real positives", w[(w.source == "real") & (w.label == "positive")])):
        idx = sub.i.to_numpy()[:per_group]
        if not len(idx):
            continue
        with rep.single_thread():
            frames = torch.stack([d.load(int(i))[0] for i in idx]).numpy().astype(np.float64)
        pad = float(np.mean([d.meta[int(i)]["pad_left_sec"] + d.meta[int(i)]["pad_right_sec"] for i in idx]))
        row = dict(n=len(idx), mean_padding_sec=round(pad, 3))
        for snr in snrs:
            mixed = []
            for f in frames:
                b, _ = body(d, bg[int(rng.integers(len(bg)))])
                b = np.resize(b, N)
                mixed.append(f + b * 10 ** ((aug.peak_db(f, SR) - snr - aug.peak_db(b, SR)) / 20))
            with rep.single_thread():
                sc_ = scorer(d.pre(torch.from_numpy(np.stack(mixed).astype(np.float32))))
            row[f"detected_at_peak_snr_{snr}db"] = float((sc_ >= thr).mean())
        with rep.single_thread():
            sc0 = scorer(d.pre(torch.from_numpy(frames.astype(np.float32))))
        row["detected_clean"] = float((sc0 >= thr).mean())
        res[name] = row
    return res


def write_report(o, outd=HERE):
    ob = o["observational"]
    L = ["# Source-confound diagnostic", "",
         f"Generated {o['created_at']} by `source_confound.py`. Model: {o['model']}. Validation windows only; "
         f"threshold {o['threshold']:.4f}. Level targets: device peak {o['targets']['device_peak_db']} dB, "
         f"device floor {o['targets']['device_floor_db']} dB, synthetic peak {o['targets']['synthetic_peak_db']} dB "
         "(augmentation_config.json measurements).", "",
         "## 1. Observational: real vs synthetic negatives", "",
         f"- {ob['n_real_negatives']} non-silent real negatives from {ob['real_negative_recordings']} recordings vs "
         f"{ob['n_synthetic_negatives']} synthetic negatives.",
         f"- Median speech-band peak / floor: real {ob['median_levels']['real']['peak_db']} / {ob['median_levels']['real']['floor_db']} dB, "
         f"synthetic {ob['median_levels']['synthetic']['peak_db']} / {ob['median_levels']['synthetic']['floor_db']} dB.",
         f"- Mean score: real {ob['mean_score']['real']:.3f}, synthetic {ob['mean_score']['synthetic']:.3f}; "
         f"above threshold: real {100 * ob['fpr_at_threshold']['real']:.1f}%, synthetic {100 * ob['fpr_at_threshold']['synthetic']:.1f}%.",
         f"- Source AUC (score separating real from synthetic negatives): unmatched **{ob['source_auc_unmatched']:.3f}**; "
         f"level-matched **{ob['source_auc_level_matched'] if ob['source_auc_level_matched'] is None else round(ob['source_auc_level_matched'], 3)}** "
         f"({ob['matched_real_windows']} real windows, {ob['matched_pairs']} synthetic matches within 6 dB).", "",
         "| Real negative | Type | Peak dB | Floor dB | Score |", "|---|---|---:|---:|---:|"]
    for r in ob["real_negatives"]:
        L.append(f"| {r['window_id']} | {r['window_type']} | {r['peak_db']} | {r['floor_db']} | {r['score']} |")
    L += ["", "## 2. Interventional: change only the level, re-score the same windows", "",
          "| Change | n (groups) | Peak before → after (dB) | Mean score before → after | Mean change (95% cluster-bootstrap CI) | Above threshold before → after |",
          "|---|---:|---|---|---|---|"]
    for r in o["interventional"]:
        L.append(f"| {r['condition']} | {r['n']} ({r['groups']}) | {r['peak_db_before']:.1f} → {r['peak_db_after']:.1f} | "
                 f"{r['mean_score_before']:.3f} → {r['mean_score_after']:.3f} | {r['mean_change']:+.3f} "
                 f"[{r['mean_change_ci95'][0]:+.3f}, {r['mean_change_ci95'][1]:+.3f}] | "
                 f"{r['above_threshold_before']} → {r['above_threshold_after']} |")
    L += ["", f"Synthetic validation positives by source group: {len(o['synthetic_positive_groups'])} groups.", ""]
    if o.get("flank_check"):
        L += ["## 3. Flank check: background over the whole window, padding included", "",
              "A model that keys on the quiet padded flanks of short clips fails here. Background = validation "
              "public-corpus speech, music and noise; SNR = speech-band peak of the window over that of the background.", "",
              "| Positives | n | Mean padding (s) | Detected clean | Detected, background 20 dB below | Detected, 10 dB below |",
              "|---|---:|---:|---:|---:|---:|"]
        for k, r in o["flank_check"].items():
            L.append(f"| {k} | {r['n']} | {r['mean_padding_sec']} | {100 * r['detected_clean']:.0f}% | "
                     f"{100 * r['detected_at_peak_snr_20db']:.0f}% | {100 * r['detected_at_peak_snr_10db']:.0f}% |")
        L.append("")
    (outd / "source_confound_report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
