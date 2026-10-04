"""
Add real human audio from public corpora to the dataset as NEGATIVES (2026-09-29).

No public corpus contains "Hey Delta", so every clip added here is negative by construction.
Raw downloads stay in external_raw/ (with their licences); this script converts a fixed,
seeded selection to 16 kHz mono PCM WAV under data/<category>/ and appends one manifest row
per file. Sources are tagged "ext_*" so the dataloader and the evaluation treat them as
public-corpus audio: neither our device recordings nor synthetic.

| source           | corpus (licence)                                  | category             | selection |
|------------------|---------------------------------------------------|----------------------|-----------|
| ext_mswc         | Multilingual Spoken Words, English (CC-BY 4.0)    | negatives_partial for "hey"/"hay"/"delta", else negatives_confusable | external_raw/mswc_en/selection.csv: 50 near-miss words, <= 300 clips per word, one per speaker |
| ext_fleurs       | FLEURS fil_ph + en_us (CC-BY 4.0)                 | negatives_general    | 400 fil_ph + 200 en_us utterances, first 6 s |
| ext_musan_speech | MUSAN speech (CC-BY 4.0 / public domain)          | negatives_general    | 200 files, one 6 s excerpt each |
| ext_musan_music  | MUSAN music (CC-BY 4.0 / public domain)           | negatives_media      | 200 files, one 6 s excerpt each |
| ext_musan_noise  | MUSAN noise (CC-BY 4.0 / public domain)           | negatives_silence    | every file, first <= 6 s |
| ext_heysnips     | Sonos "Hey Snips" KWS v1 (research use only; never redistribute modified clips) | "Hey Snips" -> negatives_confusable, other utterances -> negatives_general | external_raw/hey_snips/selection.csv: <= 2 "Hey Snips" per speaker (2,000), 1 other utterance per speaker (1,000); 0.8-8 s |

The Qualcomm Keyword Speech Dataset is NOT integrated: its licence forbids incorporating it into another
data set. It is scored separately by data/deliverables/evaluation/diagnostics/hey_phrase_negatives.py.

Split groups (recording_id -> source_group): MSWC = speaker; Hey Snips = speaker (worker_id); FLEURS = sentence id (FLEURS has
no speaker ids, so the same sentence never crosses splits); MUSAN = source file. Splits are
then assigned by segmentation_windowing's append-only split extension.

Usage: python scripts/integrate_external_datasets.py   (a source already in the manifest is skipped, never added twice)
"""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "external_raw"
SR = 16000
EXCERPT = 6.0
SEED = 20260929
rng = np.random.default_rng(SEED)
NOW = datetime.now(timezone.utc).isoformat()


def load16(path):
    y, sr = sf.read(str(path), dtype="float64", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        y = resample_poly(y, SR // g, sr // g)
    return y


def write(y, category, name):
    rel = f"data/{category}/{name}.wav"
    peak = np.abs(y).max()
    if peak > 0.999:                                   # resampling overshoot only; never louder than the source
        y = y * (0.999 / peak)
    sf.write(ROOT / rel, y.astype(np.float32), SR, subtype="PCM_16")
    return rel, len(y) / SR


def row(rel, dur, category, source, recording_id, source_id, speaker_id="", transcription=""):
    return dict(filepath=rel, label="negative", category=category, source=source, source_id=source_id,
                speaker_id=speaker_id, recording_id=recording_id, duration_sec=dur, sample_rate=SR,
                voice=None, speed=None, pitch=None, transcription=transcription or None,
                edit_distance=None, parent_filepath=None, created_at=NOW)


def excerpt(y, start=None):
    n = int(EXCERPT * SR)
    if len(y) <= n:
        return y
    a = int(rng.integers(0, len(y) - n)) if start is None else int(start * SR)
    return y[a:a + n]


def mswc():
    sel = pd.read_csv(RAW / "mswc_en/selection.csv")
    out = []
    for r in sel.itertuples(index=False):
        src = RAW / "mswc_en/opus" / r.LINK.replace("/", "_")
        y = load16(src)
        cat = "negatives_partial" if r.WORD in ("hey", "hay", "delta") else "negatives_confusable"
        spk = f"mswc-{r.SPEAKER[:16]}"
        rel, dur = write(y, cat, f"Ext-MSWC-{r.WORD}-{Path(r.LINK).stem}")
        out.append(row(rel, dur, cat, "ext_mswc", spk, f"mswc_en:{r.LINK}", spk, r.WORD))
    return out


def fleurs():
    out = []
    for lang, n in (("fil_ph", 400), ("en_us", 200)):
        d = RAW / "fleurs/data" / lang
        tsv = []
        for split in ("train", "dev", "test"):
            audio = d / "audio" / split
            if not audio.exists():
                continue
            t = pd.read_csv(d / f"{split}.tsv", sep="\t", header=None, quoting=3,
                            names=["sid", "file", "raw", "norm", "chars", "n", "gender"])
            t["path"] = [audio / f for f in t.file]
            tsv.append(t[[p.exists() for p in t.path]])
        t = pd.concat(tsv)
        t = t[~t.norm.str.contains(r"\bhey\b|delta", regex=True)]
        t = t.sample(n=min(n, len(t)), random_state=SEED)
        for r in t.itertuples(index=False):
            y = excerpt(load16(r.path), start=0)
            rel, dur = write(y, "negatives_general", f"Ext-FLEURS-{lang}-{Path(r.file).stem}")
            out.append(row(rel, dur, "negatives_general", "ext_fleurs", f"fleurs-{lang}-{r.sid}",
                           f"fleurs:{lang}/{r.file}", "", r.raw))
    return out


def musan():
    out = []
    base = RAW / "musan/musan"
    for kind, n, cat in (("speech", 200, "negatives_general"), ("music", 200, "negatives_media"),
                         ("noise", None, "negatives_silence")):
        files = sorted(base.glob(f"{kind}/*/*.wav"))
        if n is not None:
            files = [files[i] for i in sorted(rng.choice(len(files), size=min(n, len(files)), replace=False))]
        for f in files:
            y = load16(f)
            y = excerpt(y, start=0 if kind == "noise" else None)
            if np.abs(y).max() == 0:
                continue
            rel, dur = write(y, cat, f"Ext-MUSAN-{kind}-{f.stem}")
            out.append(row(rel, dur, cat, f"ext_musan_{kind}", f"musan-{f.stem}", f"musan:{f.relative_to(base)}"))
    return out


def heysnips():
    base = RAW / "hey_snips/hey_snips_research_6k_en_train_eval_clean_ter"
    out = []
    for r in pd.read_csv(RAW / "hey_snips/selection.csv").itertuples(index=False):
        cat = "negatives_confusable" if r.is_hotword == 1 else "negatives_general"
        spk = f"snips-{r.worker_id[:16]}"
        rel, dur = write(load16(base / r.audio_file_path), cat, f"Ext-HeySnips-{r.id}")
        out.append(row(rel, dur, cat, "ext_heysnips", spk, f"hey_snips_kws_4.0:{r.audio_file_path}", spk,
                       "hey snips" if r.is_hotword == 1 else ""))
    return out


SOURCES = (("MSWC", "ext_mswc", mswc), ("FLEURS", "ext_fleurs", fleurs),
           ("MUSAN", "ext_musan_", musan), ("Hey Snips", "ext_heysnips", heysnips))


def main():
    man = pd.read_csv(ROOT / "manifest.csv")
    have = man.source.astype(str)
    rows = []
    for name, prefix, fn in SOURCES:
        if have.str.startswith(prefix).any():
            print(f"{name}: already in manifest, skipped")
            continue
        r = fn()
        print(f"{name}: {len(r)} files")
        rows += r
    if not rows:
        print("nothing to add"); return
    new = pd.DataFrame(rows).reindex(columns=man.columns)
    assert not new.filepath.duplicated().any() and not new.filepath.isin(man.filepath).any()
    new.to_csv(ROOT / "manifest.csv", mode="a", header=False, index=False)
    print(f"appended {len(new)} rows -> manifest.csv ({len(man) + len(new)} rows)")
    print(new.groupby(["source", "category"]).size().to_string())


if __name__ == "__main__":
    main()
