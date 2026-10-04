"""
More synthetic speakers for "Hey Delta" (2026-09-29).

Why: BC-ResNet baselines reached 100% on the owner's RPI positives but only 10-40% on synthetic validation
positives. Train had 96 synthetic positive windows from 5 TTS voices; validation's synthetic positives are
one different voice (en_US-amy-medium). The model memorised 5 voices plus the owner's voice instead of the word.

What: offline Piper voices that are NOT already in the dataset or in streaming Set B (Lessac, Amy, Ryan,
Alan, Alba, vi_VN-vivos, id_ID-news_tts are excluded in every quality), including multi-speaker models:
LibriTTS-R (220 of 904 speakers, seeded sample), VCTK (109, UK accents), L2-ARCTIC (24 non-native English
speakers), ARCTIC (18), ARU (12), SEMAINE (4) and 18 single-speaker voices.
Per speaker: 2 "Hey Delta" clips and 2 near-miss clips, so a new speaker's voice cannot itself signal the
wakeword. Speed (length_scale 0.8-1.25), and Piper's noise_scale / noise_w_scale vary per clip.

Checks: every clip is transcribed with faster-whisper "small" (as in phase 1) on ONE GPU, and the transcript is
normalised (lower case, no punctuation). A positive is kept only if it reads exactly "hey delta" (stricter than
phase 1's edit ratio <= 0.25, which would also accept "hey delia"); a near-miss is dropped only if it reads exactly
"hey delta" (then the synthesis sounds like the wakeword). An edit-ratio rule cannot be used for near-misses:
"Hey Delia" is 0.11 from "hey delta" by construction.

Labels and groups: positives -> source tts_synth (clip-level ground truth, gt_whole_clip); near-misses ->
source tts_synthetic_phonetic_nearmiss (negatives_confusable). recording_id = "<voice>-spk<id>", so every
speaker is one source group and lands in exactly one split (append-only split extension).

Usage: CUDA_VISIBLE_DEVICES=<one gpu> python scripts/generate_tts_voices_v2.py [--dry-run]
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse, json, re, shutil, sys, wave

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "voices/piper_raw"
STAGE = ROOT / "external_raw/tts_v2_stage"
SEED = 20260929
TARGET = "hey delta"

MULTI = {"en_US-libritts_r-medium": 220, "en_GB-vctk-medium": None, "en_US-l2arctic-medium": None,
         "en_US-arctic-medium": None, "en_GB-aru-medium": None, "en_GB-semaine-medium": None}
SINGLE = ["en_GB-cori-medium", "en_GB-jenny_dioco-medium", "en_GB-northern_english_male-medium",
          "en_GB-southern_english_female-low", "en_US-bryce-medium", "en_US-danny-low", "en_US-hfc_female-medium",
          "en_US-hfc_male-medium", "en_US-joe-medium", "en_US-john-medium", "en_US-kathleen-low",
          "en_US-kristin-medium", "en_US-kusal-medium", "en_US-ljspeech-medium", "en_US-mike-medium",
          "en_US-norman-medium", "en_US-reza_ibrahim-medium", "en_US-sam-medium"]
EXCLUDED_SPEAKERS = ("lessac", "amy", "ryan", "alan", "alba", "vivos", "news_tts")
POSITIVE_TEXTS = ["Hey Delta", "Hey Delta.", "Hey Delta!", "Hey, Delta.", "Hey Delta?"]
NEARMISS_TEXTS = ["Hey Delia.", "Hey Della!", "Hey Dell.", "Hey Delhi.", "Hey Dental.", "Hey Denta.", "Hey Bella!",
                  "Hey Stella.", "Hey Martha.", "Hey Walter.", "Hey Alta.", "Hey Belta.", "Hey Melda.", "Hey Delaney!",
                  "Delta.", "Hey!", "Hey there.", "Okay Delta.", "Hey Dolly.", "Hey Delphi."]


def voice_files(v):
    from huggingface_hub import hf_hub_download
    lang, name, q = v.split("-")
    base = f"{lang.split('_')[0]}/{lang}/{name}/{q}/{v}.onnx"
    onnx = hf_hub_download("rhasspy/piper-voices", base, local_dir=str(RAW))
    hf_hub_download("rhasspy/piper-voices", base + ".json", local_dir=str(RAW))
    return onnx


def plan():
    rng = np.random.default_rng(SEED)
    jobs = []
    for v in list(MULTI) + SINGLE:
        assert not any(x in v for x in EXCLUDED_SPEAKERS), v
        onnx = voice_files(v)
        n_spk = json.loads(Path(onnx + ".json").read_text()).get("num_speakers", 1)
        spk = list(range(n_spk))
        if MULTI.get(v):
            spk = sorted(rng.choice(n_spk, MULTI[v], replace=False).tolist())
        for s in spk:
            for kind, texts in (("pos", POSITIVE_TEXTS), ("neg", NEARMISS_TEXTS)):
                for k in range(2):
                    jobs.append(dict(voice=v, onnx=onnx, speaker=int(s), multi=n_spk > 1, kind=kind, k=k,
                                     text=str(rng.choice(texts)), length_scale=round(float(rng.uniform(0.8, 1.25)), 3),
                                     noise_scale=round(float(rng.uniform(0.5, 0.8)), 3),
                                     noise_w=round(float(rng.uniform(0.6, 1.0)), 3)))
    return pd.DataFrame(jobs)


def synth_voice(group):
    from piper import PiperVoice, SynthesisConfig
    rows = group[1]
    voice = PiperVoice.load(rows.onnx.iloc[0])
    out = []
    for r in rows.itertuples(index=False):
        name = f"tts2_{r.voice}_spk{r.speaker}_{r.kind}{r.k}.wav"
        path = STAGE / name
        with wave.open(str(path), "wb") as w:
            voice.synthesize_wav(r.text, w, syn_config=SynthesisConfig(
                speaker_id=r.speaker if r.multi else None, length_scale=r.length_scale,
                noise_scale=r.noise_scale, noise_w_scale=r.noise_w))
        out.append(str(path))
    return out


def norm(t):
    """Lower-case, punctuation to spaces, single spaces: "Hey, Delta!" -> "hey delta"."""
    return " ".join(re.sub(r"[^a-z ]", " ", t.lower()).split())


def edit_ratio(a, b):
    d = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)] / max(len(b), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    STAGE.mkdir(parents=True, exist_ok=True)
    jobs = plan()
    print(f"{jobs.voice.nunique()} voices, {jobs.groupby(['voice', 'speaker']).ngroups} speakers, {len(jobs)} clips planned")
    if a.dry_run:
        print(jobs.groupby("voice").speaker.nunique().to_string())
        return
    with ProcessPoolExecutor(24) as ex:
        list(ex.map(synth_voice, list(jobs.groupby("voice"))))
    jobs["file"] = [f"tts2_{r.voice}_spk{r.speaker}_{r.kind}{r.k}.wav" for r in jobs.itertuples()]

    from faster_whisper import WhisperModel
    import soundfile as sf
    wm = WhisperModel("small", device="cuda", device_index=0, compute_type="float16")
    texts = []
    for f in jobs.file:
        seg, _ = wm.transcribe(str(STAGE / f), language="en", beam_size=5)
        texts.append(" ".join(s.text.strip() for s in seg).strip())
    jobs["transcription"] = texts
    jobs["edit_distance"] = [round(edit_ratio(norm(t), TARGET), 3) for t in texts]
    exact = np.array([norm(t) == TARGET for t in texts])
    jobs["keep"] = np.where(jobs.kind == "pos", exact, ~exact)
    jobs.drop(columns="onnx").to_csv(STAGE / "generation_log.csv", index=False)
    print(jobs.groupby(["kind", "keep"]).size().to_string())

    keep = jobs[jobs.keep]
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for r in keep.itertuples(index=False):
        cat, lab, src = (("positives", "positive", "tts_synth") if r.kind == "pos"
                         else ("negatives_confusable", "negative", "tts_synthetic_phonetic_nearmiss"))
        dst = ROOT / "data" / cat / r.file
        shutil.copy(STAGE / r.file, dst)
        info = sf.info(str(dst))
        spk = f"{r.voice}-spk{r.speaker}"
        rows.append(dict(filepath=f"data/{cat}/{r.file}", label=lab, category=cat, source=src, source_id=r.voice,
                         speaker_id=spk, recording_id=spk, duration_sec=info.duration, sample_rate=info.samplerate,
                         voice=r.voice, speed=round(1 / r.length_scale, 3), pitch=0.0, transcription=r.transcription,
                         edit_distance=r.edit_distance, parent_filepath=None, created_at=now))
    man = pd.read_csv(ROOT / "manifest.csv", low_memory=False)
    new = pd.DataFrame(rows).reindex(columns=man.columns)
    assert not new.filepath.isin(man.filepath).any(), "tts2 clips already in manifest"
    new.to_csv(ROOT / "manifest.csv", mode="a", header=False, index=False)
    print(f"appended {len(new)} rows ({new.label.value_counts().to_dict()}) from "
          f"{new.recording_id.nunique()} speakers -> manifest.csv")


if __name__ == "__main__":
    main()
