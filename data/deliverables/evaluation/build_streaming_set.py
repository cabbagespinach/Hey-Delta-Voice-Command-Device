#!/usr/bin/env python3
"""
Build the long-form streaming evaluation set.

Set A (existing, registered as is): data/streaming_eval/syn_stream_*.wav, 120 x 30 s
  from phase-1 generation. Loud (peak-normalised) synthetic continuous speech over pink
  noise, 0-2 wakewords placed in the middle 20-80 % of each stream.

Set B (composed here): new always-listening streams synthesised for evaluation only.
  - Audio: new Piper TTS (7 local voices) plus procedural noise and music. No file from
    data/ (train/validation/test) is read as audio.
  - Text: eval-only sentence, confusable and carrier banks. The build refuses any text
    that matches a training transcription (normalized manifest).
  - Wakewords: new syntheses at speeds/pitches off the training grid, checked for
    near-duplicates of training positives (normalised cross-correlation).
  - Six acoustic conditions (quiet, environmental noise, continuous speech, media,
    noisy speech, mixed), each with confusable phrases, partial wakewords, speech
    snippets and occasional complete "Hey Delta" (isolated or embedded in a sentence).
  - Every event is placed uniformly at random in the stream (rejection sampling for
    spacing only). Levels follow the deployment measurements: device floor
    U(-58.2, -51.4) dB, event peak-over-floor U(5.4, 25.7) dB.

Piper sampling is stochastic, so regenerating gives different audio: the set is built
once and frozen. streams.csv records a sha256 per file; the evaluator verifies it.

Usage: python build_streaming_set.py [--force]
Outputs: data/streaming_eval_composed/*.wav, streaming_set/{streams.csv, events.csv, build_log.json}
"""
from pathlib import Path
import argparse, datetime, hashlib, json, re, sys, time

import numpy as np
import pandas as pd
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
DELIV = HERE.parent
ROOT = DELIV.parent.parent
sys.path.insert(0, str(DELIV / "preprocessing"))
sys.path.insert(0, str(DELIV / "dataloading"))
import wakeword_preprocessing as wp      # noqa: E402
import augmentation as aug               # noqa: E402

CFG = json.loads((HERE / "eval_config.json").read_text())
B = CFG["streaming_set_b"]
SR = 16000
OUT_AUDIO = ROOT / B["audio_dir"]
OUT_META = HERE / "streaming_set"
SET_A_MANIFEST = ROOT / "data/logs/streaming_eval_manifest.csv"
TRAIN_MANIFEST = DELIV / "dataset_split/normalized_manifest.csv"
WAKE = "hey delta"

# --------------------------------------------------------------------------- eval-only text banks
SENTENCES = [
    "The library opens at nine on Saturday mornings.", "My cousin painted the fence a bright green colour.",
    "We need two more chairs for the kitchen table.", "The bus was late again because of the parade.",
    "Could you water the plants while I am away?", "The soup tastes better on the second day.",
    "She keeps her bicycle in the hallway downstairs.", "The printer ran out of paper during the meeting.",
    "Our flight lands a little after midnight.", "He bought fresh bread from the corner bakery.",
    "The museum has a new exhibit about old maps.", "I usually walk the dog before breakfast.",
    "The heater makes the room far too warm.", "Let me find the charger for my laptop.",
    "The football match was cancelled because of the storm.", "They moved into a flat near the river.",
    "Put the leftovers in the fridge after dinner.", "The pharmacy closes at six in the evening.",
    "I have not finished reading that novel yet.", "The kettle has been whistling for a minute.",
    "Grandma sent a postcard from the seaside.", "We watched a documentary about penguins.",
    "The elevator is out of order until Monday.", "Remember to lock the back gate tonight.",
    "The recipe calls for three cups of flour.", "Traffic on the bridge was slow this morning.",
    "The cat is sleeping on the warm windowsill.", "I think the neighbours are having a party.",
    "Our team finished the project ahead of schedule.", "The umbrella is behind the coat rack.",
    "Tomorrow will be cloudy with a chance of showers.", "Please turn the music down a little bit.",
    "The mechanic said the brakes need replacing.", "We are planning a picnic by the lake.",
    "That restaurant serves the best noodles in town.", "The spare key is under the flower pot.",
    "My phone battery is almost empty again.", "The garden hose was left running all afternoon.",
    "He practises the piano every evening after school.", "The concert tickets sold out in an hour.",
]
CONFUSABLES = [
    "Hey Delia", "Hey Delhi", "Hey Dalton", "Hey Dilbert", "Hey Doctor", "Hey Delphi", "Hey Deli",
    "Pay Delta", "The Delta", "Hey Dell", "Heavy Delta", "Hey Del Taco", "Hey Delete", "Hey Dealer",
    "Hey Denver", "Hey Debra", "Hey Kelta", "Hey Belda", "Hey Dad", "Hey, tell her", "Hey, tell them",
    "Hey Delaney", "Stay better", "Hey Melda", "Hey Del Ray", "Hey Del Rio", "Hey Delco", "Hey Velma",
    "Hey Deltoid", "Hello there",
]
PREFIXES = ["Okay,", "Alright then,", "So,", "Um,", "Excuse me,", "Right,", "Good morning,", "Listen,"]
SUFFIXES = ["what time is it?", "turn on the lights.", "play some music.", "set a timer for ten minutes.",
            "what is the weather like?", "call my sister.", "add milk to the list.", "stop the alarm."]
CONDITIONS = ["quiet", "environmental_noise", "continuous_speech", "media", "noisy_speech", "mixed"]


def norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(t).lower())).strip()


def check_text_novelty():
    """No eval-only text may equal a training transcription; no negative text may contain the wakeword."""
    tr = pd.read_csv(TRAIN_MANIFEST)
    seen = set(tr.transcription.dropna().map(norm_text)) | set(tr.transcription_normalized.dropna().map(norm_text))
    report = {}
    for name, bank in [("sentences", SENTENCES), ("confusables", CONFUSABLES), ("prefixes", PREFIXES),
                       ("suffixes", SUFFIXES)]:
        clash = [t for t in bank if norm_text(t) in seen]
        wake = [t for t in bank if WAKE in norm_text(t).replace("-", " ")]
        if clash or wake:
            raise SystemExit(f"{name}: text overlaps training {clash} or contains the wakeword {wake}")
        report[name] = len(bank)
    return report


# --------------------------------------------------------------------------- synthesis
class Tts:
    def __init__(self):
        from piper import PiperVoice
        self.voices = {}
        for v in B["voices"]:
            onnx = next((ROOT / "voices/piper_raw").rglob(f"{v}.onnx"))
            self.voices[v] = PiperVoice.load(str(onnx))
        self.cfg = wp.PreprocessConfig.from_json(DELIV / "preprocessing/preprocessing_config.json")

    def __call__(self, text, voice, rng, wakeword=False):
        from piper.config import SynthesisConfig
        grid = (0.85, 1.0, 1.15)
        while True:                                          # off the training speed grid
            speed = float(rng.uniform(*B["speed_range"]))
            if not wakeword or min(abs(speed - g) for g in grid) > 0.02:
                break
        noise_scale = float(rng.uniform(0.5, 0.8))
        noise_w = float(rng.uniform(0.6, 0.9))
        v = self.voices[voice]
        chunks = v.synthesize(text, syn_config=SynthesisConfig(length_scale=1 / speed, noise_scale=noise_scale,
                                                               noise_w_scale=noise_w))
        y = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        pitch = 0.0
        if rng.random() < 0.5:                               # non-integer semitones: off the training pitch grid
            pitch = float(rng.uniform(-2, 2))
            if abs(pitch - round(pitch)) < 0.1:
                pitch += 0.25
            import librosa
            y = librosa.effects.pitch_shift(y, sr=v.config.sample_rate, n_steps=pitch).astype(np.float32)
        y = wp.resample(torch.from_numpy(y), v.config.sample_rate, self.cfg).numpy().astype(np.float64)
        return y, dict(voice=voice, speed=round(speed, 4), pitch=round(pitch, 3),
                       noise_scale=round(noise_scale, 3), noise_w=round(noise_w, 3))


def trim_span(y, top_db=40.0, frame=0.01):
    """Energy span of an utterance (same rule as the TTS positives' ground truth)."""
    n = int(frame * SR)
    k = max(1, len(y) // n)
    rms = np.sqrt((y[:k * n].reshape(k, n) ** 2).mean(1) + 1e-20)
    db = 20 * np.log10(rms / rms.max())
    idx = np.where(db > -top_db)[0]
    return idx[0] * n / SR, min(len(y), (idx[-1] + 1) * n) / SR


def fade(y, ms=10):
    n = min(len(y) // 2, int(ms / 1000 * SR))
    if n > 0:
        r = np.linspace(0, 1, n)
        y = y.copy()
        y[:n] *= r
        y[-n:] *= r[::-1]
    return y


# --------------------------------------------------------------------------- beds
def scale_to_floor(noise, target_floor_db):
    return noise * 10 ** ((target_floor_db - aug.floor_db(noise, SR)) / 20)


def scale_to_peak(y, target_peak_db):
    return y * 10 ** ((target_peak_db - aug.peak_db(y, SR)) / 20)


def environmental(n, rng):
    kind = rng.choice(["synth_white", "synth_pink", "synth_hum", "synth_fan", "synth_clicks", "rain", "rumble"])
    if kind == "rain":                                       # dense filtered crackle
        from scipy.signal import butter, sosfilt
        y = sosfilt(butter(2, [1500, 6000], "bandpass", fs=SR, output="sos"), rng.standard_normal(n))
        y *= 1 + 0.5 * (rng.random(n) < 0.02)
    elif kind == "rumble":                                   # traffic-like low rumble with slow swells
        from scipy.signal import butter, sosfilt
        y = sosfilt(butter(2, 180, "lowpass", fs=SR, output="sos"), rng.standard_normal(n))
        t = np.arange(n) / SR
        y *= 1 + 0.6 * np.sin(2 * np.pi * rng.uniform(0.05, 0.2) * t)
    else:
        y = aug.synthetic_noise(str(kind), n, SR, rng)
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-12), str(kind)


def music(n, rng):
    t = np.arange(n) / SR
    root = rng.uniform(110, 220)
    y = np.zeros(n)
    beat = 60 / rng.uniform(80, 130)
    for k, ratio in enumerate([1, 1.25, 1.5, 2.0]):
        y += np.sin(2 * np.pi * root * ratio * t + rng.uniform(0, 6.28)) / (k + 1)
    y *= 0.6 + 0.4 * (np.mod(t, beat) < beat / 2)
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-12)


def speech_track(n, tts, rng, voices, level_db, gap=(0.2, 1.0)):
    """Back-to-back sentences filling ~n samples; returns track and the texts used."""
    y, pos, used = np.zeros(n), int(rng.uniform(0, 0.5) * SR), []
    while pos < n - SR:
        voice = voices[int(rng.integers(len(voices)))]
        s, meta = tts(SENTENCES[int(rng.integers(len(SENTENCES)))], voice, rng)
        s = fade(scale_to_peak(s, level_db))
        L = min(len(s), n - pos)
        y[pos:pos + L] += s[:L]
        used.append(meta["voice"])
        pos += L + int(rng.uniform(*gap) * SR)
    return y, used


def make_bed(condition, n, tts, rng):
    floor_db = float(rng.uniform(*B["device_floor_db"]))
    parts = {"device_floor_db": round(floor_db, 2)}
    bed = scale_to_floor(aug.synthetic_noise("synth_pink", n, SR, rng), floor_db)
    noise_floor = floor_db
    voices = list(tts.voices)
    if condition in ("environmental_noise", "noisy_speech", "mixed"):
        env, kind = environmental(n, rng)
        noise_floor = float(rng.uniform(*B["environmental_floor_db"]))
        bed = bed + scale_to_floor(env, noise_floor)
        parts["environmental"] = kind
        parts["environmental_floor_db"] = round(noise_floor, 2)
    if condition in ("continuous_speech", "noisy_speech"):
        lvl = noise_floor + float(rng.uniform(*B["bed_speech_over_floor_db"]))
        sp, used = speech_track(n, tts, rng, [voices[int(rng.integers(len(voices)))] for _ in range(2)], lvl)
        bed = bed + sp
        parts["speech_voices"] = sorted(set(used))
    if condition == "media":
        lvl = noise_floor + float(rng.uniform(*B["bed_speech_over_floor_db"]))
        sp, used = speech_track(n, tts, rng, [voices[int(rng.integers(len(voices)))] for _ in range(2)], lvl,
                                gap=(0.05, 0.4))
        m = music(n, rng)
        bed = bed + sp + m * 10 ** ((lvl - float(rng.uniform(3, 12)) - aug.peak_db(m, SR)) / 20)
        parts["media"] = "podcast speech + procedural music"
    if condition == "mixed":                                 # alternating speech and media segments
        seg = int(rng.uniform(8, 15) * SR)
        lvl = noise_floor + float(rng.uniform(*B["bed_speech_over_floor_db"]))
        sp, _ = speech_track(n, tts, rng, [voices[int(rng.integers(len(voices)))] for _ in range(2)], lvl)
        m = scale_to_peak(music(n, rng), lvl - 6)
        mask = (np.arange(n) // seg) % 2
        bed = bed + sp * mask + m * (1 - mask)
        parts["mixed"] = f"speech/music alternating every {seg / SR:.1f} s"
    return bed, noise_floor, parts


# --------------------------------------------------------------------------- events
def place(length, n, taken, rng, gap, tries=300):
    """Uniformly random start in [0, n - length]; the only constraint is spacing: two events stay
    max(their gaps) apart (wakewords carry a larger gap, in both directions). None if no room."""
    for _ in range(tries):
        s = int(rng.integers(0, max(1, n - length + 1)))
        e = s + length
        if all(e + max(gap, g) <= a or s >= b + max(gap, g) for a, b, g in taken):
            return s
    return None


def build_stream(stream_id, condition, tts, rng, training_positive_clips):
    D = B["stream_sec"]
    n = int(D * SR)
    bed, noise_floor, parts = make_bed(condition, n, tts, rng)
    y = bed.copy()
    events, taken = [], []                           # taken: (start, end, required gap)
    voices = list(tts.voices)
    E = B["events_per_stream"]
    plan = (["wakeword"] * int(rng.choice(E["wakeword"])) +
            ["confusable"] * int(rng.integers(E["confusable"][0], E["confusable"][1] + 1)) +
            ["partial"] * int(rng.integers(E["partial"][0], E["partial"][1] + 1)) +
            (["speech_snippet"] * int(rng.integers(E["speech_snippet"][0], E["speech_snippet"][1] + 1))
             if condition in ("quiet", "environmental_noise") else []))
    rng.shuffle(plan)
    for kind in plan:
        voice = voices[int(rng.integers(len(voices)))]
        gt = None
        if kind == "wakeword":
            embedded = rng.random() < B["embedded_wakeword_fraction"]
            w, meta = tts("Hey Delta", voice, rng, wakeword=True)
            ws, we = trim_span(w)
            w = w[int(ws * SR):int(we * SR)]
            meta["near_dup_xcorr"] = near_duplicate_score(w, training_positive_clips.get(voice, []))
            if embedded:
                p, _ = tts(PREFIXES[int(rng.integers(len(PREFIXES)))], voice, rng)
                s_, _ = tts(SUFFIXES[int(rng.integers(len(SUFFIXES)))], voice, rng)
                p = p[int(trim_span(p)[0] * SR):int(trim_span(p)[1] * SR)]
                s_ = s_[int(trim_span(s_)[0] * SR):int(trim_span(s_)[1] * SR)]
                gap1, gap2 = (np.zeros(int(rng.uniform(0.15, 0.5) * SR)) for _ in range(2))
                clip = np.concatenate([p, gap1, w, gap2, s_])
                rel = (len(p) + len(gap1)) / SR, (len(p) + len(gap1) + len(w)) / SR
                kind = "wakeword_embedded"
            else:
                clip, rel = w, (0.0, len(w) / SR)
            text = "Hey Delta"
            gap = B["min_gap_around_wakeword_sec"]
        elif kind == "confusable":
            text = CONFUSABLES[int(rng.integers(len(CONFUSABLES)))]
            clip, meta = tts(text, voice, rng)
            a, b = trim_span(clip)
            clip = clip[int(a * SR):int(b * SR)]
            gap = B["min_gap_between_events_sec"]
        elif kind == "partial":
            w, meta = tts("Hey Delta", voice, rng, wakeword=True)
            a, b = trim_span(w)
            w = w[int(a * SR):int(b * SR)]
            f = float(rng.uniform(0.3, 0.8))
            head = rng.random() < 0.5
            clip = w[:int(f * len(w))] if head else w[len(w) - int(f * len(w)):]
            kind = "partial_head" if head else "partial_tail"
            text = f"Hey Delta ({'first' if head else 'last'} {f:.0%})"
            meta["fraction_kept"] = round(f, 3)
            gap = B["min_gap_between_events_sec"]
        else:
            text = SENTENCES[int(rng.integers(len(SENTENCES)))]
            clip, meta = tts(text, voice, rng)
            a, b = trim_span(clip)
            clip = clip[int(a * SR):int(b * SR)]
            gap = B["min_gap_between_events_sec"]
        pof = float(rng.uniform(*B["event_peak_over_floor_db"]))
        clip = fade(scale_to_peak(clip, noise_floor + pof))
        L = len(clip)
        start = place(L, n, taken, rng, int(gap * SR))
        if start is None:
            continue
        y[start:start + L] += clip
        taken.append((start, start + L, int(gap * SR)))
        ev = dict(stream_id=stream_id, event_type=kind, text=text, insert_start_sec=round(start / SR, 4),
                  insert_end_sec=round((start + L) / SR, 4), target_peak_over_floor_db=round(pof, 2),
                  source="synthetic_tts_piper", **meta)
        if kind.startswith("wakeword"):
            ev["wake_start_sec"] = round(start / SR + rel[0], 4)
            ev["wake_end_sec"] = round(start / SR + rel[1], 4)
        events.append(ev)
    peak = np.abs(y).max()
    scaled = peak > 0.999
    if scaled:
        y *= 0.999 / peak
    for ev in events:                                  # achieved level in the final mix
        a, b = int(ev["insert_start_sec"] * SR), int(ev["insert_end_sec"] * SR)
        ev["achieved_peak_over_floor_db"] = round(aug.peak_db(y[a:b], SR) - noise_floor, 2)
    return y.astype(np.float32), events, dict(bed=json.dumps(parts), noise_floor_db=round(noise_floor, 2),
                                              rescaled_to_full_scale=bool(scaled))


def load_training_positive_clips():
    """Every existing Piper positive clip (all splits) per voice, at 16 kHz, for the near-duplicate check."""
    w = pd.read_csv(DELIV / "segmentation_windowing/outputs/recording_inventory.csv")
    pos = w[(w.source == "tts_synth") & (w.label == "positive")]
    cfg = wp.PreprocessConfig.from_json(DELIV / "preprocessing/preprocessing_config.json")
    out = {}
    for r in pos.itertuples(index=False):
        voice = next((v for v in B["voices"] if Path(r.filepath).name.startswith(v)), None)
        if voice:
            out.setdefault(voice, []).append(wp.load_waveform(ROOT / r.filepath, cfg).numpy().astype(np.float64))
    return out


def near_duplicate_score(w, refs):
    """Max normalised cross-correlation between a new wakeword and any training clip of the same voice."""
    from scipy.signal import fftconvolve
    best = 0.0
    for r in refs:
        c = fftconvolve(w, r[::-1])
        best = max(best, float(np.abs(c).max() / (np.linalg.norm(w) * np.linalg.norm(r) + 1e-12)))
    return round(best, 4)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register_set_a():
    a = pd.read_csv(SET_A_MANIFEST)
    streams, events = [], []
    for r in a.itertuples(index=False):
        sid = Path(r.filepath).stem
        streams.append(dict(stream_id=sid, set="A_phase1_holdout", condition="loud_continuous_speech",
                            filepath=r.filepath, duration_sec=r.duration_sec, sample_rate=sf.info(str(ROOT / r.filepath)).samplerate,
                            n_wakewords=r.wakeword_instances, bed=json.dumps({"description": "TTS continuous speech + pink noise, peak-normalised to 0.9"}),
                            noise_floor_db=None, rescaled_to_full_scale=None, source_kind="synthetic",
                            sha256=sha256(ROOT / r.filepath)))
        for k, m in enumerate(json.loads(r.wakeword_metadata)):
            events.append(dict(stream_id=sid, event_type="wakeword", text="Hey Delta", voice=m.get("voice"),
                               insert_start_sec=m["start_sec"], insert_end_sec=m["end_sec"],
                               wake_start_sec=m["start_sec"], wake_end_sec=m["end_sec"],
                               source="synthetic_tts_phase1"))
    return streams, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild Set B even if it exists (new audio!)")
    args = ap.parse_args()
    if (OUT_META / "streams.csv").exists() and not args.force:
        raise SystemExit("streaming set already built and frozen; pass --force to regenerate (audio will change)")
    t0 = time.time()
    novelty = check_text_novelty()
    OUT_AUDIO.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)
    tts = Tts()
    refs = load_training_positive_clips()
    rng = np.random.default_rng(B["seed"])
    streams, events = register_set_a()
    k = 0
    for condition in CONDITIONS:
        for _ in range(B["streams_per_condition"]):
            sid = f"evalB_{k:04d}"
            y, ev, info = build_stream(sid, condition, tts, rng, refs)
            path = OUT_AUDIO / f"{sid}.wav"
            sf.write(path, y, SR, subtype="FLOAT")
            streams.append(dict(stream_id=sid, set="B_composed", condition=condition,
                                filepath=str(path.relative_to(ROOT)), duration_sec=len(y) / SR, sample_rate=SR,
                                n_wakewords=sum(e["event_type"].startswith("wakeword") for e in ev),
                                source_kind="synthetic", sha256=sha256(path), **info))
            events += ev
            k += 1
            print(f"  {sid} {condition}: {len(ev)} events ({time.time() - t0:.0f} s)")
    s, e = pd.DataFrame(streams), pd.DataFrame(events)
    e.insert(1, "event_id", [f"{r.stream_id}#e{i:03d}" for i, r in enumerate(e.itertuples())])
    s.to_csv(OUT_META / "streams.csv", index=False)
    e.to_csv(OUT_META / "events.csv", index=False)
    wk = e[e.event_type.str.startswith("wakeword") & (e.stream_id.str.startswith("evalB"))]
    log = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               seed=B["seed"], piper_voices=B["voices"], text_banks=novelty,
               n_streams=s.groupby("set").size().to_dict(), hours=s.groupby("set").duration_sec.sum().div(3600).round(3).to_dict(),
               events=e.assign(set=np.where(e.stream_id.str.startswith("evalB"), "B", "A"))
                       .groupby(["set", "event_type"]).size().unstack(0).fillna(0).astype(int).to_dict(),
               near_duplicate_xcorr_max=float(wk.near_dup_xcorr.max()) if len(wk) else None,
               near_duplicate_xcorr_median=float(wk.near_dup_xcorr.median()) if len(wk) else None,
               build_seconds=round(time.time() - t0, 1),
               note="Piper synthesis is stochastic; this set is frozen by the sha256 values in streams.csv.")
    (OUT_META / "build_log.json").write_text(json.dumps(log, indent=2, default=str) + "\n")
    print(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
