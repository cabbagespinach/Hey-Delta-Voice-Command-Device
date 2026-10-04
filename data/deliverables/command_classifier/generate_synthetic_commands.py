#!/usr/bin/env python3
"""
Synthetic training clips for the command classifier, built to look like HeyDeltaListener captures.

A real capture (deploy/heydelta_listener.py) starts 0.3 s before the wakeword fired, which is ~0.1-0.3 s before
"Delta" ends. Then the owner waits for the chime, says the command, and the capture ends ~0.6 s after the speech.
So every synthetic clip is:

    [last 0.4-0.6 s of "Hey Delta" in the SAME voice] [pause 0.3-1.0 s] [command] [0.6-0.7 s quiet]

with a faint noise floor (-65 dBFS RMS) so no part is exact digital silence (real captures never are). Background
noise, room, level and microphone colour are left to training-time augmentation, applied to the whole clip.
`unknown_silent` clips have no command (the pause runs to the listener's 3 s wait); `unknown_other` clips say an
off-list sentence (UNKNOWN_SENTENCES, including near-miss commands).

Voices (--tier):
  ph      Microsoft neural voices with a Philippine accent via edge-tts (needs internet): en-PH-JamesNeural,
          en-PH-RosaNeural (Philippine English), fil-PH-AngeloNeural, fil-PH-BlessicaNeural (Filipino voices
          speaking English); 3 speeds x 3 pitches each = 36 voice variants. PRIORITY (owner request 2026-09-30).
  clone   Filipino-accented voices cloned from FLEURS fil_ph speakers by clone_fleurs_voices.py (Chatterbox, run in
          envs/chatterbox); this tier only assembles and checks the staged audio.
  piper   offline Piper speakers (LibriTTS-R, VCTK, L2-ARCTIC, ARCTIC, ...), --piper-speakers of them, for general
          voice variety.

Every command is transcribed with faster-whisper ("small", ONE GPU); what "small" rejects is re-checked with
"large-v3" (better on single words). A clip is kept only if a transcript matches the text (numbers and times
normalised to words; character edit ratio <= 0.2). Rejects are logged.

Output: data/commands_synthetic/<label>/<label>__<voice>.wav (16 kHz mono 16-bit) and manifest.csv
(file, label, class, text, voice, tier, speed, pitch, delta_tail_sec, pause_sec, duration_sec, whisper, edit_ratio).
Labels and classes come from deploy/label_map.csv.

Usage: CUDA_VISIBLE_DEVICES=<one gpu> python generate_synthetic_commands.py --tier ph
       CUDA_VISIBLE_DEVICES=<one gpu> python generate_synthetic_commands.py --tier piper --piper-speakers 120
       add one class to an existing tier:   ... --tier ph --labels set_alarm_7am
"""
from pathlib import Path
import argparse, asyncio, csv, io, json, re, sys, wave

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUT = ROOT / "data/commands_synthetic"
LABELS = HERE.parent / "model/deploy/label_map.csv"
SR, SEED = 16000, 20260930

from command_texts import SPOKEN, UNKNOWN_SENTENCES               # noqa: E402  (shared with clone_fleurs_voices.py)
PH_VOICES = ["en-PH-JamesNeural", "en-PH-RosaNeural", "fil-PH-AngeloNeural", "fil-PH-BlessicaNeural"]
PH_SPEEDS, PH_PITCHES = ["-15%", "+0%", "+15%"], ["-20Hz", "+0Hz", "+20Hz"]

# ---------------------------------------------------------------------------- text normalisation for checking
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
       "seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_words(n):
    n = int(n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n == 100:
        return "one hundred"
    return str(n)


def norm(t):
    t = t.lower().replace("%", " percent ").replace("°", " degrees ")
    t = re.sub(r"(\d)([a-z])", r"\1 \2", t)                          # "6am" -> "6 am"
    t = re.sub(r"\b(\d{1,2}):00\b", r"\1", t)
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: f"{num_words(m.group(1))} {num_words(m.group(2))}", t)
    t = re.sub(r"\b(a|p)\.?\s?m\.?\b", lambda m: f"{m.group(1)} m", t)
    t = re.sub(r"\b\d+\b", lambda m: num_words(m.group(0)), t)
    t = t.replace("what's", "what is").replace("it's", "it is").replace("i'm", "i am").replace("let's", "let us")
    t = re.sub(r"[^a-z ]", " ", t)
    return " ".join(t.split())


def edit_ratio(a, b):
    d = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)] / max(len(b), 1)


# ---------------------------------------------------------------------------- audio helpers
def to16k(y, sr):
    y = y.mean(axis=1) if y.ndim > 1 else y
    if sr != SR:
        g = np.gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g)
    return y.astype(np.float32)


def speech_bounds(y, rel_db=35.0):
    """First/last 10 ms frame within rel_db of the loudest frame -> (start, end) samples of the speech."""
    n = SR // 100
    k = len(y) // n
    db = 20 * np.log10(np.sqrt((y[:k * n].reshape(k, n) ** 2).mean(1)) + 1e-9)
    on = np.where(db > db.max() - rel_db)[0]
    return (on[0] * n, (on[-1] + 1) * n) if len(on) else (0, len(y))


def trim(y):
    a, b = speech_bounds(y)
    return y[a:b]


def assemble(delta, command, rng, silent=False):
    """[tail of 'Hey Delta'] [pause] [command] [quiet], peak-matched, with a faint noise floor."""
    d = trim(delta)
    tail = float(rng.uniform(0.4, 0.6))
    d = d[-int(tail * SR):]
    if silent:
        pause, cmd, after = 3.0 - 0.3, np.zeros(0, np.float32), 0.0
    else:
        pause, cmd, after = float(rng.uniform(0.3, 1.0)), trim(command), float(rng.uniform(0.6, 0.7))
        cmd = cmd * (np.abs(d).max() / (np.abs(cmd).max() + 1e-9))          # same level as the wakeword
    y = np.concatenate([d, np.zeros(int(pause * SR), np.float32), cmd, np.zeros(int(after * SR), np.float32)])
    y = y / (np.abs(y).max() + 1e-9) * 0.5
    y = y + rng.normal(0, 10 ** (-65 / 20), len(y)).astype(np.float32)
    return y.astype(np.float32), tail, pause


def write_wav(path, y):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())


# ---------------------------------------------------------------------------- TTS back ends
async def edge_say(text, voice, rate, pitch, sem):
    import edge_tts
    async with sem:
        for attempt in range(4):
            try:
                buf = io.BytesIO()
                async for ch in edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).stream():
                    if ch["type"] == "audio":
                        buf.write(ch["data"])
                buf.seek(0)
                y, sr = sf.read(buf, dtype="float32")
                return to16k(y, sr)
            except Exception:
                await asyncio.sleep(2 * (attempt + 1))
        return None


def ph_jobs():
    return [dict(voice=f"{v}_{r}_{p}".replace("%", "pct").replace("+", "p").replace("-", "m"), engine="edge",
                 edge_voice=v, rate=r, pitch=p) for v in PH_VOICES for r in PH_SPEEDS for p in PH_PITCHES]


def piper_jobs(n, rng):
    sys.path.insert(0, str(ROOT / "scripts"))
    import generate_tts_voices_v2 as g                                   # reuse the voice list / downloads
    jobs = []
    for v in list(g.MULTI) + g.SINGLE:
        onnx = g.voice_files(v)
        import json
        ns = json.loads(Path(onnx + ".json").read_text()).get("num_speakers", 1)
        for s in range(ns):
            jobs.append(dict(voice=f"{v}_spk{s}", engine="piper", onnx=onnx, speaker=s if ns > 1 else None,
                             length_scale=round(float(rng.uniform(0.85, 1.15)), 3)))
    idx = rng.choice(len(jobs), size=min(n, len(jobs)), replace=False)
    return [jobs[i] for i in sorted(idx)]


def _one_thread_onnx():
    """Piper builds its onnxruntime session without SessionOptions, i.e. with one thread per core (2026-09-30: 8
    workers drove a shared server's load to 900+). Force 1 intra-op / 1 inter-op thread per worker process."""
    import onnxruntime as ort
    if getattr(ort.InferenceSession, "_one_thread", False):
        return
    base = ort.InferenceSession

    class OneThread(base):
        _one_thread = True

        def __init__(self, path, sess_options=None, providers=None, **kw):
            so = sess_options or ort.SessionOptions()
            so.intra_op_num_threads, so.inter_op_num_threads = 1, 1
            super().__init__(path, sess_options=so, providers=providers, **kw)
    ort.InferenceSession = OneThread


def piper_say_all(job, texts):
    _one_thread_onnx()
    from piper import PiperVoice, SynthesisConfig
    v = PiperVoice.load(job["onnx"])
    out = {}
    for key, text in texts.items():
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            v.synthesize_wav(text, w, syn_config=SynthesisConfig(speaker_id=job["speaker"], length_scale=job["length_scale"]))
        buf.seek(0)
        y, sr = sf.read(buf, dtype="float32")
        out[key] = to16k(y, sr)
    return out


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["ph", "piper", "clone"], required=True)
    ap.add_argument("--piper-speakers", type=int, default=120)
    ap.add_argument("--unknown-per-voice", type=int, default=3, help="off-list sentences per voice")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="only these command labels (no unknown clips), e.g. to add a class after the list changed")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED + {"ph": 0, "piper": 1, "clone": 2}[a.tier])
    lm = pd.read_csv(LABELS)
    lm["text"] = [SPOKEN.get(l, w) for l, w in zip(lm.label, lm.words)]
    commands = lm[~lm.label.str.startswith("unknown")]
    if a.labels:
        assert set(a.labels) <= set(commands.label), f"unknown label(s): {set(a.labels) - set(commands.label)}"
        commands = commands[commands.label.isin(a.labels)]
        a.unknown_per_voice = 0
    if a.tier == "clone":                              # audio synthesized by clone_fleurs_voices.py (envs/chatterbox)
        stage = ROOT / "external_raw/commands_clone_stage"
        index = json.loads((stage / "index.json").read_text())
        keep = lambda k: k == "__delta__" or (k.split("#")[0] in set(commands.label)) or \
            (not a.labels and k.startswith("unknown_other"))
        jobs = [dict(voice=e["voice"], engine="chatterbox", reference=e["reference"], exaggeration=e["exaggeration"],
                     texts={k: t for k, t in e["texts"].items() if keep(k)}, files=e["files"]) for e in index]
    else:
        jobs = ph_jobs() if a.tier == "ph" else piper_jobs(a.piper_speakers, rng)
    print(f"tier {a.tier}: {len(jobs)} voices x ({len(commands)} commands + {a.unknown_per_voice} unknown sentences + 1 silent)")

    # 1. synthesize: per voice, "Hey Delta." + every command + some off-list sentences
    per_voice = {}
    for j in jobs:
        if a.tier == "clone":
            per_voice[j["voice"]] = (j, j["texts"])
            continue
        texts = {"__delta__": "Hey Delta."}
        texts.update({r.label: r.text for r in commands.itertuples()})
        for k, s in enumerate(rng.choice(UNKNOWN_SENTENCES, a.unknown_per_voice, replace=False) if a.unknown_per_voice else []):
            texts[f"unknown_other#{k}"] = str(s)
        per_voice[j["voice"]] = (j, texts)
    audio = {}
    if a.tier == "clone":
        for v, (j, t) in per_voice.items():
            for k in t:
                y, sr = sf.read(str(stage / j["files"][k]), dtype="float32")
                audio[(v, k)] = to16k(y, sr)
    elif a.tier == "ph":
        async def run():
            sem = asyncio.Semaphore(8)
            keys = [(v, k) for v, (j, t) in per_voice.items() for k in t]
            res = await asyncio.gather(*[edge_say(per_voice[v][1][k], per_voice[v][0]["edge_voice"],
                                                  per_voice[v][0]["rate"], per_voice[v][0]["pitch"], sem) for v, k in keys])
            return dict(zip(keys, res))
        audio = asyncio.run(run())
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(8) as ex:          # shared server: keep CPU use modest
            for (v, (j, t)), res in zip(per_voice.items(), ex.map(piper_say_all, [p[0] for p in per_voice.values()],
                                                                   [p[1] for p in per_voice.values()])):
                audio.update({(v, k): y for k, y in res.items()})
    failed = [k for k, y in audio.items() if y is None]
    print(f"synthesized {len(audio) - len(failed)} of {len(audio)} utterances ({len(failed)} failed)")

    # 2. verify every command / off-list sentence with Whisper (one GPU)
    # Two stages: "small" for everything; utterances it rejects get a second opinion from "large-v3", which is much
    # better on single isolated words ("Pause" was heard as "us"/"course" by small). Kept if either matches.
    from faster_whisper import WhisperModel
    checks = {}
    for size in ("small", "large-v3"):
        wm = WhisperModel(size, device="cuda", device_index=0, compute_type="float16")
        for (v, k), y in audio.items():
            if y is None or k == "__delta__" or ((v, k) in checks and checks[(v, k)][1] <= 0.2):
                continue
            yt = trim(y)
            seg, _ = wm.transcribe(yt, language="en", beam_size=5, word_timestamps=True)
            seg = list(seg)
            heard = " ".join(s.text.strip() for s in seg).strip()
            target = norm(per_voice[v][1][k])
            er = round(edit_ratio(norm(heard), target), 3)
            if er > 0.2:
                # The command may be there with extra sounds around it (a Chatterbox habit on short texts:
                # "Radio. Play music."). Find the contiguous words closest to the target; if they match, keep
                # only that stretch of audio. The same edit-ratio rule applies to what is kept.
                words = [w for s_ in seg for w in (s_.words or [])]
                best = (er, None)
                for i in range(len(words)):
                    for j in range(i, min(len(words), i + 12)):
                        e = edit_ratio(norm("".join(w.word for w in words[i:j + 1])), target)
                        if e < best[0]:
                            best = (e, (i, j))
                if best[1] is not None and best[0] <= 0.2:
                    i, j = best[1]
                    a0, a1 = max(0, int((words[i].start - 0.05) * SR)), min(len(yt), int((words[j].end + 0.1) * SR))
                    audio[(v, k)] = yt[a0:a1]
                    heard, er = "[cropped] " + "".join(w.word for w in words[i:j + 1]).strip(), round(best[0], 3)
            checks[(v, k)] = (heard, er)
        del wm

    # 3. assemble capture-like clips and write them
    rows, rejects = [], []
    for v, (j, texts) in per_voice.items():
        done = set()                                   # (label) already written for this voice (multi-take keys)
        delta = audio.get((v, "__delta__"))
        if delta is None:
            continue
        cls = dict(zip(lm.label, lm["class"]))
        for k, text in texts.items():
            if k == "__delta__" or audio.get((v, k)) is None:
                continue
            heard, er = checks[(v, k)]
            label = k.split("#")[0]
            take = "#t" in k                               # clone tier: several takes of a short command
            if take and label in done:
                continue
            if er > 0.2:
                last_take = take and k.endswith("#t3")
                if not take or last_take:
                    rejects.append(dict(voice=v, label=label, text=text, whisper=heard, edit_ratio=er))
                continue
            if take:
                done.add(label)
            y, tail, pause = assemble(delta, audio[(v, k)], rng)
            name = f"{label}__{v}" + (f"_{k.split('#')[1]}" if "#" in k and not take else "") + ".wav"
            write_wav(OUT / label / name, y)
            rows.append(dict(file=f"{label}/{name}", label=label, **{"class": cls[label]}, text=text, voice=v,
                             tier=a.tier, speed=j.get("rate", j.get("length_scale", j.get("exaggeration"))),
                             pitch=j.get("pitch", j.get("reference", "")), delta_tail_sec=round(tail, 3), pause_sec=round(pause, 3), duration_sec=round(len(y) / SR, 3),
                             whisper=heard, edit_ratio=er))
        if a.labels:
            continue
        y, tail, pause = assemble(delta, None, rng, silent=True)                  # unknown_silent
        name = f"unknown_silent__{v}.wav"
        write_wav(OUT / "unknown_silent" / name, y)
        rows.append(dict(file=f"unknown_silent/{name}", label="unknown_silent", **{"class": "unknown"},
                         text="(nothing)", voice=v, tier=a.tier, speed=j.get("rate", j.get("length_scale")),
                         pitch=j.get("pitch", ""), delta_tail_sec=round(tail, 3), pause_sec=round(pause, 3),
                         duration_sec=round(len(y) / SR, 3), whisper="", edit_ratio=""))
    OUT.mkdir(parents=True, exist_ok=True)
    man = OUT / "manifest.csv"
    new = pd.DataFrame(rows)
    if man.exists():
        old = pd.read_csv(man)
        new = pd.concat([old[~old.file.isin(new.file)], new], ignore_index=True)
    new.to_csv(man, index=False)
    rej = pd.DataFrame(rejects)
    if len(rej):
        rp = OUT / "rejected.csv"
        rej.assign(tier=a.tier).to_csv(rp, mode="a", header=not rp.exists(), index=False)
    print(f"wrote {len(rows)} clips ({a.tier}); rejected {len(rejects)} by Whisper; manifest now {len(new)} rows")
    if len(rej):
        print(rej.groupby("label").size().sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
