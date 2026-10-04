#!/usr/bin/env python3
"""
Other wordings of each command as TRAINING data (owner-approved 2026-10-01, after the Pi A/B test: reworded commands
~80% correct, "Switch off the lights" -> lights_on in both models, "Pause it" -> call_jane, ...).

Wordings (TRAIN below) are spoken by synthetic voices of the TRAIN and VALIDATION splits only (the voice split of
build_commands_all.voice_split), so the TEST-split voices of evaluation/phrasing_variants_test.py stay unheard.
HELD_OUT wordings (one per label) are never synthesized here: the phrasing test then measures both "wordings now in
training, new voices" and "wordings never trained". Extra near-miss sentences (NEAR_MISS) are `unknown`, so the new
wordings don't teach "any sentence with lights/stop/pause in it is a command".

    python generate_phrasing_variants.py synth [--piper-voices 60]       # ph (edge-tts, internet) + Piper, CPU
    envs/chatterbox/bin/python generate_phrasing_variants.py clone [--voices 30]   # cloned FLEURS voices, ONE GPU
    python generate_phrasing_variants.py assemble-clone                   # clone stage -> capture-like clips
    CUDA_VISIBLE_DEVICES=<gpu> python generate_phrasing_variants.py check # Whisper, same rule as the training data

Every step resumes after a disconnect (finished voices / checked clips are kept).
Output: data/commands_variants_train/<label>/<label>__<voice>__<i>.wav, _raw/ (bare phrase for Whisper), manifest.csv
(path, raw, label, class, text, voice, tier, split, heard, edit_ratio, whisper_ok).
"""
from pathlib import Path
import argparse, asyncio, json, random, sys

import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
OUT = ROOT / "data/commands_variants_train"
STAGE = ROOT / "external_raw/commands_variants_clone_stage"
CLONE_STAGE = ROOT / "external_raw/commands_clone_stage"
SEED = 20261001

# label -> wordings to TRAIN on (the trained phrasing itself is already in the data and is not repeated here)
TRAIN = {
    "lights_on": ["Turn on the lights", "Turn the lights on", "Switch on the lights", "Lights on",
                  "Turn the light on", "Switch on the light", "Lights on please", "Can you turn on the lights",
                  "Turn on all the lights"],
    "lights_off": ["Turn off the lights", "Turn the lights off", "Switch off the lights", "Lights off",
                   "Turn the light off", "Switch off the light", "Lights off please", "Can you turn off the lights",
                   "Turn off all the lights"],
    "play_music": ["Play some music", "Start the music", "Play some songs", "Music please", "I want to hear some music"],
    "weather": ["What's the weather like today?", "Weather today", "What's the weather today?", "Tell me the weather",
                "What's the forecast?"],
    "time": ["What's the time?", "Tell me the time", "What time is it now?", "Do you know the time?",
             "What's the current time?"],
    "dim_lights_20": ["Set the lights to twenty percent", "Lights at twenty percent", "Lights to twenty percent",
                      "Change the lights to twenty percent"],
    "dim_lights_50": ["Dim the lights to fifty percent", "Lights at fifty percent", "Lights to fifty percent",
                      "Change the lights to fifty percent"],
    "dim_lights_80": ["Dim the lights to eighty percent", "Set the lights to eighty percent", "Lights to eighty percent",
                      "Change the lights to eighty percent"],
    "set_timer_1": ["Set a one minute timer", "Timer for one minute", "Set timer for one minute", "One minute timer"],
    "set_timer_5": ["Set a five minute timer", "Start a timer for five minutes", "Set timer for five minutes",
                    "Five minute timer"],
    "set_timer_10": ["Start a timer for ten minutes", "Timer for ten minutes", "Set timer for ten minutes",
                     "Ten minute timer"],
    "set_alarm_6am": ["Set an alarm for six A M", "Wake me up at six A M", "Set my alarm for six A M",
                      "Alarm for six A M"],
    "set_alarm_7am": ["Set an alarm for seven A M", "Set an alarm for seven in the morning", "Set my alarm for seven A M",
                      "Alarm for seven A M"],
    "set_alarm_9pm": ["Set an alarm for nine P M", "Set an alarm for nine in the evening", "Set my alarm for nine P M",
                      "Alarm for nine P M"],
    "set_temperature_18": ["Set the temperature to eighteen degrees", "Make it eighteen degrees",
                           "Change the temperature to eighteen degrees", "Set the thermostat to eighteen"],
    "set_temperature_22": ["Make it twenty two degrees", "Change the temperature to twenty two",
                           "Change the temperature to twenty two degrees", "Set the thermostat to twenty two"],
    "set_temperature_26": ["Set the temperature to twenty six degrees", "Change the temperature to twenty six",
                           "Change the temperature to twenty six degrees", "Set the thermostat to twenty six"],
    "pause": ["Pause it", "Pause the song", "Pause playback"],
    "stop": ["Stop playing", "Stop the song", "Stop the music please"],
    "next": ["Next song", "Next track", "Go to the next song", "Next one"],
    "skip": ["Skip this song", "Skip it", "Skip this one", "Skip track"],
    "volume_up": ["Turn it up", "Increase the volume", "Raise the volume", "Volume up please"],
    "louder": ["Make it louder", "Louder please", "Can you make it louder"],
    "volume_down": ["Turn down the volume", "Turn it down", "Quieter", "Decrease the volume", "Make it quieter"],
    "remind_trash": ["Remind me to take the trash out", "Remind me to take out the garbage",
                     "Remind me to throw out the trash"],
    "remind_study": ["Remind me to study", "Remind me to study for the exam", "Remind me to review for my exam"],
    "list_reminders": ["What reminders do I have?", "Do I have any reminders?", "Show my reminders",
                       "List my reminders"],
    "call_jane": ["Phone Jane", "Call Jane please", "Make a call to Jane", "Ring Jane"],
    "text_jane": ["Send Jane a text", "Message Jane", "Send a text to Jane", "Write Jane a message"],
    "play": ["Resume", "Keep playing", "Resume playing", "Unpause", "Continue"],
    "party": ["Party mode", "Let's party", "Start the party", "Party mode on"],
}
# never trained: one wording per label, for an honest "unseen wording" measure (all are in phrasings.txt)
HELD_OUT = {
    "lights_on": "Switch the lights on", "lights_off": "Switch the lights off", "play_music": "Put on some music",
    "weather": "How's the weather?", "time": "What's the time now?", "dim_lights_20": "Dim the lights to twenty percent",
    "dim_lights_50": "Set the lights to fifty percent", "dim_lights_80": "Lights at eighty percent",
    "set_timer_1": "Start a timer for one minute", "set_timer_5": "Timer for five minutes",
    "set_timer_10": "Set a ten minute timer", "set_alarm_6am": "Set an alarm for six in the morning",
    "set_alarm_7am": "Wake me up at seven A M", "set_alarm_9pm": "Set an alarm for nine tonight",
    "set_temperature_18": "Change the temperature to eighteen", "set_temperature_22": "Set the temperature to twenty two degrees",
    "set_temperature_26": "Make it twenty six degrees", "pause": "Pause the music", "stop": "Stop the music",
    "next": "Play the next song", "skip": "Skip the song", "volume_up": "Turn up the volume", "louder": "A bit louder",
    "volume_down": "Lower the volume", "remind_trash": "Remind me about the trash",
    "remind_study": "Remind me to study for my test", "list_reminders": "Read my reminders",
    "call_jane": "Give Jane a call", "text_jane": "Send a message to Jane", "play": "Continue playing",
    "party": "Party time",
}
# unknown: sentences sharing words with the commands (contrast for the new wordings)
NEAR_MISS = [
    "Switch off the fan", "Switch on the TV", "Turn the TV off", "Turn the radio on", "The lights are too bright",
    "Pause for a second", "Stop talking please", "Call the doctor", "Message John", "Text me later",
    "Next time", "Skip lunch today", "Make it quick", "What time does the store open", "The weather was nice yesterday",
    "Turn left at the corner", "Party was fun last night", "Play basketball later", "Set the oven to two hundred",
    "Remind me later", "Keep it down", "Continue the story", "Resume the video", "Dim sum for lunch",
    "Lights out at ten", "Alarm clock is broken", "Timer is done", "Turn it off and on again",
]
for lab, w in HELD_OUT.items():
    assert w not in TRAIN[lab], (lab, w)


def texts():
    t = {"__delta__": "Hey Delta."}
    for lab, ws in TRAIN.items():
        t.update({f"{lab}#{i}": w for i, w in enumerate(ws)})
    t.update({f"unknown_other#nm{i}": w for i, w in enumerate(NEAR_MISS)})
    return t


def voice_split(v):
    from build_commands_all import voice_split as vs
    return vs(v)


def ph_piper_jobs(n_piper):
    """TRAIN/VALIDATION ph and piper voices with the settings of their training-data clips (as the phrasing test)."""
    import generate_synthetic_commands as gs
    a = pd.read_csv(ROOT / "data/commands_all.csv")
    ok = a[a.split.isin(["train", "validation"]) & a.dataset.isin(["synthetic_ph", "synthetic_piper"])]
    keep = dict(zip(ok.speaker, ok.split))
    m = pd.read_csv(gs.OUT / "manifest.csv")
    m = m[m.tier.isin(["ph", "piper"]) & m.voice.isin(keep)].drop_duplicates("voice")
    ph, pi = m[m.tier == "ph"], m[m.tier == "piper"]
    pi = pi.sample(min(n_piper, len(pi)), random_state=SEED)
    sys.path.insert(0, str(ROOT / "scripts"))
    import generate_tts_voices_v2 as g
    jobs = []
    for r in ph.itertuples(index=False):
        ev = next(v for v in gs.PH_VOICES if r.voice.startswith(v.replace("-", "m")))
        jobs.append(dict(voice=r.voice, tier="ph", edge_voice=ev, rate=r.speed, pitch=r.pitch, split=keep[r.voice]))
    for r in pi.itertuples(index=False):
        base, s = r.voice.rsplit("_spk", 1)
        onnx = g.voice_files(base)
        ns = json.loads(Path(onnx + ".json").read_text()).get("num_speakers", 1)
        jobs.append(dict(voice=r.voice, tier="piper", onnx=onnx, speaker=int(s) if ns > 1 else None,
                         length_scale=float(r.speed), split=keep[r.voice]))
    return jobs


def lab_class():
    lm = pd.read_csv(HERE.parent / "model/deploy/label_map.csv")
    return dict(zip(lm.label, lm["class"]))


def write_voice(gs, v, tier, split, audio, rng, cls):
    """One voice's phrases -> capture-like clips (+ bare phrase for Whisper); returns manifest rows."""
    delta = audio.get("__delta__")
    rows = []
    if delta is None:
        return rows
    for k, text in texts().items():
        y = audio.get(k)
        if k == "__delta__" or y is None or len(y) < 1600:
            continue
        lab, i = k.split("#")
        raw = OUT / "_raw" / f"{lab}__{v}__{i}.wav"
        gs.write_wav(raw, y)
        clip, _, _ = gs.assemble(delta, y, rng)
        name = f"{lab}/{lab}__{v}__{i}.wav"
        gs.write_wav(OUT / name, clip)
        rows.append(dict(path=str((OUT / name).relative_to(ROOT)), raw=str(raw.relative_to(ROOT)), label=lab,
                         **{"class": cls[lab]}, text=text, voice=v, tier=tier, split=split))
    return rows


def save_rows(rows):
    man = OUT / "manifest.csv"
    new = pd.DataFrame(rows)
    if man.exists():
        old = pd.read_csv(man)
        new = pd.concat([old, new[~new.path.isin(set(old.path))]], ignore_index=True)
    new.to_csv(man, index=False)
    return new


def synth(a):
    import generate_synthetic_commands as gs
    jobs = ph_piper_jobs(a.piper_voices)
    man = OUT / "manifest.csv"
    done = set(pd.read_csv(man).voice) if man.exists() else set()
    jobs = [j for j in jobs if j["voice"] not in done]
    t, cls = texts(), lab_class()
    print(f"{len(jobs)} voices to do ({sum(j['tier'] == 'ph' for j in jobs)} ph) x {len(t) - 1} phrases; "
          f"{len(done)} voices already done", flush=True)
    rng = np.random.default_rng(SEED)
    ph = [j for j in jobs if j["tier"] == "ph"]
    for j in ph:                                         # one voice at a time, saved as it finishes
        async def run():
            sem = asyncio.Semaphore(8)
            res = await asyncio.gather(*[gs.edge_say(s, j["edge_voice"], j["rate"], j["pitch"], sem) for s in t.values()])
            return dict(zip(t, res))
        audio = asyncio.run(run())
        save_rows(write_voice(gs, j["voice"], "ph", j["split"], audio, rng, cls))
        print(f"  ph {j['voice']} done", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    pj = [j for j in jobs if j["tier"] == "piper"]
    with ProcessPoolExecutor(6) as ex:                   # shared server: 6 workers, 1 onnx thread each
        for j, audio in zip(pj, ex.map(gs.piper_say_all, pj, [t] * len(pj))):
            save_rows(write_voice(gs, j["voice"], "piper", j["split"], audio, rng, cls))
            print(f"  piper {j['voice']} done", flush=True)
    print(f"manifest: {len(pd.read_csv(man))} clips", flush=True)


def clone(a):
    """Runs in envs/chatterbox. Cloned FLEURS voices of the TRAIN/VALIDATION split say a random 60% of the phrases
    (all near-misses kept). Their 'Hey Delta.' is reused from the clone stage."""
    import torch
    torch.set_num_threads(a.threads)
    from chatterbox.tts import ChatterboxTTS
    idx = json.loads((CLONE_STAGE / "index.json").read_text())
    rng = random.Random(SEED)
    pool = [e for e in idx if voice_split(e["voice"]) in ("train", "validation")]
    rng.shuffle(pool)
    pool = pool[:a.voices]
    STAGE.mkdir(parents=True, exist_ok=True)
    ip = STAGE / "index.json"
    index = json.loads(ip.read_text()) if ip.exists() else []
    done = {e["voice"] for e in index}
    model = ChatterboxTTS.from_pretrained(device="cuda")
    t = {k: v for k, v in texts().items() if k != "__delta__"}
    for n, e in enumerate(pool):
        if e["voice"] in done:
            continue
        r = random.Random(f"{SEED}|{e['voice']}")
        keys = [k for k in t if k.startswith("unknown") or r.random() < 0.6]
        out = STAGE / e["voice"]
        out.mkdir(exist_ok=True)
        files = {"__delta__": str(Path("..") / "commands_clone_stage" / e["files"]["__delta__"])}
        for k in keys:
            f = out / f"{k.replace('#', '_')}.wav"
            if not f.exists():                           # resume inside a voice
                with torch.inference_mode():
                    wav = model.generate(t[k], audio_prompt_path=str(ROOT / e["reference"]),
                                         exaggeration=e["exaggeration"], cfg_weight=0.5)
                sf.write(str(f), wav.squeeze(0).cpu().numpy(), model.sr)
            files[k] = f"{e['voice']}/{f.name}"
        index.append(dict(voice=e["voice"], texts={k: t[k] for k in keys}, files=files))
        ip.write_text(json.dumps(index, indent=1))
        print(f"clone voice {n + 1}/{len(pool)} {e['voice']} done ({len(keys)} phrases)", flush=True)


def assemble_clone(a):
    import generate_synthetic_commands as gs
    index = json.loads((STAGE / "index.json").read_text())
    man = OUT / "manifest.csv"
    done = set(pd.read_csv(man).voice) if man.exists() else set()
    rng, cls = np.random.default_rng(SEED + 2), lab_class()
    for e in index:
        if e["voice"] in done:
            continue
        audio = {}
        for k, f in e["files"].items():
            y, sr = sf.read(str((STAGE / f).resolve()), dtype="float32")
            audio[k] = gs.to16k(y, sr)
        save_rows(write_voice(gs, e["voice"], "clone", voice_split(e["voice"]), audio, rng, cls))
    print(f"manifest: {len(pd.read_csv(man))} clips", flush=True)


def check(a):
    """Same rule as the training data (generate_synthetic_commands): Whisper small, then large-v3 for what small
    rejects; kept if the normalised character edit ratio <= 0.2. `checked` records the passes done, so a restart
    continues where it stopped; progress is saved every 500 clips."""
    import generate_synthetic_commands as gs
    from faster_whisper import WhisperModel
    man = OUT / "manifest.csv"
    m = pd.read_csv(man)
    for c, v in (("heard", ""), ("edit_ratio", 9.0), ("checked", "")):
        if c not in m:
            m[c] = v
    m["heard"], m["checked"] = m.heard.fillna("").astype(str), m.checked.fillna("").astype(str)
    for size, todo in (("small", lambda: m.index[m.checked == ""]),
                       ("large-v3", lambda: m.index[(m.checked == "small") & (m.edit_ratio > 0.2)])):
        todo = list(todo())
        print(f"whisper {size}: {len(todo)} clips", flush=True)
        if not todo:
            continue
        wm = WhisperModel(size, device="cuda", device_index=0, compute_type="float16")
        for n, i in enumerate(todo):
            y = gs.trim(gs.to16k(*sf.read(str(ROOT / m.at[i, "raw"]), dtype="float32")))
            seg, _ = wm.transcribe(y, language="en", beam_size=5)
            heard = " ".join(s.text.strip() for s in seg).strip()
            er = round(gs.edit_ratio(gs.norm(heard), gs.norm(m.at[i, "text"])), 3)
            if er < m.at[i, "edit_ratio"]:
                m.at[i, "heard"], m.at[i, "edit_ratio"] = heard, er
            m.at[i, "checked"] = "small" if size == "small" else "large"
            if n % 500 == 499:
                m.assign(whisper_ok=m.edit_ratio <= 0.2).to_csv(man, index=False)
                print(f"  {n + 1}/{len(todo)}", flush=True)
        del wm
    m["whisper_ok"] = m.edit_ratio <= 0.2
    m.to_csv(man, index=False)
    print(f"kept {int(m.whisper_ok.sum())} of {len(m)} ({m.whisper_ok.mean():.0%})")
    print(m.groupby("tier").whisper_ok.mean().round(2).to_string())
    print(m.groupby("text").whisper_ok.mean().sort_values().head(15).round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synth", "clone", "assemble-clone", "check"])
    ap.add_argument("--piper-voices", type=int, default=60)
    ap.add_argument("--voices", type=int, default=30, help="clone: cloned voices")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    {"synth": synth, "clone": clone, "assemble-clone": assemble_clone, "check": check}[a.mode](a)


if __name__ == "__main__":
    main()
