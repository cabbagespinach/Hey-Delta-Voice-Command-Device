#!/usr/bin/env python3
"""
Phrasing test (2026-10-01): does the command model recognise OTHER WAYS of saying a command ("turn the lights on",
"switch the lights on", "lights on" for lights_on), or only the one phrasing it was trained on?

TEST ONLY: nothing here is used for training. Every phrasing in VARIANTS (the trained one first) is spoken by the
synthetic voices of the TEST split (never heard in training): 6 Philippine-accent edge-tts voices and 14 Piper voices,
with the same voice settings as their training-data clips, assembled like a capture ("Delta" tail, pause, command,
quiet; generate_synthetic_commands.assemble). Whisper checks each phrase as in the training data; rejects are dropped
and counted.

    python phrasing_variants_test.py synth                      # CPU + internet (edge-tts); no GPU
    CUDA_VISIBLE_DEVICES=<one gpu> python phrasing_variants_test.py score ../model/runs/<run> [...]

score: Whisper check (first time only), then each model at its validation-chosen cutoffs (evaluate_commands.py).
Writes results/phrasing_variants_<run>.csv (per clip) and results/phrasing_variants_summary.md (per phrasing:
correct / wrong command / rejected as unknown at the balanced cutoff; trained phrasing vs other phrasings).
"""
from pathlib import Path
import asyncio, json, sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CC = HERE.parent
sys.path.insert(0, str(CC))
import generate_synthetic_commands as gs                                  # noqa: E402  (TTS, assembly, Whisper text check)

ROOT = gs.ROOT
OUT = ROOT / "data/commands_variants_test"
RES = HERE / "results"
SEED = 20261001

# label -> phrasings, the TRAINED one first (numbers in words, as in command_texts.SPOKEN). Labels map to classes
# through deploy/label_map.csv (skip -> next, louder -> volume_up).
VARIANTS = {
    "lights_on": ["Turn on lights", "Turn on the lights", "Turn the lights on", "Switch on the lights",
                  "Switch the lights on", "Lights on"],
    "lights_off": ["Turn off lights", "Turn off the lights", "Turn the lights off", "Switch off the lights",
                   "Switch the lights off", "Lights off"],
    "play_music": ["Play music", "Play some music", "Put on some music", "Start the music"],
    "weather": ["What's the weather?", "How's the weather?", "What's the weather like today?", "Weather today"],
    "time": ["What time is it?", "What's the time?", "Tell me the time", "What's the time now?"],
    "dim_lights_20": ["Dim lights to twenty percent", "Dim the lights to twenty percent",
                      "Set the lights to twenty percent", "Lights at twenty percent"],
    "dim_lights_50": ["Dim lights to fifty percent", "Dim the lights to fifty percent",
                      "Set the lights to fifty percent", "Lights at fifty percent"],
    "dim_lights_80": ["Dim lights to eighty percent", "Dim the lights to eighty percent",
                      "Set the lights to eighty percent", "Lights at eighty percent"],
    "set_timer_1": ["Set a timer for one minute", "Set a one minute timer", "Start a timer for one minute",
                    "Timer for one minute"],
    "set_timer_5": ["Set a timer for five minutes", "Set a five minute timer", "Start a timer for five minutes",
                    "Timer for five minutes"],
    "set_timer_10": ["Set a timer for ten minutes", "Set a ten minute timer", "Start a timer for ten minutes",
                     "Timer for ten minutes"],
    "set_alarm_6am": ["Set an alarm at six A M", "Set an alarm for six A M", "Wake me up at six A M",
                      "Set an alarm for six in the morning"],
    "set_alarm_7am": ["Set an alarm at seven A M", "Set an alarm for seven A M", "Wake me up at seven A M",
                      "Set an alarm for seven in the morning"],
    "set_alarm_9pm": ["Set an alarm at nine P M", "Set an alarm for nine P M", "Set an alarm for nine tonight",
                      "Set an alarm for nine in the evening"],
    "set_temperature_18": ["Set temperature to eighteen degrees", "Set the temperature to eighteen degrees",
                           "Make it eighteen degrees", "Change the temperature to eighteen"],
    "set_temperature_22": ["Set temperature to twenty two degrees", "Set the temperature to twenty two degrees",
                           "Make it twenty two degrees", "Change the temperature to twenty two"],
    "set_temperature_26": ["Set temperature to twenty six degrees", "Set the temperature to twenty six degrees",
                           "Make it twenty six degrees", "Change the temperature to twenty six"],
    "pause": ["Pause", "Pause the music", "Pause it"],
    "stop": ["Stop", "Stop the music", "Stop playing"],
    "next": ["Next", "Next song", "Play the next song"],
    "skip": ["Skip", "Skip this song", "Skip the song"],
    "volume_up": ["Volume up", "Turn up the volume", "Turn it up"],
    "louder": ["Louder", "Make it louder", "A bit louder"],
    "volume_down": ["Volume down", "Turn down the volume", "Turn it down", "Quieter", "Lower the volume"],
    "remind_trash": ["Remind me to take out the trash", "Remind me to take the trash out",
                     "Remind me about the trash"],
    "remind_study": ["Remind me to study for my exam", "Remind me to study for my test", "Remind me to study"],
    "list_reminders": ["What are my reminders?", "What reminders do I have?", "Do I have any reminders?",
                       "Read my reminders"],
    "call_jane": ["Call Jane", "Phone Jane", "Give Jane a call", "Call Jane please"],
    "text_jane": ["Text Jane", "Send Jane a text", "Message Jane", "Send a message to Jane"],
    "play": ["Play", "Resume", "Keep playing", "Continue playing"],
    "party": ["Party party", "Party mode", "Let's party", "Party time"],
}


def test_voices():
    """TEST-split ph and piper voices with the settings of their training-data clips."""
    a = pd.read_csv(ROOT / "data/commands_all.csv")
    test = set(a[(a.split == "test") & a.dataset.isin(["synthetic_ph", "synthetic_piper"])].speaker)
    m = pd.read_csv(gs.OUT / "manifest.csv")
    m = m[m.tier.isin(["ph", "piper"]) & m.voice.isin(test)].drop_duplicates("voice")
    sys.path.insert(0, str(ROOT))
    import generate_tts_voices_v2 as g
    jobs = []
    for r in m.itertuples(index=False):
        if r.tier == "ph":
            ev = next(v for v in gs.PH_VOICES if r.voice.startswith(v.replace("-", "m")))
            jobs.append(dict(voice=r.voice, tier="ph", engine="edge", edge_voice=ev, rate=r.speed, pitch=r.pitch))
        else:
            base, s = r.voice.rsplit("_spk", 1)
            onnx = g.voice_files(base)
            ns = json.loads(Path(onnx + ".json").read_text()).get("num_speakers", 1)
            jobs.append(dict(voice=r.voice, tier="piper", engine="piper", onnx=onnx,
                             speaker=int(s) if ns > 1 else None, length_scale=float(r.speed)))
    return jobs


def synth():
    jobs = test_voices()
    texts = {"__delta__": "Hey Delta."}
    for lab, ps in VARIANTS.items():
        texts.update({f"{lab}#{i}": p for i, p in enumerate(ps)})
    print(f"{len(jobs)} test voices ({sum(j['tier'] == 'ph' for j in jobs)} ph) x {len(texts) - 1} phrasings")
    audio = {}
    ph = [j for j in jobs if j["tier"] == "ph"]

    async def run():
        sem = asyncio.Semaphore(8)
        keys = [(j["voice"], k) for j in ph for k in texts]
        res = await asyncio.gather(*[gs.edge_say(texts[k], j["edge_voice"], j["rate"], j["pitch"], sem)
                                     for j in ph for k in texts])
        return dict(zip(keys, res))
    audio.update(asyncio.run(run()))
    from concurrent.futures import ProcessPoolExecutor
    pj = [j for j in jobs if j["tier"] == "piper"]
    with ProcessPoolExecutor(8) as ex:                 # shared server: 8 workers, 1 onnx thread each
        for j, res in zip(pj, ex.map(gs.piper_say_all, pj, [texts] * len(pj))):
            audio.update({(j["voice"], k): y for k, y in res.items()})
    rng = np.random.default_rng(SEED)
    lm = pd.read_csv(gs.LABELS)
    cls = dict(zip(lm.label, lm["class"]))
    rows, failed = [], 0
    for j in jobs:
        v, delta = j["voice"], audio.get((j["voice"], "__delta__"))
        for k, text in texts.items():
            if k == "__delta__":
                continue
            y = audio.get((v, k))
            if y is None or delta is None:
                failed += 1
                continue
            lab, i = k.split("#")
            raw = OUT / "_raw" / f"{lab}__{v}__{i}.wav"     # the bare phrase, for the Whisper check
            gs.write_wav(raw, y)
            clip, _, _ = gs.assemble(delta, y, rng)
            name = f"{lab}/{lab}__{v}__{i}.wav"
            gs.write_wav(OUT / name, clip)
            rows.append(dict(path=str((OUT / name).relative_to(ROOT)), raw=str(raw.relative_to(ROOT)), label=lab,
                             **{"class": cls[lab]}, phrasing=text, trained=(i == "0"), speaker=v, tier=j["tier"]))
    pd.DataFrame(rows).to_csv(OUT / "manifest.csv", index=False)
    print(f"wrote {len(rows)} clips; {failed} failed to synthesize -> {OUT / 'manifest.csv'}")


def whisper_check(m):
    """Same rule as the training data: small, then large-v3 for what small rejects; keep if edit ratio <= 0.2."""
    from faster_whisper import WhisperModel
    m = m.copy()
    m["heard"], m["edit_ratio"] = "", 9.0
    for size in ("small", "large-v3"):
        wm = WhisperModel(size, device="cuda", device_index=0, compute_type="float16")
        for i in m.index[m.edit_ratio > 0.2]:
            y = gs.trim(gs.to16k(*gs.sf.read(str(ROOT / m.at[i, "raw"]), dtype="float32")))
            seg, _ = wm.transcribe(y, language="en", beam_size=5)
            heard = " ".join(s.text.strip() for s in seg).strip()
            er = gs.edit_ratio(gs.norm(heard), gs.norm(m.at[i, "phrasing"]))
            if er < m.at[i, "edit_ratio"]:
                m.at[i, "heard"], m.at[i, "edit_ratio"] = heard, round(er, 3)
        del wm
    m["whisper_ok"] = m.edit_ratio <= 0.2
    return m


def score(runs):
    import torch
    sys.path.insert(0, str(CC / "model")), sys.path.insert(0, str(CC / "dataloading")), sys.path.insert(0, str(HERE))
    import command_data as cd
    from command_model import CommandNet
    from train_commands import predict
    from evaluate_commands import TARGETS, choose_cutoff, decide
    man = OUT / "manifest.csv"
    m = pd.read_csv(man)
    if "whisper_ok" not in m:
        m = whisper_check(m)
        m.to_csv(man, index=False)
    print(f"Whisper: {int(m.whisper_ok.sum())} of {len(m)} clips say their phrasing; the rest are left out")
    rows = m[m.whisper_ok].reset_index(drop=True).assign(dataset="variants_test", split="test", group="synthetic",
                                                         needs_tail=False)
    cfg = cd.load_config()
    ds = cd.EvalClips(cfg, "test", rows=rows)
    classes = cd.classes()
    unk = len(classes) - 1
    md = ["# Phrasing test: other ways of saying each command", "",
          f"{len(rows)} clips: {rows.phrasing.nunique()} phrasings x {rows.speaker.nunique()} TEST-split synthetic "
          f"voices ({(rows.drop_duplicates('speaker').tier == 'ph').sum()} Philippine-accent edge-tts, the rest "
          "Piper), assembled like captures. `trained` = the phrasing in the training data (same wording, voices "
          f"never heard). {int((~m.whisper_ok).sum())} clips dropped because Whisper did not hear the phrasing.", ""]
    for run in [Path(r).resolve() for r in runs]:
        ck = torch.load(run / "best.pt", map_location="cpu")
        assert ck["classes"] == classes
        model = CommandNet(len(classes), ck["tau"]).to("cuda")
        model.load_state_dict(ck["state_dict"])
        vp = pd.read_csv(run / "val_predictions.csv")
        cut = {k: choose_cutoff(vp[[f"p_{c}" for c in classes]].to_numpy(), vp, classes, t) for k, t in TARGETS.items()}
        P = predict(model, ds, cfg, "cuda")
        r = rows.copy()
        y = r["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        for name, c in [("argmax", None)] + list(cut.items()):
            p = decide(P, classes, c)
            r[f"pred_{name}"] = [classes[k] for k in p]
            r[f"out_{name}"] = np.where(p == y, "correct", np.where(p == unk, "rejected", "wrong"))
        r["p_max"] = P.max(1).round(3)
        r.drop(columns=["raw"]).to_csv(RES / f"phrasing_variants_{run.name}.csv", index=False)

        o = "out_balanced"
        rate = lambda g: pd.Series({"clips": len(g), "correct": (g[o] == "correct").mean(),
                                    "wrong command": (g[o] == "wrong").mean(),
                                    "rejected": (g[o] == "rejected").mean(),
                                    "wrong as": ", ".join(sorted(set(g.loc[g[o] == "wrong", "pred_balanced"])))})
        pct = lambda d: d.assign(**{k: (d[k] * 100).round(0).astype(int).astype(str) + "%"
                                    for k in ("correct", "wrong command", "rejected")})
        md += [f"## {run.name} (balanced cutoff {cut['balanced']}; cautious {cut['cautious']})", ""]
        overall = r.groupby(np.where(r.trained, "trained phrasing", "other phrasings")).apply(rate)
        md += [pct(overall).to_markdown(), "", "Per class, other phrasings only (worst first):", ""]
        pc = r[~r.trained].groupby("class").apply(rate).sort_values("correct")
        pc.insert(1, "trained phrasing correct",
                  r[r.trained].groupby("class").apply(lambda g: f"{(g[o] == 'correct').mean() * 100:.0f}%"))
        md += [pct(pc).to_markdown(), "", "Per phrasing:", ""]
        pp = r.groupby(["class", "phrasing", "trained"], sort=False).apply(rate).reset_index()
        md += [pct(pp).to_markdown(index=False), ""]
        print(pct(overall).to_string())
    (RES / "phrasing_variants_summary.md").write_text("\n".join(md))
    print(f"-> {RES / 'phrasing_variants_summary.md'}")


if __name__ == "__main__":
    {"synth": lambda: synth(), "score": lambda: score(sys.argv[2:])}[sys.argv[1]]()
