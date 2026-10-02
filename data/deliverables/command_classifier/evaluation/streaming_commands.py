#!/usr/bin/env python3
"""
Phase 4b: streaming evaluation of the whole command path, as on the Pi:
continuous audio -> HeyDeltaListener (BC-ResNet-6 wakeword ONNX, 2-of-3 rule, loudness end-of-speech 6/4 dB,
1.0 s pause) -> captured command -> command classifier (+ confidence cutoff) -> action.

    CUDA_VISIBLE_DEVICES=<one gpu> python streaming_commands.py ../model/runs/bcresnet3 [--streams 40]

Streams are built from TEST-split material only (fixed seed), 60 s each, over the owner's real room background
(test-split room recordings, unknown_room):
  owner      the owner's real "Hey Delta" (wakeword project, test-split Pi recordings) + a pause + the owner's real
             test-session command (capture audio from the first command word on)
  clone      a test-split cloned voice saying "Hey Delta." + a pause + a command or an off-list sentence (raw clone
             audio, scaled to the measured Pi speech level)
  distractor FLEURS test speech with no wakeword (should cause no wake-up, or at least no action)
Per event: was the device woken; then, for commands: correct / wrong command / rejected; for off-list sentences: was
an action triggered. Wake-ups at no scheduled event are counted as false wake-ups, with the action they caused.
Cutoffs are the validation-chosen ones in results/test_results.json (run evaluate_commands.py first).
Writes results/<model>_streaming.json and results/<model>_streaming_events.csv.
"""
from pathlib import Path
import argparse, json, sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
CC = HERE.parent
DEPLOY = CC.parent / "model/deploy"
sys.path.insert(0, str(CC / "model"))
sys.path.insert(0, str(CC / "dataloading"))
sys.path.insert(0, str(DEPLOY))
sys.path.insert(0, str(CC.parent / "dataloading"))
import command_data as cd                                                 # noqa: E402
import command_preprocessing as cp                                        # noqa: E402
from command_model import CommandNet                                       # noqa: E402
from heydelta_listener import HeyDeltaListener, Settings                   # noqa: E402
import augmentation as wk                                                  # noqa: E402

ROOT = cd.ROOT
SR = 16000
STAGE = ROOT / "external_raw/commands_clone_stage"
OUT = HERE / "results"


def load16(p):
    y, sr = sf.read(str(p), dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        y = resample_poly(y, SR, sr).astype(np.float32)
    return y


def at_level(y, target_db):
    pk = wk.peak_db(y.astype(np.float64), SR)
    return (y * 10 ** ((target_db - pk) / 20)).astype(np.float32)


def materials(rng):
    a = pd.read_csv(ROOT / "data/commands_all.csv")
    te = a[a.split == "test"]
    # background: owner room recordings (test split), joined
    room = te[te.dataset == "reuse_owner"].path.tolist()
    bed = np.concatenate([load16(ROOT / p) for p in room])
    # owner real wakewords (wakeword project test split, Pi recordings)
    inv = pd.read_csv(CC.parent / "segmentation_windowing/outputs/recording_inventory.csv", low_memory=False)
    wakes = inv[(inv.source == "manual_recording") & (inv.label == "positive") & (inv.split == "test")
                & inv.filepath.str.contains("RPI")].filepath.tolist()
    owner_wakes = [load16(ROOT / p) for p in wakes]
    # owner test-session commands, from the first command word on
    chk = pd.read_csv(CC.parent / "model/deploy/recordings/_check/check.csv").set_index("file")
    own = te[(te.dataset == "owner_recordings") & (te.speaker == "owner") & (te["class"] != "unknown")]
    owner_cmds = []
    for path, lab, c in zip(own.path, own.label, own["class"]):
        f = path.split("recordings/", 1)[1]
        if f in chk.index and pd.notna(chk.loc[f, "cmd_start"]):
            y = load16(ROOT / path)
            owner_cmds.append((lab, c, y[int(max(0, chk.loc[f, "cmd_start"] - 0.15) * SR):]))
    # cloned test voices
    idx = json.loads((STAGE / "index.json").read_text())
    syn = te[te.dataset == "synthetic_clone"].speaker.unique().tolist()
    lm = pd.read_csv(DEPLOY / "label_map.csv")
    cls = dict(zip(lm.label, lm["class"]))
    clone = []
    for e in idx:
        if e["voice"] not in syn:
            continue
        keys = [k for k in e["files"] if k != "__delta__" and (k.split("#")[0] in cls or k.startswith("unknown_other"))
                and not k.endswith(("#t2", "#t3"))]
        clone.append((e["voice"], STAGE / e["files"]["__delta__"], [(k, STAGE / e["files"][k]) for k in keys]))
    fleurs = te[te.dataset == "reuse_fleurs"].path.tolist()
    return bed, owner_wakes, owner_cmds, clone, fleurs, cls


def build_stream(rng, M, sec=60.0):
    bed, owner_wakes, owner_cmds, clone, fleurs, cls = M
    n = int(sec * SR)
    off = int(rng.integers(0, max(1, len(bed) - n)))
    x = np.array(bed[off:off + n] if len(bed) >= n else np.resize(bed, n), dtype=np.float32)
    events, t = [], float(rng.uniform(4, 6))
    while t < sec - 10:
        kind = rng.choice(["owner", "clone", "distractor"], p=[0.4, 0.45, 0.15])
        if kind == "distractor":
            y = at_level(load16(ROOT / fleurs[int(rng.integers(len(fleurs)))]), rng.uniform(-40, -30))
            a = int(t * SR)
            L = min(len(y), n - a)
            x[a:a + L] += y[:L]
            events.append(dict(kind=kind, t_wake_start=t, t_wake_end=None, t_end=t + L / SR, label="(none)", cls="none"))
            t += len(y) / SR + rng.uniform(3, 5)
            continue
        if kind == "owner":
            w = owner_wakes[int(rng.integers(len(owner_wakes)))]
            lab, c, cmd = owner_cmds[int(rng.integers(len(owner_cmds)))]
        else:
            v, wp, keys = clone[int(rng.integers(len(clone)))]
            lvl = rng.uniform(-40, -30)
            w = at_level(load16(wp), lvl)
            k, p = keys[int(rng.integers(len(keys)))]
            lab = k.split("#")[0]
            c = cls.get(lab, "unknown")
            cmd = at_level(load16(p), lvl + rng.uniform(-3, 3))
        gap = rng.uniform(0.5, 0.9)                         # the user waits for the chime
        seq = np.concatenate([w, np.zeros(int(gap * SR), np.float32), cmd])
        a = int(t * SR)
        if a + len(seq) > n:
            break
        x[a:a + len(seq)] += seq
        events.append(dict(kind=kind, t_wake_start=t, t_wake_end=t + len(w) / SR, t_end=t + len(seq) / SR, label=lab, cls=c))
        t += len(seq) / SR + rng.uniform(3, 5)
    return np.clip(x, -1, 1), events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--streams", type=int, default=40)
    a = ap.parse_args()
    run = Path(a.run).resolve()
    res = json.loads((OUT / "test_results.json").read_text())[run.name]
    cut = res["cutoffs"]
    ck = torch.load(run / "best.pt", map_location="cpu")
    classes = ck["classes"]
    unk = len(classes) - 1
    model = CommandNet(len(classes), ck["tau"]).eval().to("cuda")
    model.load_state_dict(ck["state_dict"])
    pcfg = cp.load_config()
    rng = np.random.default_rng(20260930)
    M = materials(rng)
    rows, false_wakes, total_sec = [], [], 0.0
    for si in range(a.streams):
        x, events = build_stream(rng, M)
        total_sec += len(x) / SR
        caps = []
        L = HeyDeltaListener(on_command=lambda c: caps.append(c), settings=Settings(chimes=False), verbose=False)
        L.feed(x)
        dec = []
        for c in caps:
            if c.audio is None:
                dec.append((c.wake_time, None, None, None))
                continue
            w = cp.frame_command(torch.from_numpy(c.audio), pcfg).unsqueeze(0).to("cuda")
            with torch.no_grad():
                P = torch.softmax(model(w), 1)[0].cpu().numpy()
            dec.append((c.wake_time, int(P.argmax()), float(P.max()), c.reason))
        used = set()
        for e in events:
            m = None
            if e["t_wake_end"] is not None:
                for j, (tw, *_rest) in enumerate(dec):
                    if j not in used and e["t_wake_start"] - 0.2 <= tw <= e["t_wake_end"] + 1.5:
                        m = j
                        used.add(j)
                        break
            else:
                for j, (tw, *_rest) in enumerate(dec):
                    if j not in used and e["t_wake_start"] - 0.2 <= tw <= e["t_end"] + 1.5:
                        m = j
                        used.add(j)
                        break
            r = dict(stream=si, **e, woke=m is not None)
            if m is not None:
                tw, p, conf, reason = dec[m]
                r.update(wake_time=tw, capture_reason=reason)
                for rule, cval in [("argmax", 0.0)] + list(cut.items()):
                    if p is None:
                        r[rule] = "unknown"
                    else:
                        r[rule] = classes[p] if (p == unk or conf >= cval) else "unknown"
            rows.append(r)
        for j, (tw, p, conf, reason) in enumerate(dec):
            if j not in used:
                f = dict(stream=si, wake_time=tw, capture_reason=reason)
                for rule, cval in [("argmax", 0.0)] + list(cut.items()):
                    f[rule] = "unknown" if p is None else (classes[p] if (p == unk or conf >= cval) else "unknown")
                false_wakes.append(f)
        print(f"stream {si + 1}/{a.streams}: {len(events)} events, {len(caps)} wake-ups", flush=True)

    ev = pd.DataFrame(rows)
    ev.to_csv(OUT / f"{run.name}_streaming_events.csv", index=False)
    fw = pd.DataFrame(false_wakes)
    hours = total_sec / 3600
    out = dict(model=run.name, streams=a.streams, hours=round(hours, 3), cutoffs=cut, by_kind={})
    for kind, g in ev.groupby("kind"):
        d = dict(events=int(len(g)), woke=round(float(g.woke.mean()), 3))
        woke = g[g.woke]
        for rule in ["argmax"] + list(cut):
            if kind == "distractor" or len(woke) == 0:
                continue
            cmd = woke[woke.cls != "unknown"]
            off = woke[woke.cls == "unknown"]
            d[rule] = dict(
                command_correct=round(float((cmd[rule] == cmd.cls).mean()), 3) if len(cmd) else None,
                command_wrong=round(float(((cmd[rule] != cmd.cls) & (cmd[rule] != "unknown")).mean()), 3) if len(cmd) else None,
                command_rejected=round(float((cmd[rule] == "unknown").mean()), 3) if len(cmd) else None,
                offlist_triggered_action=round(float((off[rule] != "unknown").mean()), 3) if len(off) else None,
                end_to_end_correct=round(float(((g.cls != "unknown") & g.woke & (g[rule] == g.cls)).sum()
                                               / max(1, (g.cls != "unknown").sum())), 3))
        out["by_kind"][kind] = d
    out["false_wakeups_per_hour"] = round(len(fw) / hours, 2)
    out["false_wakeup_actions_per_hour"] = {rule: round(float((fw[rule] != "unknown").sum()) / hours, 2) if len(fw) else 0.0
                                            for rule in ["argmax"] + list(cut)}
    dist = ev[ev.kind == "distractor"]
    out["distractor_woke_rate"] = round(float(dist.woke.mean()), 3) if len(dist) else None
    (OUT / f"{run.name}_streaming.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
