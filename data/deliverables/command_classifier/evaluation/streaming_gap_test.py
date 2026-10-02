"""Streaming gap ablation: same owner test commands, one change at a time.
A  test clip as in evaluate_commands (full recording)
B  trimmed: command from cmd_start-0.15 (what the stream inserts), alone
C  ideal capture: 0.3 s pre-roll + 'Delta' end + chime gap + command + 1 s, over the room bed, cut by hand
D  real capture: same mini-stream through HeyDeltaListener
Only short (<=2.5 s) owner wake clips are used.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
torch.set_num_threads(4)
EV = Path("/mnt/jfs_hpc/home/arvir.jane.redondo/sandbox/AI231/ME2 v2/data/deliverables/command_classifier/evaluation")
sys.path.insert(0, str(EV))
import streaming_commands as sc
from streaming_commands import cp, CommandNet, HeyDeltaListener, Settings, SR, ROOT, load16

run = sc.CC / "model/runs" / sys.argv[1]
ck = torch.load(run / "best.pt", map_location="cpu")
classes = ck["classes"]; unk = len(classes) - 1
model = CommandNet(len(classes), ck["tau"]).eval(); model.load_state_dict(ck["state_dict"])
cut = json.loads((sc.OUT / "test_results.json").read_text())[run.name]["cutoffs"]
pcfg = cp.load_config()

def classify(y):
    w = cp.frame_command(torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)), pcfg).unsqueeze(0)
    with torch.no_grad():
        P = torch.softmax(model(w), 1)[0].numpy()
    return int(P.argmax()), float(P.max())

rng = np.random.default_rng(1)
bed, wakes, _, _, _, _ = sc.materials(rng)
wakes = [w for w in wakes if len(w) / SR <= 2.5]
a = pd.read_csv(ROOT / "data/commands_all.csv")
chk = pd.read_csv(sc.CC.parent / "model/deploy/recordings/_check/check.csv").set_index("file")
own = a[(a.split == "test") & (a.dataset == "owner_recordings") & (a.speaker == "owner") & (a["class"] != "unknown")]
rows = []
for path, c in zip(own.path, own["class"]):
    f = path.split("recordings/", 1)[1]
    if f not in chk.index or pd.isna(chk.loc[f, "cmd_start"]):
        continue
    full = load16(ROOT / path)
    cmd = full[int(max(0, chk.loc[f, "cmd_start"] - 0.15) * SR):]
    for rep in range(3):
        w = wakes[int(rng.integers(len(wakes)))]
        gap = rng.uniform(0.5, 0.9)
        lead, tail = 2.0, 4.0
        seq = np.concatenate([w, np.zeros(int(gap * SR), np.float32), cmd])
        n = int((lead + tail) * SR) + len(seq)
        off = int(rng.integers(0, len(bed) - n))
        x = bed[off:off + n].copy()
        s0 = int(lead * SR); x[s0:s0 + len(seq)] += seq; x = np.clip(x, -1, 1)
        wake_end = s0 + len(w)
        ideal = x[max(0, wake_end - int(0.48 * SR)): wake_end + int((gap + len(cmd) / SR + 1.0) * SR)]
        y0 = np.concatenate([w, np.zeros(int(gap * SR), np.float32), cmd])
        nobed = y0[max(0, len(w) - int(0.48 * SR)):]
        bedonly = np.clip(cmd + bed[off:off + len(cmd)], -1, 1)
        rec_db = 10 * np.log10(np.mean(full[:int(0.3 * SR)] ** 2) + 1e-12)
        bed_db = 10 * np.log10(np.mean(bed[off:off + n] ** 2) + 1e-12)
        caps = []
        L = HeyDeltaListener(on_command=lambda k: caps.append(k), settings=Settings(chimes=False), verbose=False)
        L.feed(x)
        cap = next((k for k in caps if k.audio is not None), None)
        r = dict(path=path, cls=c, rep=rep)
        for name, y in [("A_clip", full), ("B_trimmed", cmd), ("C_ideal_capture", ideal), ("C1_prefix_no_bed", nobed), ("C2_bed_no_prefix", bedonly),
                        ("D_listener", None if cap is None else cap.audio)]:
            if y is None:
                r[name] = "no_wake"; continue
            p, conf = classify(y)
            r[name] = classes[p]; r[name + "_conf"] = conf
        r["reason"] = None if cap is None else cap.reason
        r["cap_sec"] = None if cap is None else len(cap.audio) / SR
        r["rec_db"] = rec_db; r["bed_db"] = bed_db
        r["ideal_sec"] = len(ideal) / SR
        rows.append(r)
df = pd.DataFrame(rows)
out = sc.OUT / f"gap_ablation_{run.name}.csv"; df.to_csv(out, index=False)
print(f"{len(df)} trials, {df.path.nunique()} commands")
for name in ["A_clip", "B_trimmed", "C1_prefix_no_bed", "C2_bed_no_prefix", "C_ideal_capture", "D_listener"]:
    v = df[df[name] != "no_wake"]
    line = [f"{name:16s} n={len(v):3d}"]
    for rule, cv in [("argmax", 0.0)] + list(cut.items()):
        pred = np.where((v[name] != "unknown") & (v[name + "_conf"] < cv), "unknown", v[name])
        line.append(f"{rule}: ok {np.mean(pred == v.cls):.0%} wrong {np.mean((pred != v.cls) & (pred != 'unknown')):.0%} ask {np.mean(pred == 'unknown'):.0%}")
    print(" | ".join(line))
v = df[df.D_listener != "no_wake"]
print("listener by reason:", v.groupby("reason").apply(lambda g: f"{(g.D_listener == g.cls).mean():.0%} of {len(g)}").to_dict())
print("capture length (s):", v.cap_sec.describe().round(2).to_dict(), "ideal:", df.ideal_sec.describe().round(2).to_dict())
print("room level in owner clips (dB):", df.rec_db.describe().round(1).to_dict())
print("added bed level (dB):", df.bed_db.describe().round(1).to_dict())
