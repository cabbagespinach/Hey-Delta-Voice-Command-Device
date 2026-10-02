#!/usr/bin/env python3
"""
"Hey Delta" on the Raspberry Pi 5: correctness check and a quick speed benchmark for every model in models.json.
Needs only:  pip install numpy onnxruntime

    python3 heydelta_pi.py check     # does the Pi give the same scores as the training server? (bundled clips)
    python3 heydelta_pi.py bench     # time per window, CPU share and memory for each model

For listening through the microphone (all models at once, with detection and efficiency reports) use
heydelta_live.py (see README_PI.md). Every command writes a JSON file in results/.
"""
from pathlib import Path
import argparse, datetime, json, platform, resource, sys, time

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
MODELS = json.loads((HERE / "models.json").read_text())["models"]
SR, WIN, HOP = 16000, 24000, 1600
RESULTS = HERE / "results"


def session(fname):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1           # one core: the other three stay free for speech-to-text etc.
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(HERE / fname), so, providers=["CPUExecutionProvider"])


def device_info():
    info = dict(python=sys.version.split()[0], onnxruntime=ort.__version__, machine=platform.machine(),
                platform=platform.platform())
    try:
        info["device"] = Path("/proc/device-tree/model").read_text().strip("\x00")
    except OSError:
        pass
    return info


def save(name, obj):
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(obj, indent=2) + "\n")
    print(f"\nsaved {p}")


def cmd_check(_):
    ref = np.load(HERE / "reference_windows.npz")
    out = dict(device=device_info(), models={})
    for m in MODELS:
        s = session(m["file"])
        p = np.concatenate([s.run(["prob"], {"audio": ref["audio"][i:i + 1]})[0] for i in range(len(ref["audio"]))])
        exp = ref[f"prob_{m['name']}"]
        d = np.abs(p - exp)
        flips = int(((p >= m["threshold"]) != (exp >= m["threshold"])).sum())
        ok = bool(d.max() < 1e-3 and flips == 0)
        out["models"][m["name"]] = dict(max_abs_diff=float(d.max()), decisions_changed=flips, ok=ok)
        print(f"{m['name']:<14} max difference {d.max():.2e}, decisions changed {flips} -> {'OK' if ok else 'MISMATCH'}")
    save("check", out)


def cmd_bench(a):
    x = np.random.default_rng(0).normal(0, 0.01, (1, WIN)).astype(np.float32)
    out = dict(device=device_info(), models={})
    for m in MODELS:
        s = session(m["file"])
        for _ in range(20):
            s.run(["prob"], {"audio": x})
        lat, c0 = [], time.process_time()
        for _ in range(a.n):
            t = time.perf_counter()
            s.run(["prob"], {"audio": x})
            lat.append((time.perf_counter() - t) * 1000)
        cpu_ms = (time.process_time() - c0) / a.n * 1000
        lat = np.array(lat)
        out["models"][m["name"]] = dict(ms_per_window_median=round(float(np.median(lat)), 2),
                                        ms_per_window_p95=round(float(np.percentile(lat, 95)), 2),
                                        cpu_share_of_one_core_pct=round(cpu_ms / (HOP / SR * 1000) * 100, 2))
        r = out["models"][m["name"]]
        print(f"{m['name']:<14} {r['ms_per_window_median']:6.2f} ms per window (p95 {r['ms_per_window_p95']:.2f}), "
              f"{r['cpu_share_of_one_core_pct']:.1f}% of one core")
    out["peak_memory_mb_all_models_loaded"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    print(f"peak memory with all models loaded: {out['peak_memory_mb_all_models_loaded']} MB")
    save("bench", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    b = sub.add_parser("bench")
    b.add_argument("--n", type=int, default=1000, help="windows to time per model")
    a = ap.parse_args()
    {"check": cmd_check, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    main()
