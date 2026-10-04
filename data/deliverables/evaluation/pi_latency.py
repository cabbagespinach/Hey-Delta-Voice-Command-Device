#!/usr/bin/env python3
"""
Raspberry Pi 5 latency of every deployed model, from a clone of the repository (owner, 2026-10-04).
Needs only:  pip install numpy onnxruntime

    python3 data/deliverables/evaluation/pi_latency.py            # about 2 minutes on a Pi 5
    python3 data/deliverables/evaluation/pi_latency.py --quick    # fewer repetitions (a smoke test)

Times the ONNX files exactly as the Pi runs them (ONNX Runtime, CPU, ONE thread; each file holds the front end and
the network, so the time is audio in -> probabilities out):

* wakeword  data/deliverables/model/deploy/heydelta_bcresnet6.onnx: one 1.5 s window, run every 100 ms while listening
* commands  data/deliverables/command_classifier/model/export/<run>/*.onnx: one 5 s capture, run once per command

Writes data/deliverables/evaluation/pi_latency/<device>_<date>.json and .md (median / p95 / max, real-time factor,
share of one core, memory, CPU temperature, throttling). Input is low-level noise: these networks have no
data-dependent branches, so the time does not depend on what is said.
"""
from pathlib import Path
import argparse, datetime, json, platform, re, resource, socket, subprocess, sys, time

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
DELIV = HERE.parent
OUT = HERE / "pi_latency"
SR = 16000
WAKE = dict(name="wakeword BC-ResNet-6", file=DELIV / "model/deploy/heydelta_bcresnet6.onnx", seconds=1.5,
            every_ms=100)
COMMANDS = ["bcresnet6_hf_plus", "bcresnet6_hf_only", "bcresnet6_schema_b", "dscnn_schema_b"]


def device_info():
    info = dict(host=socket.gethostname(), python=sys.version.split()[0], onnxruntime=ort.__version__,
                numpy=np.__version__, machine=platform.machine(), platform=platform.platform())
    try:
        info["device"] = Path("/proc/device-tree/model").read_text().strip("\x00")
    except OSError:
        info["device"] = "not a Raspberry Pi (no /proc/device-tree/model)"
    try:
        info["cpu"] = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text()).group(1)
    except (OSError, AttributeError):
        pass
    return info


def temp_c():
    try:
        return round(int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000, 1)
    except (OSError, ValueError):
        return None


def throttled():
    try:
        return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True,
                              timeout=2).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1                      # as deployed: one core, the other three stay free
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def bench(path, seconds, warmup, n):
    s = session(path)
    inp, out = s.get_inputs()[0].name, s.get_outputs()[0].name
    x = np.random.default_rng(0).normal(0, 0.01, (1, int(seconds * SR))).astype(np.float32)
    for _ in range(warmup):
        s.run([out], {inp: x})
    lat, c0 = [], time.process_time()
    for _ in range(n):
        t = time.perf_counter()
        s.run([out], {inp: x})
        lat.append((time.perf_counter() - t) * 1000)
    cpu_ms = (time.process_time() - c0) / n * 1000
    lat = np.array(lat)
    return dict(ms_median=round(float(np.median(lat)), 2), ms_p95=round(float(np.percentile(lat, 95)), 2),
                ms_max=round(float(lat.max()), 2), cpu_ms_per_run=round(cpu_ms, 2),
                real_time_factor=round(float(np.median(lat)) / (seconds * 1000), 4), runs=n,
                onnx_mb=round(path.stat().st_size / 1e6, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer repetitions (smoke test, not for reporting)")
    a = ap.parse_args()
    n_wake, n_cmd, warm = (30, 5, 3) if a.quick else (600, 100, 20)

    info = device_info()
    print(f"device: {info['device']}  ({info.get('cpu', info['machine'])}), onnxruntime {info['onnxruntime']}, "
          f"1 thread")
    res = dict(date=datetime.datetime.now().isoformat(timespec="seconds"), device=info, threads=1,
               quick=a.quick, temp_c_start=temp_c(), throttled_start=throttled(), models={})

    r = bench(WAKE["file"], WAKE["seconds"], warm, n_wake)
    r["share_of_one_core_pct"] = round(r["cpu_ms_per_run"] / WAKE["every_ms"] * 100, 1)
    res["models"]["wakeword_bcresnet6"] = dict(r, file=str(WAKE["file"].relative_to(DELIV.parents[1])),
                                               input_s=WAKE["seconds"], runs_every_ms=WAKE["every_ms"])
    print(f"{WAKE['name']:<28} {r['ms_median']:7.2f} ms per 1.5 s window (p95 {r['ms_p95']:.2f}), "
          f"{r['share_of_one_core_pct']:.1f}% of one core while listening")

    for run in COMMANDS:
        ex = DELIV / "command_classifier/model/export" / run
        cfg = json.loads((ex / "command_config.json").read_text())
        f = ex / cfg["model_file"]
        r = bench(f, cfg["num_samples"] / cfg["sample_rate"], warm, n_cmd)
        res["models"][run] = dict(r, file=str(f.relative_to(DELIV.parents[1])), input_s=cfg["num_samples"] / SR)
        print(f"command {run:<20} {r['ms_median']:7.2f} ms per 5 s capture (p95 {r['ms_p95']:.2f}), "
              f"real-time factor {r['real_time_factor']:.4f}")

    res["temp_c_end"], res["throttled_end"] = temp_c(), throttled()
    res["peak_memory_mb_all_models_loaded"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    print(f"peak memory {res['peak_memory_mb_all_models_loaded']} MB; CPU temperature {res['temp_c_start']} -> "
          f"{res['temp_c_end']} C; throttled {res['throttled_end']}")

    OUT.mkdir(exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9]+", "-", info["device"])[:40].strip("-") + ("_quick" if a.quick else "")
    stem = OUT / f"{tag}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    stem.with_suffix(".json").write_text(json.dumps(res, indent=2) + "\n")
    rows = ["| model | input | median ms | p95 ms | max ms | real-time factor | ONNX MB |", "|---|---|---:|---:|---:|---:|---:|"]
    for k, m in res["models"].items():
        rows.append(f"| {k} | {m['input_s']:g} s | {m['ms_median']} | {m['ms_p95']} | {m['ms_max']} | "
                    f"{m['real_time_factor']} | {m['onnx_mb']} |")
    w = res["models"]["wakeword_bcresnet6"]
    stem.with_suffix(".md").write_text(
        f"# Latency on {info['device']}\n\n{res['date']}; ONNX Runtime {info['onnxruntime']}, 1 thread, CPU; "
        f"{'QUICK (smoke test)' if a.quick else f'{n_wake} wakeword / {n_cmd} command runs after {warm} warm-up runs'}"
        f"; script `data/deliverables/evaluation/pi_latency.py`.\n\n" + "\n".join(rows) +
        f"\n\nWakeword while listening (one window every 100 ms): **{w['share_of_one_core_pct']}% of one core**. "
        f"Peak memory with all models loaded: {res['peak_memory_mb_all_models_loaded']} MB. CPU temperature "
        f"{res['temp_c_start']} -> {res['temp_c_end']} C; throttled: {res['throttled_end']}.\n")
    print(f"\nsaved {stem.with_suffix('.json')}\n      {stem.with_suffix('.md')}")


if __name__ == "__main__":
    main()
