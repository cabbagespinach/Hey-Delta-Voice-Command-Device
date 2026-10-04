#!/usr/bin/env python3
"""
"Hey Delta" voice assistant for the class benchmark (github.com/airimonda/vcm-benchmark), run on the Raspberry Pi.

    python3 vcm_bench_assistant.py --model hf_plus              # listen; one log line per command -> bench.log
    python3 vcm_bench_assistant.py --model hf_only --rule balanced
    python3 vcm_bench_assistant.py --model hf_plus --also hf_only                # several models, same captures
    python3 vcm_bench_assistant.py --selftest                   # no microphone: classify the reference clips once

Pipeline = the deployed one: wakeword listener (listener/, BC-ResNet-6, threshold 0.7335, 2 of 3 windows), then
ONE command model (models/<name>/). Every decision is written to --log as one JSON line, flushed at once:

    {"event": "wake", "wake_prob": 0.97}                                         when "Hey Delta" fires
    {"intent": "TIMER_30S", "slot": "", "infer_ms": 61.2, "audio_ms": 5000, ...}  when the command is decided

* intent: our class name; the benchmark splits joint names itself (TIMER_30S -> TIMER, "30s"; checked for all 31
  classes). `unknown` is written as OUT_OF_SCOPE.
* infer_ms: feature extraction + model (the ONNX file contains both), timed around the model call only.
* audio_ms: 5000 = the audio the model processes (the first 5 s of the capture, padded if shorter). The real capture
  length is logged too (capture_ms), for reference.
* No speech after the wake word: nothing is written (the benchmark counts that as no response).

--also (2026-10-03): other models answer the SAME capture. The --model line goes to --log first (the benchmark
scores it; its timing is not delayed by the others) with a "seq" number; then every model (the --model one too) is
run and its answer under all three rules is written to --others-log as one line with the same "seq":

    {"seq": 3, "models": {"bcresnet6_hf_only": {"argmax": "NEXT", "cautious": "OUT_OF_SCOPE", "balanced": "NEXT",
                                               "confidence": 0.72, "infer_ms": 64.0}, ...}}

The benchmark never reads --others-log; score_multi.py (on the laptop) joins it to the benchmark's trials by "seq".
"""
from pathlib import Path
import argparse, json, sys, time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "listener"))
from command_pi import CommandClassifier                                    # noqa: E402

MODELS = {"hf_plus": "bcresnet6_hf_plus", "hf_only": "bcresnet6_hf_only", "hf_ourlabels": "bcresnet6_hf_ourlabels"}
RULES = ("argmax", "cautious", "balanced")


def bench_intent(clf, p, cutoff):
    """The model's answer under one cutoff (None = argmax), as the benchmark reads it."""
    k = int(np.argmax(p))
    label = clf.classes[k]
    if cutoff is not None and p[k] < cutoff:
        label = "unknown"
    label = clf.cfg.get("class_map", {}).get(label, label)
    return "OUT_OF_SCOPE" if label == "unknown" else label


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), default="hf_plus")
    ap.add_argument("--rule", choices=list(RULES), help="default: the model's default (cautious)")
    ap.add_argument("--also", default="", help="comma-separated other models run on the same captures, e.g. "
                                               "hf_only (answers -> --others-log)")
    ap.add_argument("--log", default=str(HERE / "bench.log"))
    ap.add_argument("--others-log", help="default: next to --log, bench_others.jsonl")
    ap.add_argument("--device", help="microphone (number or name); default = system default")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    run = MODELS[a.model]
    clf = CommandClassifier(HERE / "models" / run / "command_config.json", rule=a.rule, threads=a.threads)
    also = [m.strip() for m in a.also.split(",") if m.strip()]
    for m in also:
        if m not in MODELS:
            ap.error(f"--also: unknown model {m!r} (choose from {', '.join(sorted(MODELS))})")
    extra = {MODELS[m]: CommandClassifier(HERE / "models" / MODELS[m] / "command_config.json", threads=a.threads)
             for m in also if m != a.model}
    log = open(a.log, "a", buffering=1)
    others_path = Path(a.others_log) if a.others_log else Path(a.log).with_name("bench_others.jsonl")
    others = open(others_path, "a", buffering=1) if extra else None
    print(f"model {run}  rule {clf.rule}  cutoff {clf.cutoff}  log {a.log}", flush=True)
    if extra:
        print(f"also {', '.join(extra)}  -> {others_path}", flush=True)
    window_ms = clf.cfg["num_samples"] / clf.cfg["sample_rate"] * 1000
    seq = [int(time.time())]           # unique across restarts that append to the same logs

    def emit(obj, f=log):
        line = json.dumps(obj)
        f.write(line + "\n")
        f.flush()
        print(line, flush=True)

    def decide(audio, info):
        t0 = time.perf_counter()
        p = clf.probs(audio)                                   # front end + network (one ONNX call)
        infer_ms = (time.perf_counter() - t0) * 1000
        label, conf = clf.decide(p)
        line = dict(intent="OUT_OF_SCOPE" if label == "unknown" else label, slot="", infer_ms=round(infer_ms, 1),
                    audio_ms=round(window_ms), capture_ms=round(len(audio) / clf.cfg["sample_rate"] * 1000),
                    confidence=round(conf, 3), model=run, rule=clf.rule, **info)
        if not extra:
            emit(line)
            return
        seq[0] += 1
        emit(dict(line, seq=seq[0]))                           # the benchmark's answer goes out first
        answers = {}
        for name, c in [(run, clf)] + list(extra.items()):
            if c is clf:
                q, ms = p, infer_ms
            else:
                t0 = time.perf_counter()
                q = c.probs(audio)
                ms = (time.perf_counter() - t0) * 1000
            ans = {r: bench_intent(c, q, None if r == "argmax" else c.cfg["cutoffs"][r]) for r in RULES}
            answers[name] = dict(ans, confidence=round(float(np.max(q)), 3), infer_ms=round(ms, 1))
        emit(dict(seq=seq[0], models=answers, **info), others)

    if a.selftest:
        ref = HERE / "models" / run / "reference_clips.npz"
        if ref.exists():
            r = np.load(ref)
            clips = list(zip(r["audio"][:5], r["classes"][:5]))
        else:                                          # kit built from the public repository: no audio in it
            print("(no reference clips in this kit: using 3 low-noise captures, which should be OUT_OF_SCOPE)")
            g = np.random.default_rng(0)
            clips = [(g.normal(0, 0.003, 80000).astype(np.float32), "noise") for _ in range(3)]
        for x, want in clips:
            decide(x, {"expected": str(want)})
        print("selftest done: the lines above are exactly what the benchmark will read"
              + (f" (and what score_multi.py reads from {others_path})" if extra else ""))
        return

    from heydelta_listener import HeyDeltaListener

    def on_wake(p, t):
        emit({"event": "wake", "wake_prob": round(float(p), 3)})

    def on_command(cap):
        if cap.audio is None or cap.reason in ("no_speech", "no_speech_detected"):
            return                                             # nothing said: no line = no response
        decide(cap.audio, {"reason": cap.reason})

    dev = int(a.device) if a.device and a.device.isdigit() else a.device
    HeyDeltaListener(on_command=on_command, on_wake=on_wake, device=dev,
                     config_path=HERE / "listener/deploy_config.json").run()


if __name__ == "__main__":
    main()
