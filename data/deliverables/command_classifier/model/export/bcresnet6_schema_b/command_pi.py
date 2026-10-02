#!/usr/bin/env python3
"""
Command classifier on the Raspberry Pi. Needs numpy + onnxruntime (and the wakeword deploy folder for `live`).

    from command_pi import CommandClassifier
    clf = CommandClassifier()                          # reads command_config.json next to this file
    label, prob = clf(cap.audio)                       # 16 kHz mono float32 -1..1 -> ("lights_on", 0.97) or ("unknown", ...)

    python command_pi.py check                         # ONNX on this Pi vs the server's probabilities (16 clips)
    python command_pi.py wav file.wav [...]            # classify 16 kHz mono 16-bit WAV files
    python command_pi.py live [--deploy DIR]           # "Hey Delta" -> command, using heydelta_listener.py in DIR
                                                       # (default ../deploy or this folder)
    add --rule argmax|cautious|balanced to change the decision rule (default from command_config.json)
"""
from pathlib import Path
import argparse, json, sys, time, wave

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent


class CommandClassifier:
    def __init__(self, config_path=HERE / "command_config.json", rule=None, threads=1):
        self.cfg = json.loads(Path(config_path).read_text())
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(Path(config_path).parent / self.cfg["model_file"]), so,
                                         providers=["CPUExecutionProvider"])
        self.classes = self.cfg["classes"]
        self.rule = rule or self.cfg["default_rule"]
        self.cutoff = None if self.rule == "argmax" else self.cfg["cutoffs"][self.rule]
        self._rng = np.random.default_rng()

    def frame(self, audio):
        """First 5 s; shorter captures padded on the right with noise at the Pi's measured floor (as in training)."""
        n = self.cfg["num_samples"]
        x = np.asarray(audio, dtype=np.float32).reshape(-1)[:n]
        if len(x) < n:
            pad = self._rng.standard_normal(n - len(x)).astype(np.float32) * 10 ** (self.cfg["pad_noise_dbfs"] / 20)
            x = np.concatenate([x, pad])
        return x

    def probs(self, audio):
        return self.sess.run(["probs"], {"audio": self.frame(audio)[None]})[0][0]

    def decide(self, p):
        k = int(np.argmax(p))
        label = self.classes[k]
        if self.cutoff is not None and label != "unknown" and p[k] < self.cutoff:
            label = "unknown"
        return label, float(p[k])

    def __call__(self, audio):
        return self.decide(self.probs(audio))


def read_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2, f"{path}: need 16 kHz 16-bit WAV"
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    return x.mean(1).astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "wav", "live"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--rule", choices=["argmax", "cautious", "balanced"])
    ap.add_argument("--deploy", help="folder with heydelta_listener.py + deploy_config.json (live)")
    a = ap.parse_args()
    clf = CommandClassifier(rule=a.rule)
    print(f"model {clf.cfg['model_file']}  rule {clf.rule}  cutoff {clf.cutoff}")

    if a.mode == "check":
        r = np.load(HERE / "reference_clips.npz")
        t0 = time.perf_counter()
        p = np.stack([clf.sess.run(["probs"], {"audio": x[None]})[0][0] for x in r["audio"]])
        ms = (time.perf_counter() - t0) / len(p) * 1000
        same = (p.argmax(1) == r["probs"].argmax(1)).sum()
        print(f"max |prob difference| vs server: {np.abs(p - r['probs']).max():.2e}   "
              f"same top class: {same}/{len(p)}   {ms:.0f} ms per capture")
        for want, q in zip(r["classes"], p):
            print(f"  true {want:22s} -> {clf.decide(q)[0]:22s} {q.max():.3f}")
    elif a.mode == "wav":
        for f in a.files:
            print(f"{f}: {clf(read_wav(f))}")
    else:
        d = Path(a.deploy) if a.deploy else next((p for p in (HERE.parent / "deploy", HERE)
                                                    if (p / "heydelta_listener.py").exists()), None)
        assert d is not None, "heydelta_listener.py not found: pass --deploy DIR"
        sys.path.insert(0, str(d))
        from heydelta_listener import HeyDeltaListener

        def on_command(cap):
            if cap.audio is None:
                print(f"  -> nothing said ({cap.reason})")
                return
            t0 = time.perf_counter()
            label, p = clf(cap.audio)
            print(f"  -> {label}  ({p:.2f}, {cap.duration_sec:.1f} s capture, "
                  f"{(time.perf_counter() - t0) * 1000:.0f} ms)")
        HeyDeltaListener(on_command=on_command, config_path=d / "deploy_config.json").run()


if __name__ == "__main__":
    main()
