#!/usr/bin/env python3
"""
Export a trained command classifier as ONE self-contained ONNX model for the Raspberry Pi:

    input  "audio": float32 [B, 80000]  (first 5 s of the capture, 16 kHz mono, +-1; shorter captures padded on the
                                         right with Gaussian noise at -50.6 dBFS RMS, see command_pi.py)
    output "probs": float32 [B, 30]     (softmax over the classes in command_config.json)

The front end is the wakeword export's OnnxFrontEnd (../../model/export_onnx.py: STFT as a Conv1d with a fixed DFT
basis, mel matrix, 10*log10, frozen TRAIN normalisation), so the Pi needs only numpy + onnxruntime.

Checks (written to <out>/export_report.json), on TEST clips built exactly as the evaluator builds them:
  1. front-end parity: ONNX features vs command_preprocessing
  2. probability parity: ONNX fp32 vs the PyTorch model, and decisions changed at argmax / cautious / balanced

Writes <out>/command_<run>.onnx, command_config.json (classes + validation-chosen cutoffs), command_pi.py,
reference_clips.npz (16 frames + server probabilities, for `python command_pi.py check` on the Pi).

Usage: python export_commands_onnx.py runs/bcresnet6 --out export/bcresnet6
"""
from pathlib import Path
import argparse, json, shutil, sys

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
CC = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CC / "dataloading"))
sys.path.insert(0, str(CC / "evaluation"))
sys.path.insert(0, str(CC.parent / "model"))
import command_data as cd                                                 # noqa: E402
from command_model import CommandNet                                       # noqa: E402
from evaluate_commands import decide, choose_cutoff, TARGETS               # noqa: E402
from export_onnx import OnnxFrontEnd                                       # noqa: E402


class Deployable(nn.Module):
    def __init__(self, front, net):
        super().__init__()
        self.front, self.net = front, net

    def forward(self, audio):
        return torch.softmax(self.net(self.front(audio)), 1)


def run_onnx(path, frames, bs=64):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return np.concatenate([s.run(["probs"], {"audio": frames[i:i + bs]})[0] for i in range(0, len(frames), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-check", type=int, default=600, help="test clips used for the parity check")
    a = ap.parse_args()
    run, out = Path(a.run).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    ck = torch.load(run / "best.pt", map_location="cpu")
    classes = ck["classes"]
    model = CommandNet(len(classes), ck["tau"], arch=ck.get("arch", "bcresnet")).eval()
    model.load_state_dict(ck["state_dict"])
    vp = pd.read_csv(run / "val_predictions.csv")
    cut = {k: float(choose_cutoff(vp[[f"p_{c}" for c in classes]].to_numpy(), vp, classes, t)) for k, t in TARGETS.items()}

    front = OnnxFrontEnd(model.front).eval()
    full = Deployable(front, model.net).eval()
    N = model.front.cfg.num_samples
    onnx_path = out / f"command_{run.name}.onnx"
    torch.onnx.export(full, torch.zeros(1, N), str(onnx_path), input_names=["audio"], output_names=["probs"],
                      dynamic_axes={"audio": {0: "batch"}, "probs": {0: "batch"}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    cfg = cd.load_config()
    test = cd.EvalClips(cfg, "test")
    idx = np.linspace(0, len(test) - 1, min(a.n_check, len(test))).round().astype(int)
    frames = torch.stack([test[int(i)][0] for i in idx])
    rows = test.rows.iloc[idx].reset_index(drop=True)
    with torch.no_grad():
        f_ref, f_onnx = model.front(frames[:128]), front(frames[:128])
        p_torch = torch.cat([torch.softmax(model(frames[i:i + 64]), 1) for i in range(0, len(frames), 64)]).numpy()
    p_onnx = run_onnx(onnx_path, frames.numpy())

    rep = dict(run=str(run), best_epoch=ck["epoch"], tau=ck["tau"], classes=len(classes), opset=17,
               input=f"audio float32 [B, {N}] @16 kHz", output=f"probs float32 [B, {len(classes)}]",
               check_clips=len(frames), cutoffs=cut,
               frontend_max_abs_diff_normalised_units=float((f_ref - f_onnx).abs().max()),
               prob_max_abs_diff_onnx_vs_torch=float(np.abs(p_onnx - p_torch).max()))
    for rule, c in [("argmax", None)] + list(cut.items()):
        dt, do = decide(p_torch, classes, c), decide(p_onnx, classes, c)
        rep[f"decisions_changed_{rule}"] = int((dt != do).sum())
        rep[f"accuracy_{rule}_onnx"] = float((np.array(classes)[do] == rows["class"].to_numpy()).mean())
    rep["file_bytes"] = onnx_path.stat().st_size
    (out / "export_report.json").write_text(json.dumps(rep, indent=2) + "\n")

    pc = model.front.cfg
    (out / "command_config.json").write_text(json.dumps(dict(
        model_file=onnx_path.name, run=run.name, tau=ck["tau"], best_epoch=ck["epoch"], classes=classes,
        # class-benchmark models deploy with balanced (owner 2026-10-04: Pi benchmark 84.7% vs 75.2% correct, 14% vs 25% ignored)
        cutoffs=cut, default_rule="balanced" if run.name.startswith("bcresnet6_hf_") else "cautious",
        rule="take the most likely class; if it is a command and its probability is below the cutoff, answer unknown",
        _cutoffs="chosen on validation: cautious = at most 2% of unknown clips trigger a command, balanced = at most 5%",
        sample_rate=pc.sample_rate, num_samples=N, pad_noise_dbfs=pc.pad_noise_dbfs,
        _input="the capture from the wakeword listener (Capture.audio), first 5 s, padded on the right with "
               "Gaussian noise at pad_noise_dbfs RMS"), indent=2) + "\n")
    sel = np.linspace(0, len(frames) - 1, 16).round().astype(int)
    np.savez_compressed(out / "reference_clips.npz", audio=frames.numpy()[sel], probs=p_torch[sel],
                        classes=np.array(rows["class"].to_numpy()[sel], dtype=str))
    shutil.copy(HERE / "command_pi.py", out / "command_pi.py")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
