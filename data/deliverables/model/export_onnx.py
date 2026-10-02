#!/usr/bin/env python3
"""
Export a trained BC-ResNet as ONE self-contained ONNX model for the Raspberry Pi:

    input  "audio": float32 [B, 24000]  (1.5 s of 16 kHz mono, full scale +-1)
    output "prob":  float32 [B]         (wakeword probability)

The graph contains the whole front end, rebuilt from plain ONNX ops so the Pi needs only onnxruntime
(no torch/torchaudio): STFT as a strided Conv1d with a fixed Hann-windowed DFT basis (n_fft 512, win 400,
hop 160, center=False), power, the HTK mel matrix taken from torchaudio, 10*log10(max(x, 1e-10)), and the
frozen train-split normalization. Then the BC-ResNet and a sigmoid.

Checks (written to <out>/export_report.json):
  1. front-end parity: ONNX features vs wakeword_preprocessing on real validation windows
  2. probability parity: ONNX fp32 vs the PyTorch model on the whole frozen validation set
  3. int8: static QDQ quantization of the network (front end kept in fp32), calibrated on 512 unaugmented
     TRAIN windows; kept only if validation AUC drops <= 0.005 and detection at the operating threshold
     changes by at most one positive per subset.

Usage: python export_onnx.py --model bcresnet3 --checkpoint runs/bcresnet3_hn/best.pt --results <eval dir> --out export/
"""
from pathlib import Path
import argparse, json, math, sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchaudio
from torch import nn

HERE = Path(__file__).resolve().parent
DELIV = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DELIV / "evaluation"))
sys.path.insert(0, str(DELIV / "dataloading"))
import bc_resnet as bcr                 # noqa: E402
import wakeword_data as wd              # noqa: E402
import isolated_eval as iso             # noqa: E402
import detection as det                 # noqa: E402
import reproducibility as rep           # noqa: E402

wp = wd.wp


class OnnxFrontEnd(nn.Module):
    """Waveform [B, N] -> normalised log-mel [B, 1, n_mels, frames] using only Conv1d / MatMul / Log."""

    def __init__(self, pre: "wp.WakewordPreprocessor"):
        super().__init__()
        c = pre.cfg
        n_fft, win, hop = c.n_fft, c.win_length, c.hop_length
        w = torch.hann_window(win, periodic=True, dtype=torch.float64)
        lp = (n_fft - win) // 2                                         # torch.stft centres a short window
        window = torch.zeros(n_fft, dtype=torch.float64)
        window[lp:lp + win] = w
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64).view(-1, 1)
        n = torch.arange(n_fft, dtype=torch.float64).view(1, -1)
        ang = 2 * math.pi * k * n / n_fft
        basis = torch.cat([torch.cos(ang) * window, -torch.sin(ang) * window])      # [2*(F), n_fft]
        self.register_buffer("basis", basis.float().unsqueeze(1))                  # [2F, 1, n_fft]
        self.hop, self.nbins = hop, n_fft // 2 + 1
        fb = pre.extract.mel.mel_scale.fb                                           # [F, n_mels]
        self.register_buffer("fb", fb.float())
        self.register_buffer("mean", pre.normalizer.mean.float())                   # [n_mels, 1]
        self.register_buffer("std", pre.normalizer.std.float())
        self.amin, self.mult = c.db_amin, c.db_multiplier

    def forward(self, audio):
        spec = nn.functional.conv1d(audio.unsqueeze(1), self.basis, stride=self.hop)   # [B, 2F, frames]
        re, im = spec[:, :self.nbins], spec[:, self.nbins:]
        power = re * re + im * im                                                      # [B, F, frames]
        mel = torch.matmul(power.transpose(1, 2), self.fb).transpose(1, 2)            # [B, n_mels, frames]
        db = self.mult * torch.log10(torch.clamp(mel, min=self.amin))
        return ((db - self.mean) / self.std).unsqueeze(1)


class Deployable(nn.Module):
    def __init__(self, front: nn.Module, net: nn.Module):
        super().__init__()
        self.front, self.net = front, net

    def forward(self, audio):
        return torch.sigmoid(self.net(self.front(audio)))


def validation_frames(split="validation"):
    """Frozen isolated-set windows as raw frames [N, 24000] + manifest (exactly what the evaluator scores)."""
    d, man = iso.load_isolated_set(split)
    with rep.single_thread():
        frames = torch.stack([d.load(i)[0] for i in range(len(d))])
    return d, man, frames


def run_onnx(path, frames, bs=512):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return np.concatenate([s.run(["prob"], {"audio": frames[i:i + bs].numpy()})[0] for i in range(0, len(frames), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(bcr.MODELS), required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--results", required=True, help="run_evaluation output dir of this checkpoint (threshold)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-int8", action="store_true", help="skip the int8 variant (fp32 only)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    thr = json.loads((Path(args.results) / "results.json").read_text())["isolated"]["threshold"]

    pre = wp.WakewordPreprocessor.from_files()
    net = bcr.MODELS[args.model]()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    net.eval()
    front = OnnxFrontEnd(pre).eval()
    full = Deployable(front, net).eval()
    fp32 = out / "heydelta_fp32.onnx"
    torch.onnx.export(full, torch.zeros(1, pre.cfg.num_samples), str(fp32), input_names=["audio"],
                      output_names=["prob"], dynamic_axes={"audio": {0: "batch"}, "prob": {0: "batch"}},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(fp32)))

    d, man, frames = validation_frames()
    rep_ = dict(model=args.model, checkpoint=args.checkpoint, threshold=thr, opset=17,
                parameters=bcr.count_params(net), input="audio float32 [B, 24000] @16 kHz", output="prob float32 [B]")

    # 1. front-end parity on real windows
    idx = np.linspace(0, len(frames) - 1, 256).round().astype(int)
    with torch.no_grad():
        ref = pre(frames[idx])
        mine = front(frames[idx])
    rep_["frontend_max_abs_diff_normalised_units"] = float((ref - mine).abs().max())
    rep_["frontend_mean_abs_diff_normalised_units"] = float((ref - mine).abs().mean())

    # 2. probability parity (whole validation set)
    with torch.no_grad(), rep.single_thread():
        p_torch = torch.cat([torch.sigmoid(net(pre(frames[i:i + 512]))) for i in range(0, len(frames), 512)]).numpy()
    p_fp32 = run_onnx(fp32, frames)
    y = (man.label == "positive").to_numpy().astype(int)

    def summary(p):
        s = man.assign(score=p)
        pos = s[s.label == "positive"]
        return dict(auc=det.roc_auc(y, p), detected_by_subset={k: int((g.score >= thr).sum()) for k, g in pos.groupby("eval_subset")},
                    false_positives=int(((s.label == "negative") & (s.score >= thr)).sum()))
    rep_["prob_max_abs_diff_fp32_vs_torch"] = float(np.abs(p_fp32 - p_torch).max())
    rep_["decisions_changed_fp32_vs_torch"] = int(((p_fp32 >= thr) != (p_torch >= thr)).sum())
    rep_["torch"], rep_["onnx_fp32"] = summary(p_torch), summary(p_fp32)

    if args.no_int8:
        rep_["int8_accepted"] = False
        rep_["files"] = {fp32.name: fp32.stat().st_size}
        (out / "export_report.json").write_text(json.dumps(rep_, indent=2, default=float) + "\n")
        print(json.dumps({k: rep_[k] for k in ("frontend_max_abs_diff_normalised_units", "prob_max_abs_diff_fp32_vs_torch",
                                               "decisions_changed_fp32_vs_torch", "files")}, indent=1))
        return
    # 3. int8 (network only; the front end stays fp32: log/normalisation are precision-sensitive)
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    _, tds = wd.build_datasets(splits=["train"], augment=False)
    td = tds["train"]
    cal_idx = np.random.default_rng(0).choice(len(td), 512, replace=False)
    with rep.single_thread():
        cal = torch.stack([td.load(int(i))[0] for i in cal_idx]).numpy()

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"audio": cal[i:i + 32]} for i in range(0, len(cal), 32)])

        def get_next(self):
            return next(self.it, None)

    prep = out / "heydelta_fp32_prep.onnx"
    quant_pre_process(str(fp32), str(prep))
    m = onnx.load(str(prep))
    front_nodes = [n.name for n in m.graph.node if n.name.startswith("/front/")]
    int8 = out / "heydelta_int8.onnx"
    quantize_static(str(prep), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, nodes_to_exclude=front_nodes)
    prep.unlink()
    p_int8 = run_onnx(int8, frames)
    rep_["onnx_int8"] = summary(p_int8)
    rep_["decisions_changed_int8_vs_fp32"] = int(((p_int8 >= thr) != (p_fp32 >= thr)).sum())
    a, b = rep_["onnx_fp32"], rep_["onnx_int8"]
    ok = (a["auc"] - b["auc"] <= 0.005 and
          all(abs(a["detected_by_subset"][k] - b["detected_by_subset"].get(k, 0)) <= 1 for k in a["detected_by_subset"]))
    rep_["int8_accepted"] = bool(ok)
    rep_["files"] = {f.name: f.stat().st_size for f in (fp32, int8)}
    (out / "export_report.json").write_text(json.dumps(rep_, indent=2, default=float) + "\n")
    print(json.dumps({k: rep_[k] for k in ("frontend_max_abs_diff_normalised_units", "prob_max_abs_diff_fp32_vs_torch",
                                           "decisions_changed_fp32_vs_torch", "decisions_changed_int8_vs_fp32",
                                           "int8_accepted", "files")}, indent=1))
    print("torch", rep_["torch"], "\nint8 ", rep_["onnx_int8"])


if __name__ == "__main__":
    main()
