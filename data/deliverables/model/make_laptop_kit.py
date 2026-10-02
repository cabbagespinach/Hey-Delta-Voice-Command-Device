#!/usr/bin/env python3
"""
Assemble laptop_kit/ (+ laptop_kit.zip): every exported BC-ResNet (export/laptop/<run>/heydelta_fp32.onnx), each
with the threshold and trigger rule its own evaluation chose (runs/<run>/eval/results.json), plus
laptop/heydelta_laptop.py and README_LAPTOP.md.

Usage: python make_laptop_kit.py
"""
from pathlib import Path
import json, shutil

HERE = Path(__file__).resolve().parent
RUNS = ["bcresnet1", "bcresnet3", "bcresnet6", "bcresnet6_hn"]


def main():
    out = HERE / "laptop_kit"
    if out.exists():
        shutil.rmtree(out)
    (out / "models").mkdir(parents=True)
    models = []
    for r in RUNS:
        onnx = HERE / "export/laptop" / r / "heydelta_fp32.onnx"
        rep = json.loads((HERE / "export/laptop" / r / "export_report.json").read_text())
        res = json.loads((HERE / "runs" / r / "eval/results.json").read_text())
        d = res["streaming"]["detection_config"]
        shutil.copy(onnx, out / "models" / f"{r}.onnx")
        models.append(dict(name=r, file=f"models/{r}.onnx", parameters=rep["parameters"],
                           threshold=res["isolated"]["threshold"], k_of_n=d.get("k_of_n", [1, 1]),
                           refractory_sec=d["refractory_sec"],
                           parity_max_abs_diff_vs_pytorch=rep["prob_max_abs_diff_fp32_vs_torch"]))
    (out / "models.json").write_text(json.dumps(dict(
        note="Thresholds chosen on isolated validation (per-category FPR <= 1% on deployment-like negatives); "
             "trigger rule and refractory period as in the server evaluation.", models=models), indent=2) + "\n")
    for f in ("heydelta_laptop.py", "README_LAPTOP.md"):
        shutil.copy(HERE / "laptop" / f, out / f)
    z = shutil.make_archive(str(HERE / "laptop_kit"), "zip", root_dir=HERE, base_dir="laptop_kit")
    print(f"{out} and {z}:", sorted(p.name for p in out.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
