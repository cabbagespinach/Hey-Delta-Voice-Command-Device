#!/usr/bin/env python3
"""
Assemble the Raspberry Pi kit (pi_kit/ + pi_kit.zip) with every BC-ResNet model:
    models/<run>.onnx, models.json     fp32 models (front end included), each with its validated threshold and rule
    reference_windows.npz              48 frozen validation windows + each model's server-side probabilities (`check`)
    heydelta_pi.py                     check / bench
    heydelta_live.py                   all models live on the microphone, detection + efficiency reports
                                       (same script as the laptop kit's heydelta_laptop.py)
    README_PI.md

Usage: python make_pi_bundle.py      (after make_laptop_kit.py; uses the same exports)
"""
from pathlib import Path
import json, shutil, sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_onnx as ex                # noqa: E402
import make_laptop_kit as lk            # noqa: E402


def main():
    out = HERE / "pi_kit"
    if out.exists():
        shutil.rmtree(out)
    (out / "models").mkdir(parents=True)
    lap = HERE / "laptop_kit"
    cfg = json.loads((lap / "models.json").read_text())
    assert [m["name"] for m in cfg["models"]] == lk.RUNS, "rebuild laptop_kit first (python make_laptop_kit.py)"
    for m in cfg["models"]:
        shutil.copy(lap / m["file"], out / m["file"])
    (out / "models.json").write_text(json.dumps(cfg, indent=2) + "\n")
    shutil.copy(HERE / "laptop/heydelta_laptop.py", out / "heydelta_live.py")
    for f in ("heydelta_pi.py", "README_PI.md"):
        shutil.copy(HERE / "pi" / f, out / f)

    # Reference windows: 16 positives + 32 negatives of the frozen validation set, with server-side probabilities.
    _, man, frames = ex.validation_frames()
    rng = np.random.default_rng(0)
    pos = np.where(man.label == "positive")[0]
    neg = np.where(man.label == "negative")[0]
    idx = np.sort(np.concatenate([rng.choice(pos, 16, replace=False), rng.choice(neg, 32, replace=False)]))
    probs = {f"prob_{m['name']}": ex.run_onnx(out / m["file"], frames[idx]).astype(np.float32) for m in cfg["models"]}
    np.savez_compressed(out / "reference_windows.npz", audio=frames[idx].numpy().astype(np.float32),
                        window_id=man.window_id.to_numpy()[idx].astype(str), **probs)
    z = shutil.make_archive(str(HERE / "pi_kit"), "zip", root_dir=HERE, base_dir="pi_kit")
    print(f"{out} and {z}:", sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
