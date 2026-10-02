#!/usr/bin/env python3
"""
Assemble the side-by-side command test kit for the Raspberry Pi (ab_kit/ + ab_kit.zip), self-contained:
    command_ab.py, command_pi.py        both models on every live capture (check / live / replay / devices)
    models/<run>.onnx, <run>.json        the exported models (export/<run>/), each with its validated cutoffs
    models/<run>_reference.npz           16 clips + server-side probabilities, for `check`
    models.json                          which models, in report order
    listener/                            the "Hey Delta" listener + wakeword model (../../model/deploy)
    label_map.csv, commands.txt          prompt label -> class; the trained phrasings
    phrasings.txt                        other ways of saying each command (evaluation/phrasing_variants_test.py)
    README_AB.md

Usage: python make_ab_kit.py [run ...]      (default: bcresnet6 bcresnet6_ownernoise)
"""
from pathlib import Path
import json, shutil, sys

HERE = Path(__file__).resolve().parent
DEPLOY = HERE.parents[1] / "model/deploy"
SRC = HERE / "ab_kit_src"
sys.path.insert(0, str(HERE.parent / "evaluation"))


def main():
    runs = sys.argv[1:] or ["bcresnet6", "bcresnet6_ownernoise"]
    out = HERE / "ab_kit"
    if out.exists():
        shutil.rmtree(out)
    (out / "models").mkdir(parents=True)
    models = []
    for run in runs:
        ex = HERE / "export" / run
        cfg = json.loads((ex / "command_config.json").read_text())
        shutil.copy(ex / cfg["model_file"], out / "models" / cfg["model_file"])
        (out / "models" / f"{run}.json").write_text(json.dumps(cfg, indent=2) + "\n")
        shutil.copy(ex / "reference_clips.npz", out / "models" / f"{run}_reference.npz")
        models.append(dict(name=run, config=f"{run}.json", cutoffs=cfg["cutoffs"], best_epoch=cfg["best_epoch"]))
    (out / "models.json").write_text(json.dumps(dict(models=models), indent=2) + "\n")
    shutil.copy(HERE / "export" / runs[0] / "command_pi.py", out / "command_pi.py")
    for f in ("command_ab.py", "README_AB.md"):
        shutil.copy(SRC / f, out / f)
    (out / "listener").mkdir()
    dcfg = json.loads((DEPLOY / "deploy_config.json").read_text())
    for f in ("heydelta_listener.py", "deploy_config.json", dcfg["model_file"]):
        shutil.copy(DEPLOY / f, out / "listener" / f)
    shutil.copy(DEPLOY / "label_map.csv", out / "label_map.csv")
    shutil.copy(DEPLOY / "commands.txt", out / "commands.txt")
    from phrasing_variants_test import VARIANTS                    # trained phrasing first, then the others
    lines = ["# Other ways of saying each command: label | words to say. The first line of each label is the trained one."]
    for lab, ps in VARIANTS.items():
        lines += [f"{lab} | {p}" for p in ps]
    lines += ["unknown_other | (say something that is NOT a command)"]
    (out / "phrasings.txt").write_text("\n".join(lines) + "\n")
    z = shutil.make_archive(str(HERE / "ab_kit"), "zip", root_dir=HERE, base_dir="ab_kit")
    print(f"{out} and {z}:", sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
