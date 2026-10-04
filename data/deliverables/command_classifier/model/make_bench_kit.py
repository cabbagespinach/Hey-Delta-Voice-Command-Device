#!/usr/bin/env python3
"""
Assemble the class-benchmark kit (vcm_bench_kit/ + vcm_bench_kit.zip) for github.com/airimonda/vcm-benchmark (2026-10-02):
    vcm_bench_assistant.py, command_pi.py     Pi runner: wakeword listener + one command model -> JSON log lines
                                              (--also: more models on the same captures -> bench_others.jsonl)
    score_multi.py                            laptop: one benchmark report per model from a --also run (2026-10-03)
    models/<run>/                             command ONNX + command_config.json (official cutoffs) + reference clips
    listener/                                 "Hey Delta" listener + wakeword model (../../model/deploy)
    holdout_da92a79.parquet                   the class holdout pinned to HF revision da92a79 (laptop: --holdout)
    README_BENCH.md

Usage: python make_bench_kit.py [run ...]    (default: the runs in export/ among bcresnet6_hf_plus, bcresnet6_hf_only,
                                              bcresnet6_hf_ourlabels)

Works in a fresh clone of the public repository: the holdout is downloaded once from Hugging Face at the pinned
revision when it is not there yet, and reference clips (audio, not in the repository) are added only if present.
"""
from pathlib import Path
import json, shutil, sys, urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEPLOY = HERE.parents[1] / "model/deploy"
SRC = HERE / "bench_kit_src"
HOLDOUT = ROOT / "external_raw/hf_class_dataset_da92a79/holdout-00000-of-00001.parquet"
HOLDOUT_URL = ("https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands/resolve/da92a79/"
               "data/holdout-00000-of-00001.parquet")
DEFAULT_RUNS = ["bcresnet6_hf_plus", "bcresnet6_hf_only", "bcresnet6_hf_ourlabels"]


def fetch_holdout():
    if not HOLDOUT.exists():
        HOLDOUT.parent.mkdir(parents=True, exist_ok=True)
        print("downloading the class holdout (Hugging Face airimonda/ai231-me2-voice-commands, revision da92a79) ...")
        tmp = HOLDOUT.with_name(HOLDOUT.name + ".part")
        urllib.request.urlretrieve(HOLDOUT_URL, tmp)
        tmp.rename(HOLDOUT)


def main():
    runs = sys.argv[1:] or [r for r in DEFAULT_RUNS if (HERE / "export" / r / "command_config.json").exists()]
    fetch_holdout()
    out = HERE / "vcm_bench_kit"
    if out.exists():
        shutil.rmtree(out)
    for run in runs:
        ex, d = HERE / "export" / run, out / "models" / run
        d.mkdir(parents=True)
        cfg = json.loads((ex / "command_config.json").read_text())
        for f in (cfg["model_file"], "command_config.json", "reference_clips.npz"):
            if (ex / f).exists() or f != "reference_clips.npz":
                shutil.copy(ex / f, d / f)
        print(f"{run}: cutoffs {cfg['cutoffs']}, default rule {cfg['default_rule']}")
    shutil.copy(HERE / "export" / runs[0] / "command_pi.py", out / "command_pi.py")
    for f in ("vcm_bench_assistant.py", "score_multi.py", "README_BENCH.md"):
        shutil.copy(SRC / f, out / f)
    (out / "listener").mkdir()
    dcfg = json.loads((DEPLOY / "deploy_config.json").read_text())
    for f in ("heydelta_listener.py", "deploy_config.json", dcfg["model_file"]):
        shutil.copy(DEPLOY / f, out / "listener" / f)
    shutil.copy(HOLDOUT, out / "holdout_da92a79.parquet")
    z = shutil.make_archive(str(HERE / "vcm_bench_kit"), "zip", root_dir=HERE, base_dir="vcm_bench_kit")
    print(f"{out} and {z}:", sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
