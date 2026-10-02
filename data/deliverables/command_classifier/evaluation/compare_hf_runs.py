#!/usr/bin/env python3
"""
Side-by-side HF-test results of bcresnet6_hf_only and bcresnet6_hf_plus (run_hf_compare.sh, owner 2026-10-02).
Reads evaluation/results_hf_only/ and results_hf_plus/ (evaluate_commands.py output). -> evaluation/results_hf_compare.md
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = {"hf_only": "HF data only", "hf_plus": "HF + our data"}


def main():
    res, per = {}, {}
    for k in RUNS:
        r = json.loads((HERE / f"results_{k}/test_results.json").read_text())[f"bcresnet6_{k}"]
        res[k] = r
        tp = pd.read_csv(HERE / f"results_{k}/bcresnet6_{k}_test_predictions.csv")
        cls = [c[2:] for c in tp.columns if c.startswith("p_")]
        tp["pred"] = np.array(cls)[tp[[f"p_{c}" for c in cls]].to_numpy().argmax(1)]
        per[k] = tp.groupby("class").apply(lambda g: (g.pred == g.name).mean(), include_groups=False)
    L = ["# Class benchmark (HF test): HF data only vs HF + our data", "",
         "Both: BC-ResNet-6, same recipe, validation = our filtered validation clips (never HF), cutoffs chosen on "
         "validation. Test = HF test split (4,418 clips; no test speaker in either model's training).", "",
         "| | " + " | ".join(RUNS.values()) + " |", "|---|---|---|",
         "| best epoch | " + " | ".join(str(res[k]["best_epoch"]) for k in RUNS) + " |",
         "| cutoffs (cautious / balanced) | " + " | ".join(f"{res[k]['cutoffs']['cautious']:.3f} / {res[k]['cutoffs']['balanced']:.3f}" for k in RUNS) + " |", ""]
    L += ["| rule | who | " + " | ".join(f"{v}: correct / wrong / rejected" for v in RUNS.values()) + " |", "|---|---|---|---|"]
    for rule in ("argmax", "balanced", "cautious"):
        whos = [w for w in res["hf_only"]["test"][rule] if isinstance(res["hf_only"]["test"][rule][w], dict) and "clips" in res["hf_only"]["test"][rule][w]]
        for w in whos:
            cells = []
            for k in RUNS:
                x = res[k]["test"][rule].get(w)
                cells.append(f"{x['correct']:.1%} / {x['wrong_command']:.1%} / {x['rejected']:.1%} ({x['clips']} clips)" if x else "-")
            L.append(f"| {rule} | {w} | " + " | ".join(cells) + " |")
        L.append(f"| {rule} | non-commands that trigger an action (mean over groups) | " +
                 " | ".join(f"{res[k]['test'][rule]['unknown_false_action_mean']:.1%}" for k in RUNS) + " |")
    t = pd.DataFrame(per).rename(columns=RUNS)
    t["difference"] = t[RUNS["hf_plus"]] - t[RUNS["hf_only"]]
    L += ["", "## Per class, argmax accuracy on HF test", "", (t * 100).round(1).to_markdown()]
    out = HERE / "results_hf_compare.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
