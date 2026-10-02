#!/usr/bin/env python3
"""
One-time held-out TEST evaluation of the chosen model (owner decision 2026-09-30: BC-ResNet-6, round 1c).

The test split was never read before this run: not for training, the threshold, the diagnostics or the choice
between models. The threshold is the one chosen on validation. The script refuses to run a second time
(runs/<run>/test/ exists), so the test result can't feed back into any decision.

Usage: python evaluate_test.py
Writes runs/bcresnet6/test/test_results.json, test_scores.csv and test_report.md.
"""
from pathlib import Path
import datetime, json, os, sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "evaluation"))
import detection as det               # noqa: E402
import isolated_eval as iso           # noqa: E402
import run_evaluation as re_          # noqa: E402

# WAKEWORD_RUN: another run folder, e.g. a reproduction (bcresnet6_repro, 2026-10-02)
RUN, CLS = os.environ.get("WAKEWORD_RUN", "bcresnet6"), "bc_resnet:BCResNet6"


def main():
    out = HERE / "runs" / RUN / "test"
    if out.exists():
        sys.exit(f"{out} exists: the test split has already been evaluated once; not running it again.")
    thr = json.loads((HERE / "runs" / RUN / "eval/results.json").read_text())["isolated"]["threshold"]
    scorer = det.TorchScorer(re_.load_model(str(HERE / "runs" / RUN / "best.pt"), CLS))
    I, s = iso.evaluate(scorer, "test", threshold=thr)
    out.mkdir(parents=True)
    I["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    I["model"] = f"{CLS} from runs/{RUN}/best.pt"
    (out / "test_results.json").write_text(json.dumps(I, indent=2, default=float) + "\n")
    s.to_csv(out / "test_scores.csv", index=False)
    pct = lambda v: f"{100 * v:.1f}%"
    L = ["# Held-out test result: BC-ResNet-6 (one-time run)", "",
         f"Generated {I['created_at']}. Threshold {thr:.4f}, chosen on validation (not on test). "
         f"{I['n_windows']} test windows, {I['n_positive']} positive.", "",
         f"ROC AUC **{I['roc_auc']:.4f}**, average precision **{I['average_precision']:.4f}**.", "",
         "| Positives | n | Detected | Rate (95% CI) |", "|---|---:|---:|---|"]
    for k, v in I["positives_by_subset"].items():
        L.append(f"| {k} | {v['n']} | {v['detected']} | {pct(v['detection_rate'])} [{pct(v['ci95'][0])}, {pct(v['ci95'][1])}] |")
    L += ["", "| Negatives by category | n | False positives | FPR (95% CI) |", "|---|---:|---:|---|"]
    for k, v in sorted(I["negatives_by_category"].items(), key=lambda kv: -kv[1]["fpr"]):
        L.append(f"| {k} | {v['n']} | {v['false_positives']} | {100 * v['fpr']:.2f}% "
                 f"[{100 * v['ci95'][0]:.2f}, {100 * v['ci95'][1]:.2f}]% |")
    L += ["", "| Negatives by source | n | False positives | FPR |", "|---|---:|---:|---:|"]
    for k, v in I["negatives_by_subset"].items():
        L.append(f"| {k} | {v['n']} | {v['false_positives']} | {100 * v['fpr']:.2f}% |")
    (out / "test_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
