#!/usr/bin/env python3
"""
Reported vs reproduced results (called by reproduce.sh).    python repro/compare_results.py _repro
-> repro_outputs/REPRODUCTION_REPORT.md. A part that was not reproduced is listed as "not run".
Exact equality is not expected (GPU arithmetic and data-loader timing differ between runs); with all datasets
present the numbers should agree within a few points. Rows left out for missing datasets: repro_outputs/missing_data.md.
"""
from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/deliverables"
SFX = sys.argv[1] if len(sys.argv) > 1 else "_repro"


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def wakeword(L):
    L += ["## Wakeword (BC-ResNet-6, round 1c)", ""]
    rep, new = D / "model/runs/bcresnet6", D / f"model/runs/bcresnet6{SFX}"
    a, b = load(rep / "eval/results.json"), load(new / "eval/results.json")
    if b is None:
        L += ["not run", ""]
        return
    ta, tb = load(rep / "test/test_results.json"), load(new / "test/test_results.json")

    def rows(r, t):
        i = r["isolated"]
        worst = max(v["fpr"] for v in i["negatives_by_category"].values())
        out = dict(threshold=f"{i['threshold']:.3f}", auc=f"{i['roc_auc']:.3f}",
                   rpi=f"{i['positives_by_subset']['real_device_rpi']['detected']}/{i['positives_by_subset']['real_device_rpi']['n']}",
                   syn=pct(i["positives_by_subset"].get("synthetic", {}).get("detection_rate")),
                   worst=pct(worst), fa=f"{r['streaming']['overall']['fa_per_hour']:.1f}")
        if t:
            out["test_rpi"] = f"{t['positives_by_subset']['real_device_rpi']['detected']}/{t['positives_by_subset']['real_device_rpi']['n']}"
            out["test_worst"] = pct(max(v["fpr"] for v in t["negatives_by_category"].values()))
        return out
    ra, rb = rows(a, ta), rows(b, tb)
    names = dict(threshold="threshold (validation)", auc="validation ROC AUC", rpi="validation real Pi positives detected",
                 syn="validation synthetic positives detected", worst="validation worst negative-category FPR",
                 fa="streaming false accepts / hour", test_rpi="TEST real Pi positives detected",
                 test_worst="TEST worst negative-category FPR")
    L += ["| metric | reported | reproduced |", "|---|---|---|"]
    L += [f"| {names[k]} | {ra.get(k, '–')} | {rb.get(k, '–')} |" for k in names]
    q = new / "eval/diagnostics/hey_phrase_negatives_SKIPPED.txt"
    L += ["", "Qualcomm check: " + ("skipped (dataset not present)" if q.exists() else "run, see eval/diagnostics/"), ""]


def commands(L, title, runs, results):
    L += [f"## {title}", ""]
    for run in runs:
        a = load(D / f"command_classifier/evaluation/{results}/test_results.json")
        b = load(D / f"command_classifier/evaluation/{results}{SFX}/test_results.json")
        if b is None or f"{run}{SFX}" not in b:
            L += [f"{run}: not run", ""]
            continue
        a, b = a[run], b[f"{run}{SFX}"]
        L += [f"### {run}", "", f"cutoffs (cautious / balanced): reported {a['cutoffs']['cautious']:.3f} / "
              f"{a['cutoffs']['balanced']:.3f}, reproduced {b['cutoffs']['cautious']:.3f} / {b['cutoffs']['balanced']:.3f}", "",
              "| rule | who | reported correct / wrong | reproduced correct / wrong |", "|---|---|---|---|"]
        for rule in ("argmax", "balanced"):
            for who, x in a["test"][rule].items():
                if isinstance(x, dict) and "correct" in x:
                    y = b["test"][rule].get(who)
                    L.append(f"| {rule} | {who} | {pct(x['correct'])} / {pct(x['wrong_command'])} | " +
                             (f"{pct(y['correct'])} / {pct(y['wrong_command'])} |" if y else "– |"))
            L.append(f"| {rule} | non-commands triggering an action | {pct(a['test'][rule]['unknown_false_action_mean'])} | "
                     f"{pct(b['test'][rule]['unknown_false_action_mean'])} |")
        L.append("")


def main():
    L = ["# Reproduction report", "", "Reported = the results committed in this repository; reproduced = this run "
         f"(suffix `{SFX}`). Data left out (datasets not obtained): `repro_outputs/missing_data.md`.", ""]
    miss = ROOT / "repro_outputs/missing_data.md"
    if miss.exists():
        L += ["Data status: " + ("exact data (nothing left out)" if "Nothing was left out" in miss.read_text()
                                  else "**reduced data** (see missing_data.md): differences are expected"), ""]
    wakeword(L)
    commands(L, "Command classifier, schema B (results_schema_b)", ["bcresnet6_schema_b", "dscnn_schema_b"], "results_schema_b")
    for k in ("hf_only", "hf_plus"):
        commands(L, f"Class benchmark: {k} (HF test)", [f"bcresnet6_{k}"], f"results_{k}")
    out = ROOT / "repro_outputs/REPRODUCTION_REPORT.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
