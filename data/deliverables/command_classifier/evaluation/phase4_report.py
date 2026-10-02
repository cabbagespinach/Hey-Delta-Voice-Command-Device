#!/usr/bin/env python3
"""
Phase 4 summary: one page with the small-scale test, the test results at both cutoffs and the streaming results.

    python phase4_report.py            # -> results/phase4_summary.md

The recommended model is the one with the better VALIDATION selection score (test results never choose), unless the
two are within 0.005, in which case the smaller (faster on the Pi) one is recommended.
"""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MODEL = HERE.parent / "model"


def pct(x):
    return "-" if x is None else f"{x:.1%}"


def main():
    t = json.loads((RES / "test_results.json").read_text())
    names = sorted(t, key=lambda n: t[n]["params"])
    small, big = names[0], names[-1]
    vs = {n: t[n]["validation"]["selection_score"] for n in names}
    rec = big if vs[big] - vs[small] > 0.005 else small
    L = ["# Command classifier: phase 4 summary", "",
         f"Recommended model: **{rec}** (validation selection score {vs[rec]:.3f}; "
         + ", ".join(f"{n} {vs[n]:.3f}" for n in names) + "). Test results never chose anything.", ""]
    for n in names:
        r = t[n]
        L += [f"## {n}: {r['params']:,} parameters, best epoch {r['best_epoch']}", "",
              f"Cutoffs (validation): cautious {r['cutoffs']['cautious']:.3f} (unknown -> action <= 2%), "
              f"balanced {r['cutoffs']['balanced']:.3f} (<= 5%)", "",
              "| rule | owner correct / wrong / asks again | speaker2 correct / wrong / asks again | public | synthetic | unknown -> action |",
              "|---|---|---|---|---|---|"]
        for rule, s in r["test"].items():
            cell = lambda w: (f"{s[w]['correct']:.1%} / {s[w]['wrong_command']:.1%} / {s[w]['rejected']:.1%}"
                              if w in s else "-")
            L.append(f"| {rule} | {cell('owner')} | {cell('speaker2')} | {s['web']['correct']:.1%} | "
                     f"{s['synthetic']['correct']:.1%} | {s['unknown_false_action_mean']:.1%} |")
        sp = RES / f"{n}_streaming.json"
        if sp.exists():
            st = json.loads(sp.read_text())
            L += ["", f"Streaming ({st['streams']} streams, {st['hours'] * 60:.0f} min; whole Pi path):", "",
                  "| events | woke | rule | command correct | wrong | asks again | off-list -> action | end-to-end correct |",
                  "|---|---|---|---|---|---|---|---|"]
            for kind, d in st["by_kind"].items():
                for rule in ("argmax", "cautious", "balanced"):
                    if rule in d:
                        x = d[rule]
                        L.append(f"| {kind} ({d['events']}) | {d['woke']:.0%} | {rule} | {pct(x['command_correct'])} | "
                                 f"{pct(x['command_wrong'])} | {pct(x['command_rejected'])} | "
                                 f"{pct(x['offlist_triggered_action'])} | {pct(x['end_to_end_correct'])} |")
                if kind == "distractor":
                    L.append(f"| distractor ({d['events']}) | {d['woke']:.0%} | - | - | - | - | - | - |")
            L += ["", f"False wake-ups: {st['false_wakeups_per_hour']}/hour; actions they caused per hour: "
                      f"{st['false_wakeup_actions_per_hour']}", ""]
    pr = MODEL / "runs/probes/probe_report.md"
    if pr.exists():
        L += ["---", "", "Small-scale test: see `model/runs/probes/probe_report.md` and `model/runs/probes/shortcuts.json`."]
    (RES / "phase4_summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
