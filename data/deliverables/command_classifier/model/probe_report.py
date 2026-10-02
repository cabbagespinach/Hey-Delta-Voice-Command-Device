#!/usr/bin/env python3
"""
Small-scale test, part 2: compare two short trainings (validation split only).

    python probe_report.py runs/probe_all runs/probe_no_owner      # -> runs/probes/probe_report.md

probe_all       trained on everything
probe_no_owner  trained WITHOUT the owner's recordings and anything derived from them (converted voices, owner
                fragments): only public datasets and synthetic voices
Both are scored on the owner's validation session (13:36-13:55, real Pi captures). Per command:
- probe_all low                       -> hard even with the owner's recordings: more real data needed
- probe_no_owner much lower than all  -> the class depends on the owner's own recordings, i.e. the public and
                                          synthetic data do not transfer to real Pi captures for it: recordings by
                                          more people (or more varied sessions) would help most here
Also: the most frequent confusions, and how often the owner's `unknown` validation clips trigger a command.
"""
from pathlib import Path
import json, sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load(run):
    d = pd.read_csv(Path(run) / "val_predictions.csv")
    cls = [c[2:] for c in d.columns if c.startswith("p_")]
    d["pred"] = np.array(cls)[d[[f"p_{c}" for c in cls]].to_numpy().argmax(1)]
    d["ok"] = d.pred == d["class"]
    return d, cls


def main():
    a, b = sys.argv[1], sys.argv[2]
    A, cls = load(HERE / a)
    B, _ = load(HERE / b)
    oa, ob = A[A.group == "owner"], B[B.group == "owner"]
    t = pd.DataFrame({"clips": oa.groupby("class").size(), "acc_all": oa.groupby("class").ok.mean(),
                      "acc_no_owner": ob.groupby("class").ok.mean()})
    t["drop"] = t.acc_all - t.acc_no_owner
    t = t.round(2).sort_values(["acc_no_owner", "acc_all"])
    conf = oa[~oa.ok].groupby(["class", "pred"]).size().sort_values(ascending=False).head(15)
    conf_b = ob[~ob.ok].groupby(["class", "pred"]).size().sort_values(ascending=False).head(15)
    ua = oa[oa["class"] == "unknown"]
    ub = ob[ob["class"] == "unknown"]
    ha = json.loads((HERE / a / "history.json").read_text())
    hb = json.loads((HERE / b / "history.json").read_text())
    best = lambda h: max(h, key=lambda m: m["selection_score"])
    lines = [
        "# Small-scale test: which commands need more real recordings?", "",
        f"Validation split only. `{a}` = all data; `{b}` = no owner recordings or anything derived from them.", "",
        "| | all data | without owner data |", "|---|---|---|",
        *[f"| {k} | {best(ha)[k]:.3f} | {best(hb)[k]:.3f} |" for k in
          ("owner_macro_acc", "web_macro_acc", "synthetic_macro_acc", "unknown_false_action", "selection_score")],
        "", f"Owner `unknown` validation clips taken for a command: all data {(ua.pred != 'unknown').mean():.0%} "
        f"({len(ua)} clips), without owner data {(ub.pred != 'unknown').mean():.0%}.", "",
        "## Per command, owner validation session (sorted by accuracy without owner data)", "",
        t.to_markdown(), "", "## Most frequent confusions (all data)", "", conf.to_frame("clips").to_markdown(), "",
        "## Most frequent confusions (without owner data)", "", conf_b.to_frame("clips").to_markdown(), ""]
    out = HERE / "runs/probes/probe_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
