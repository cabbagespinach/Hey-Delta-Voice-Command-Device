#!/usr/bin/env python3
"""
Test results on UNSEEN speakers only (prof's criterion, 2026-10-01): test clips whose speaker has no clip at all in the
training split of the clip list, per source. Reads what evaluate_commands.py wrote (COMMAND_RESULTS_DIR):
<run>_test_predictions.csv and test_results.json (validation-chosen cutoffs); no model is run.

    COMMAND_RESULTS_DIR=... python unseen_speakers_report.py bcresnet6_schema_b dscnn_schema_b
    # -> <results>/unseen_speakers.md, unseen_speakers.csv
"""
from pathlib import Path
import json, os, sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dataloading"))
import command_data as cd                                                 # noqa: E402

OUT = Path(os.environ.get("COMMAND_RESULTS_DIR", HERE / "results"))


def main():
    cfg = cd.load_config()
    classes = cd.classes()
    unk = len(classes) - 1
    a = pd.read_csv(cd.ROOT / cfg["clips_csv"])
    seen = set(a[a.split == "train"].speaker.dropna())
    res = json.loads((OUT / "test_results.json").read_text())
    rows = []
    for run in sys.argv[1:]:
        tp = pd.read_csv(OUT / f"{run}_test_predictions.csv")
        tp = tp[~tp.speaker.isin(seen) & tp.speaker.notna()]
        P = tp[[f"p_{c}" for c in classes]].to_numpy()
        y = tp["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        for rule, cut in [("argmax", None)] + list(res[run]["cutoffs"].items()):
            pred = P.argmax(1)
            if cut is not None:
                pred = np.where((pred != unk) & (P.max(1) < cut), unk, pred)
            r = tp.assign(pred=pred, y=y)
            for ds, g in r.groupby("dataset"):
                c = g[g["class"] != "unknown"]
                u = g[g["class"] == "unknown"]
                per = c.groupby("class").apply(lambda s: pd.Series(dict(ok=(s.pred == s.y).mean(),
                                                                        wrong=((s.pred != s.y) & (s.pred != unk)).mean())),
                                               include_groups=False) if len(c) else None
                rows.append(dict(model=run, rule=rule, dataset=ds, unseen_speakers=g.speaker.nunique(),
                                 command_clips=len(c), command_classes=c["class"].nunique(),
                                 correct=round(float(per.ok.mean()), 4) if per is not None else np.nan,
                                 wrong_command=round(float(per.wrong.mean()), 4) if per is not None else np.nan,
                                 unknown_clips=len(u),
                                 unknown_false_action=round(float((u.pred != unk).mean()), 4) if len(u) else np.nan))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "unseen_speakers.csv", index=False)
    md = ["# Test results on unseen speakers", "",
          "Only test clips whose speaker never appears in the training split. Command rates are macro averages over "
          "classes; `unknown_false_action` = share of unknown clips that trigger a command.", "",
          t.to_markdown(index=False)]
    (OUT / "unseen_speakers.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
