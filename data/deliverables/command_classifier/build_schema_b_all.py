#!/usr/bin/env python3
"""
Schema-B training list (owner, 2026-10-01): the owner's data relabelled by schema_b.py (data/commands_schema_b.csv)
+ the classmates' datasets labelled by label_classmate_datasets.py (data/commands_classmates/manifest.csv).

    python build_schema_b_all.py      # -> data/commands_schema_b_all.csv (read via dataloader_config_schema_b.json)

Checks: every file exists, paths unique, classes in the schema-B label map, splits train/validation/test. Reports
clips per class x split and speakers shared between splits (reported, not changed: splits are each source's own).
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
D = ROOT / "data"


def main():
    own = pd.read_csv(D / "commands_schema_b.csv")
    cm = pd.read_csv(D / "commands_classmates/manifest.csv")
    a = pd.concat([own, cm[own.columns]], ignore_index=True)
    lm = pd.read_csv(HERE.parent / "model/deploy/label_map_schema_b.csv")
    assert a.path.is_unique, a[a.path.duplicated()].head()
    assert set(a["class"]) <= set(lm["class"]) | {"unknown"}
    assert set(a.split) <= {"train", "validation", "test"}, set(a.split)
    missing = [p for p in a.path if not (ROOT / p).exists()]
    assert not missing, f"{len(missing)} missing files, e.g. {missing[:3]}"
    a.to_csv(D / "commands_schema_b_all.csv", index=False)
    t = pd.crosstab(a["class"], a.split).reindex(list(lm["class"]) + ["unknown"]).fillna(0).astype(int)
    print(t.to_string())
    print(pd.crosstab(a.dataset, a.split).to_string())
    for x, y in (("train", "validation"), ("train", "test"), ("validation", "test")):
        s = set(a[a.split == x].speaker.dropna()) & set(a[a.split == y].speaker.dropna())
        print(f"speakers in both {x} and {y}: {len(s)}" + (f" e.g. {sorted(s)[:8]}" if s else ""))
    print(f"{len(a)} clips -> {D / 'commands_schema_b_all.csv'}")


if __name__ == "__main__":
    main()
