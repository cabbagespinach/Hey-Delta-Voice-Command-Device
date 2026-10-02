#!/usr/bin/env python3
"""Automated leakage assertions for an already normalized manifest."""
from pathlib import Path
import sys
import pandas as pd

def check(path):
    df = pd.read_csv(path)
    splits = ["train","validation","test"]

    source_sets = {s:set(df.loc[df.split==s,"source_group"]) for s in splits}
    recording_sets = {s:set(df.loc[df.split==s,"recording_id"]) for s in splits}

    for a,b in [("train","validation"),("train","test"),("validation","test")]:
        assert not source_sets[a] & source_sets[b], f"source_group overlap: {a}/{b}"
        assert not recording_sets[a] & recording_sets[b], f"recording_id overlap: {a}/{b}"

    by_file = df.set_index("filepath")["split"].to_dict()
    for _, row in df[df.parent_filepath.notna()].iterrows():
        assert by_file[row.parent_filepath] == row.split, (
            f"child/parent split mismatch: {row.filepath}"
        )

    assert (df.groupby("source_group")["split"].nunique() == 1).all()
    print("PASS: source_group, recording_id, and parent-child leakage checks.")
    print("Rows:", len(df))
    print("Splits:", df.groupby("split").size().to_dict())

if __name__ == "__main__":
    check(sys.argv[1] if len(sys.argv) > 1 else "normalized_manifest.csv")
