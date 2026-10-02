#!/usr/bin/env python3
"""
OWNER-SIDE: build the read-only data folder the professor reproduces from on this HPC server (2026-10-02).

    python repro/make_shared_folder.py [--dest /home/arvir.jane.redondo/AI231_ME2_reproduce]

Every file in repro_common.DATA_DIRS is HARD-LINKED (same file on disk, no extra space) into --dest, which lies
outside the private ~/sandbox, so other users reach it without passing through ~/sandbox. Never shared: Qualcomm,
Hey Snips clips, FSC clips, classmates' recordings (repro_common.NEVER_SHARE): the professor obtains those himself.
Re-runnable: adds what is new, removes links whose source is gone. Then checks that every audio path referenced by
a tracked CSV is either shared or a known optional dataset, and that other users can read the folder.
"""
from pathlib import Path
import argparse, os, stat, subprocess, sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro_common import REPO, DEFAULT_SHARED, DATA_DIRS, NEVER_SHARE, AUDIO   # noqa: E402

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(DEFAULT_SHARED))
    a = ap.parse_args()
    dest = Path(a.dest)
    dest.mkdir(mode=0o755, exist_ok=True)
    os.chmod(dest, 0o755)
    want, skipped, unreadable = set(), 0, []
    # files in git reach the professor through the clone (anonymised in the public version): not shared again
    tracked = set(subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout.split("\n"))
    for d in DATA_DIRS:
        for root, dirs, files in os.walk(REPO / d):
            for f in files:
                src = Path(root) / f
                rel = src.relative_to(REPO).as_posix()
                if NEVER_SHARE.search(rel) or rel in tracked:
                    skipped += 1
                    continue
                if not os.stat(src).st_mode & stat.S_IROTH:
                    unreadable.append(rel)
                    continue
                want.add(rel)
    made = 0
    for rel in sorted(want):
        t = dest / rel
        if t.exists():
            continue
        t.parent.mkdir(parents=True, exist_ok=True)
        os.link(REPO / rel, t)
        made += 1
    gone = 0
    for root, dirs, files in os.walk(dest):
        for f in files:
            rel = (Path(root) / f).relative_to(dest).as_posix()
            if rel not in want:
                (Path(root) / f).unlink()
                gone += 1
    for root, dirs, files in os.walk(dest):                    # read-only for everyone else
        os.chmod(root, 0o755)
    print(f"{len(want)} files shared ({made} new, {gone} removed), {skipped} never-shared files skipped")
    if unreadable:
        print(f"WARNING: {len(unreadable)} files are not readable by other users, not shared, e.g. {unreadable[:3]}")

    # coverage: every audio path in a tracked CSV must be shared, or belong to an optional dataset
    tracked = subprocess.run(["git", "ls-files", "*.csv"], cwd=REPO, capture_output=True, text=True).stdout.split()
    missing = {}
    for f in tracked:
        try:
            df = pd.read_csv(REPO / f, low_memory=False, dtype=str)
        except Exception:
            continue
        for c in df.columns:
            v = df[c].dropna()
            v = v[v.str.startswith("data/") & v.str.contains(r"\.(?:wav|flac|mp3|opus|ogg)$", case=False, regex=True)]
            for p in v:
                if p not in want and not NEVER_SHARE.search(p):
                    missing.setdefault(f, set()).add(p)
    if missing:
        print("audio referenced by tracked CSVs but NOT shared (and not an optional dataset):")
        for f, s in missing.items():
            print(f"  {f}: {len(s)} e.g. {sorted(s)[:2]}")
    else:
        print("coverage OK: every audio path in the tracked CSVs is shared or belongs to an optional dataset")
    chain = subprocess.run(["namei", "-m", str(dest)], capture_output=True, text=True).stdout
    print(chain)


if __name__ == "__main__":
    main()
