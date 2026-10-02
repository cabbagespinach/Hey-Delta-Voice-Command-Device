#!/usr/bin/env python3
"""
Link the shared data folder into this clone (called by setup_data.sh --server).

    python repro/link_data.py [/home/arvir.jane.redondo/AI231_ME2_reproduce]

Nothing is copied: a folder of the shared data that does not exist in the clone becomes one symbolic link; a folder
that exists in the clone (or that restore_optional.py writes into, repro_common.WRITABLE_DIRS) stays a real folder
and gets one link per file. Existing files in the clone are never replaced. Re-runnable.
"""
from pathlib import Path
import os, sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro_common import REPO, DEFAULT_SHARED, WRITABLE_DIRS   # noqa: E402


def must_recurse(rel: str) -> bool:
    return any(w == rel or w.startswith(rel + "/") or rel.startswith(w + "/") for w in WRITABLE_DIRS)


def link_tree(src: Path, dst: Path, rel: str, counts):
    if not dst.exists() and not dst.is_symlink() and not must_recurse(rel):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src, target_is_directory=src.is_dir())
        counts["dir" if src.is_dir() else "file"] += 1
        return
    if src.is_dir():
        if dst.is_symlink():                       # an earlier whole-folder link where a real folder is needed
            dst.unlink()
        dst.mkdir(parents=True, exist_ok=True)
        for c in os.scandir(src):
            link_tree(Path(c.path), dst / c.name, f"{rel}/{c.name}", counts)
    elif not dst.exists():
        dst.symlink_to(src)
        counts["file"] += 1


def main():
    shared = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SHARED
    if not (shared / "data").is_dir():
        sys.exit(f"shared data folder not found or not readable: {shared}/data")
    counts = {"dir": 0, "file": 0}
    link_tree(shared / "data", REPO / "data", "data", counts)
    print(f"linked {counts['dir']} folders and {counts['file']} files from {shared}")


if __name__ == "__main__":
    main()
