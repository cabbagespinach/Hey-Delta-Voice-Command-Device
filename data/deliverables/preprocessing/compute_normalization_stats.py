#!/usr/bin/env python3
"""
Compute the frozen per-mel-bin normalization statistics from the training split only.

Reads segmentation_windowing/outputs/windows.csv, keeps split == "train" windows
(never streaming_eval / validation / test), cross-checks every kept window's
source_group against the frozen split (dataset_split/split_assignments.csv plus
segmentation_windowing/outputs/split_extension.csv), and accumulates the dB
log-mel of each window (no augmentation, no normalization) in float64, in a
fixed order, so the result is deterministic.

Writes normalization_stats.json: mean/std per mel bin, the feature-config hash
they belong to, and fingerprints of exactly which windows were used.

Usage: python compute_normalization_stats.py [--limit N] [--out PATH]
"""
from pathlib import Path
import argparse, datetime, hashlib, json, sys, time

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wakeword_preprocessing as wp

SPLIT_ASSIGNMENTS = HERE / "../dataset_split/split_assignments.csv"
SPLIT_EXTENSION = HERE / "../segmentation_windowing/outputs/split_extension.csv"
TRAIN = "train"


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def window_ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def frozen_split_map() -> dict:
    m = {}
    for path in (SPLIT_ASSIGNMENTS, SPLIT_EXTENSION):
        d = pd.read_csv(path)
        m.update(zip(d.source_group, d.split))
    return m


def select_train_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """Train-split, non-eval windows, verified against the frozen split. Raises on any disagreement."""
    t = windows[(windows.split == TRAIN) & ~windows.is_eval_only.astype(bool)].copy()
    frozen = t.source_group.map(frozen_split_map())
    if frozen.isna().any():
        raise ValueError(f"{int(frozen.isna().sum())} train windows have a source_group missing from the frozen split")
    bad = t[frozen != TRAIN]
    if len(bad):
        raise ValueError(f"{len(bad)} windows labelled train belong to non-train source groups, e.g. "
                         f"{bad.source_group.iloc[0]}")
    return t.sort_values(["filepath", "start_sec", "window_id"]).reset_index(drop=True)


def accumulate(rows: pd.DataFrame, cfg: wp.PreprocessConfig, log_every: int = 1000):
    """Per-bin mean, std, frame count and fraction of values at the dB floor, accumulated
    in float64 over all frames of the given windows."""
    if (rows.split != TRAIN).any():
        raise ValueError("normalization statistics may only use train-split windows")
    extract = wp.FeatureExtractor(cfg)
    s1 = torch.zeros(cfg.n_mels, dtype=torch.float64)
    s2 = torch.zeros(cfg.n_mels, dtype=torch.float64)
    at_floor = torch.zeros(cfg.n_mels, dtype=torch.float64)
    floor_db = cfg.db_multiplier * torch.log10(torch.tensor(cfg.db_amin, dtype=torch.float64)) + 0.01
    n_frames, n_done, t0 = 0, 0, time.time()
    for fp, g in rows.groupby("filepath", sort=True):
        wave = wp.load_waveform(wp.PROJECT_ROOT / fp, cfg)
        frames = torch.stack([wp.frame_at(wave, int(round(r.start_sec * cfg.sample_rate)), cfg, pad_key=r.window_id)
                              for r in g.itertuples(index=False)])
        with torch.no_grad():
            x = extract(frames).to(torch.float64)          # [W, n_mels, n_frames]
        s1 += x.sum(dim=(0, 2))
        s2 += (x * x).sum(dim=(0, 2))
        at_floor += (x <= floor_db).sum(dim=(0, 2))
        n_frames += x.shape[0] * x.shape[2]
        if log_every and (n_done + len(g)) // log_every > n_done // log_every:
            print(f"  {n_done + len(g)}/{len(rows)} windows ({time.time() - t0:.0f} s)")
        n_done += len(g)
    mean = s1 / n_frames
    std = (s2 / n_frames - mean * mean).clamp_min(0).sqrt()
    return mean, std, n_frames, at_floor / n_frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(wp.DEFAULT_CONFIG))
    ap.add_argument("--windows", default=str(wp.DEFAULT_WINDOWS))
    ap.add_argument("--out", default=str(wp.DEFAULT_STATS))
    ap.add_argument("--limit", type=int, help="use only the first N train windows (smoke test; do not ship)")
    args = ap.parse_args()

    if args.limit and Path(args.out).resolve() == wp.DEFAULT_STATS.resolve():
        ap.error("--limit writes partial statistics; pass a different --out so the frozen artifact is not replaced")
    cfg = wp.PreprocessConfig.from_json(args.config)
    windows = pd.read_csv(args.windows)
    train = select_train_windows(windows)
    if args.limit:
        train = train.head(args.limit)
    print(f"[stats] {len(train)} train windows from {train.filepath.nunique()} files; "
          f"excluded (never read): {windows[windows.split != TRAIN].split.value_counts().to_dict()}")
    mean, std, n_frames, at_floor = accumulate(train, cfg)

    out = dict(
        description="Per-mel-bin mean/std of dB log-mel features over all frames of the unaugmented "
                    "train-split windows. Frozen: validation, test and inference apply these values unchanged.",
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        config_feature_hash=cfg.feature_hash(),
        config=cfg.feature_dict(),
        splits_used=[TRAIN],
        windows_csv=str(Path(args.windows).resolve().relative_to(wp.PROJECT_ROOT)),
        windows_csv_sha256=sha256_file(args.windows),
        n_windows=int(len(train)),
        n_windows_by_label=train.label.value_counts().to_dict(),
        n_files=int(train.filepath.nunique()),
        n_frames=int(n_frames),
        train_window_ids_sha256=window_ids_hash(train.window_id),
        mean=[round(float(v), 6) for v in mean],
        std=[round(float(v), 6) for v in std],
        diagnostic_fraction_at_db_floor=[round(float(v), 4) for v in at_floor],
        diagnostic_note="Fraction of frame values per mel bin at the -100 dB floor (exact digital silence: "
                        "zero padding and silent synthetic audio). Not used for normalization.",
    )
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"[stats] {n_frames} frames; mean range {mean.min():.2f}..{mean.max():.2f} dB, "
          f"std range {std.min():.2f}..{std.max():.2f} dB -> {args.out}")


if __name__ == "__main__":
    main()
