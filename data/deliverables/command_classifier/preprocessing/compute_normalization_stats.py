#!/usr/bin/env python3
"""
Frozen per-mel-bin normalisation statistics for the command classifier, from TRAINING draws only.

    python compute_normalization_stats.py [--draws 4000]

The statistics are taken over what the model is trained on: --draws weighted training draws (dataloading
TrainDraws, epoch 0, WITH augmentation and format alignment), log-mel in dB, accumulated in float64 in draw order.
Validation and test clips are never read. Output: normalization_stats.json in the format the wakeword Normalizer
checks (feature-config hash, splits_used = ["train"]).
"""
from pathlib import Path
import argparse, datetime, json, sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dataloading"))
import command_preprocessing as cp                                     # noqa: E402
import command_data as cd                                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    cfg = cd.load_config()
    cfg["epoch_draws"] = a.draws
    ds = cd.TrainDraws(cfg)
    pcfg = cp.load_config()
    ext = cp.FeatureExtractor(pcfg)
    s1 = torch.zeros(pcfg.n_mels, dtype=torch.float64)
    s2 = torch.zeros(pcfg.n_mels, dtype=torch.float64)
    n = 0
    for w, _, _ in cd.make_loader(ds, cfg, num_workers=a.workers, batch_size=64):
        with torch.no_grad():
            x = ext(w).to(torch.float64)
        s1 += x.sum(dim=(0, 2))
        s2 += (x * x).sum(dim=(0, 2))
        n += x.shape[0] * x.shape[2]
    mean = s1 / n
    std = (s2 / n - mean ** 2).clamp_min(0).sqrt()
    out = dict(mean=mean.tolist(), std=std.tolist(), config_feature_hash=pcfg.feature_hash(), splits_used=["train"],
               source="dataloading.TrainDraws epoch 0 (augmented, format-aligned)", draws=a.draws, frames=int(n),
               clips_csv=cfg["clips_csv"], created=datetime.datetime.now().isoformat(timespec="seconds"))
    Path(cp.DEFAULT_STATS).write_text(json.dumps(out, indent=1))
    print(f"{a.draws} draws, {n} frames; mean {mean.mean():.1f} dB, std {std.mean():.1f} dB -> {cp.DEFAULT_STATS}")


if __name__ == "__main__":
    main()
