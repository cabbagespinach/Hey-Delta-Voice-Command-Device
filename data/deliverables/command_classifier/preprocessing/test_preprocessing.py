#!/usr/bin/env python3
"""Tests for the command front end.  python test_preprocessing.py  (or pytest)"""
from pathlib import Path
import json, sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import command_preprocessing as cp                                        # noqa: E402

CFG = cp.load_config()


def test_shapes():
    assert CFG.num_samples == 80000 and CFG.n_frames == 497
    pre = cp.CommandPreprocessor.from_files()
    assert tuple(pre(torch.zeros(80000)).shape) == (1, 40, 497)
    assert tuple(pre(torch.zeros(3, 80000)).shape) == (3, 1, 40, 497)


def test_short_clip_padded_right_with_noise():
    w = cp.frame_command(torch.ones(16000) * 0.5, CFG, pad_key="x")
    assert w.shape == (80000,)
    assert torch.all(w[:16000] == 0.5)                  # the capture stays at the start
    tail = w[16000:]
    assert tail.abs().max() > 0                         # noise, not digital silence
    rms_db = 20 * np.log10(float(tail.pow(2).mean().sqrt()))
    assert abs(rms_db - CFG.pad_noise_dbfs) < 1.0


def test_long_clip_keeps_start():
    x = torch.arange(100000, dtype=torch.float32) / 1e6
    w = cp.frame_command(x, CFG)
    assert torch.equal(w, x[:80000])


def test_padding_deterministic():
    a = cp.frame_command(torch.zeros(1000), CFG, pad_key="k")
    b = cp.frame_command(torch.zeros(1000), CFG, pad_key="k")
    assert torch.equal(a, b)


def test_same_features_as_wakeword_definition():
    ww = cp.wp.PreprocessConfig.from_json(HERE.parents[1] / "preprocessing/preprocessing_config.json")
    for k in ("sample_rate", "n_fft", "win_length", "hop_length", "n_mels", "f_min", "f_max", "power", "mel_scale",
              "center", "db_amin", "db_multiplier", "pad_noise_dbfs"):
        assert getattr(ww, k) == getattr(CFG, k), k


def test_stats_are_train_only_and_match_config():
    s = json.loads((HERE / "normalization_stats.json").read_text())
    assert s["splits_used"] == ["train"] and s["config_feature_hash"] == CFG.feature_hash()
    assert len(s["mean"]) == 40 and min(s["std"]) > 0


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
