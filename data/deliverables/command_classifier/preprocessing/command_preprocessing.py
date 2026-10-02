#!/usr/bin/env python3
"""
Command-classifier front end: the wakeword project's preprocessing (../../preprocessing/wakeword_preprocessing.py)
with a 5 s window starting at the capture start.

    capture waveform (any rate, mono/stereo)
      -> mono float32, 16 kHz                              load_waveform (wakeword module)
      -> first 5 s; shorter clips padded on the right      frame_command
         with noise at the measured RPI floor
      -> log-mel [40, 497] in dB, fixed floor               FeatureExtractor (wakeword module)
      -> per-mel-bin normalisation, TRAIN statistics        Normalizer (wakeword module; normalization_stats.json)

    import command_preprocessing as cp
    pre = cp.CommandPreprocessor.from_files()             # config + frozen stats
    x = pre(cp.frame_command(wave))                       # [1, 40, 497];  batched: [B, 80000] -> [B, 1, 40, 497]

The same module is used for training (features computed on the GPU from augmented waveforms), evaluation and
the ONNX export, so there is one definition of the features.
"""
from __future__ import annotations

from pathlib import Path
import os, sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "preprocessing"))
import wakeword_preprocessing as wp                                    # noqa: E402

DEFAULT_CONFIG = HERE / "preprocessing_config.json"
# COMMAND_NORM_STATS: another stats file (schema-B run, 2026-10-01), so the current model's file is left alone
DEFAULT_STATS = Path(os.environ.get("COMMAND_NORM_STATS", HERE / "normalization_stats.json"))
PROJECT_ROOT = HERE.parents[3]

PreprocessConfig = wp.PreprocessConfig
FeatureExtractor = wp.FeatureExtractor
Normalizer = wp.Normalizer
load_waveform = wp.load_waveform


def load_config(path=DEFAULT_CONFIG) -> PreprocessConfig:
    return PreprocessConfig.from_json(path)


def frame_command(wave: torch.Tensor, cfg: PreprocessConfig | None = None, pad_key="clip") -> torch.Tensor:
    """First cfg.num_samples samples of a capture; shorter captures padded on the right (cfg.pad_mode)."""
    cfg = cfg or load_config()
    wave = torch.as_tensor(wave, dtype=torch.float32).reshape(-1)
    return wp.frame_at(wave[:cfg.num_samples], 0, cfg, pad_key)


class CommandPreprocessor(torch.nn.Module):
    """[num_samples] -> [1, 40, T];  [B, num_samples] -> [B, 1, 40, T]."""

    def __init__(self, cfg: PreprocessConfig, stats_path=DEFAULT_STATS, normalize: bool = True):
        super().__init__()
        self.cfg = cfg
        self.extract = FeatureExtractor(cfg)
        self.normalizer = Normalizer.from_stats(stats_path, cfg) if normalize else None

    @classmethod
    def from_files(cls, config_path=DEFAULT_CONFIG, stats_path=DEFAULT_STATS, normalize=True) -> "CommandPreprocessor":
        return cls(load_config(config_path), stats_path, normalize)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = self.extract(frames)
        if self.normalizer is not None:
            x = self.normalizer(x)
        return x.unsqueeze(-3)

    @torch.no_grad()
    def process_clip(self, wave_or_path, sample_rate: int | None = None) -> torch.Tensor:
        if isinstance(wave_or_path, (str, Path)):
            wave = load_waveform(wave_or_path, self.cfg)
        else:
            wave = torch.as_tensor(wave_or_path, dtype=torch.float32)
            if wave.dim() == 2:
                wave = wave.mean(dim=0)
            wave = wp.resample(wave, sample_rate or self.cfg.sample_rate, self.cfg)
        return self(frame_command(wave, self.cfg))
