#!/usr/bin/env python3
"""
Deterministic "Hey Delta" preprocessing shared by training, validation, test and
streaming inference.

    raw waveform (any rate, any channel count)
      -> mono float32, resampled to cfg.sample_rate           load_waveform / StreamResampler
      -> training-only waveform augmentation (caller opt-in)   WindowDataset(augment=...)
      -> fixed-duration framing to cfg.num_samples             frame_at / frame_clip / StreamingPreprocessor
      -> MelSpectrogram (power)                                FeatureExtractor
      -> dB with a fixed floor (no top_db, no reference max)   FeatureExtractor
      -> per-mel-bin normalization, frozen training stats      Normalizer
      -> model-ready features [1, n_mels, n_frames]            WakewordPreprocessor

Every stage after loading is a pure function of its input and the config: no
per-utterance statistics, no peak normalisation and no randomness unless the
caller passes an augmentation. An isolated clip and a streaming window go
through the same WakewordPreprocessor.__call__.

The policy (and why) is documented in README.md and preprocessing_config.json.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import hashlib, json, math

import numpy as np
import soundfile as sf
import torch
import torchaudio

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = (HERE / "../../..").resolve()
DEFAULT_CONFIG = HERE / "preprocessing_config.json"
DEFAULT_STATS = HERE / "normalization_stats.json"
DEFAULT_WINDOWS = HERE / "../segmentation_windowing/outputs/windows.csv"

# Fields that do not change the features of a fixed-length window; excluded from
# feature_hash so that e.g. changing the streaming hop does not invalidate stats.
NON_FEATURE_FIELDS = {"stream_hop_sec", "short_clip_placement", "long_clip_policy"}


def stable_hash(*parts) -> int:
    """Process-independent 63-bit hash (Python's hash() is salted per process)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


# ----------------------------------------------------------------------------
# Central configuration
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PreprocessConfig:
    version: str = "1.0"
    # Waveform
    sample_rate: int = 16000
    window_sec: float = 1.5
    pad_mode: str = "noise"            # "noise" | "zeros"
    pad_noise_dbfs: float = -50.6      # RMS of the Gaussian fill; = measured median RPI floor on the deployment metric
    short_clip_placement: str = "center"   # isolated clips shorter than a window: "center" | "start" | "end"
    long_clip_policy: str = "error"        # isolated clips longer than a window: "error" | "sliding"
    # Resampling (torchaudio.functional.resample)
    resampling_method: str = "sinc_interp_hann"
    lowpass_filter_width: int = 6
    rolloff: float = 0.99
    # MelSpectrogram
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 40
    f_min: float = 20.0
    f_max: float = 7600.0
    power: float = 2.0
    mel_scale: str = "htk"
    center: bool = False
    # dB
    db_amin: float = 1e-10
    db_multiplier: float = 10.0
    # Normalization
    norm_eps: float = 1e-5
    # Streaming
    stream_hop_sec: float = 0.1

    def __post_init__(self):
        self.validate()

    # Derived sizes -----------------------------------------------------------
    @property
    def num_samples(self) -> int:
        return int(round(self.window_sec * self.sample_rate))

    @property
    def n_frames(self) -> int:
        if self.center:
            return self.num_samples // self.hop_length + 1
        return (self.num_samples - self.n_fft) // self.hop_length + 1

    @property
    def feature_shape(self) -> tuple:
        return (1, self.n_mels, self.n_frames)

    @property
    def stream_hop_samples(self) -> int:
        return int(round(self.stream_hop_sec * self.sample_rate))

    # Checks ------------------------------------------------------------------
    def validate(self):
        def need(ok, msg):
            if not ok:
                raise ValueError(f"PreprocessConfig: {msg}")
        need(self.sample_rate > 0, "sample_rate must be positive")
        need(abs(self.window_sec * self.sample_rate - round(self.window_sec * self.sample_rate)) < 1e-6,
             "window_sec * sample_rate must be a whole number of samples")
        need(self.win_length <= self.n_fft, "win_length must be <= n_fft")
        need(0 < self.hop_length <= self.win_length, "hop_length must be in (0, win_length]")
        need(self.num_samples >= self.n_fft, "window shorter than n_fft")
        need(0 <= self.f_min < self.f_max <= self.sample_rate / 2, "need 0 <= f_min < f_max <= sample_rate / 2")
        need(self.pad_mode in ("zeros", "noise"), "pad_mode must be 'zeros' or 'noise'")
        need(self.short_clip_placement in ("center", "start", "end"), "bad short_clip_placement")
        need(self.long_clip_policy in ("error", "sliding"), "bad long_clip_policy")
        need(self.db_amin > 0, "db_amin must be positive")
        need(abs(self.stream_hop_sec * self.sample_rate - round(self.stream_hop_sec * self.sample_rate)) < 1e-6
             and self.stream_hop_samples > 0, "stream_hop_sec must be a positive whole number of samples")

    # Serialisation -----------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def feature_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in NON_FEATURE_FIELDS}

    def feature_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.feature_dict(), sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessConfig":
        known = {f.name for f in fields(cls)}
        unknown = {k for k in d if not k.startswith("_")} - known
        if unknown:
            raise ValueError(f"PreprocessConfig: unknown keys {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, path=DEFAULT_CONFIG) -> "PreprocessConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ----------------------------------------------------------------------------
# Waveform loading and resampling
# ----------------------------------------------------------------------------
def resample(wave: torch.Tensor, orig_sr: int, cfg: PreprocessConfig) -> torch.Tensor:
    if int(orig_sr) == cfg.sample_rate:
        return wave
    return torchaudio.functional.resample(
        wave, int(orig_sr), cfg.sample_rate, lowpass_filter_width=cfg.lowpass_filter_width,
        rolloff=cfg.rolloff, resampling_method=cfg.resampling_method)


def load_waveform(path, cfg: PreprocessConfig) -> torch.Tensor:
    """Mono float32 [T] at cfg.sample_rate. Amplitude is kept as stored (no normalisation).

    soundfile decides the format from the file content, so the MP3 data stored
    under .wav names (17 edge-tts positives) decodes the same way every time.
    """
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wave = torch.from_numpy(np.ascontiguousarray(y.mean(axis=1)))
    return resample(wave, sr, cfg)


class StreamResampler:
    """Chunked resampler whose output matches resample() on the whole signal.

    Input is processed in blocks of whole resampling periods (orig/g input
    samples -> new/g output samples). Each block is resampled together with C
    periods of real left and right context, so the kernel never sees an
    artificial edge except at stream start, where the zero history equals the
    zero padding resample() applies to a whole file. Latency: C periods.
    """

    def __init__(self, input_sr: int, cfg: PreprocessConfig):
        self.cfg, self.input_sr = cfg, int(input_sr)
        g = math.gcd(self.input_sr, cfg.sample_rate)
        self.u_in, self.u_out = self.input_sr // g, cfg.sample_rate // g
        base = min(self.u_in, self.u_out) * cfg.rolloff
        width = math.ceil(cfg.lowpass_filter_width * self.u_in / base)  # kernel half-width, input samples
        self.ctx = math.ceil(width / self.u_in) + 1                     # context, in periods
        self.hist = torch.zeros(self.ctx * self.u_in)
        self.pending = torch.zeros(0)

    def push(self, chunk: torch.Tensor) -> torch.Tensor:
        self.pending = torch.cat([self.pending, chunk.to(torch.float32)])
        k = self.pending.numel() // self.u_in - self.ctx
        if k <= 0:
            return torch.zeros(0)
        n = k * self.u_in
        x = torch.cat([self.hist, self.pending[:n + self.ctx * self.u_in]])
        y = resample(x, self.input_sr, self.cfg)
        out = y[self.ctx * self.u_out: self.ctx * self.u_out + k * self.u_out]
        self.hist = torch.cat([self.hist, self.pending[:n]])[-self.ctx * self.u_in:]
        self.pending = self.pending[n:]
        return out


# ----------------------------------------------------------------------------
# Fixed-duration framing
# ----------------------------------------------------------------------------
def _pad_fill(n: int, cfg: PreprocessConfig, key, side: str) -> torch.Tensor:
    if n <= 0:
        return torch.zeros(0)
    if cfg.pad_mode == "zeros":
        return torch.zeros(n)
    g = torch.Generator().manual_seed(stable_hash("pad", key, side))
    return torch.randn(n, generator=g) * (10.0 ** (cfg.pad_noise_dbfs / 20.0))


def frame_at(wave: torch.Tensor, start: int, cfg: PreprocessConfig, pad_key="") -> torch.Tensor:
    """The cfg.num_samples samples starting at sample `start` (may be negative or run
    past the end); positions outside the waveform are padded per cfg.pad_mode.
    Never truncates a window: the window length is fixed and the caller chooses start."""
    N, T = cfg.num_samples, wave.numel()
    a, b = max(0, start), min(T, start + N)
    body = wave[a:b] if b > a else torch.zeros(0)
    left = _pad_fill(a - start if b > a else N, cfg, pad_key, "left")
    right = _pad_fill(N - left.numel() - body.numel(), cfg, pad_key, "right")
    out = torch.cat([left, body.to(torch.float32), right])
    assert out.numel() == N
    return out


def sliding_starts(total: int, cfg: PreprocessConfig, hop: int | None = None) -> list:
    """Window starts 0, hop, 2*hop, ... for every window fully inside `total` samples.
    Identical to the schedule StreamingPreprocessor follows on the same audio."""
    hop = hop or cfg.stream_hop_samples
    return list(range(0, total - cfg.num_samples + 1, hop))


def frame_clip(wave: torch.Tensor, cfg: PreprocessConfig, pad_key="clip") -> torch.Tensor:
    """Frame an isolated clip into [K, num_samples].

    Shorter clips: one window, placed per cfg.short_clip_placement (default centre,
    the same rule segmentation_windowing uses for short clips).
    Longer clips: an error by default, because cutting could drop part of a
    wakeword; with long_clip_policy="sliding", the streaming window schedule plus a
    final end-aligned window so the tail is covered.
    """
    N, T = cfg.num_samples, wave.numel()
    if T <= N:
        start = {"center": -((N - T) // 2), "start": 0, "end": -(N - T)}[cfg.short_clip_placement]
        return frame_at(wave, start, cfg, pad_key).unsqueeze(0)
    if cfg.long_clip_policy == "error":
        raise ValueError(f"clip is {T / cfg.sample_rate:.3f} s, longer than the {cfg.window_sec} s window; "
                         "use long_clip_policy='sliding' or StreamingPreprocessor")
    starts = sliding_starts(T, cfg)
    if starts[-1] + N < T:
        starts.append(T - N)
    return torch.stack([frame_at(wave, s, cfg, pad_key) for s in starts])


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
class FeatureExtractor(torch.nn.Module):
    """[..., num_samples] waveform -> [..., n_mels, n_frames] log-mel in dB.

    dB uses a fixed floor (db_amin) and reference 1.0, and no top_db: torchaudio's
    top_db clamps relative to each input's own maximum, which would make a window's
    features depend on its loudest frame and remove absolute level.
    """

    def __init__(self, cfg: PreprocessConfig):
        super().__init__()
        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, win_length=cfg.win_length,
            hop_length=cfg.hop_length, f_min=cfg.f_min, f_max=cfg.f_max, n_mels=cfg.n_mels,
            power=cfg.power, center=cfg.center, mel_scale=cfg.mel_scale, norm=None,
            window_fn=torch.hann_window)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[-1] != self.cfg.num_samples:
            raise ValueError(f"expected {self.cfg.num_samples} samples per window, got {frames.shape[-1]}")
        m = self.mel(frames.to(torch.float32))
        return torchaudio.functional.amplitude_to_DB(
            m, multiplier=self.cfg.db_multiplier, amin=self.cfg.db_amin, db_multiplier=0.0, top_db=None)


class Normalizer(torch.nn.Module):
    """(x - mean) / std per mel bin, with statistics frozen from the training split."""

    def __init__(self, mean, std, cfg: PreprocessConfig):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1, 1)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(-1, 1)
        if mean.shape[0] != cfg.n_mels or std.shape[0] != cfg.n_mels:
            raise ValueError("normalization stats do not match n_mels")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(cfg.norm_eps))

    @classmethod
    def from_stats(cls, path, cfg: PreprocessConfig) -> "Normalizer":
        s = json.loads(Path(path).read_text())
        if s["config_feature_hash"] != cfg.feature_hash():
            raise ValueError(f"{path} was computed with feature config {s['config_feature_hash']}, "
                             f"current config is {cfg.feature_hash()}; rerun compute_normalization_stats.py")
        if s.get("splits_used") != ["train"]:
            raise ValueError(f"{path}: statistics must come from the train split only, got {s.get('splits_used')}")
        return cls(s["mean"], s["std"], cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class WakewordPreprocessor(torch.nn.Module):
    """Framed waveform(s) -> model-ready features.

    [num_samples]      -> [1, n_mels, n_frames]
    [B, num_samples]   -> [B, 1, n_mels, n_frames]
    """

    def __init__(self, cfg: PreprocessConfig, stats_path=DEFAULT_STATS, normalize: bool = True):
        super().__init__()
        self.cfg = cfg
        self.extract = FeatureExtractor(cfg)
        self.normalizer = Normalizer.from_stats(stats_path, cfg) if normalize else None

    @classmethod
    def from_files(cls, config_path=DEFAULT_CONFIG, stats_path=DEFAULT_STATS) -> "WakewordPreprocessor":
        return cls(PreprocessConfig.from_json(config_path), stats_path)

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = self.extract(frames)
        if self.normalizer is not None:
            x = self.normalizer(x)
        return x.unsqueeze(-3)

    def process_clip(self, wave_or_path, sample_rate: int | None = None) -> torch.Tensor:
        """Isolated clip (path, or waveform at `sample_rate`) -> [K, 1, n_mels, n_frames]."""
        if isinstance(wave_or_path, (str, Path)):
            wave = load_waveform(wave_or_path, self.cfg)
        else:
            wave = torch.as_tensor(wave_or_path, dtype=torch.float32)
            if wave.dim() == 2:
                wave = wave.mean(dim=0)
            wave = resample(wave, sample_rate or self.cfg.sample_rate, self.cfg)
        return self(frame_clip(wave, self.cfg))


# ----------------------------------------------------------------------------
# Streaming inference
# ----------------------------------------------------------------------------
class StreamingPreprocessor:
    """Sliding-window front end for a live stream.

    push() audio chunks of any size at `input_sample_rate`; it returns the
    features of every window that became complete: windows start at 0, hop,
    2*hop, ... (hop = cfg.stream_hop_sec) in resampled stream time, exactly the
    schedule of sliding_starts(). Each window goes through the same
    WakewordPreprocessor as training/validation windows. No window is emitted
    before a full window of audio has arrived, so streaming never pads.
    """

    def __init__(self, preprocessor: WakewordPreprocessor, input_sample_rate: int | None = None):
        self.pre, self.cfg = preprocessor, preprocessor.cfg
        sr = input_sample_rate or self.cfg.sample_rate
        self.resampler = None if sr == self.cfg.sample_rate else StreamResampler(sr, self.cfg)
        self.buf = torch.zeros(0)
        self.buf_start = 0     # stream sample index (target rate) of buf[0]
        self.next_start = 0    # start sample of the next window to emit

    def push(self, chunk) -> tuple:
        """Returns (window_start_samples: list[int], features: [K, 1, n_mels, n_frames])."""
        chunk = torch.as_tensor(chunk, dtype=torch.float32).reshape(-1)
        if self.resampler is not None:
            chunk = self.resampler.push(chunk)
        self.buf = torch.cat([self.buf, chunk])
        N, hop = self.cfg.num_samples, self.cfg.stream_hop_samples
        starts, frames = [], []
        while self.next_start + N <= self.buf_start + self.buf.numel():
            a = self.next_start - self.buf_start
            frames.append(self.buf[a:a + N])
            starts.append(self.next_start)
            self.next_start += hop
        drop = self.next_start - self.buf_start
        if drop > 0:
            self.buf, self.buf_start = self.buf[drop:], self.buf_start + drop
        if not frames:
            return [], torch.zeros(0, *self.cfg.feature_shape)
        return starts, self.pre(torch.stack(frames))


# ----------------------------------------------------------------------------
# Dataset over segmentation_windowing/outputs/windows.csv
# ----------------------------------------------------------------------------
class WindowDataset(torch.utils.data.Dataset):
    """Model-ready examples for the windows of one or more splits.

    Returns (features [1, n_mels, n_frames], label float {0,1}, window_id).

    augment: None (default) or a callable
        augment(waveform [T] float32, sample_rate, generator, meta: dict) -> waveform [T]
    applied to the real-audio excerpt of the window (before framing). Only the
    train split accepts it; the generator is seeded from (seed, epoch, window_id),
    so augmented training runs are reproducible and validation/test never change.
    """

    def __init__(self, preprocessor: WakewordPreprocessor, splits, windows_csv=DEFAULT_WINDOWS,
                 augment=None, seed: int = 0, cache_files: int = 256):
        import pandas as pd
        self.pre, self.cfg = preprocessor, preprocessor.cfg
        splits = [splits] if isinstance(splits, str) else list(splits)
        if augment is not None and splits != ["train"]:
            raise ValueError(f"augmentation is training-only; got splits={splits}")
        w = pd.read_csv(windows_csv)
        self.rows = w[w.split.isin(splits)].reset_index(drop=True)
        self.augment, self.seed, self.epoch = augment, seed, 0
        self._cache, self._cache_size = OrderedDict(), cache_files

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def _wave(self, filepath: str) -> torch.Tensor:
        if filepath in self._cache:
            self._cache.move_to_end(filepath)
            return self._cache[filepath]
        wave = load_waveform(PROJECT_ROOT / filepath, self.cfg)
        self._cache[filepath] = wave
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return wave

    def frame(self, i: int) -> torch.Tensor:
        """The framed (optionally augmented) waveform of window i, [num_samples]."""
        r = self.rows.iloc[i]
        wave = self._wave(r.filepath)
        start = int(round(r.start_sec * self.cfg.sample_rate))
        a, b = max(0, start), min(wave.numel(), start + self.cfg.num_samples)
        excerpt = wave[a:b]
        if self.augment is not None:
            g = torch.Generator().manual_seed(stable_hash(self.seed, self.epoch, r.window_id))
            out = self.augment(excerpt.clone(), self.cfg.sample_rate, g, r.to_dict())
            if out.shape != excerpt.shape:
                raise ValueError("augmentation must preserve waveform length")
            excerpt = out
        return frame_at(excerpt, start - a, self.cfg, pad_key=r.window_id)

    def __getitem__(self, i: int):
        r = self.rows.iloc[i]
        return self.pre(self.frame(i)), torch.tensor(float(r.label == "positive")), r.window_id
