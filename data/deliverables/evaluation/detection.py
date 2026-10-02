#!/usr/bin/env python3
"""
Shared evaluation machinery: scoring, sliding-window streaming inference, trigger
post-processing, ground-truth matching and metrics.

Streaming inference uses the deployment front end unchanged:
preprocessing.StreamingPreprocessor (1.5 s windows, 100 ms hop, same features and
frozen normalization as training), fed in fixed-size audio chunks.
"""
from __future__ import annotations

from pathlib import Path
import math, sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.stats import beta as beta_dist, chi2

HERE = Path(__file__).resolve().parent
DELIV = HERE.parent
ROOT = DELIV.parent.parent
sys.path.insert(0, str(DELIV / "preprocessing"))
sys.path.insert(0, str(DELIV / "dataloading"))
import wakeword_preprocessing as wp      # noqa: E402
import reproducibility as rep            # noqa: E402


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
class TorchScorer:
    """features [B, 1, n_mels, n_frames] -> wakeword probability [B] (float64 numpy).

    Runs on CPU with one thread by default so scores are bit-reproducible."""

    def __init__(self, model: torch.nn.Module, batch_size: int = 512, device: str = "cpu"):
        self.model, self.bs, self.device = model.to(device).eval(), batch_size, device

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> np.ndarray:
        out = [torch.sigmoid(self.model(x[i:i + self.bs].to(self.device)).reshape(-1)).double().cpu()
               for i in range(0, len(x), self.bs)]
        return torch.cat(out).numpy() if out else np.zeros(0)


def stream_scores(path, pre: wp.WakewordPreprocessor, scorer, chunk_sec: float = 0.5):
    """Score a stream exactly as the device would: StreamingPreprocessor fed chunk by chunk.
    Returns (window_end_times_sec, scores)."""
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    sp = wp.StreamingPreprocessor(pre, input_sample_rate=sr)
    chunk = max(1, int(chunk_sec * sr))
    starts, feats = [], []
    with rep.single_thread():
        for i in range(0, len(y), chunk):
            s, f = sp.push(torch.from_numpy(y[i:i + chunk]))
            starts += s
            if len(s):
                feats.append(f)
        scores = scorer(torch.cat(feats)) if feats else np.zeros(0)
    times = (np.asarray(starts) + pre.cfg.num_samples) / pre.cfg.sample_rate
    return times, scores


# ----------------------------------------------------------------------------
# Triggers and matching
# ----------------------------------------------------------------------------
def smooth(scores: np.ndarray, k: int) -> np.ndarray:
    """Causal moving average over the last k windows (k=1: unchanged)."""
    if k <= 1:
        return scores
    c = np.cumsum(np.insert(scores, 0, 0.0))
    out = np.empty_like(scores)
    for i in range(len(scores)):
        a = max(0, i + 1 - k)
        out[i] = (c[i + 1] - c[a]) / (i + 1 - a)
    return out


def detect(times, scores, threshold: float, smoothing_windows: int = 1, refractory_sec: float = 1.0,
           k: int = 1, n: int = 1):
    """Trigger when at least k of the last n (smoothed) window scores reach the threshold, then stay
    silent for refractory_sec. k = n = 1 is a plain threshold."""
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    s = smooth(np.asarray(scores, dtype=np.float64), smoothing_windows)
    above = s >= threshold
    trig, last = [], -math.inf
    for i, (t, v) in enumerate(zip(times, s)):
        if above[i] and above[max(0, i + 1 - n):i + 1].sum() >= k and t - last >= refractory_sec:
            trig.append((float(t), float(v)))
            last = t
    return trig


def union_length(intervals, lo, hi) -> float:
    total, cur = 0.0, lo
    for a, b in sorted(intervals):
        a, b = max(a, cur), min(b, hi)
        if b > a:
            total += b - a
            cur = b
    return total


def match(triggers, wakes, duration: float, window_sec: float, before: float, after: float):
    """wakes: list of (event_id, wake_start, wake_end). Returns dict with hits, misses,
    false accepts, duplicates and the time during which a false accept was possible."""
    zones = [(eid, ws - before, we + after, we) for eid, ws, we in sorted(wakes, key=lambda w: w[1])]
    hit = {}
    fas, dups = [], 0
    for t, v in triggers:
        inside = [z for z in zones if z[1] <= t <= z[2]]
        if not inside:
            fas.append((t, v))
            continue
        free = [z for z in inside if z[0] not in hit]
        if free:
            z = free[0]
            hit[z[0]] = dict(trigger_sec=t, score=v, latency_sec=t - z[3])
        else:
            dups += 1
    # A window can only fire once a full window of audio exists: exposure starts at window_sec.
    exposure = max(0.0, duration - window_sec) - union_length([(a, b) for _, a, b, _ in zones], window_sec, duration)
    return dict(hits=hit, misses=[z[0] for z in zones if z[0] not in hit], false_accepts=fas,
                duplicates=dups, exposure_sec=exposure)


def attribute(t: float, window_sec: float, events: pd.DataFrame, background: str) -> tuple:
    """Category of a false accept at time t: the non-wakeword event overlapping the triggering
    window [t - window_sec, t] the most, else the stream background condition."""
    if len(events):
        ov = (np.minimum(events.insert_end_sec, t) - np.maximum(events.insert_start_sec, t - window_sec)).clip(lower=0)
        if ov.max() > 0:
            i = int(ov.values.argmax())
            return events.iloc[i].event_type, events.iloc[i].event_id
    return f"background:{background}", None


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def poisson_rate_ci(k: int, exposure_h: float, conf=0.95):
    """Rate k/exposure with an exact (Garwood) Poisson interval."""
    if exposure_h <= 0:
        return (float("nan"),) * 3
    a = 1 - conf
    lo = 0.0 if k == 0 else chi2.ppf(a / 2, 2 * k) / 2
    hi = chi2.ppf(1 - a / 2, 2 * k + 2) / 2
    return k / exposure_h, lo / exposure_h, hi / exposure_h


def proportion_ci(k: int, n: int, conf=0.95):
    """k/n with an exact (Clopper-Pearson) interval."""
    if n == 0:
        return (float("nan"),) * 3
    a = 1 - conf
    lo = 0.0 if k == 0 else beta_dist.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - a / 2, k + 1, n - k)
    return k / n, float(lo), float(hi)


def roc_auc(labels, scores) -> float:
    """Mann-Whitney AUC with tie handling."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def average_precision(labels, scores) -> float:
    labels, scores = np.asarray(labels), np.asarray(scores)
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(1, y.sum()))


def threshold_at_fpr(neg_scores, target_fpr: float) -> float:
    """Lowest threshold t with #(neg >= t) / #neg <= target_fpr."""
    s = np.sort(np.asarray(neg_scores, dtype=np.float64))[::-1]
    k = int(math.floor(target_fpr * len(s)))
    if k >= len(s):
        return float(s[-1])
    return float(np.nextafter(s[k], np.inf))
