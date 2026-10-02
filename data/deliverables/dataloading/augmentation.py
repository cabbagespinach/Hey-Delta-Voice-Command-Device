#!/usr/bin/env python3
"""
Training-only waveform augmentation, implementing
../deployment_driven_augmentation_strategy/augmentation_config.json.

    random_mic_coloring -> device_level_gain -> rir_reverberation -> background_noise_mixing -> device_noise_floor

Levels use the strategy's own metric (measure_deployment_levels.py): 150-4000 Hz
band-pass, 20 ms frames, peak_db = 98th percentile, floor_db = 20th percentile.

The augmenter is a pure function of (waveform, rng, metadata): all randomness
comes from the numpy Generator the caller passes, so a given seed always
produces the same augmented example, independent of worker count or order.

rir_reverberation uses the curated RIR bank passed in by the caller (data/rir/rir_bank.csv);
without a bank it is skipped (the config forbids inventing RIRs).
"""
from __future__ import annotations

from pathlib import Path
import json, math

import numpy as np
import pandas as pd
from scipy.signal import butter, fftconvolve, sosfilt

HERE = Path(__file__).resolve().parent
DELIV = HERE.parent
AUG_CONFIG = DELIV / "deployment_driven_augmentation_strategy/augmentation_config.json"
NOISE_BANK = DELIV / "deployment_driven_augmentation_strategy/real_noise_bank.csv"

BAND = (150.0, 4000.0)
FRAME_SEC = 0.02
SILENT_DB = -90.0            # peak below this: nothing to scale against
SYNTHETIC_NOISES = ("synth_white", "synth_pink", "synth_hum", "synth_fan", "synth_clicks")


# ----------------------------------------------------------------------------
# Level metric (same definition as measure_deployment_levels.py)
# ----------------------------------------------------------------------------
_SOS = {}


def band_frames_db(y: np.ndarray, sr: int) -> np.ndarray:
    if sr not in _SOS:
        _SOS[sr] = butter(4, [BAND[0], min(BAND[1], 0.45 * sr)], "bandpass", fs=sr, output="sos")
    yb = sosfilt(_SOS[sr], np.asarray(y, dtype=np.float64))
    n = int(FRAME_SEC * sr)
    k = len(yb) // n
    if k == 0:
        return np.array([-120.0])
    return 20 * np.log10(np.sqrt((yb[:k * n].reshape(k, n) ** 2).mean(1)) + 1e-9)


def peak_db(y, sr) -> float:
    return float(np.percentile(band_frames_db(y, sr), 98))


def floor_db(y, sr) -> float:
    return float(np.percentile(band_frames_db(y, sr), 20))


def longest_zero_run(y: np.ndarray) -> int:
    z = np.concatenate([[0], (np.asarray(y) == 0).astype(np.int8), [0]])
    d = np.diff(z)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return int((ends - starts).max()) if len(starts) else 0


# ----------------------------------------------------------------------------
# Synthetic noise (procedural; no audio content, so no split concerns)
# ----------------------------------------------------------------------------
def coloring_gain_db(freqs: np.ndarray, p: dict, rng: np.random.Generator) -> tuple:
    """Random microphone colouring (config random_mic_coloring): a low shelf, a high shelf and one
    peak, each drawn uniformly in its range (frequencies uniform in log2). Returns (gain_db per
    frequency, sampled parameters)."""
    def logu(spec):
        return float(2 ** rng.uniform(math.log2(spec["min"]), math.log2(spec["max"])))

    def u(spec):
        return float(rng.uniform(spec["min"], spec["max"]))

    k = p["shelf_slope_per_octave"]
    lf = np.log2(np.maximum(freqs, 1.0))
    lo_f, lo_g = logu(p["low_shelf"]["corner_hz"]), u(p["low_shelf"]["gain_db"])
    hi_f, hi_g = logu(p["high_shelf"]["corner_hz"]), u(p["high_shelf"]["gain_db"])
    pk_f, pk_g, pk_bw = logu(p["peak"]["center_hz"]), u(p["peak"]["gain_db"]), u(p["peak"]["bandwidth_octaves"])
    g = (lo_g / (1 + np.exp(k * (lf - math.log2(lo_f))))            # low shelf: full gain well below the corner
         + hi_g / (1 + np.exp(-k * (lf - math.log2(hi_f))))         # high shelf: full gain well above the corner
         + pk_g * np.exp(-0.5 * ((lf - math.log2(pk_f)) / (pk_bw / 2)) ** 2))
    return g, dict(low_shelf_hz=round(lo_f), low_shelf_db=round(lo_g, 2), high_shelf_hz=round(hi_f),
                   high_shelf_db=round(hi_g, 2), peak_hz=round(pk_f), peak_db=round(pk_g, 2),
                   peak_bw_oct=round(pk_bw, 2))


def synthetic_noise(kind: str, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / sr
    white = rng.standard_normal(n)
    if kind == "synth_white":
        y = white
    elif kind == "synth_pink":
        spec = np.fft.rfft(white)
        f = np.fft.rfftfreq(n, 1 / sr)
        spec[1:] /= np.sqrt(f[1:])
        spec[0] = 0
        y = np.fft.irfft(spec, n)
    elif kind == "synth_hum":
        f0 = rng.choice([50.0, 60.0])
        y = sum(rng.uniform(0.2, 1.0) / h * np.sin(2 * np.pi * h * f0 * t + rng.uniform(0, 2 * np.pi))
                for h in range(1, 9)) + 0.02 * white
    elif kind == "synth_fan":
        sos = butter(2, rng.uniform(200, 800), "lowpass", fs=sr, output="sos")
        y = sosfilt(sos, white) * (1 + 0.2 * np.sin(2 * np.pi * rng.uniform(5, 30) * t))
    elif kind == "synth_clicks":
        y = 0.02 * white                                    # faint floor so floor_db is defined
        for _ in range(rng.poisson(rng.uniform(2, 10) * n / sr)):
            i = int(rng.integers(0, n))
            L = int(rng.uniform(0.002, 0.005) * sr)
            seg = rng.standard_normal(min(L, n - i)) * np.exp(-np.arange(min(L, n - i)) / (0.3 * L))
            y[i:i + len(seg)] += rng.uniform(0.3, 1.0) * seg
    else:
        raise ValueError(kind)
    rms = np.sqrt(np.mean(y ** 2)) + 1e-12
    return (y / rms).astype(np.float64)


# ----------------------------------------------------------------------------
# Augmenter
# ----------------------------------------------------------------------------
class DeploymentAugmenter:
    """augment(x, rng, meta) -> (y, applied): x float32 [T] at `sample_rate`,
    meta needs window_type, parent_source and file_id. `applied` records every
    transform that ran and its sampled parameters (for traceability)."""

    def __init__(self, sample_rate: int, noise_bank_waves: dict, config_path=AUG_CONFIG, rir_bank=None):
        cfg = json.loads(Path(config_path).read_text())
        self.sr = sample_rate
        self.config_version = cfg.get("version")
        self.T = {t["name"]: t for t in cfg["enabled_transforms"]}
        self.order = [n for n in cfg["composition_rules"]["order"] if n != "preprocessing"]
        self.bank = noise_bank_waves                    # window_id -> float64 waveform at sample_rate
        self.bank_ids = sorted(noise_bank_waves)
        if not self.bank_ids:
            raise ValueError("empty real noise bank")
        # Draw a recording uniformly, then a window inside it, so long recordings do not dominate the noise.
        by_file = {}
        for wid in self.bank_ids:
            by_file.setdefault(wid.split("@")[0], []).append(wid)
        self.bank_files = [by_file[f] for f in sorted(by_file)]
        self.rirs = rir_bank or []                      # list of dict(id, wave, rt60_s)
        mix = self.T["background_noise_mixing"]["parameters"]["noise_sources"]
        self.p_real_noise = mix["real_device_background"]["weight"] / (
            mix["real_device_background"]["weight"] + mix["synthetic"]["weight"])
        self.synthetic_kinds = tuple(mix["synthetic"]["sources"])
        assert set(self.synthetic_kinds) <= set(SYNTHETIC_NOISES), self.synthetic_kinds

    # -- helpers ---------------------------------------------------------------
    def _applies(self, name, window_type) -> bool:
        a = self.T[name]["apply_to"]
        return "all categories" in a or window_type in a

    def _bank_noise(self, n, rng):
        ids = self.bank_files[int(rng.integers(len(self.bank_files)))]
        wid = ids[int(rng.integers(len(ids)))]
        w = self.bank[wid]
        if len(w) < n:
            w = np.tile(w, int(math.ceil(n / len(w))))
        off = int(rng.integers(0, len(w) - n + 1))
        return w[off:off + n], wid

    @staticmethod
    def _uniform(rng, spec):
        return float(rng.uniform(spec["min"], spec["max"]))

    # -- main ------------------------------------------------------------------
    def __call__(self, x, rng: np.random.Generator, meta: dict):
        y = np.asarray(x, dtype=np.float64).copy()
        sr, wt, applied = self.sr, meta["window_type"], {}
        n = len(y)
        is_real_rpi = meta.get("parent_source") == "manual_recording" and "RPI" in str(meta.get("file_id", ""))
        had_digital_silence = longest_zero_run(y) >= int(FRAME_SEC * sr)
        rt60 = None
        for name in self.order:
            t = self.T[name]
            if not self._applies(name, wt):
                continue
            if name == "rir_reverberation" and (is_real_rpi or not self.rirs):
                continue
            if name == "device_noise_floor" and not had_digital_silence:
                continue
            if rng.random() >= t["probability"]:
                continue
            p = t["parameters"]

            if name == "random_mic_coloring":
                if n == 0:
                    continue
                g_db, params = coloring_gain_db(np.fft.rfftfreq(n, 1 / sr), p, rng)
                y = np.fft.irfft(np.fft.rfft(y) * 10 ** (g_db / 20), n)   # zero-phase: no time shift
                applied["random_mic_coloring"] = params

            elif name == "device_level_gain":
                pk = peak_db(y, sr)
                if pk <= SILENT_DB:
                    continue
                target = self._uniform(rng, p["target_peak_db"])
                g = 10 ** ((target - pk) / 20)
                m = np.abs(y).max()
                if m * g > 0.999:                           # never above full scale
                    g = 0.999 / m
                y *= g
                applied["device_level_gain"] = dict(target_peak_db=round(target, 2), gain_db=round(20 * math.log10(g), 2))

            elif name == "rir_reverberation":
                r = self.rirs[int(rng.integers(len(self.rirs)))]
                wet = fftconvolve(y, r["wave"])
                d = int(np.argmax(np.abs(r["wave"])))       # keep the direct path aligned
                y = wet[d:d + n]
                rt60 = r.get("rt60_s")
                applied["rir_reverberation"] = dict(rir_id=r["id"], rt60_s=rt60)

            elif name == "background_noise_mixing":
                spec = p["target_peak_over_floor_db"]
                target = self._uniform(rng, spec)
                if rt60 is not None and rt60 > 0.45 and target < 10:    # disallowed combination
                    target = float(rng.uniform(10, spec["max"]))
                target = max(target, spec["hard_min"])
                pk = peak_db(y, sr)
                if pk <= SILENT_DB:
                    continue
                if p.get("only_if_cleaner_than_target") and pk - floor_db(y, sr) <= target:
                    # already at/below the target signal-over-background (e.g. real device audio)
                    applied["background_noise_mixing"] = dict(skipped="already at or below target",
                                                              target_peak_over_floor_db=round(target, 2))
                    continue
                if rng.random() < self.p_real_noise:
                    noise, src = self._bank_noise(n, rng)
                    src = f"real:{src}"
                else:
                    kind = self.synthetic_kinds[int(rng.integers(len(self.synthetic_kinds)))]
                    noise, src = synthetic_noise(kind, n, sr, rng), kind
                # Start with the noise floor at (speech peak - target), then correct in closed loop: with
                # non-stationary noise (clicks, real background with transients) the mixture can land on either
                # side of the target, because noise peaks raise the mixture peak and noise events in the speech's
                # quiet frames raise the mixture floor. hard_min is enforced on the achieved value.
                gdb = (pk - target) - floor_db(noise, sr)
                for _ in range(4):
                    mix = y + noise * 10 ** (gdb / 20)
                    achieved = peak_db(mix, sr) - floor_db(mix, sr)
                    if abs(achieved - target) < 0.5:
                        break
                    gdb += achieved - target                # noisier than target -> quieter noise, and vice versa
                while achieved < spec["hard_min"]:          # never below the verified-audible minimum
                    gdb -= spec["hard_min"] - achieved + 0.5
                    mix = y + noise * 10 ** (gdb / 20)
                    achieved = peak_db(mix, sr) - floor_db(mix, sr)
                y = mix
                applied["background_noise_mixing"] = dict(
                    source=src, target_peak_over_floor_db=round(target, 2), achieved_peak_over_floor_db=round(achieved, 2))

            elif name == "device_noise_floor":
                target = self._uniform(rng, p["target_floor_db"])
                if "background_noise_mixing" in applied and floor_db(y, sr) >= target:
                    applied["device_noise_floor"] = dict(skipped="mixing floor already >= target")
                    continue
                noise, wid = self._bank_noise(n, rng)
                y = y + noise * 10 ** ((target - floor_db(noise, sr)) / 20)
                applied["device_noise_floor"] = dict(target_floor_db=round(target, 2), source=f"real:{wid}")

        m = np.abs(y).max() if n else 0.0
        if m > 0.999:                                       # mixing pushed a loud example over full scale
            y *= 0.999 / m
            applied["rescaled_to_full_scale"] = True
        return y.astype(np.float32), applied


def load_noise_bank(load_window, bank_csv=NOISE_BANK) -> tuple:
    """Rows of real_noise_bank.csv and their waveforms, via load_window(row) -> float64 array."""
    rows = pd.read_csv(bank_csv)
    return rows, {r.window_id: np.asarray(load_window(r), dtype=np.float64) for r in rows.itertuples(index=False)}
