#!/usr/bin/env python3
"""
Unit / sanity tests for the shared preprocessing.

  python test_preprocessing.py        (or: pytest test_preprocessing.py)

Synthetic-signal tests need nothing else. Tests marked "real data" read audio
from data/ and the frozen normalization_stats.json (run
compute_normalization_stats.py first).
"""
from pathlib import Path
import json, math, sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wakeword_preprocessing as wp
import compute_normalization_stats as cns

CFG = wp.PreprocessConfig.from_json()
ZERO_PAD = wp.PreprocessConfig.from_dict({**CFG.to_dict(), "pad_mode": "zeros"})
SR, N = CFG.sample_rate, CFG.num_samples
AUG_CONFIG = HERE / "../deployment_driven_augmentation_strategy/augmentation_config.json"


def tone(sec, freq=1000.0, amp=0.1, sr=SR):
    t = torch.arange(int(round(sec * sr))) / sr
    return (amp * torch.sin(2 * math.pi * freq * t)).to(torch.float32)


def windows():
    return pd.read_csv(wp.DEFAULT_WINDOWS)


def preprocessor():
    return wp.WakewordPreprocessor(CFG)


# --------------------------------------------------------------------------- config
def test_config_values_and_derived_shapes():
    assert (SR, N) == (16000, 24000)
    assert CFG.n_frames == (N - CFG.n_fft) // CFG.hop_length + 1 == 147
    assert CFG.feature_shape == (1, CFG.n_mels, 147)
    assert CFG.stream_hop_samples == 1600


def test_config_roundtrip_and_hash():
    again = wp.PreprocessConfig.from_dict(CFG.to_dict())
    assert again == CFG and again.feature_hash() == CFG.feature_hash()
    # Streaming hop does not change window features, so it must not invalidate the statistics.
    assert wp.PreprocessConfig.from_dict({**CFG.to_dict(), "stream_hop_sec": 0.05}).feature_hash() == CFG.feature_hash()
    assert wp.PreprocessConfig.from_dict({**CFG.to_dict(), "n_mels": 64}).feature_hash() != CFG.feature_hash()


def test_config_rejects_invalid_values():
    for bad in [dict(win_length=1024), dict(f_max=9000.0), dict(pad_mode="edge"), dict(window_sec=1.50001),
                dict(nonexistent_key=1)]:
        try:
            wp.PreprocessConfig.from_dict({**CFG.to_dict(), **bad})
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid config {bad}")


# --------------------------------------------------------------------------- loading / sample rates
def test_load_resamples_every_source_rate_and_keeps_amplitude():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for sr in (16000, 22050, 24000, 48000):
            p = Path(d) / f"t{sr}.wav"
            sf.write(p, np.stack([tone(1.0, sr=sr).numpy()] * 2, axis=1), sr, subtype="FLOAT")  # stereo
            w = wp.load_waveform(p, CFG)
            assert w.dtype == torch.float32 and w.dim() == 1
            assert w.numel() == SR, (sr, w.numel())
            mid = w[2000:-2000]
            rms = float(mid.pow(2).mean().sqrt())
            assert abs(rms - 0.1 / math.sqrt(2)) < 1e-3, (sr, rms)   # no level normalisation


def test_real_files_decode_at_target_rate():
    """real data: one file per on-disk rate, including an MP3-in-.wav edge-tts positive."""
    for rel, sr_on_disk in [("data/positives/Manual-Fil-RPI10.wav", 48000),
                            ("data/positives/fil-PH-AngeloNeural_spd1.0_pit0.wav", None)]:
        path = wp.PROJECT_ROOT / rel
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if sr_on_disk:
            assert sr == sr_on_disk
        w = wp.load_waveform(path, CFG)
        assert w.numel() == math.ceil(len(y) * SR / sr), (rel, w.numel())
        assert torch.isfinite(w).all()


# --------------------------------------------------------------------------- framing
def test_frame_at_pads_and_never_truncates():
    w = tone(1.0)
    f = wp.frame_at(w, -4000, ZERO_PAD)
    assert f.numel() == N
    assert torch.all(f[:4000] == 0) and torch.equal(f[4000:4000 + SR], w) and torch.all(f[4000 + SR:] == 0)
    f2 = wp.frame_at(w, 100, ZERO_PAD)
    assert torch.equal(f2[:SR - 100], w[100:]) and torch.all(f2[SR - 100:] == 0)
    assert torch.all(wp.frame_at(w, 10 * SR, ZERO_PAD) == 0)   # entirely outside -> all padding
    # Same geometry with the configured fill: the real audio is untouched.
    assert torch.equal(wp.frame_at(w, -4000, CFG)[4000:4000 + SR], w)


def test_noise_padding_is_deterministic_and_only_in_pads():
    cfg = CFG
    assert cfg.pad_mode == "noise"
    w = tone(1.0)
    a, b = wp.frame_at(w, -4000, cfg, "win-1"), wp.frame_at(w, -4000, cfg, "win-1")
    assert torch.equal(a, b)
    assert not torch.equal(a, wp.frame_at(w, -4000, cfg, "win-2"))
    assert torch.equal(a[4000:4000 + SR], w)
    rms = float(a[:4000].pow(2).mean().sqrt())
    assert abs(20 * math.log10(rms) - cfg.pad_noise_dbfs) < 1.0
    ext = wp.FeatureExtractor(cfg)
    assert ext(wp.frame_at(torch.zeros(0), 0, cfg, "w")).min() > -90   # no digital-silence frames


def test_pad_noise_matches_measured_device_floor():
    """The fill's floor_db on the deployment metric equals the measured median RPI floor (+-0.5 dB)."""
    from scipy.signal import butter, sosfilt
    target = json.loads(AUG_CONFIG.read_text())["evidence_scope"]["measured_values"]["rpi_positive_floor_db"]["median"]
    y = wp.frame_at(torch.zeros(0), 0, CFG, "calibration").numpy().astype(np.float64)
    yb = sosfilt(butter(4, [150.0, 4000.0], "bandpass", fs=SR, output="sos"), y)
    n = int(0.02 * SR)
    frames = 20 * np.log10(np.sqrt((yb[:len(yb) // n * n].reshape(-1, n) ** 2).mean(1)) + 1e-9)
    floor = np.percentile(frames, 20)
    assert abs(floor - target) < 0.5, (floor, target)


def test_frame_clip_short_centered_long_policy():
    f = wp.frame_clip(tone(1.0), ZERO_PAD)
    assert f.shape == (1, N) and torch.all(f[0, :4000] == 0) and torch.all(f[0, -4000:] == 0)
    try:
        wp.frame_clip(tone(2.0), CFG)
        raise AssertionError("long clip was framed with long_clip_policy=error")
    except ValueError:
        pass
    cfg = wp.PreprocessConfig.from_dict({**CFG.to_dict(), "long_clip_policy": "sliding"})
    w = tone(2.05)
    fs = wp.frame_clip(w, cfg)
    starts = wp.sliding_starts(w.numel(), cfg) + [w.numel() - N]
    assert fs.shape == (len(starts), N)
    assert torch.equal(fs[-1], w[-N:])     # tail covered by an end-aligned window


def test_dataset_framing_matches_windows_csv_padding():
    """real data: pad amounts implied by start_sec equal windows.csv pad_left/right_sec."""
    w = windows()
    sample = pd.concat([w[w.pad_left_sec > 0].head(20), w[w.pad_right_sec > 0].head(20),
                        w[(w.pad_left_sec == 0) & (w.pad_right_sec == 0)].head(10)])
    for r in sample.itertuples(index=False):
        wave = wp.load_waveform(wp.PROJECT_ROOT / r.filepath, CFG)
        start = int(round(r.start_sec * SR))
        pad_l = max(0, -start)
        pad_r = max(0, start + N - wave.numel())
        assert abs(pad_l / SR - r.pad_left_sec) <= 1.5 / SR + 1e-4, (r.window_id, pad_l / SR, r.pad_left_sec)
        assert abs(pad_r / SR - r.pad_right_sec) <= 1.5 / SR + 1e-3, (r.window_id, pad_r / SR, r.pad_right_sec)
        f = wp.frame_at(wave, start, ZERO_PAD)
        assert f.numel() == N
        if pad_l:
            assert torch.all(f[:pad_l] == 0)
        if pad_r:
            assert torch.all(f[N - pad_r:] == 0)
        body = slice(pad_l, N - pad_r)
        assert torch.equal(wp.frame_at(wave, start, CFG, r.window_id)[body], f[body])   # fill only touches pads


# --------------------------------------------------------------------------- features
def test_feature_shapes_single_and_batch():
    pre = preprocessor()
    assert pre(tone(1.5)).shape == CFG.feature_shape
    assert pre(torch.stack([tone(1.5)] * 3)).shape == (3, *CFG.feature_shape)
    try:
        pre(tone(1.0))
        raise AssertionError("accepted a window of the wrong length")
    except ValueError:
        pass


def test_db_floor_and_no_top_db():
    ext = wp.FeatureExtractor(CFG)
    silent = ext(torch.zeros(N))
    assert torch.allclose(silent, torch.full_like(silent, -100.0))
    # A loud tone must not raise the floor elsewhere (top_db would clamp it to max - top_db).
    x = torch.zeros(N)
    x[:8000] = tone(0.5, amp=0.9)
    assert ext(x)[:, -10:].max() <= -99.9


def test_amplitude_is_preserved_through_normalization():
    pre = preprocessor()
    quiet = tone(1.5, freq=700.0, amp=0.01)
    a, b = pre(quiet), pre(quiet * 2.0)   # +6.02 dB
    shift = (b - a) * pre.normalizer.std.unsqueeze(0)      # back to dB, per bin
    band = pre.extract(quiet * 2.0) > -60                    # bins that carry the tone
    assert torch.allclose(shift[band.unsqueeze(0)], torch.full_like(shift[band.unsqueeze(0)], 20 * math.log10(2)),
                          atol=1e-3)


def test_validation_and_test_preprocessing_is_deterministic():
    """real data: the same val/test windows give bit-identical features across datasets and calls."""
    pre = preprocessor()
    for split in ("validation", "test"):
        d1, d2 = wp.WindowDataset(pre, split), wp.WindowDataset(pre, split)
        pos = d1.rows.index[d1.rows.label == "positive"][:5].tolist()
        idx = pos + [0, len(d1) - 1]
        for i in idx:
            x1, y1, id1 = d1[i]
            x2, y2, id2 = d2[i]
            assert id1 == id2 and torch.equal(y1, y2)
            assert torch.equal(x1, x2), (split, id1)
            assert torch.equal(x1, d1[i][0])
            assert x1.shape == CFG.feature_shape and torch.isfinite(x1).all()


# --------------------------------------------------------------------------- augmentation
class CountingGain:
    def __init__(self):
        self.calls = 0

    def __call__(self, wave, sr, generator, meta):
        assert sr == SR
        self.calls += 1
        return wave * (0.5 + torch.rand(1, generator=generator).item())


def test_augmentation_only_when_explicitly_enabled_for_train():
    pre = preprocessor()
    aug = CountingGain()
    plain = wp.WindowDataset(pre, "train")
    i = int(plain.rows.index[plain.rows.label == "positive"][0])   # real speech, not silence
    plain[i]
    assert aug.calls == 0
    augd = wp.WindowDataset(pre, "train", augment=aug, seed=1)
    x1 = augd[i][0]
    assert aug.calls == 1 and not torch.equal(x1, plain[i][0])
    assert torch.equal(x1, wp.WindowDataset(pre, "train", augment=CountingGain(), seed=1)[i][0])  # reproducible
    augd.set_epoch(1)
    assert not torch.equal(x1, augd[i][0])                                                      # new draw per epoch
    for split in ("validation", "test", "streaming_eval_holdout", ["train", "validation"]):
        try:
            wp.WindowDataset(pre, split, augment=aug)
            raise AssertionError(f"augmentation accepted for {split}")
        except ValueError:
            pass


# --------------------------------------------------------------------------- normalization statistics
def test_frozen_stats_use_train_windows_only():
    """real data: the artifact matches exactly the train windows of the current windows.csv."""
    s = json.loads(wp.DEFAULT_STATS.read_text())
    w = windows()
    train = cns.select_train_windows(w)
    assert s["splits_used"] == ["train"]
    assert s["windows_csv_sha256"] == cns.sha256_file(wp.DEFAULT_WINDOWS), "stats are stale: rerun compute_normalization_stats.py"
    assert s["n_windows"] == len(train) == int(((w.split == "train") & ~w.is_eval_only).sum())
    assert s["train_window_ids_sha256"] == cns.window_ids_hash(train.window_id)
    non_train = set(w.loc[w.split != "train", "window_id"])
    assert not non_train & set(train.window_id)
    assert s["n_frames"] == len(train) * CFG.n_frames
    assert s["config_feature_hash"] == CFG.feature_hash()
    assert len(s["mean"]) == len(s["std"]) == CFG.n_mels and min(s["std"]) > 0


def test_accumulate_refuses_non_train_windows():
    w = windows()
    mixed = pd.concat([w[w.split == "train"].head(2), w[w.split == "validation"].head(1)])
    try:
        cns.accumulate(mixed, CFG)
        raise AssertionError("statistics accepted a validation window")
    except ValueError:
        pass


def test_accumulate_matches_direct_computation():
    """real data: streaming float64 accumulation equals mean/std of the stacked features."""
    rows = cns.select_train_windows(windows()).head(12)
    mean, std, n, _ = cns.accumulate(rows, CFG, log_every=0)
    ext = wp.FeatureExtractor(CFG)
    feats = []
    for r in rows.itertuples(index=False):
        wave = wp.load_waveform(wp.PROJECT_ROOT / r.filepath, CFG)
        feats.append(ext(wp.frame_at(wave, int(round(r.start_sec * SR)), CFG, r.window_id)).double())
    x = torch.stack(feats).permute(1, 0, 2).reshape(CFG.n_mels, -1)
    assert n == x.shape[1]
    assert torch.allclose(mean, x.mean(dim=1), atol=1e-6)
    assert torch.allclose(std, x.std(dim=1, unbiased=False), atol=1e-6)


def test_normalizer_rejects_mismatched_or_non_train_stats():
    import tempfile
    s = json.loads(wp.DEFAULT_STATS.read_text())
    with tempfile.TemporaryDirectory() as d:
        for bad in [{**s, "config_feature_hash": "0" * 16}, {**s, "splits_used": ["train", "validation"]}]:
            p = Path(d) / "s.json"
            p.write_text(json.dumps(bad))
            try:
                wp.Normalizer.from_stats(p, CFG)
                raise AssertionError("accepted bad statistics")
            except ValueError:
                pass


def test_train_features_are_standardized():
    """real data: over a sample of train windows the normalized features are ~zero-mean, unit-std per bin."""
    pre = preprocessor()
    d = wp.WindowDataset(pre, "train")
    idx = np.linspace(0, len(d) - 1, 300).round().astype(int)
    x = torch.stack([d[int(i)][0] for i in idx]).squeeze(1)      # [n, n_mels, frames]
    m = x.mean(dim=(0, 2))
    s = x.std(dim=(0, 2))
    assert m.abs().max() < 0.5 and (s - 1).abs().max() < 0.5, (m.abs().max(), s.min(), s.max())


# --------------------------------------------------------------------------- streaming
def _stream(pre, wave, sr, chunk_sizes):
    sp = wp.StreamingPreprocessor(pre, input_sample_rate=sr)
    starts, feats, i, k = [], [], 0, 0
    while i < wave.numel():
        n = chunk_sizes[k % len(chunk_sizes)]
        s, f = sp.push(wave[i:i + n])
        starts += s
        feats.append(f)
        i, k = i + n, k + 1
    return starts, torch.cat(feats)


def test_streaming_windows_equal_isolated_windows_at_16k():
    pre = preprocessor()
    g = torch.Generator().manual_seed(0)
    wave = torch.randn(int(4.3 * SR), generator=g) * 0.05
    starts, feats = _stream(pre, wave, SR, [317, 1600, 5, 4001])
    expected = wp.sliding_starts(wave.numel(), CFG)
    assert starts == expected and feats.shape == (len(expected), *CFG.feature_shape)
    iso = pre(torch.stack([wp.frame_at(wave, s, CFG) for s in expected]))
    assert torch.equal(feats, iso)                                   # bit-identical
    for j, s in enumerate(expected[:3]):
        assert torch.equal(feats[j], pre(wave[s:s + N]))             # one isolated window at a time


def test_streaming_from_48k_matches_offline_resampling():
    """real data: a 48 kHz RPI recording streamed in odd-sized chunks matches offline load + framing."""
    pre = preprocessor()
    y, sr = sf.read(str(wp.PROJECT_ROOT / "data/positives/Manual-Fil-RPI23.wav"), dtype="float32", always_2d=True)
    raw = torch.from_numpy(np.ascontiguousarray(y.mean(axis=1)))
    assert sr == 48000
    starts, feats = _stream(pre, raw, sr, [4801, 960, 12345])
    offline = wp.load_waveform(wp.PROJECT_ROOT / "data/positives/Manual-Fil-RPI23.wav", CFG)
    iso = pre(torch.stack([wp.frame_at(offline, s, CFG) for s in starts]))
    assert len(starts) >= 100 and starts == [k * CFG.stream_hop_samples for k in range(len(starts))]
    err = (feats - iso).abs().max().item()
    print(f"      48 kHz stream vs offline: max |diff| = {err:.2e} (normalized units)")
    assert err < 1e-3, err


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except Exception as e:
            failed += 1; print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
