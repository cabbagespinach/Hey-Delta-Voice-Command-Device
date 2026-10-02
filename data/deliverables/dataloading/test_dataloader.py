#!/usr/bin/env python3
"""
Dataset / DataLoader sanity tests.

  python test_dataloader.py        (or: pytest test_dataloader.py)

Reads the real deliverables (windows.csv, split files, preprocessing stats,
noise bank). Datasets are built once and shared between tests.
"""
from pathlib import Path
import copy, functools, json, math, sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wakeword_data as wd
import augmentation as aug
import reproducibility as rep


@functools.lru_cache(maxsize=None)
def built():
    return wd.build_datasets(splits=["train", "validation", "test"])


def datasets():
    return built()[1]


def cfg():
    return built()[0]


@functools.lru_cache(maxsize=None)
def windows():
    return wd.load_windows(cfg())


def small_loader(split, num_workers=0, batch_size=16, augment=None):
    return wd.build_dataloaders(splits=[split], batch_size=batch_size, num_workers=num_workers,
                                augment=augment)[split]


# --------------------------------------------------------------------------- batches, shapes, labels
def test_batches_load_with_valid_shapes_and_labels():
    ds = datasets()
    shape = ds["train"].cfg.feature_shape
    for split in ("train", "validation", "test"):
        loader = torch.utils.data.DataLoader(ds[split], batch_size=16, collate_fn=wd.collate,
                                             sampler=wd.BucketQuotaSampler(ds[split].rows, cfg()["sampling"], 1)
                                             if split == "train" else None)
        b = next(iter(loader))
        assert b["features"].shape == (16, *shape) and b["features"].dtype == torch.float32
        assert torch.isfinite(b["features"]).all()
        assert b["labels"].shape == (16,) and set(b["labels"].tolist()) <= {0.0, 1.0}
        for y, m in zip(b["labels"].tolist(), b["meta"]):
            assert m["split"] == split
            assert y == m["label_int"] == (m["label"] == "positive") == (m["window_type"] == "positive_wakeword")


def test_factory_honours_batch_size_and_workers():
    loader = small_loader("validation", num_workers=2, batch_size=8)
    assert loader.batch_size == 8 and loader.num_workers == 2
    b = next(iter(loader))
    assert b["features"].shape[0] == 8


# --------------------------------------------------------------------------- traceability
def test_metadata_traceable_to_source():
    """Every metadata record matches windows.csv, the file exists, its source group is frozen in the
    same split, and the example can be regenerated from (filepath, start_sec, window_id) alone."""
    import wakeword_preprocessing as wp
    ds = datasets()
    w = windows().set_index("window_id")
    guard = wd.SplitGuard(cfg())
    pre = ds["validation"].pre
    for split in ("train", "validation", "test"):
        d = ds[split]
        for key in ([(i, 3, 100 + i) for i in range(0, len(d), len(d) // 6)] if split == "train"
                    else list(range(0, len(d), len(d) // 6))):
            x, y, m = d[key]
            r = w.loc[m["window_id"]]
            for col in ("filepath", "start_sec", "end_sec", "label", "window_type", "source_group", "split",
                        "recording_id", "file_id"):
                assert m[col] == r[col], (m["window_id"], col)
            assert (wp.PROJECT_ROOT / m["filepath"]).exists()
            assert guard.frozen[m["source_group"]] == split
            if split == "train":                      # regenerate the augmented example from its key
                assert torch.equal(x, d[(m["index"], m["epoch"], m["draw"])][0])
                assert json.loads(m["augmentation"] or "{}") == json.loads(d[(m["index"], m["epoch"], m["draw"])][2]["augmentation"] or "{}")
            else:                                     # regenerate from the source file alone
                with rep.single_thread():
                    wave = wp.load_waveform(wp.PROJECT_ROOT / m["filepath"], pre.cfg)
                    frame = wp.frame_at(wave, int(round(m["start_sec"] * pre.cfg.sample_rate)), pre.cfg, m["window_id"])
                    assert torch.equal(x, pre(frame))


def test_metadata_fields_present():
    m = datasets()["train"].meta
    required = {"label", "window_type", "category", "source", "recording_id", "source_group", "speaker_id",
                "session_id", "split", "filepath", "start_sec"}
    assert required <= set(m[0])
    s = pd.DataFrame(m)
    assert s.filepath.notna().all() and s.source_group.notna().all() and s.category.notna().all()
    # Every synthetic *speech* window has a speaker (TTS voice). Procedural noise clips (synth_*) and
    # manual recordings have none: no speaker exists / none was recorded (documented).
    speech = s[s.source.str.startswith("tts_") | (s.source == "positive_clip_truncated")]
    assert len(speech) and speech.speaker_id.notna().all() and speech.speaker_id_normalized.notna().all()
    assert s.session_id.isna().all()      # no session field exists in any input manifest


# --------------------------------------------------------------------------- augmentation
def test_training_augmentation_is_stochastic_and_reproducible():
    d = datasets()["train"]
    i = int(d.rows.index[d.rows.window_type == "positive_wakeword"][0])
    a1, _, m1 = d[(i, 0, 0)]
    a2, _, m2 = d[(i, 0, 1)]
    a3, _, _ = d[(i, 1, 0)]
    assert torch.equal(a1, d[(i, 0, 0)][0])                          # same key -> same example
    draws = [d[(i, 0, k)] for k in range(12)]
    assert len({x.numpy().tobytes() for x, _, _ in draws}) > 6        # different draws -> different examples
    assert any(m["augmented"] for _, _, m in draws) and not all(m["augmentation"] == draws[0][2]["augmentation"] for _, _, m in draws)
    assert not torch.equal(a1, a3) or not torch.equal(a1, a2)         # epoch or draw changes the example


def test_augmentation_rates_match_configuration():
    d = datasets()["train"]
    T = {t["name"]: t for t in json.loads(Path(cfg()["_inputs"]["augmentation_config"]).read_text())["enabled_transforms"]}
    rows = d.rows.index[d.rows.window_type.isin(["negative_general_speech", "negative_confusable", "positive_wakeword"])]
    idx = rows[np.linspace(0, len(rows) - 1, 300).round().astype(int)]
    applied = [json.loads(d.load((int(i), 0, k))[1]["augmentation"] or "{}") for k, i in enumerate(idx)]
    for name in ("device_level_gain", "background_noise_mixing"):
        rate = np.mean([name in a for a in applied])
        p = T[name]["probability"]
        assert abs(rate - p) < 4 * math.sqrt(p * (1 - p) / len(idx)) + 0.02, (name, rate, p)
    # Reverberation (MIT IR Survey bank): applied at its rate to eligible examples, never to real RPI audio,
    # which already carries the deployment room.
    rpi = np.array([d.rows.loc[i, "parent_source"] == "manual_recording" and "RPI" in d.rows.loc[i, "file_id"]
                    for i in idx])
    elig = np.array([d.rows.loc[i, "window_type"] in T["rir_reverberation"]["apply_to"] for i in idx]) & ~rpi
    rev = np.array(["rir_reverberation" in a for a in applied])
    assert not rev[rpi].any()
    p = T["rir_reverberation"]["probability"]
    rate = rev[elig].mean()
    assert abs(rate - p) < 4 * math.sqrt(p * (1 - p) / elig.sum()) + 0.02, ("rir_reverberation", rate, p)


def test_mic_coloring_is_label_agnostic_and_bounded():
    """random_mic_coloring: same rate for both labels (real RPI included), gain within the configured ranges."""
    d = datasets()["train"]
    T = {t["name"]: t for t in json.loads(Path(cfg()["_inputs"]["augmentation_config"]).read_text())["enabled_transforms"]}
    spec, p = T["random_mic_coloring"]["parameters"], T["random_mic_coloring"]["probability"]
    rates = {}
    for lab in ("positive", "negative"):
        rows = d.rows.index[d.rows.label == lab]
        idx = rows[np.linspace(0, len(rows) - 1, 200).round().astype(int)]
        applied = [json.loads(d.load((int(i), 3, k))[1]["augmentation"] or "{}") for k, i in enumerate(idx)]
        col = [a["random_mic_coloring"] for a in applied if "random_mic_coloring" in a]
        rates[lab] = len(col) / len(idx)
        for c in col:
            assert spec["low_shelf"]["corner_hz"]["min"] - 1 <= c["low_shelf_hz"] <= spec["low_shelf"]["corner_hz"]["max"] + 1
            assert abs(c["low_shelf_db"]) <= spec["low_shelf"]["gain_db"]["max"] and abs(c["peak_db"]) <= spec["peak"]["gain_db"]["max"]
    for lab, r in rates.items():
        assert abs(r - p) < 4 * math.sqrt(p * (1 - p) / 200) + 0.02, (lab, r, p)
    # the transform itself: zero-phase (no delay) and the curve stays within the summed ranges
    rng = np.random.default_rng(0)
    f = np.fft.rfftfreq(24000, 1 / 16000)
    for _ in range(50):
        g, _ = aug.coloring_gain_db(f, spec, rng)
        assert np.abs(g).max() <= spec["low_shelf"]["gain_db"]["max"] + spec["high_shelf"]["gain_db"]["max"] + spec["peak"]["gain_db"]["max"]
    x = np.zeros(24000); x[12000] = 1.0
    g, _ = aug.coloring_gain_db(f, spec, np.random.default_rng(1))
    y = np.fft.irfft(np.fft.rfft(x) * 10 ** (g / 20), 24000)
    assert int(np.argmax(np.abs(y))) == 12000


def test_augmentation_disabled_gives_fixed_preprocessing():
    cfg_, ds = wd.build_datasets(splits=["train", "validation"], augment=False)
    d, v = ds["train"], ds["validation"]
    assert d.augmenter is None
    assert torch.equal(d[(5, 0, 0)][0], d[(5, 7, 99)][0])
    assert v.augmenter is None and datasets()["validation"].augmenter is None and datasets()["test"].augmenter is None


def test_augmentation_refused_outside_train():
    d = datasets()
    for split in ("validation", "test", "streaming_eval_holdout"):
        try:
            wd.WakewordWindowDataset(split, windows(), wd.SplitGuard(cfg()), d["train"].pre, d["train"].augmenter)
            raise AssertionError(f"augmenter accepted for {split}")
        except ValueError:
            pass


# --------------------------------------------------------------------------- determinism
def test_validation_and_test_are_deterministic():
    for split in ("validation", "test"):
        runs = []
        for nw in (0, 0, 2):
            loader = small_loader(split, num_workers=nw, batch_size=32, augment=False)
            it = iter(loader)
            runs.append([next(it) for _ in range(3)])
        for other in runs[1:]:
            for b1, b2 in zip(runs[0], other):
                assert [m["window_id"] for m in b1["meta"]] == [m["window_id"] for m in b2["meta"]]
                assert torch.equal(b1["labels"], b2["labels"])
                diff = (b1["features"] - b2["features"]).abs().max().item()
                assert diff == 0.0, (split, diff)


def test_training_batches_independent_of_worker_count():
    b = []
    for nw in (0, 2):
        loader = small_loader("train", num_workers=nw, batch_size=16)
        loader.sampler.set_epoch(3)
        b.append(next(iter(loader)))
    assert [m["window_id"] for m in b[0]["meta"]] == [m["window_id"] for m in b[1]["meta"]]
    assert [m["augmentation"] for m in b[0]["meta"]] == [m["augmentation"] for m in b[1]["meta"]]
    assert (b[0]["features"] - b[1]["features"]).abs().max().item() == 0.0


# --------------------------------------------------------------------------- split boundaries
def _expect_boundary_error(fn, what):
    try:
        fn()
    except wd.SplitBoundaryError:
        return
    raise AssertionError(f"not rejected: {what}")


def test_split_guard_accepts_real_data_and_rejects_tampering():
    guard = wd.SplitGuard(cfg())
    w = windows()
    guard.check_windows(w)
    t = w.copy()
    i = t.index[t.split == "train"][0]
    t.loc[i, "split"] = "validation"                                   # one window moved across splits
    _expect_boundary_error(lambda: guard.check_windows(t), "window moved to validation")
    t = w.copy()
    g = t.loc[t.split == "test", "source_group"].iloc[0]
    j = t.index[t.split == "train"][0]
    t.loc[j, "source_group"] = g                                       # train window claims a test source group
    _expect_boundary_error(lambda: guard.check_windows(t), "test source group in train")
    t = w.copy()
    k = t.index[t.is_eval_only.astype(bool)][0]
    t.loc[k, "split"] = "train"                                        # streaming-eval window leaks into train
    _expect_boundary_error(lambda: guard.check_windows(t), "streaming eval window in train")
    rows = w[w.split == "train"].head(5).copy()
    rows.iloc[0, rows.columns.get_loc("split")] = "validation"
    _expect_boundary_error(lambda: guard.check_rows(rows, "train"), "mixed rows in a train dataset")


def test_noise_bank_is_train_only_and_checked():
    guard = wd.SplitGuard(cfg())
    bank = pd.read_csv(cfg()["_inputs"]["real_noise_bank"])
    guard.check_noise_bank(bank)
    val_file = windows().loc[windows().split == "validation", "filepath"].iloc[0]
    bad = pd.concat([bank, pd.DataFrame([dict(window_id="x", file_id="x", filepath=val_file, split="train",
                                              src_start_sec=0.0, src_end_sec=1.5)])])
    _expect_boundary_error(lambda: guard.check_noise_bank(bad), "validation audio in the noise bank")


def test_no_source_group_or_file_in_two_loaded_splits():
    ds = datasets()
    g = {s: set(ds[s].rows.source_group) for s in ds}
    f = {s: set(ds[s].rows.filepath) for s in ds}
    for a in ds:
        for b in ds:
            if a < b:
                assert not g[a] & g[b] and not f[a] & f[b], (a, b)


# --------------------------------------------------------------------------- sampling
def test_sampler_quotas_reproducibility_and_scope():
    d = datasets()["train"]
    s = wd.BucketQuotaSampler(d.rows, cfg()["sampling"], seed=11)
    idx = s.epoch_indices(0)
    assert len(idx) == len(s) == len(d.rows)
    counts = pd.Series(s.bucket_of[idx]).value_counts()
    for name, b in s.buckets.items():
        assert counts.get(name, 0) == b["count"], name
        assert abs(b["count"] / len(s) - b["share"]) < 1e-3
    assert np.array_equal(idx, s.epoch_indices(0))                       # reproducible
    assert not np.array_equal(idx, s.epoch_indices(1))                   # new epoch, new draw
    assert not np.array_equal(idx, wd.BucketQuotaSampler(d.rows, cfg()["sampling"], seed=12).epoch_indices(0))
    assert idx.min() >= 0 and idx.max() < len(d.rows)
    keys = list(s)
    assert keys[0] == (int(idx[0]), 0, 0) and len({k[2] for k in keys}) == len(keys)


def test_sampler_limits_single_source_group_dominance():
    d = datasets()["train"]
    s = wd.BucketQuotaSampler(d.rows, cfg()["sampling"], seed=0)
    pos = s.buckets["positive_real"]
    groups = d.rows.source_group.to_numpy()[pos["index"]]
    share_windows = pd.Series(groups).value_counts(normalize=True)
    share_sampled = pd.Series(pos["weight"].numpy(), index=groups).groupby(level=0).sum()
    top = share_windows.index[0]
    assert share_sampled[top] < share_windows[top]                        # biggest group is down-weighted
    alpha = cfg()["sampling"]["source_group_exponent"]
    n = pd.Series(groups).value_counts()
    expect = (n ** (1 - alpha)) / (n ** (1 - alpha)).sum()
    assert np.allclose(share_sampled.sort_index(), expect.sort_index())


def test_sampler_source_split_buckets():
    """Source-restricted buckets hold only their source; positives are drawn half real, half synthetic."""
    d = datasets()["train"]
    s = wd.BucketQuotaSampler(d.rows, cfg()["sampling"], seed=0)
    kind = wd.source_kind(d.rows)
    for name, b in cfg()["sampling"]["buckets"].items():
        if "sources" in b:
            assert set(kind[s.buckets[name]["index"]]) == set(b["sources"]), name
    idx = s.epoch_indices(0)
    pos = idx[d.rows.label.to_numpy()[idx] == "positive"]
    assert abs((kind[pos] == "real").mean() - 0.5) < 0.01


def test_sampler_rejects_bad_configuration():
    d = datasets()["train"]
    base = cfg()["sampling"]
    for bad in [dict(buckets={**base["buckets"], "positive_real": {**base["buckets"]["positive_real"], "share": 0.5}}),
                dict(buckets={k: v for k, v in base["buckets"].items() if k != "media"})]:
        try:
            wd.BucketQuotaSampler(d.rows, {**base, **bad}, 0)
            raise AssertionError(f"accepted {list(bad)}")
        except ValueError:
            pass


# --------------------------------------------------------------------------- augmentation units
def _augmenter():
    return datasets()["train"].augmenter


def _speech_like(sec=1.2, sr=16000, amp=0.3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * sr)) / sr
    env = (np.sin(2 * np.pi * 3 * t) > 0).astype(float)
    return (amp * env * np.sin(2 * np.pi * 440 * t) + 1e-3 * rng.standard_normal(len(t))).astype(np.float32)


def test_gain_hits_target_and_never_clips():
    A, sr = _augmenter(), 16000
    x = _speech_like()
    for seed in range(20):
        y, applied = A(x, np.random.default_rng(seed), dict(window_type="negative_silence_noise", parent_source="x", file_id="x"))
        assert np.abs(y).max() <= 0.999 + 1e-6
        if "device_level_gain" in applied:
            assert abs(aug.peak_db(y, sr) - applied["device_level_gain"]["target_peak_db"]) < 0.5
        assert "background_noise_mixing" not in applied                  # never on silence_noise


def test_noise_mixing_reaches_target_level():
    A, sr = _augmenter(), 16000
    x = _speech_like(amp=0.05)
    hits = 0
    for seed in range(40):
        y, applied = A(x, np.random.default_rng(seed), dict(window_type="negative_confusable", parent_source="x", file_id="x"))
        mix = applied.get("background_noise_mixing")
        if mix and "skipped" not in mix:
            hits += 1
            assert mix["target_peak_over_floor_db"] >= 4.5
            err = mix["achieved_peak_over_floor_db"] - mix["target_peak_over_floor_db"]
            assert mix["achieved_peak_over_floor_db"] >= 4.5, mix            # hard_min holds on the achieved value
            if mix["source"] == "synth_clicks" or mix["source"].startswith("real:"):
                # non-stationary: closed-loop correction; may stay cleaner when the noise's own peaks dominate
                assert err > -3.0, mix
            else:
                assert abs(err) < 1.0, mix
    assert hits > 10


def test_device_noise_floor_replaces_digital_silence():
    # Test the transform in isolation: random_mic_coloring (applies to all categories) is switched off, so
    # y - x is exactly the added device noise. Its floor must equal the target (the transform scales the noise
    # by its own whole-window floor). Checking only the zero-filled half-second instead would depend on which
    # real background excerpt was drawn: real background is not stationary.
    A, sr = copy.copy(_augmenter()), 16000
    A.T = dict(A.T)
    A.T["random_mic_coloring"] = dict(A.T["random_mic_coloring"], probability=0.0)
    x = _speech_like()
    x[: sr // 2] = 0.0                                                   # 0.5 s of exact zeros
    for seed in range(10):
        y, applied = A(x, np.random.default_rng(seed), dict(window_type="negative_real_background", parent_source="x", file_id="x"))
        assert "device_noise_floor" in applied and "skipped" not in applied["device_noise_floor"]
        assert "random_mic_coloring" not in applied
        assert aug.longest_zero_run(y) < int(0.02 * sr)
        assert abs(aug.floor_db(y - x, sr) - applied["device_noise_floor"]["target_floor_db"]) < 0.5
        assert "device_level_gain" not in applied and "background_noise_mixing" not in applied   # not in apply_to


def test_augmentation_reaches_padding():
    """Training augmentation runs on the framed window: when noise is mixed into a padded window, the padding
    changes too (otherwise quiet padded flanks become a positive cue). Unaugmented framing is unchanged."""
    d = datasets()["train"]
    rows = d.rows.reset_index(drop=True)
    idx = rows.index[(rows.label == "positive") & (rows.pad_left_sec + rows.pad_right_sec > 0.3)][:40]
    assert len(idx)
    sr = 16000
    reached = mixed = 0
    for k, i in enumerate(idx):
        m = d.meta[int(i)]
        clean = wd.wp.frame_at(d._wave(m["filepath"])[max(0, int(round(m["start_sec"] * sr))):],
                               int(round(m["start_sec"] * sr)) - max(0, int(round(m["start_sec"] * sr))), d.cfg,
                               pad_key=m["window_id"]).numpy()
        y, meta = d.load((int(i), 5, k))
        a = json.loads(meta["augmentation"] or "{}")
        pl = int(m["pad_left_sec"] * sr)
        region = slice(0, pl) if pl > 800 else slice(sr * 3 // 2 - int(m["pad_right_sec"] * sr), sr * 3 // 2)
        if "background_noise_mixing" in a and "skipped" not in a["background_noise_mixing"]:
            mixed += 1
            reached += int(not np.allclose(y.numpy()[region], clean[region]))
    assert mixed > 0 and reached == mixed, (reached, mixed)


def test_synthetic_noises_are_deterministic():
    for kind in aug.SYNTHETIC_NOISES:
        a = aug.synthetic_noise(kind, 8000, 16000, np.random.default_rng(1))
        b = aug.synthetic_noise(kind, 8000, 16000, np.random.default_rng(1))
        assert np.array_equal(a, b) and np.isfinite(a).all() and abs(np.sqrt(np.mean(a ** 2)) - 1) < 1e-6


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
