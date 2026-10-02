#!/usr/bin/env python3
"""
Evaluation pipeline tests (acceptance checks + unit tests of the metric logic).

  python test_evaluation.py        (or: pytest test_evaluation.py)

Needs the frozen streaming set (build_streaming_set.py) and the isolated manifest
(created on the first isolated evaluation).
"""
from pathlib import Path
import functools, json, sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import kstest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import detection as det
import isolated_eval as iso
import streaming_eval as se
import build_streaming_set as bss
import wakeword_data as wd


@functools.lru_cache(maxsize=None)
def stream_set():
    return se.load_set()


class MeanFeatureScorer:
    """Deterministic stand-in scorer: sigmoid of the mean feature value (no model needed)."""

    def __call__(self, x):
        return torch.sigmoid(x.mean(dim=(1, 2, 3))).double().numpy()


# --------------------------------------------------------------------------- acceptance: independence
def test_no_evaluation_source_overlaps_training():
    streams, _ = stream_set()
    se.verify_set(streams)                                   # checksums + no train/val/test overlap
    manifest = pd.read_csv(iso.manifest_path("validation"))
    guard = wd.SplitGuard(wd.load_config())
    assert (manifest.split == "validation").all()
    assert (manifest.source_group.map(guard.frozen) == "validation").all()
    train = pd.read_csv(wd.load_config()["_inputs"]["windows_csv"]).query("split == 'train'")
    assert not set(manifest.source_group) & set(train.source_group)
    assert not set(manifest.filepath) & set(train.filepath)


def test_streaming_independent_of_isolated_validation():
    streams, _ = stream_set()
    se.verify_set(streams, [iso.manifest_path("validation")])
    manifest = pd.read_csv(iso.manifest_path("validation"))
    assert not set(streams.filepath) & set(manifest.filepath)
    assert not set(streams.stream_id) & set(manifest.source_group)


def test_verify_set_rejects_tampering():
    streams, _ = stream_set()
    train_file = pd.read_csv(wd.load_config()["_inputs"]["windows_csv"]).query("split == 'train'").filepath.iloc[0]
    bad = streams.copy()
    bad.loc[bad.index[0], "filepath"] = train_file
    try:
        se.verify_set(bad)
        raise AssertionError("training audio accepted into the streaming set")
    except RuntimeError:
        pass


def test_set_b_text_and_audio_are_new():
    bss.check_text_novelty()                                  # raises on any overlap with training text
    log = json.loads((HERE / "streaming_set/build_log.json").read_text())
    assert log["near_duplicate_xcorr_max"] < 0.9, log["near_duplicate_xcorr_max"]


# --------------------------------------------------------------------------- acceptance: determinism
def test_isolated_preprocessing_is_deterministic():
    d, manifest = iso.load_isolated_set("validation")
    d2, manifest2 = iso.load_isolated_set("validation")
    assert manifest.equals(manifest2)
    for i in (0, len(d) // 2, len(d) - 1):
        assert torch.equal(d[i][0], d2[i][0])
    s1 = iso.score_dataset(d, MeanFeatureScorer(), num_workers=0)
    s2 = iso.score_dataset(d2, MeanFeatureScorer(), num_workers=4)
    assert np.array_equal(s1, s2)


def test_streaming_preprocessing_is_deterministic_and_chunk_invariant():
    streams, _ = stream_set()
    pre = det.wp.WakewordPreprocessor.from_files()
    for sid in (streams[streams.set == "B_composed"].stream_id.iloc[0], streams[streams.set != "B_composed"].stream_id.iloc[0]):
        path = det.ROOT / streams.set_index("stream_id").filepath[sid]
        t1, s1 = det.stream_scores(path, pre, MeanFeatureScorer(), 0.5)
        t2, s2 = det.stream_scores(path, pre, MeanFeatureScorer(), 0.5)
        t3, s3 = det.stream_scores(path, pre, MeanFeatureScorer(), 0.37)
        assert np.array_equal(t1, t2) and np.array_equal(s1, s2)             # repeatable
        n = min(len(s1), len(s3))
        assert np.array_equal(t1[:n], t3[:n]) and np.abs(s1[:n] - s3[:n]).max() < 1e-6   # chunk size irrelevant
        assert np.allclose(np.diff(t1), 0.1)                                 # 100 ms hop


# --------------------------------------------------------------------------- acceptance: time normalization
def test_false_accepts_are_normalized_to_exposure_time():
    # 60 s stream, one wakeword at 20-21 s; hit zone [20, 22.5]; triggers: FA at 5, hit at 21.2, dup at 22.3, FA at 40
    trig = [(5.0, .9), (21.2, .9), (22.3, .9), (40.0, .9)]
    m = det.match(trig, [("w1", 20.0, 21.0)], duration=60.0, window_sec=1.5, before=0.0, after=1.5)
    assert set(m["hits"]) == {"w1"} and abs(m["hits"]["w1"]["latency_sec"] - 0.2) < 1e-9
    assert [t for t, _ in m["false_accepts"]] == [5.0, 40.0] and m["duplicates"] == 1
    assert abs(m["exposure_sec"] - (60.0 - 1.5 - 2.5)) < 1e-9
    rate, lo, hi = det.poisson_rate_ci(2, m["exposure_sec"] / 3600)
    assert abs(rate - 2 / (56 / 3600)) < 1e-6 and lo < rate < hi
    streams, events = stream_set()
    # With a scorer that never fires, every stream contributes its full exposure and zero false accepts.
    scores = {r.stream_id: (np.arange(1.5, r.duration_sec + 1e-9, 0.1), np.zeros(int(round((r.duration_sec - 1.5) / 0.1)) + 1))
              for r in streams.itertuples(index=False)}
    res, ps, dt, fa, _ = se.evaluate(streams, events, scores, threshold=0.5)
    assert res["overall"]["false_accepts"] == 0 and res["overall"]["fa_per_hour"] == 0
    assert res["overall"]["detection_rate"] == 0 and res["overall"]["misses"] == res["overall"]["wakewords"]
    assert 1.5 < res["overall"]["exposure_hours"] < streams.duration_sec.sum() / 3600


def test_poisson_and_proportion_intervals():
    r, lo, hi = det.poisson_rate_ci(0, 1.0)
    assert r == 0 and lo == 0 and abs(hi - 3.689) < 1e-3                    # exact upper bound for k=0
    p, lo, hi = det.proportion_ci(0, 10)
    assert p == 0 and lo == 0 and abs(hi - 0.3085) < 1e-3


# --------------------------------------------------------------------------- acceptance: wakeword positions
def test_wakeword_positions_are_not_fixed():
    streams, events = stream_set()
    b = events[events.stream_id.str.startswith("evalB") & events.event_type.str.startswith("wakeword")]
    dur = b.stream_id.map(streams.set_index("stream_id").duration_sec)
    lens = b.insert_end_sec - b.insert_start_sec
    u = b.insert_start_sec / (dur - lens)                                    # position within the feasible range
    assert u.min() < 0.1 and u.max() > 0.9
    assert kstest(u.clip(0, 1), "uniform").pvalue > 0.01                     # consistent with uniform placement
    frac = (b.wake_start_sec % 0.1) / 0.1                                    # offset relative to the 100 ms window grid
    assert frac.std() > 0.2
    assert b.event_type.nunique() == 2                                       # isolated and embedded wakewords


# --------------------------------------------------------------------------- unit: detection logic
def test_detect_threshold_refractory_and_smoothing():
    t = np.arange(1.5, 5.0, 0.1)
    s = np.zeros_like(t)
    s[5:8] = 0.9                                                             # 3 windows above threshold
    s[20] = 0.9
    trig = det.detect(t, s, 0.5, 1, refractory_sec=1.0)
    assert [round(x, 1) for x, _ in trig] == [2.0, 3.5]
    sm = det.detect(t, s, 0.5, 3, 1.0)                                       # causal mean of 3 windows
    assert len(sm) == 1 and abs(sm[0][0] - t[6]) < 1e-9 and abs(sm[0][1] - 0.6) < 1e-9   # spike at 20 suppressed


def test_detect_k_of_n():
    t = np.arange(1.5, 5.0, 0.1)
    s = np.zeros_like(t)
    s[5] = 0.9                                                               # isolated single window: suppressed by 2-of-3
    s[15], s[17] = 0.9, 0.9                                                  # 2 of 3 (with a gap): fires at the 2nd
    s[30:33] = 0.9                                                           # outside the 1 s refractory period
    trig = det.detect(t, s, 0.5, 1, 1.0, k=2, n=3)
    assert [round(x, 1) for x, _ in trig] == [round(t[17], 1), round(t[31], 1)]
    assert det.detect(t, s, 0.5, 1, 1.0, k=1, n=1) == det.detect(t, s, 0.5, 1, 1.0)


def test_per_category_threshold_policy():
    """Every deployment-like category ends at FPR <= target; public-corpus negatives never set the threshold."""
    rng = np.random.default_rng(0)
    neg = pd.DataFrame(dict(
        window_type=["negative_confusable"] * 400 + ["negative_real_background"] * 400 + ["negative_media"] * 30
                    + ["negative_confusable"] * 3000,
        eval_subset=["synthetic"] * 400 + ["real_device_rpi"] * 400 + ["synthetic"] * 30 + ["real_external"] * 3000,
        score=np.concatenate([rng.uniform(0, 0.6, 400), rng.uniform(0, 0.3, 400), rng.uniform(0, 1, 30),
                              rng.uniform(0, 0.05, 3000)])))
    pol = dict(mode="per_category_fpr", target_fpr=0.01, exclude_subsets=["real_external"], min_category_n=100)
    thr, d = iso.choose_threshold(neg, pol)
    assert d["excluded_windows"] == 3000 and "other (pooled small categories)" in d["per_group_threshold"]
    groups = iso.policy_groups(neg, pol)
    for g in groups.dropna().unique():
        assert (neg.score[groups == g] >= thr).mean() <= 0.01
    pooled, _ = iso.choose_threshold(neg, dict(mode="validation_fpr", target_fpr=0.01))
    assert pooled < thr                                                      # easy external negatives lower a pooled threshold


def test_false_accept_attribution():
    ev = pd.DataFrame([dict(event_id="c1", event_type="negative_confusable", insert_start_sec=10.0, insert_end_sec=11.0),
                       dict(event_id="p1", event_type="partial_head", insert_start_sec=30.0, insert_end_sec=30.4)])
    assert det.attribute(11.2, 1.5, ev, "quiet") == ("negative_confusable", "c1")
    assert det.attribute(30.9, 1.5, ev, "quiet") == ("partial_head", "p1")
    assert det.attribute(50.0, 1.5, ev, "quiet") == ("background:quiet", None)


def test_threshold_at_fpr_respects_target():
    rng = np.random.default_rng(0)
    neg = rng.random(1000)
    thr = det.threshold_at_fpr(neg, 0.01)
    assert (neg >= thr).mean() <= 0.01 and (neg >= np.nextafter(thr, -1) - 1e-3).mean() > 0.01 - 1e-9


def test_auc_and_ap():
    y = np.array([0, 0, 1, 1])
    assert det.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert det.roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert det.average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


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
