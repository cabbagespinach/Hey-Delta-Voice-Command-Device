#!/usr/bin/env python3
"""
Always-listening streaming evaluation.

Every stream in streaming_set/streams.csv is run through the deployment front end
(StreamingPreprocessor: 1.5 s windows every 100 ms, fed in 0.5 s chunks), scored,
turned into triggers (threshold, optional smoothing, refractory period) and matched
against the ground-truth wakeword spans in streaming_set/events.csv.

Measures: false accepts per hour (with exact Poisson CIs), false accepts by category
(the event or background under the triggering window), detection rate, misses,
duplicate triggers, detection latency, all broken down by acoustic condition, wakeword
level, isolated vs embedded wakeword, voice, and set. Every stream is synthetic
(Set A: phase-1 holdout; Set B: composed for evaluation); real device audio is covered
only by isolated evaluation.
"""
from pathlib import Path
import hashlib, json, multiprocessing as mp, sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dataloading"))
import detection as det             # noqa: E402
import wakeword_data as wd          # noqa: E402

CFG = json.loads((HERE / "eval_config.json").read_text())
SET_DIR = HERE / "streaming_set"
WAKE_TYPES = ("wakeword", "wakeword_embedded")


def load_set():
    s = pd.read_csv(SET_DIR / "streams.csv")
    e = pd.read_csv(SET_DIR / "events.csv")
    return s, e


def verify_set(streams: pd.DataFrame, isolated_manifests=()):
    """The frozen set is intact and independent of training and of isolated evaluation."""
    problems = []
    for r in streams.itertuples(index=False):
        p = det.ROOT / r.filepath
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != r.sha256:
            problems.append(f"{r.stream_id}: missing or modified audio")
    cfg = wd.load_config()
    guard = wd.SplitGuard(cfg)
    windows = pd.read_csv(cfg["_inputs"]["windows_csv"])
    non_eval = windows[~windows.is_eval_only.astype(bool)]
    ids = set(streams.stream_id)
    if ids & set(guard.frozen):
        problems.append("a stream id is a source group of train/validation/test")
    if set(streams.filepath) & set(non_eval.filepath):
        problems.append("a stream file is also a train/validation/test window source")
    for m in isolated_manifests:
        iso = pd.read_csv(m)
        if set(streams.filepath) & set(iso.filepath) or ids & set(iso.source_group):
            problems.append(f"streaming set overlaps isolated set {Path(m).name}")
    b = streams[streams.set == "B_composed"]
    if not b.filepath.str.startswith(CFG["streaming_set_b"]["audio_dir"] + "/").all():
        problems.append("Set B audio outside its eval-only directory")
    if problems:
        raise RuntimeError("streaming set verification failed: " + "; ".join(problems))
    return dict(streams=len(streams), sha256_verified=True, independent_of=["train", "validation", "test", "isolated sets"])


# --------------------------------------------------------------------------- scoring (parallel)
_WORKER = {}


def _init(model_factory):
    import torch
    torch.set_num_threads(1)
    _WORKER["pre"] = det.wp.WakewordPreprocessor.from_files()
    _WORKER["scorer"] = det.TorchScorer(model_factory(), batch_size=512)


def _score(args):
    sid, path = args
    t, s = det.stream_scores(det.ROOT / path, _WORKER["pre"], _WORKER["scorer"], CFG["detection"]["stream_chunk_sec"])
    return sid, t, s


def score_streams(streams: pd.DataFrame, model_factory, workers: int = 16) -> dict:
    """{stream_id: (window_end_times, scores)}. Each worker is single-threaded, so results do not
    depend on the worker count."""
    jobs = list(zip(streams.stream_id, streams.filepath))
    with mp.get_context("fork").Pool(workers, initializer=_init, initargs=(model_factory,)) as pool:
        return {sid: (t, s) for sid, t, s in pool.imap(_score, jobs, chunksize=2)}


# --------------------------------------------------------------------------- evaluation
def snr_bucket(v):
    if pd.isna(v):
        return "unmeasured (Set A)"
    return "low (<10 dB)" if v < 10 else ("mid (10-18 dB)" if v < 18 else "high (>=18 dB)")


def evaluate(streams, events, scores: dict, threshold: float, k_of_n=None):
    D = dict(CFG["detection"])
    if k_of_n is not None:
        D["k_of_n"] = list(k_of_n)
    k, n = D.get("k_of_n", [1, 1])
    W = 1.5
    trig_rows, fa_rows, det_rows, per_stream = [], [], [], []
    for st in streams.itertuples(index=False):
        t, s = scores[st.stream_id]
        ev = events[events.stream_id == st.stream_id]
        wk = ev[ev.event_type.isin(WAKE_TYPES)]
        other = ev[~ev.event_type.isin(WAKE_TYPES)]
        trig = det.detect(t, s, threshold, D["smoothing_windows"], D["refractory_sec"], k, n)
        m = det.match(trig, list(zip(wk.event_id, wk.wake_start_sec, wk.wake_end_sec)), st.duration_sec, W,
                      D["hit_zone"]["before_start_sec"], D["hit_zone"]["after_end_sec"])
        for tt, v in m["false_accepts"]:
            cat, eid = det.attribute(tt, D["fa_attribution_window_sec"], other, st.condition)
            fa_rows.append(dict(stream_id=st.stream_id, set=st.set, condition=st.condition, trigger_sec=tt, score=v,
                                category=cat, event_id=eid))
        for r in wk.itertuples(index=False):
            h = m["hits"].get(r.event_id)
            det_rows.append(dict(event_id=r.event_id, stream_id=st.stream_id, set=st.set, condition=st.condition,
                                 wake_type=r.event_type, voice=r.voice, wake_start_sec=r.wake_start_sec,
                                 wake_end_sec=r.wake_end_sec, relative_position=r.wake_start_sec / st.duration_sec,
                                 level=snr_bucket(getattr(r, "achieved_peak_over_floor_db", np.nan)),
                                 detected=h is not None, latency_sec=h["latency_sec"] if h else np.nan,
                                 trigger_score=h["score"] if h else np.nan))
        trig_rows += [dict(stream_id=st.stream_id, trigger_sec=tt, score=v) for tt, v in trig]
        per_stream.append(dict(stream_id=st.stream_id, set=st.set, condition=st.condition,
                               exposure_sec=m["exposure_sec"], false_accepts=len(m["false_accepts"]),
                               wakewords=len(wk), detected=len(m["hits"]), duplicates=m["duplicates"]))
    ps, dt = pd.DataFrame(per_stream), pd.DataFrame(det_rows)
    fa = pd.DataFrame(fa_rows, columns=["stream_id", "set", "condition", "trigger_sec", "score", "category", "event_id"])

    def summary(p, d):
        hours = p.exposure_sec.sum() / 3600
        k = int(p.false_accepts.sum())
        r, lo, hi = det.poisson_rate_ci(k, hours)
        n, h = len(d), int(d.detected.sum()) if len(d) else 0
        dr, dlo, dhi = det.proportion_ci(h, n)
        lat = d.latency_sec.dropna()
        return dict(streams=len(p), exposure_hours=round(hours, 4), false_accepts=k, fa_per_hour=r, fa_per_hour_ci95=[lo, hi],
                    wakewords=n, detected=h, misses=n - h, detection_rate=dr, detection_rate_ci95=[dlo, dhi],
                    false_reject_rate=(1 - dr) if n else float("nan"), duplicate_triggers=int(p.duplicates.sum()),
                    latency_sec=dict(median=float(lat.median()) if len(lat) else None,
                                     p90=float(lat.quantile(0.9)) if len(lat) else None,
                                     mean=float(lat.mean()) if len(lat) else None,
                                     min=float(lat.min()) if len(lat) else None,
                                     max=float(lat.max()) if len(lat) else None))

    res = dict(threshold=threshold, detection_config=D, overall=summary(ps, dt),
               by_set={k: summary(g, dt[dt.set == k]) for k, g in ps.groupby("set")},
               by_condition={k: summary(g, dt[dt.condition == k]) for k, g in ps.groupby("condition")})
    for col in ("wake_type", "level", "voice"):
        res[f"detection_by_{col}"] = {k: dict(zip(("rate", "lo", "hi"), det.proportion_ci(int(g.detected.sum()), len(g))),
                                              n=len(g)) for k, g in dt.groupby(col)}
    # false accepts by category
    total_h = ps.exposure_sec.sum() / 3600
    ev_other = events[~events.event_type.isin(WAKE_TYPES)]
    cats = []
    for cat, g in fa.groupby("category"):
        row = dict(category=cat, false_accepts=len(g), share_of_false_accepts=len(g) / max(1, len(fa)),
                   fa_per_hour_of_all_audio=len(g) / total_h, sets=", ".join(sorted(g.set.unique())))
        if not cat.startswith("background:"):
            n_ev = int((ev_other.event_type == cat).sum())
            k_ev = g.event_id.nunique()
            r, lo, hi = det.proportion_ci(k_ev, n_ev)
            row.update(events_of_this_type=n_ev, events_triggering_fa=k_ev, fa_per_event=r, fa_per_event_ci95=[lo, hi])
        else:
            cond = cat.split(":", 1)[1]
            h = ps[ps.condition == cond].exposure_sec.sum() / 3600
            r, lo, hi = det.poisson_rate_ci(len(g), h)
            row.update(condition_exposure_hours=h, fa_per_hour_in_condition=r, fa_per_hour_in_condition_ci95=[lo, hi])
        cats.append(row)
    for cat in sorted(set(ev_other.event_type) - set(fa.category)):          # categories with zero FAs
        n_ev = int((ev_other.event_type == cat).sum())
        r, lo, hi = det.proportion_ci(0, n_ev)
        cats.append(dict(category=cat, false_accepts=0, share_of_false_accepts=0.0, fa_per_hour_of_all_audio=0.0,
                         events_of_this_type=n_ev, events_triggering_fa=0, fa_per_event=0.0, fa_per_event_ci95=[lo, hi]))
    res["false_accepts_by_category"] = sorted(cats, key=lambda c: -c["false_accepts"])
    res["wakeword_position"] = dict(relative_start_min=float(dt.relative_position.min()),
                                    relative_start_max=float(dt.relative_position.max()),
                                    relative_start_quartiles=[float(q) for q in dt.relative_position.quantile([.25, .5, .75])],
                                    by_set={k: [float(g.relative_position.min()), float(g.relative_position.max())]
                                            for k, g in dt.groupby("set")})
    return res, ps, dt, fa, pd.DataFrame(trig_rows)


def sweep(streams, events, scores, thresholds, k_of_n=None):
    """Detection-error trade-off per set: FA/hour vs miss rate across thresholds."""
    rows = []
    for thr in thresholds:
        r, *_ = evaluate(streams, events, scores, float(thr), k_of_n)
        for k, v in r["by_set"].items():
            rows.append(dict(threshold=float(thr), set=k, fa_per_hour=v["fa_per_hour"], miss_rate=v["false_reject_rate"]))
        rows.append(dict(threshold=float(thr), set="ALL", fa_per_hour=r["overall"]["fa_per_hour"],
                         miss_rate=r["overall"]["false_reject_rate"]))
    return pd.DataFrame(rows)
