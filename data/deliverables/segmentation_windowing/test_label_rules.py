#!/usr/bin/env python3
"""
Rule tests: try to break the labelling invariants on purpose.

  python test_label_rules.py        (or: pytest test_label_rules.py)

Unit tests use synthetic fixtures. test_review_roundtrip runs resolve + windows +
validate_windows.py on a temporary copy of outputs/ with a scripted set of
review overrides, so the real outputs/ and review/ are never modified.
Requires `python segment_annotate.py all` to have been run once.
"""
from pathlib import Path
import json, shutil, subprocess, sys, tempfile
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import segment_annotate as sa

CFG = sa.load_config()


def chunk(words, c0=0.0, c1=2.0):
    ws, t = [], c0 + 0.1
    for w in words:
        ws.append(dict(word=w, start=t, end=t + 0.3, probability=0.9))
        t += 0.32
    return dict(chunk_start=c0, chunk_end=c1, text=" ".join(words), words=ws)


def types(ch, positive=True):
    return [d["match_type"] for d in sa.detect_in_chunk(ch, CFG, positive)]


def test_only_exact_pair_is_complete():
    assert types(chunk(["Hey", "Delta!"])) == [sa.MATCH_COMPLETE]
    assert types(chunk(["Hey,", "Delta."])) == [sa.MATCH_COMPLETE]
    assert sa.MATCH_COMPLETE not in types(chunk(["Hey", "Dota!"]))
    assert types(chunk(["Hey", "Dota!"]))[0] == sa.MATCH_FUZZY
    assert types(chunk(["Delta"]))[0] == sa.MATCH_PARTIAL
    assert types(chunk(["Hey,", "there's", "a"]))[0] == sa.MATCH_PARTIAL
    assert types(chunk(["He", "built", "that."])) == [sa.MATCH_UNMATCHED]
    assert types(chunk(["Thank", "you."]), positive=False) == []


def test_complete_in_negative_recording_needs_review():
    d = sa.detect_in_chunk(chunk(["Hey", "Delta"]), CFG, False)[0]
    assert "wakeword_detected_in_negative_recording" in d["reasons"]


def ann_frame(rows):
    base = dict(origin=sa.ORIGIN_ASR, status=sa.ST_CANDIDATE, match_type=sa.MATCH_COMPLETE,
                bounds_verified=False, boundary_method="asr_energy_refined")
    return sa.finalize_annotations(pd.DataFrame([{**base, **r} for r in rows]), CFG)


def test_partial_fuzzy_unmatched_and_needs_review_never_eligible():
    a = ann_frame([dict(match_type=sa.MATCH_PARTIAL), dict(match_type=sa.MATCH_FUZZY),
                   dict(match_type=sa.MATCH_UNMATCHED), dict(match_type=sa.MATCH_NOT_WW),
                   dict(status=sa.ST_NEEDS_REVIEW), dict(status=sa.ST_REMOVED),
                   dict(origin=sa.ORIGIN_GT, status=sa.ST_GT, match_type=sa.MATCH_PARTIAL)])
    assert not a.positive_eligible.any()


def test_asr_is_labelled_distinctly_and_policy_respected():
    a = ann_frame([dict(), dict(origin=sa.ORIGIN_GT, status=sa.ST_GT, bounds_verified=True)])
    assert list(a.label_source) == ["asr_candidate_unreviewed", "synthetic_ground_truth"]
    strict = json.loads(json.dumps({k: v for k, v in CFG.items() if not k.startswith("_")}))
    strict["annotation_policy"]["asr_unreviewed_positive_policy"] = "require_review"
    a2 = sa.finalize_annotations(pd.DataFrame([dict(origin=sa.ORIGIN_ASR, status=sa.ST_CANDIDATE,
                                                    match_type=sa.MATCH_COMPLETE, bounds_verified=False,
                                                    boundary_method="asr_raw")]), strict)
    assert not a2.positive_eligible.any()


class Rec:
    file_id, filepath, recording_id, source_id, source_group, speaker_id = "f", "x.wav", "r", "s", "g", ""
    split, is_eval_only, category, source, source_recording_offset_sec = "train", False, "positives", "t", float("nan")
    annotation_mode, recording_is_positive, canonical_category = "asr", True, "positive_wakeword"

    def __init__(self, duration):
        self.duration_sec = duration


def one(**kw):
    a = ann_frame([dict(annotation_id="f#a", start_sec=2.0, end_sec=2.7, **kw)])
    return next(a.itertuples(index=False))


def test_positive_window_contains_expanded_span_and_caps_duplicates():
    a, rej = one(), []
    ws = sa.positive_windows(Rec(10.0), a, [], CFG, rej)
    assert 1 <= len(ws) <= CFG["positive"]["max_windows_per_occurrence"]
    for w in ws:
        assert w["start_sec"] <= a.start_sec - a.tol_pre_sec - CFG["positive"]["min_pre_context_sec"] + 1e-6
        assert w["end_sec"] >= a.end_sec + a.tol_post_sec + CFG["positive"]["min_post_context_sec"] - 1e-6
    starts = sorted(w["start_sec"] for w in ws)
    assert all(b - a_ >= CFG["positive"]["min_shift_between_windows_sec"] - 1e-6 for a_, b in zip(starts, starts[1:]))


def test_too_long_occurrence_is_rejected_not_truncated():
    a = ann_frame([dict(annotation_id="f#a", start_sec=1.0, end_sec=2.4)]).iloc[0]
    rej = []
    assert sa.positive_windows(Rec(10.0), a, [], CFG, rej) == []
    assert rej[0]["reason"] == "occurrence_too_long_for_window"


def test_positive_window_rejected_when_another_wakeword_like_annotation_overlaps():
    a = one()
    other = one(match_type=sa.MATCH_FUZZY)._replace(annotation_id="f#b", start_sec=3.0, end_sec=3.4)
    rej = []
    ws = sa.positive_windows(Rec(10.0), a, [other], CFG, rej)
    assert all(w["end_sec"] <= other.start_sec - other.tol_pre_sec + 1e-9 for w in ws)
    assert any(r["reason"] == "overlaps_other_wakeword_like_annotation" for r in rej)


def test_fragments_never_reach_complete_coverage_and_need_verified_bounds():
    rej = []
    assert sa.partial_windows(Rec(10.0), one(), [], CFG, rej) == []  # unverified ASR bounds
    g = ann_frame([dict(annotation_id="f#g", origin=sa.ORIGIN_GT, status=sa.ST_GT, bounds_verified=True,
                        boundary_method="synthetic_insertion", start_sec=4.0, end_sec=4.8)]).iloc[0]
    ws = sa.partial_windows(Rec(10.0), g, [], CFG, rej)
    assert ws and all(w["label"] == "negative" for w in ws)
    assert all(0 < w["wakeword_coverage"] <= CFG["partial_negative"]["max_coverage_fraction"] for w in ws)


def test_review_roundtrip():
    """Scripted review overrides -> resolve -> windows -> full validation, in a temp copy."""
    out = CFG["_out"]
    ann = pd.read_csv(out / "candidate_annotations.csv")
    asr = ann[ann.origin == sa.ORIGIN_ASR]
    # Confirmed as a wakeword below, so it must come from a positive recording
    # (background near-misses such as Manual-BG-RPI16's "Hey, Marta" are fuzzy too).
    inv = pd.read_csv(out / "recording_inventory.csv")
    fuzzy = asr[(asr.match_type == sa.MATCH_FUZZY)
                & asr.file_id.isin(inv.loc[inv.label == "positive", "file_id"])].iloc[0]
    comp = asr[(asr.match_type == sa.MATCH_COMPLETE) & (asr.status == sa.ST_CANDIDATE)]
    to_remove, to_partial = comp.iloc[0], comp.iloc[1]
    gt = ann[ann.origin == sa.ORIGIN_GT].iloc[0]
    flags = pd.read_csv(out / "recording_flags.csv").fillna("")
    flagged = flags[flags.recording_flag != ""].file_id.iloc[0]
    fs, fe = round(fuzzy.start_sec, 3), round(fuzzy.end_sec + 0.1, 3)
    ov = pd.DataFrame([
        dict(override_id="t1", action="confirm_complete", annotation_id=fuzzy.annotation_id, new_start_sec=fs, new_end_sec=fe),
        dict(override_id="t2", action="remove", annotation_id=to_remove.annotation_id),
        dict(override_id="t3", action="mark_partial", annotation_id=to_partial.annotation_id),
        dict(override_id="t4", action="add", file_id=flagged, new_start_sec=0.2, new_end_sec=0.9, new_match_type="complete"),
        dict(override_id="t5", action="clear_recording_flag", file_id=flagged),
        dict(override_id="bad1", action="correct_bounds", annotation_id=gt.annotation_id, new_start_sec=0, new_end_sec=1),
        dict(override_id="bad2", action="confirm_complete", annotation_id="does-not-exist#asr999"),
    ]).reindex(columns=sa.OVERRIDE_COLUMNS).assign(reviewer="test", reviewed_at="2026-09-25T00:00:00Z")

    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        tmp = Path(tmp)
        (tmp / "out").mkdir(); (tmp / "review").mkdir()
        for f in ["recording_inventory.csv", "candidate_annotations.csv", "recording_flags.csv",
                  "asr_boundary_calibration.json"]:
            shutil.copy(out / f, tmp / "out" / f)
        (tmp / "out" / "asr_raw").symlink_to(out / "asr_raw")
        ov.to_csv(tmp / "review" / "review_overrides.csv", index=False)
        shutil.copy(CFG["_review"] / "recording_label_overrides.csv", tmp / "review")
        cfg_raw = json.loads((HERE / "window_config.json").read_text())
        cfg_raw["paths"].update(project_root=str(CFG["_root"]), output_dir=str(tmp / "out"),
                                review_dir=str(tmp / "review"))
        (tmp / "cfg.json").write_text(json.dumps(cfg_raw))
        cfg = sa.load_config(tmp / "cfg.json")

        res = sa.stage_resolve(cfg).set_index("annotation_id")
        errs = (tmp / "out" / "override_errors.txt").read_text()
        assert "bad1" in errs and "only be removed" in errs, errs
        assert "bad2" in errs and "orphaned" in errs, errs
        assert (res.loc[gt.annotation_id, ["start_sec", "end_sec"]].values == [gt.start_sec, gt.end_sec]).all()
        f = res.loc[fuzzy.annotation_id]
        assert f.match_type == sa.MATCH_COMPLETE and f.label_source == "asr_candidate_reviewed" and f.bounds_verified
        assert f.tol_pre_sec == 0 and f.tol_post_sec == 0 and f.positive_eligible
        assert res.loc[to_remove.annotation_id].status == sa.ST_REMOVED
        assert not res.loc[to_partial.annotation_id].positive_eligible
        human = res[res.origin == sa.ORIGIN_HUMAN]
        assert len(human) == 1 and human.label_source.iloc[0] == "human" and human.positive_eligible.iloc[0]

        win = sa.stage_windows(cfg)
        assert (win[win.annotation_id == fuzzy.annotation_id].label_source
                .isin(["asr_candidate_reviewed", "asr_candidate_reviewed_fragment"])).all()
        assert (win.annotation_id == fuzzy.annotation_id).any()
        assert not (win.annotation_id == to_remove.annotation_id).any()
        assert not ((win.annotation_id == to_partial.annotation_id) & (win.label == "positive")).any()
        assert (win[(win.file_id == flagged) & (win.label == "positive")].label_source == "human").all()

        # Remove the deliberately invalid overrides, then the full validator must pass.
        ov[~ov.override_id.str.startswith("bad")].to_csv(tmp / "review" / "review_overrides.csv", index=False)
        sa.stage_resolve(cfg); sa.stage_windows(cfg)
        r = subprocess.run([sys.executable, str(HERE / "validate_windows.py"), "--config", str(tmp / "cfg.json")],
                           capture_output=True, text=True)
        assert r.returncode == 0, (r.stdout[-2000:] + r.stderr[-2000:])


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
