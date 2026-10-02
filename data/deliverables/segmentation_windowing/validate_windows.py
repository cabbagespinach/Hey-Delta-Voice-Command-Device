#!/usr/bin/env python3
"""
Validation checks for timestamps, labels and source inheritance.

Re-derives every rule from the inputs (manifest, frozen split, streaming-eval
manifest, audio headers, config) instead of trusting generator columns, and
writes outputs/validation_report.json + outputs/validation_report.md.
Exit code 1 if any check fails.

Usage: python validate_windows.py [--config window_config.json]
"""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_annotate import (load_config, energy_trim, config_hash, ORIGIN_GT, ORIGIN_ASR,
                              MATCH_COMPLETE, MATCH_NOT_WW, ST_REMOVED, ST_NEEDS_REVIEW)

EPS = 1e-3


class Report:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail="", examples=None):
        self.rows.append(dict(check=name, status="PASS" if ok else "FAIL", detail=detail,
                              examples=list(examples)[:5] if examples is not None else []))
        return ok

    @property
    def failed(self):
        return [r for r in self.rows if r["status"] == "FAIL"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "window_config.json"))
    cfg = load_config(ap.parse_args().config)
    out, root = cfg["_out"], cfg["_root"]
    W = cfg["window"]["duration_sec"]
    R = Report()

    man = pd.read_csv(cfg["_manifest"])
    norm = pd.read_csv(cfg["_normalized_manifest"])
    frozen = pd.read_csv(cfg["_split_assignments"])
    se = pd.read_csv(cfg["_streaming_eval_manifest"])
    inv = pd.read_csv(out / "recording_inventory.csv")
    cand = pd.read_csv(out / "candidate_annotations.csv")
    ann = pd.read_csv(out / "resolved_annotations.csv")
    win = pd.read_csv(out / "windows.csv")
    ann["review_reasons"] = ann["review_reasons"].fillna("")

    # ---------------------------------------------------------------- inventory / splits
    expected_files = set(man.filepath) | (set(se.filepath) if cfg["sources"]["include_streaming_eval"] else set())
    R.check("inventory_covers_every_manifest_recording", set(inv.filepath) == expected_files,
            f"{len(inv)} inventory rows vs {len(expected_files)} expected",
            sorted(expected_files ^ set(inv.filepath)))

    frozen_map = dict(zip(frozen.source_group, frozen.split))
    known = inv[inv.source_group.isin(frozen_map)]
    bad = known[known.split != known.source_group.map(frozen_map)]
    R.check("frozen_dataset_split_assignments_unchanged", bad.empty,
            f"{len(known)} files in pre-existing source groups", bad.file_id)
    norm_split = norm.set_index("filepath").split
    bad = inv[inv.filepath.isin(norm_split.index) & (inv.split != inv.filepath.map(norm_split))]
    R.check("normalized_manifest_splits_inherited", bad.empty, "", bad.file_id)

    parent_rec = man.set_index("filepath").recording_id.to_dict()
    exp_group = man.parent_filepath.map(parent_rec).fillna(man.recording_id)
    g = dict(zip(man.filepath, exp_group))
    bad = inv[inv.filepath.isin(g) & (inv.source_group != inv.filepath.map(g))]
    R.check("source_group_rule_reproduced", bad.empty, "parent recording_id if parent_filepath else recording_id",
            bad.file_id)

    for key in ["source_group", "recording_id"]:
        per = inv.groupby(key).split.nunique()
        R.check(f"no_{key}_crosses_splits", (per == 1).all(), f"{per.size} {key}s", per[per > 1].index)
    path_split = inv.set_index("filepath").split.to_dict()
    kids = inv[inv.parent_filepath.notna()]
    bad = kids[kids.split != kids.parent_filepath.map(path_split)]
    R.check("parent_child_same_split", bad.empty, f"{len(kids)} derived files", bad.file_id)
    ev = inv[inv.is_eval_only]
    R.check("streaming_eval_isolated_from_train_val_test",
            ev.split.eq(cfg["sources"]["streaming_eval_split"]).all()
            and not (set(ev.source_group) & set(inv[~inv.is_eval_only].source_group)),
            f"{len(ev)} eval-only files")

    durs = {fp: sf.info(str(root / fp)).duration for fp in inv.filepath}
    dd = (inv.duration_sec - inv.filepath.map(durs)).abs()
    R.check("recording_durations_match_audio_headers", (dd < EPS).all(), "", inv.file_id[dd >= EPS])

    # ---------------------------------------------------------------- annotations
    invi = inv.set_index("file_id")
    a_dur = ann.file_id.map(invi.duration_sec)
    bad = ann[(ann.start_sec < -EPS) | (ann.end_sec > a_dur + EPS) | (ann.end_sec <= ann.start_sec)]
    R.check("annotation_timestamps_within_recording_and_ordered", bad.empty, f"{len(ann)} annotations",
            bad.annotation_id)
    R.check("annotation_ids_unique", not ann.annotation_id.duplicated().any())
    for col in ["filepath", "recording_id", "source_group", "split"]:
        bad = ann[ann[col].astype(str) != ann.file_id.map(invi[col]).astype(str)]
        R.check(f"annotation_inherits_{col}", bad.empty, "", bad.annotation_id)

    # Synthetic GT is never replaced by ASR.
    synth = inv[inv.annotation_mode.isin(["gt_insertions", "gt_whole_clip"])]
    a_synth = ann[ann.file_id.isin(synth.file_id)]
    R.check("synthetic_recordings_have_no_asr_annotations", a_synth.origin.eq(ORIGIN_GT).all(),
            f"{len(a_synth)} annotations on {len(synth)} synthetic GT files",
            a_synth[a_synth.origin != ORIGIN_GT].annotation_id)
    real = inv[inv.annotation_mode == "asr"]
    R.check("real_recordings_have_no_gt_annotations",
            not ann[ann.file_id.isin(real.file_id)].origin.eq(ORIGIN_GT).any())
    mism = []
    for r in se.itertuples(index=False):
        fid = Path(r.filepath).stem
        gt = sorted(json.loads(r.wakeword_metadata), key=lambda m: m["start_sec"])
        got = ann[(ann.file_id == fid) & (ann.origin == ORIGIN_GT)].sort_values("start_sec")
        if len(gt) != len(got) or any(abs(m["start_sec"] - x.start_sec) > 1e-6 or abs(m["end_sec"] - x.end_sec) > 1e-6
                                      for m, x in zip(gt, got.itertuples())):
            mism.append(fid)
    R.check("streaming_gt_timestamps_verbatim", not mism, f"{len(se)} streams", mism)
    t = cfg["synthetic_ground_truth"]["tts_positive_whole_clip"]
    mism = []
    for r in inv[inv.annotation_mode == "gt_whole_clip"].itertuples(index=False):
        s, e = energy_trim(root / r.filepath, t["frame_sec"], t["top_db"])
        got = ann[(ann.file_id == r.file_id) & (ann.origin == ORIGIN_GT)]
        if len(got) != 1 or abs(got.start_sec.iloc[0] - s) > 1e-6 or abs(got.end_sec.iloc[0] - e) > 1e-6:
            mism.append(r.file_id)
    R.check("tts_whole_clip_gt_reproduced", not mism, "energy-trim rule re-derived from audio", mism)
    cgt = cand[cand.origin == ORIGIN_GT].set_index("annotation_id")[["start_sec", "end_sec"]]
    rgt = ann[ann.origin == ORIGIN_GT].set_index("annotation_id")[["start_sec", "end_sec"]]
    both = cgt.join(rgt, rsuffix="_r", how="inner")
    R.check("review_did_not_modify_gt_bounds",
            ((both.start_sec - both.start_sec_r).abs().max() < 1e-9 if len(both) else True)
            and ((both.end_sec - both.end_sec_r).abs().max() < 1e-9 if len(both) else True))

    # ASR annotations are visibly distinguishable.
    asr_ann = ann[ann.origin == ORIGIN_ASR]
    R.check("asr_annotations_labelled_as_candidates",
            asr_ann.label_source.str.startswith("asr_candidate").all() and
            not ann[ann.origin != ORIGIN_ASR].label_source.str.startswith("asr").any(),
            f"{len(asr_ann)} ASR annotations; label_sources={sorted(ann.label_source.unique())}")
    ov_err = (out / "override_errors.txt").read_text().strip() if (out / "override_errors.txt").exists() else ""
    R.check("no_orphaned_or_invalid_review_overrides", not ov_err, ov_err[:300])

    # Positive eligibility follows the explicit rule.
    policy = cfg["annotation_policy"]["asr_unreviewed_positive_policy"]
    exp = (ann.match_type.eq(MATCH_COMPLETE) & ann.status.isin(["ground_truth", "candidate", "reviewed"]) &
           (ann.label_source.ne("asr_candidate_unreviewed") | (policy == "allow_flagged")))
    R.check("positive_eligibility_rule", (exp == ann.positive_eligible).all(), "",
            ann.annotation_id[exp != ann.positive_eligible])
    R.check("needs_review_never_positive_eligible",
            not ann[ann.status == ST_NEEDS_REVIEW].positive_eligible.any())

    # Tolerances cover the measured ASR boundary error.
    tol = cfg["annotation_policy"]["boundary_tolerance_sec"]
    cal_p = out / "asr_boundary_calibration.json"
    if cal_p.exists():
        cal = json.loads(cal_p.read_text())
        bad = [f"{k}: need pre>={cal[k]['required_pre_sec']} post>={cal[k]['required_post_sec']}, have {tol[k]}"
               for k in ("asr_raw", "asr_energy_refined") if k in cal and
               (tol[k]["pre"] + EPS < cal[k]["required_pre_sec"] or tol[k]["post"] + EPS < cal[k]["required_post_sec"])]
        R.check("asr_tolerance_covers_calibrated_error", not bad, json.dumps({k: tol[k] for k in tol if k != 'note'}), bad)
    exp_key = np.where(ann.bounds_verified, "verified_bounds",
                       np.where(ann.boundary_method.eq("asr_energy_refined"), "asr_energy_refined", "asr_raw"))
    bad = ann[(ann.tol_pre_sec != [tol[k]["pre"] for k in exp_key]) |
              (ann.tol_post_sec != [tol[k]["post"] for k in exp_key])]
    R.check("annotation_tolerances_match_config", bad.empty, "", bad.annotation_id)
    R.check("asr_bounds_never_marked_verified_without_correction",
            not ann[(ann.origin == ORIGIN_ASR) & ann.bounds_verified &
                    ~ann.boundary_method.eq("human_corrected")].shape[0])

    # ---------------------------------------------------------------- windows
    R.check("window_ids_unique", not win.window_id.duplicated().any(), f"{len(win)} windows",
            win.window_id[win.window_id.duplicated()])
    for col in ["filepath", "recording_id", "source_group", "split", "is_eval_only"]:
        bad = win[win[col].astype(str) != win.file_id.map(invi[col]).astype(str)]
        R.check(f"window_inherits_{col}", bad.empty, "", bad.window_id)
    sid = win.file_id.map(invi.source_id)
    bad = win[~((win.source_id.astype(str) == sid.astype(str)) | (win.source_id.isna() & sid.isna()))]
    R.check("window_inherits_source_id", bad.empty, "", bad.window_id)
    per = win.groupby("source_group").split.nunique()
    R.check("no_window_source_group_crosses_splits", (per == 1).all(), "", per[per > 1].index)

    D = win.file_id.map(invi.duration_sec)
    R.check("window_duration_exact", ((win.end_sec - win.start_sec - W).abs() < EPS).all())
    ok = ((win.src_start_sec - win.start_sec.clip(lower=0)).abs() < EPS) & \
         ((win.src_end_sec - np.minimum(win.end_sec, D)).abs() < EPS) & \
         ((win.pad_left_sec - (-win.start_sec).clip(lower=0)).abs() < EPS) & \
         ((win.pad_right_sec - (win.end_sec - D).clip(lower=0)).abs() < EPS) & \
         (win.src_end_sec > win.src_start_sec)
    R.check("window_source_span_and_padding_consistent", ok.all(), "", win.window_id[~ok])
    maxpad = cfg["window"]["max_pad_fraction"] * W
    pad = win.pad_left_sec + win.pad_right_sec
    bad = win[(pad > maxpad + EPS) & (win.note.fillna("") != "short_clip_centered")]
    R.check("padding_within_limit_or_short_clip_policy", bad.empty, f"max_pad={maxpad:.3f}s", bad.window_id)

    # Positive windows: re-derive containment from the annotation and config.
    P = cfg["positive"]
    annI = ann.set_index("annotation_id")
    pos = win[win.label == "positive"]
    bad_ref, bad_contain, bad_type, bad_src, bad_other = [], [], [], [], []
    live = ann[(ann.status != ST_REMOVED) & (ann.match_type != MATCH_NOT_WW)]
    live_by_file = {k: v for k, v in live.groupby("file_id")}
    for w in pos.itertuples(index=False):
        if w.annotation_id not in annI.index:
            bad_ref.append(w.window_id); continue
        a = annI.loc[w.annotation_id]
        if not (a.positive_eligible and a.match_type == MATCH_COMPLETE and a.file_id == w.file_id):
            bad_type.append(w.window_id)
        s, e = a.start_sec - a.tol_pre_sec, a.end_sec + a.tol_post_sec
        if not (s - w.start_sec >= P["min_pre_context_sec"] - EPS and w.end_sec - e >= P["min_post_context_sec"] - EPS):
            bad_contain.append(w.window_id)
        if w.label_source != a.label_source:
            bad_src.append(w.window_id)
        for o in live_by_file[w.file_id].itertuples(index=False):
            if o.annotation_id != w.annotation_id and \
                    min(w.end_sec, o.end_sec + o.tol_post_sec) - max(w.start_sec, o.start_sec - o.tol_pre_sec) > 1e-9:
                bad_other.append(w.window_id); break
    R.check("positive_windows_reference_existing_annotation", not bad_ref, f"{len(pos)} positive windows", bad_ref)
    R.check("positive_windows_only_from_eligible_complete_annotations", not bad_type, "", bad_type)
    R.check("positive_windows_fully_contain_tolerance_expanded_wakeword", not bad_contain, "", bad_contain)
    R.check("positive_window_label_source_matches_annotation", not bad_src, "", bad_src)
    R.check("positive_windows_contain_no_other_wakeword_like_annotation", not bad_other, "", bad_other)
    R.check("positive_label_sources_are_explicit",
            pos.label_source.isin(["synthetic_ground_truth", "human", "asr_candidate_reviewed",
                                   "asr_candidate_unreviewed"]).all(), str(pos.label_source.value_counts().to_dict()))
    R.check("no_synthetic_recording_has_asr_labelled_windows",
            not win[win.file_id.isin(synth.file_id)].label_source.str.startswith("asr").any())

    # Near-duplicate control.
    per_occ = pos.groupby("annotation_id")
    R.check("max_positive_windows_per_occurrence", (per_occ.size() <= P["max_windows_per_occurrence"]).all(),
            f"max={int(per_occ.size().max()) if len(pos) else 0}")
    dup = [k for k, g in per_occ if len(g) > 1 and np.diff(np.sort(g.start_sec.values)).min()
           < P["min_shift_between_windows_sec"] - EPS]
    R.check("positive_windows_min_shift", not dup, "", dup)

    # Negative windows never contain a wakeword-like annotation (except explicit fragments).
    neg = win[win.label == "negative"]
    frag = neg[neg.window_type == "negative_partial_wakeword"]
    plain = neg[neg.window_type != "negative_partial_wakeword"]
    guard = cfg["negative"]["guard_sec"]
    bad = []
    for w in plain.itertuples(index=False):
        for o in live_by_file.get(w.file_id, ann.iloc[0:0]).itertuples(index=False):
            if min(w.end_sec, o.end_sec + o.tol_post_sec + guard) - max(w.start_sec, o.start_sec - o.tol_pre_sec - guard) > 1e-9:
                bad.append(w.window_id); break
    R.check("negative_windows_outside_guarded_annotations", not bad, f"{len(plain)} plain negatives", bad)

    # Partial fragments: coverage bound + only verified-bounds sources.
    PN = cfg["partial_negative"]
    bad_cov, bad_src = [], []
    for w in frag.itertuples(index=False):
        if not isinstance(w.annotation_id, str) or w.annotation_id not in annI.index:
            if not (w.parent_category == "negatives_partial"):
                bad_cov.append(w.window_id)
            continue
        a = annI.loc[w.annotation_id]
        cov = max(0.0, min(w.end_sec, a.end_sec) - max(w.start_sec, a.start_sec)) / (a.end_sec - a.start_sec)
        if cov > PN["max_coverage_fraction"] + EPS or cov >= 1 - EPS or abs(cov - w.wakeword_coverage) > 1e-3:
            bad_cov.append(w.window_id)
        if PN["require_verified_bounds"] and not a.bounds_verified:
            bad_src.append(w.window_id)
    R.check("partial_fragment_windows_never_cover_complete_wakeword", not bad_cov,
            f"{len(frag)} fragment windows; max coverage {PN['max_coverage_fraction']}", bad_cov)
    R.check("partial_fragment_windows_only_from_verified_bounds", not bad_src, "", bad_src)
    loose = set(PN["exclude_boundary_methods"])
    bad = [w.window_id for w in frag.itertuples(index=False)
           if isinstance(w.annotation_id, str) and w.annotation_id in annI.index
           and annI.loc[w.annotation_id].boundary_method in loose]
    R.check("no_fragments_from_loose_clip_extent_bounds", not bad, f"excluded methods: {sorted(loose)}", bad)
    ce = ann[ann.boundary_method == "human_clip_extent"]
    bad = ce[(ce.start_sec.abs() > 1e-6) | ((ce.end_sec - ce.file_id.map(invi.duration_sec)).abs() > 1e-6) |
             (ce.origin != "human")]
    R.check("clip_level_annotations_span_whole_file", bad.empty, f"{len(ce)} clip-level annotations", bad.annotation_id)
    lo_p = cfg["_review"] / "recording_label_overrides.csv"
    lo = (pd.read_csv(lo_p, dtype=str).fillna("") if lo_p.exists()
          else pd.DataFrame(columns=["file_id", "new_label", "new_category"]))
    rel = inv[inv.label_origin == "human_relabel"]
    exp = lo.set_index("file_id")
    ok = set(rel.file_id) == set(exp.index) and all(
        r.label == exp.loc[r.file_id, "new_label"] and r.category == exp.loc[r.file_id, "new_category"]
        and r.annotation_mode == "human_label" for r in rel.itertuples(index=False))
    R.check("recording_relabels_applied_and_recorded", ok, f"{len(rel)} relabelled recordings", rel.file_id)
    man_lab = man.set_index("filepath")
    bad = rel[(rel.manifest_label != rel.filepath.map(man_lab.label)) | (rel.manifest_category != rel.filepath.map(man_lab.category))]
    R.check("relabelled_recordings_keep_manifest_values", bad.empty, "", bad.file_id)
    R.check("relabelled_recordings_have_no_machine_annotations", not ann.file_id.isin(rel.file_id).any())
    hw = win[win.file_id.isin(rel.file_id)]
    R.check("relabelled_recording_windows_use_human_label",
            hw.label_source.eq("human_recording_label").all() and
            (hw.label == hw.file_id.map(invi.label)).all(), f"{len(hw)} windows")
    bad = pos[pos.file_id.map(invi.label).eq("negative")]
    R.check("no_positive_windows_from_negative_labelled_recordings", bad.empty, "", bad.window_id)
    R.check("partial_manifest_clips_are_negative",
            win[win.parent_category == "negatives_partial"].label.eq("negative").all())

    # Every positive window traces to an annotation; nothing is positive by manifest label alone.
    R.check("no_positive_without_annotation", pos.annotation_id.notna().all() and (pos.annotation_id != "").all())
    fl = pd.read_csv(out / "recording_review_state.csv").fillna("")
    blocked = fl[(fl.recording_flag != "") & (fl.flag_cleared_by_review.astype(str) != "True")].file_id
    R.check("no_negatives_from_unresolved_flagged_recordings", not neg.file_id.isin(blocked).any(),
            f"{len(blocked)} flagged recordings")
    R.check("windows_generated_with_current_config", win.config_hash.eq(config_hash(cfg)).all())

    # ---------------------------------------------------------------- write report
    summary = dict(generated_by="validate_windows.py", config_hash=config_hash(cfg),
                   n_checks=len(R.rows), n_failed=len(R.failed), checks=R.rows,
                   windows_by_split_label=win.groupby(["split", "label"]).size().rename("n").reset_index()
                   .to_dict("records"))
    (out / "validation_report.json").write_text(json.dumps(summary, indent=2, default=str))
    lines = ["# Validation report", "", f"Config hash `{config_hash(cfg)}` · {len(R.rows)} checks · "
             f"**{len(R.failed)} failed**", "", "| check | status | detail |", "|---|---|---|"]
    for r in R.rows:
        ex = f" e.g. {', '.join(map(str, r['examples']))}" if r["examples"] and r["status"] == "FAIL" else ""
        lines.append(f"| {r['check']} | {r['status']} | {str(r['detail']).replace('|', '/')}{ex} |")
    (out / "validation_report.md").write_text("\n".join(lines) + "\n")
    for r in R.rows:
        print(f"{r['status']}  {r['check']}" + (f"  -> {r['examples']}" if r["status"] == "FAIL" else ""))
    print(f"\n{len(R.rows) - len(R.failed)}/{len(R.rows)} checks passed")
    sys.exit(1 if R.failed else 0)


if __name__ == "__main__":
    main()
