#!/usr/bin/env python3
"""
Encode the recording owner's confirmation of the manual recordings' labels as
review overrides (review/review_overrides.csv). Rows are tagged override_id
'owner-*'; rerunning replaces only those rows, so hand-made overrides survive.

Assertion supplied by the person who recorded and labelled the files (2026-09-25):
  every manual recording labelled positive contains "Hey Delta", and the labels are correct.

How the assertion is applied (labels come from the owner; bounds come from ASR,
from the file extent, or not at all):
  * Files <= asr.whole_file_max_sec are treated as one-utterance clips.
      - duration <= window - min_pre - min_post: ASR annotations removed, one
        clip_level_positive annotation [0, duration] added. A window holding the
        whole clip is then guaranteed to contain the complete wakeword.
      - longer, with an ASR complete/fuzzy match: that match is confirmed
        (ASR bounds + ASR tolerance kept); other ASR items in the file removed.
      - longer, no match: ASR items removed and a clip_level_positive added to
        record the label. It cannot fit a window, so it is logged as
        occurrence_too_long_for_window and blocks negatives from the file.
  * Longer session recordings: per VAD chunk, complete and fuzzy matches are
    confirmed; if a chunk has neither, its unmatched_speech span is confirmed
    (the owner states every utterance is the wakeword); remaining items in the
    chunk are removed as duplicates of the same utterance.
  * Files with a hand-made `add` override (utterance bounds marked by ear) keep
    those bounds: their ASR items are removed and no clip_level_positive is added.
  * Flags on handled recordings are cleared.
  * Background (negative) recordings: wakeword-like ASR items are marked
    non_wakeword (owner: no wakeword in those recordings). Confirmed for
    Manual-BG-RPI1..13 on 2026-09-25 and for Manual-BG-RPI14..19 on 2026-09-28,
    including RPI16's "Hey!", "Hey Michelle", "Hey mishmash", "Hey Martha" and
    "Hey, Marta" (near-misses, kept as real hard negatives); and for
    Manual-BG-RPI20..36 on 2026-09-29 (near-misses "Hey Michelle", "Hey Bobby",
    "Hey Martha", "Hey Marta" in RPI20, RPI24, RPI32); and for Manual-BG-Macmic2 and Manual-BG-Phonemic1 on
    2026-09-29: deliberate near-misses by the owner ("Hey Delia", "Hey Denta", "Hey Kenta", "Hey Martha",
    "Hey Michelle", "Hey Marian", ...), none of them "Hey Delta" (owner confirmed).
  * Positive sessions Manual-Fil-RPI65..82 (2026-09-29) contain background media
    speech, so the "every utterance is Hey Delta" session rule does NOT apply to
    them: their positives come only from the owner's hand-marked times (95 `human-*`
    add rows from review/owner_timing_2026-09-29.csv). Manual-Fil-RPI69 holds no
    Hey Delta and is relabelled to negatives_media in recording_label_overrides.csv.
  * EXCLUDE lists files the assertion cannot resolve; they stay in the review queue.
  * Recording-level relabels (e.g. Manual-Fil-RPI13 -> negatives_confusable) live in
    review/recording_label_overrides.csv; relabelled files are not handled here.
  * LISTENED: low-signal clips the owner listened to and confirmed as complete
    positives despite Whisper hearing nothing (kept deliberately: they match the
    deployment device's capture level).

Usage: python make_owner_overrides.py && python segment_annotate.py resolve && \\
       python segment_annotate.py windows && python segment_annotate.py review-tool
"""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment_annotate as sa

REVIEWER = "recording_owner"
REVIEWED_AT = "2026-09-25T00:00:00Z"
LISTENED = {"Manual-Fil-RPI19", "Manual-Fil-RPI20", "Manual-Fil-RPI21"}
# Manual-Fil-RPI23 was excluded until the owner marked its 9 wakewords by ear (2026-09-28).
EXCLUDE = {}


def main():
    cfg = sa.load_config()
    out = cfg["_out"]
    inv = pd.read_csv(out / "recording_inventory.csv")
    ann = pd.read_csv(out / "candidate_annotations.csv")
    flags = pd.read_csv(out / "recording_flags.csv").fillna("").set_index("file_id")
    W = cfg["window"]["duration_sec"]
    fit_max = W - cfg["positive"]["min_pre_context_sec"] - cfg["positive"]["min_post_context_sec"]
    short_max = cfg["asr"]["whole_file_max_sec"]

    path = sa.overrides_path(cfg)
    existing = pd.read_csv(path, dtype=str).fillna("")
    kept = existing[~existing.override_id.str.startswith("owner-")]
    # Files whose utterance was marked by ear (a hand-made `add` override) keep
    # those bounds: their ASR items are removed and no clip-level span is added.
    timed = set(kept.loc[kept.action == "add", "file_id"])

    rows = []

    def ov(action, file_id, annotation_id="", note=""):
        rows.append(dict(override_id=f"owner-{len(rows):04d}", action=action, annotation_id=annotation_id,
                         file_id=file_id, new_start_sec="", new_end_sec="", new_match_type="",
                         reviewer=REVIEWER, reviewed_at=REVIEWED_AT, note=note))

    missing = LISTENED - set(inv.file_id)
    assert not missing, missing
    # Held recordings (window_config.json hold_recordings) get no owner rows until the owner's timing
    # arrives; once they have hand-made `add` rows they are handled like any timed file below.
    held = set(cfg.get("hold_recordings", {})) - timed
    pos = inv[(inv.annotation_mode == "asr") & (inv.label == "positive") & ~inv.file_id.isin(EXCLUDE)
              & ~inv.file_id.isin(held)]
    for r in pos.itertuples(index=False):
        a = ann[(ann.file_id == r.file_id) & (ann.origin == sa.ORIGIN_ASR)]
        if r.file_id in timed:
            for x in a.itertuples(index=False):
                ov("remove", r.file_id, x.annotation_id, "superseded by human-marked bounds")
        elif r.duration_sec <= short_max:
            match = a[a.match_type.isin([sa.MATCH_COMPLETE, sa.MATCH_FUZZY])]
            match = match.assign(_p=match.match_type.ne(sa.MATCH_COMPLETE)).sort_values(["_p", "start_sec"])
            if r.duration_sec <= fit_max or match.empty:
                for x in a.itertuples(index=False):
                    ov("remove", r.file_id, x.annotation_id, "superseded by owner clip-level label")
                why = ("whole clip fits one window" if r.duration_sec <= fit_max
                       else "records owner label; no located utterance, too long for one window")
                heard = "; owner listened: low signal, complete Hey Delta" if r.file_id in LISTENED else ""
                ov("clip_level_positive", r.file_id, note=f"owner: file holds one complete Hey Delta ({why}){heard}")
            else:
                keep = match.iloc[0]
                ov("confirm_complete", r.file_id, keep.annotation_id, "owner: one complete Hey Delta per clip")
                for x in a[a.annotation_id != keep.annotation_id].itertuples(index=False):
                    ov("remove", r.file_id, x.annotation_id, "duplicate of the clip's single utterance")
        else:
            for _, c in a.groupby("vad_chunk_start_sec"):
                words = c[c.match_type.isin([sa.MATCH_COMPLETE, sa.MATCH_FUZZY])]
                keep = words if len(words) else c[c.match_type == sa.MATCH_UNMATCHED]
                if keep.empty:
                    keep = c[c.match_type == sa.MATCH_PARTIAL].head(1)
                for x in keep.itertuples(index=False):
                    ov("confirm_complete", r.file_id, x.annotation_id, "owner: every utterance is Hey Delta")
                for x in c[~c.annotation_id.isin(keep.annotation_id)].itertuples(index=False):
                    ov("remove", r.file_id, x.annotation_id, "duplicate of the same utterance")
        if r.file_id in flags.index and flags.loc[r.file_id, "recording_flag"]:
            ov("clear_recording_flag", r.file_id, note="owner confirmed label")

    # Owner: background (negative) recordings contain no wakeword, so any
    # wakeword-like ASR item there (e.g. "Hey, Michelle.") is ordinary speech.
    neg = inv[(inv.annotation_mode == "asr") & (inv.label == "negative") & ~inv.file_id.isin(EXCLUDE)]
    for x in ann[ann.file_id.isin(neg.file_id) & (ann.origin == sa.ORIGIN_ASR)].itertuples(index=False):
        ov("mark_not_wakeword", x.file_id, x.annotation_id, "owner: background recordings contain no wakeword")

    new = pd.DataFrame(rows).reindex(columns=sa.OVERRIDE_COLUMNS)
    pd.concat([kept, new], ignore_index=True).to_csv(path, index=False)
    print(f"wrote {len(new)} owner overrides ({new.action.value_counts().to_dict()}); "
          f"kept {len(kept)} other overrides -> {path}")
    for f, why in EXCLUDE.items():
        print(f"left for manual review: {f}: {why}")


if __name__ == "__main__":
    main()
