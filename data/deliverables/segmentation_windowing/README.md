# Segmentation, annotation and window generation — Hey Delta

Converts every recording referenced by `manifest.csv` (plus the held-out
`streaming_eval` streams) into fixed-duration **candidate positive / negative windows**,
without losing the identity of the source recording, and makes every wakeword
annotation reviewable, correctable and removable.

Builds on `../dataset_split/` (frozen split, `source_group` rule) and
`../deployment_driven_augmentation_strategy/` (train-only augmentation, and the
note that real recordings were missing — the 83 `manual_recording` rows are the first real audio).

## Deliverables

| Requirement | File |
|---|---|
| Segmentation/annotation module | `segment_annotate.py` (stages `inventory → annotate → resolve → windows → review-tool`, optional `materialize`) |
| Candidate-annotation manifest | `outputs/candidate_annotations.csv` (as produced by ASR / GT), `outputs/resolved_annotations.csv` (after review overrides; this one feeds windowing) |
| Window-generation configuration | `window_config.json` |
| Annotation-review mechanism | `review/review_tool.html`, `review/review_queue.csv`, `review/review_overrides.csv`, `make_owner_overrides.py` (bulk confirmation by the recording owner), `review/recording_label_overrides.csv` (recording-level relabels) |
| Validation checks | `validate_windows.py` → `outputs/validation_report.{md,json}` (61 checks); `test_label_rules.py` (9 rule tests incl. a scripted review round-trip) |
| Window manifest | `outputs/windows.csv` (+ `window_summary.csv`, `rejected_windows.csv`) |

Supporting outputs: `recording_inventory.csv` (one row per audio file, with inherited split and provenance),
`split_extension.csv`, `recording_flags.csv` / `recording_review_state.csv`, `asr_raw/*.json` (cached
per-chunk Whisper words), `asr_crosscheck_synthetic.csv`, `asr_boundary_calibration.json`.

## Running

```bash
cd data/deliverables/segmentation_windowing
python segment_annotate.py all          # ~1 min with the ASR cache; first ASR run needs a GPU for a few minutes
python validate_windows.py              # exit 1 on any failed check
python test_label_rules.py              # or: pytest test_label_rules.py
# after review:
python segment_annotate.py resolve && python segment_annotate.py windows && python segment_annotate.py review-tool
python validate_windows.py
# optional audio export (16 kHz mono WAVs, see window.output_sample_rate):
python segment_annotate.py materialize --split validation --limit 50
```

Requires `faster-whisper` (1.2.1 tested; model `large-v3`), `soundfile`, `scipy`, `pandas`, `numpy`.
ASR results are cached per file under `outputs/asr_raw/`, keyed by a hash of the `asr` config section.

**Added 2026-09-29: public-corpus negatives** (`../../../scripts/integrate_external_datasets.py`; raw data and licences in
`external_raw/README.md`). There are 9,253 files, all negative and all in `generation_label` mode:
- MSWC near-miss words: 6,663 confusables and 660 partials ("hey", "hay", "delta").
- FLEURS read speech, 600 files (fil_ph and en_us).
- MUSAN: 200 speech excerpts, 200 music excerpts and 930 noise clips.

Split groups are the MSWC speaker, the FLEURS sentence id (FLEURS has no speaker ids) and the MUSAN source file. The
append-only extension assigned 6,934 new groups. No earlier window changed its label or split. These files add 19,559
windows: train 15,737, validation 1,910, test 1,912.

**Added 2026-09-29 (night): 405 new synthetic speakers** (`../../../scripts/generate_tts_voices_v2.py`). BC-ResNet baselines
had memorised the 5 training TTS voices (validation synthetic detection 10–40%, real RPI 100%). New offline Piper
voices, none already in the dataset or in streaming Set B, add 2 "Hey Delta" + 2 near-miss clips per speaker:
- LibriTTS-R (220 speakers), VCTK (109), L2-ARCTIC (24 non-native), ARCTIC (18), ARU (12), SEMAINE (4) and 18
  single-speaker voices.
- Whisper-verified, strictly: a positive is kept only if it transcribes exactly as "hey delta" (625 of 810 kept); a
  near-miss is dropped only if it does (1 of 810 dropped).
- Positives are `tts_synth` (clip-level GT); near-misses are `tts_synthetic_phonetic_nearmiss`.
- Each speaker is one source group, assigned to one split: train 304, validation 51, test 50 speakers.
- Windows: train 1,409 positive and 618 negative; validation 167 / 102; test 172 / 99.

**ASR calibration reference unchanged.** `asr_boundary_calibration.json` is still computed on the phase-1 TTS positives
only, and only on spans the energy trim tightened at both ends. 329 of the new voices' ASR-matched clips could not be
trimmed at all: some Piper voices (VCTK especially) carry a constant noise floor. Such a span is a safe label, because
a window must contain all of it, but it is not a timing reference; it had pushed the required tolerance above 1 s.
The required tolerances are unchanged (raw 0.00 / 0.51 s, refined 0.08 / 0.38 s).

## How recordings are handled

Each audio file is one row of `recording_inventory.csv` and gets one `annotation_mode`:

| Mode | Files | Annotation source |
|---|---:|---|
| `asr` — real recordings (`source = manual_recording`) | 117 (83 labelled positive, 34 background) plus 5 relabelled recordings windowed from their human label | Faster-Whisper word timestamps → **candidates** |
| `gt_insertions` — `streaming_eval` streams | 120 (128 inserted wakewords) | `wakeword_metadata` insertion timestamps, used verbatim |
| `gt_whole_clip` — TTS positives (`tts_synth`) | 50 | Clip-level GT: one complete, ASR-verified utterance per clip; span = clip trimmed of near-silence by a fixed energy rule |
| `generation_label` — synthetic negatives | 6,911 | Negative by construction (generation metadata) |

**Source identity.** Every annotation and window carries `file_id`, `filepath`, `recording_id`,
`source_id`, `source_group`, `speaker_id` and `split` copied from the inventory; windows carry
`start_sec`/`end_sec` in the parent file, the actually-read `src_start_sec`/`src_end_sec`, and padding.
Pre-sliced synthetic speech (`tts_synthetic_continuous_speech`, originally cut from long
`_raw_sources/*.wav`) also gets `source_recording_offset_sec`, so a window maps back to its
position in the original long recording.

**Splits.** Splits are inherited, never recomputed. 6,961 files take their split from
`dataset_split` unchanged (checked). The 83 manual recordings were not in that split, so
`split_extension.csv` assigns them using the same `source_group` rule (here = `recording_id`),
stratified by category × recording device (Macmic / Phonemic / RPI, parsed from the filename and used
only for stratification). The largest-absolute-deficit rule prevents single long files from filling validation or test.
Result: positives 49 / 10 / 10 recordings, background 8 / 4 / 2 (train / validation / test).

**The split extension is append-only (2026-09-28).** Earlier assignments in `split_extension.csv` are frozen, and only
recordings never seen before are assigned, against their stratum's existing totals. `stage_inventory` raises if any
earlier group would move. Before this change, adding recordings could reshuffle existing manual recordings, because
the whole extension was recomputed on every run. With the current manifest the change is a verified no-op.

**Added 2026-09-28: Manual-BG-RPI14..19** (real RPI negatives, 16.3 min). The owner confirmed none contains
"Hey Delta". RPI16's ASR hits ("Hey!", "Hey Michelle", "Hey mishmash", "Hey Martha", "Hey, Marta") are marked
`non_wakeword`, so they are kept as real near-miss negatives. The content is conversation, singing and room sound.
Splits: RPI14, 15, 18, 19 → train; RPI16 → validation; RPI17 → test. An earlier upload of four files that were
entirely digital zeros was rejected before integration.
**Added 2026-09-29: Manual-Fil-RPI65..82 (except RPI72)**, RPI positive sessions with background media, music and AEC.
ASR cannot time them (repeated phrase plus media speech), so they were held (`hold_recordings`) until the owner marked
every "Hey Delta" by ear in `review/owner_timing_2026-09-29.csv`. That gives 95 `human-*` `add` rows (human-0040..0134)
over 16 files. RPI69 has no wakeword and is relabelled (below). Result: 173 positive windows from all 95
occurrences, plus 344 negatives, 152 of them from RPI69. Seven first-pass spans were longer than 1.4 s and could not fit a
window. The owner re-marked them tighter the same day (human-0120, 0124–0127, 0129, 0134; the old bounds are kept in each row's note). Manual-Fil-RPI72.wav is on disk but not in `manifest.csv`, so it is not integrated.
Streams get split `streaming_eval_holdout` and never mix with train/validation/test.

## Real recordings: ASR as candidate annotation

1. **Decoding unit.** Whole files ≤ 8 s are decoded whole. Longer files are cut with Silero VAD and **each
   speech chunk is decoded independently**. Decoding a 30 s file of repeated "Hey Delta" in one pass
   caused a Whisper repetition loop with zero-length word timestamps.
2. **Matching** (`detect_in_chunk`). Only an adjacent, exact `hey` + `delta` token pair is
   `complete`. Near matches ("Hey Dota", "Hey, Doga") are `fuzzy_complete`; a lone `hey`/`delta`-like
   word is `partial_fragment`; speech in a positive recording with no wakeword-like text is
   `unmatched_speech` (it may be a missed wakeword). Only `complete` can ever become positive.
3. **Quality gates** send a `complete` match to `needs_review`: word probability < 0.3, gap between words > 0.35 s,
   collapsed word timestamps, implausible duration, or a wakeword found in a negative recording.
4. **Boundary refinement.** Whisper word bounds are snapped to the nearby acoustic onset/offset
   (`refine_bounds`: local noise floor, rejects the ~40 ms click at the start of every RPI file).
   Raw Whisper bounds stay in `raw_start_sec`/`raw_end_sec`; `boundary_method` records which was used.
5. **Calibrated tolerance.** ASR bounds are widened by `[start − pre, end + post]` before any
   containment test. Tolerances come from `asr_crosscheck_synthetic.csv`, which runs the same ASR on the
   synthetic GT files. Those results are **never used as labels**.

   | Bounds | Needed on TTS GT (pre / post) | Configured |
   |---|---|---|
   | raw Whisper | 0.00 / 0.51 s (word ends are median 0.35 s early) | 0.10 / 0.55 |
   | energy-refined | 0.08 / 0.38 s (median end error −0.03 s) | 0.09 / 0.40 |
   | GT, human-drawn, human-corrected | — | 0 / 0 |

   `validate_windows.py` fails if a configured tolerance drops below the current calibration.
6. **Recording flags.** A positive-labelled recording with no `complete` match is flagged
   `positive_recording_without_complete_detection`. No negatives are taken from it until a reviewer
   clears the flag.

## Labelling rules (all in `window_config.json`)

| Parameter | Value | Rule |
|---|---|---|
| `window.duration_sec` | 1.5 | Matches `clip_len_sec` of every fixed clip already in the corpus |
| **Positive rule** | — | Window fully contains `[start − pre, end + post]` of exactly one *positive-eligible* annotation, with ≥ 0.05 s context before and after, and overlaps no other live wakeword-like annotation (also widened) |
| Positive eligibility | — | `match_type = complete` and status ∈ {ground_truth, candidate, reviewed}. Unreviewed ASR candidates are allowed but labelled `asr_candidate_unreviewed` (`asr_unreviewed_positive_policy = allow_flagged`; set `require_review` to exclude them) |
| Alignment | end-anchored | Wakeword end placed 0.10 / 0.25 / 0.40 s before window end; a single centred fallback when only a tight fit exists |
| Near-duplicates | ≤ 3 per occurrence, ≥ 0.15 s apart | Extra offsets logged as `near_duplicate` |
| Padding / truncation | zero-pad; ≤ 60 % of a window | Positives are never truncated: a span that does not fit is rejected (`occurrence_too_long_for_window`), not cut. Shorter-than-window negative clips get one centred padded window |
| Partial fragments | head/tail windows covering 35 % (max 50 %) of a wakeword, labelled **negative** | Only from verified bounds (GT, human, human-corrected). Unreviewed ASR bounds are too loose to guarantee the rest of the word is excluded |
| Negatives | hop 0.75 s (50 % overlap), guard 0.5 s around every live wakeword-like annotation (after tolerance), ≤ 700 per file (was 120 until 2026-09-28; see `window_config.json` `max_windows_note`) | Real positive recordings: non-speech regions only. Region shorter than a window → skipped (logged) |

Every positive window stores the `annotation_id` it came from and the wakeword position inside the window.
Every window has a `label_source`:
`synthetic_ground_truth`, `asr_candidate_unreviewed`, `asr_candidate_reviewed`, `human`,
`*_fragment`, `synthetic_generation_label`, `synthetic_ground_truth_absence`, `asr_no_detection_unreviewed`.
Anything not emitted is listed with a reason in `rejected_windows.csv`.

## Review workflow

1. Open `review/review_tool.html` **locally** (it plays the source WAVs via relative paths, so keep it inside the
   project tree). Recordings are sorted with flagged or `needs_review` ones first. The page shows the waveform, VAD chunks,
   colour-coded annotations, the ASR text and the review reasons.
2. For each annotation: ▶ play, edit start/end, then **✓ complete**, **bounds**, **partial**,
   **not wakeword** or **remove**. To add a missed wakeword, drag across the waveform and click **Add annotation**.
   Use **Clear flag** once a flagged recording has been checked.
3. **Export review_overrides.csv** and save it over `review/review_overrides.csv`. The file can also be edited by hand:
   columns `override_id, action, annotation_id, file_id, new_start_sec, new_end_sec, new_match_type, reviewer, reviewed_at, note`.
4. Re-run `resolve`, `windows`, `review-tool`, then `validate_windows.py`.

Guarantees, tested in `test_review_roundtrip`:
- Synthetic GT can only be `remove`d, never re-bounded.
- An override pointing at an unknown annotation is reported in `override_errors.txt` and fails validation.
- Confirming an ASR candidate without correcting its bounds keeps the ASR tolerance.
- Nothing edits `candidate_annotations.csv`; all changes are replayed from the overrides file.

## Current output (after the recording owner's label confirmation)

The person who recorded and labelled the manual recordings confirmed their labels.
`make_owner_overrides.py` turns that confirmation into 148 `owner-*` rows in `review/review_overrides.csv`
(59 confirm, 44 removals, 8 clip-level positives, 36 flags cleared, 1 not-wakeword); its docstring
spells out the rules. The label comes from the owner. The timing comes from ASR (with its tolerance), from
bounds the owner marked by ear, or, for short clips, from the whole file. Rerunning the script replaces only its own rows.

**Owner-marked bounds** are the 40 hand-made `human-*` `add` rows (2026-09-28): one span for each of 31 low-signal
RPI clips that ASR could not locate, plus 9 spans in `Manual-Fil-RPI23`. They are verified bounds (zero tolerance).
When a file has a hand-made `add` row, `make_owner_overrides.py` removes that file's ASR items and adds no
clip-level span. The `add` rows do not refer to generated annotation ids, so regenerating the `owner-*` rows cannot orphan them.

**Recording-level relabels** go in `review/recording_label_overrides.csv`
(`file_id, new_label, new_category, reviewer, reviewed_at, note`). They are applied after split assignment, so
they never move any recording between splits, and the manifest values are kept in `manifest_label`/`manifest_category`.
A relabelled file is windowed from its human label (`label_source = human_recording_label`), not from ASR.
Current relabels:
- **Manual-Fil-RPI13 → `negatives_confusable`**. After listening, the owner found that the
  recording timing cut off the "Hey"; the clip holds only "Delta".
- **Manual-Fil-RPI40 → `negatives_confusable`** (2026-09-28). After listening, the owner found it is not a complete "Hey Delta".
- **Manual-Fil-RPI69 → `negatives_media`** (2026-09-29). The owner listened: 115 s of background media and some confusables, no "Hey Delta".

| Split | Positive windows (GT / ASR-confirmed / owner-marked bounds / owner clip-level) | Negative windows |
|---|---|---|
| train | 400 (96 / 89 / 212 / 3) | 9,438 |
| validation | 64 (27 / 5 / 29 / 3) | 1,249 |
| test | 60 (27 / 4 / 25 / 4) | 1,290 |
| streaming_eval_holdout | 288 (288 / 0 / 0 / 0) | 4,354 |

- **Low-signal RPI positives the owner listened to and confirmed:** RPI19 (validation), RPI20 and RPI21 (test).
  Whisper scores these 0.000; they are kept on purpose because they reflect the deployment device's capture level.
- **Low-signal RPI clips timed by ear (2026-09-28).** Neither ASR, Whisper forced alignment nor a Whisper window scan
  could locate the utterance in 32 RPI clips of 1.49–2.68 s (best window score ≤ 0.07 on 30 of 32, versus a median
  of 0.74 on located real wakewords). The owner marked start/end for 31 of them, and RPI40 was relabelled (above).
  Each now gives 1–3 positive windows and, because its bounds are verified, partial-wakeword negatives.
- **`Manual-Fil-RPI23` (13.5 s)** holds 9 "Hey Delta"s (owner count and bounds). ASR had heard "Hey, Darla" and
  "King Dog House"; those items are removed. The file's spans are close together, so several windows are rejected for
  overlapping a neighbouring wakeword, and no negatives come from it.
- The review queue is empty.
- Clips ≤ 1.4 s use a `human_clip_extent` span [0, duration]. A window holding the whole clip is guaranteed to
  contain the wakeword. Because this span is not tight, no partial-wakeword fragments are cut from it.
- **Not yet hand-timed (optional; adds positives, nothing is mislabelled):** confirmed ASR wakewords whose span plus
  ASR tolerance is longer than a window (`occurrence_too_long_for_window`), so they give no positive windows:
  - 8 short clips: RPI2, 5, 28, 38, 39, 44, 46, 64 (RPI5 and RPI64 are test). Phonemic1 (test, 3–4.1 s) and Phonemic2
    (validation, 0.7–2.1 s) were timed by ear on 2026-09-29 (human-0135, human-0136) and now give 2 and 1 positive windows.
    Bounds marked by ear (an `add` row) would make them usable.
  - Macmic2: 4 of 14 wakewords (≈1.05, 22.86, 28.49, 35.89 s). Phonemic3: ≈17.82 s, and ≈4.3–7.7 s where ASR merged
    two wakewords into one 3.15 s span. For these, a `correct_bounds` row on the ASR annotation id (e.g.
    `Manual-Fil-Macmic2#asr000`) replaces the bounds; for the merged Phonemic3 stretch, `correct_bounds` on `#asr002` and `#asr003` gives the two separate spans.
- The owner confirmed by listening that the doubtful ASR transcripts that do give positive windows
  (Macmic2 ≈11.3 s "Bye Delta!", RPI15 "3 Delta") and Macmic2 ≈35.9 s ("Bye!") are complete "Hey Delta"s.

## Findings and limitations

- **Many RPI positives cannot be located by ASR.** 38 of 69 positive-labelled real recordings
  have no `complete` match with `large-v3` (`small` found fewer). Their labels are confirmed by the owner, and the owner has now marked bounds by ear for the ones that were too long to window. Most short RPI clips are at −46 to −50 dBFS RMS,
  and Whisper returns "Thank you." (a known hallucination) or nothing. Measured in the speech band, these clips peak about
  9 dB above the noise floor, versus 22 dB for RPI clips where the wakeword was found. This is how the deployment
  device captured them, and the owner confirmed several by listening. The capture level is a measured deployment
  condition that v1 of `../deployment_driven_augmentation_strategy` treated as unknown. Its v2 (2026-09-25) now
  measures it (`measure_deployment_levels.py`) and enables device-level gain, noise mixing at measured levels, and a device noise floor.
  The other 32 had no positive windows until the owner marked the utterance bounds by ear (2026-09-28; see above).
- **ASR calibration is on synthetic speech.** Tolerances are measured against TTS positives. Error on real
  speech is unknown until reviewed bounds exist; once some are corrected, compare them with `raw_*`/refined bounds and
  update `boundary_tolerance_sec`.
- **ASR recall on streaming_eval is low** (59 / 128 exact matches). This supports keeping ASR
  output as candidates rather than labels.
- **25 streaming-eval insertions are 1.872 s long**: Edge-TTS output including its silence padding. GT is used verbatim,
  so these cannot form a 1.5 s positive window and are logged as `occurrence_too_long_for_window`.
  The GT is not trimmed.
- **`manifest.csv` sample rates are wrong for 1,249 files.** All TTS positives and confusables are
  16 / 22.05 / 24 kHz on disk, but the manifest lists 48 kHz. The inventory stores both
  (`sample_rate`, `manifest_sample_rate`). Windows are defined in seconds, so labels are unaffected.
  `materialize` resamples everything to 16 kHz so bandwidth cannot act as a class cue.
- `source_id` is blank for manual recordings in the manifest and is preserved as blank; `source_group = recording_id`.
- Speaker-disjoint splitting is still not possible (see `dataset_split`). The manual recordings probably
  share one speaker, and this is reported, not solved.
- Window audio is not materialised by default: `windows.csv` is the manifest, and `materialize` writes WAVs on demand.
