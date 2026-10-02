# Validation report

Config hash `4b08315cb6fe` · 61 checks · **0 failed**

| check | status | detail |
|---|---|---|
| inventory_covers_every_manifest_recording | PASS | 20892 inventory rows vs 20892 expected |
| frozen_dataset_split_assignments_unchanged | PASS | 6961 files in pre-existing source groups |
| normalized_manifest_splits_inherited | PASS |  |
| source_group_rule_reproduced | PASS | parent recording_id if parent_filepath else recording_id |
| no_source_group_crosses_splits | PASS | 12207 source_groups |
| no_recording_id_crosses_splits | PASS | 12257 recording_ids |
| parent_child_same_split | PASS | 800 derived files |
| streaming_eval_isolated_from_train_val_test | PASS | 120 eval-only files |
| recording_durations_match_audio_headers | PASS |  |
| annotation_timestamps_within_recording_and_ordered | PASS | 1177 annotations |
| annotation_ids_unique | PASS |  |
| annotation_inherits_filepath | PASS |  |
| annotation_inherits_recording_id | PASS |  |
| annotation_inherits_source_group | PASS |  |
| annotation_inherits_split | PASS |  |
| synthetic_recordings_have_no_asr_annotations | PASS | 803 annotations on 795 synthetic GT files |
| real_recordings_have_no_gt_annotations | PASS |  |
| streaming_gt_timestamps_verbatim | PASS | 120 streams |
| tts_whole_clip_gt_reproduced | PASS | energy-trim rule re-derived from audio |
| review_did_not_modify_gt_bounds | PASS |  |
| asr_annotations_labelled_as_candidates | PASS | 229 ASR annotations; label_sources=['asr_candidate_reviewed', 'human', 'synthetic_ground_truth'] |
| no_orphaned_or_invalid_review_overrides | PASS |  |
| positive_eligibility_rule | PASS |  |
| needs_review_never_positive_eligible | PASS |  |
| asr_tolerance_covers_calibrated_error | PASS | {"verified_bounds": {"pre": 0.0, "post": 0.0}, "asr_energy_refined": {"pre": 0.09, "post": 0.4}, "asr_raw": {"pre": 0.1, "post": 0.55}} |
| annotation_tolerances_match_config | PASS |  |
| asr_bounds_never_marked_verified_without_correction | PASS |  |
| window_ids_unique | PASS | 46606 windows |
| window_inherits_filepath | PASS |  |
| window_inherits_recording_id | PASS |  |
| window_inherits_source_group | PASS |  |
| window_inherits_split | PASS |  |
| window_inherits_is_eval_only | PASS |  |
| window_inherits_source_id | PASS |  |
| no_window_source_group_crosses_splits | PASS |  |
| window_duration_exact | PASS |  |
| window_source_span_and_padding_consistent | PASS |  |
| padding_within_limit_or_short_clip_policy | PASS | max_pad=0.900s |
| positive_windows_reference_existing_annotation | PASS | 2563 positive windows |
| positive_windows_only_from_eligible_complete_annotations | PASS |  |
| positive_windows_fully_contain_tolerance_expanded_wakeword | PASS |  |
| positive_window_label_source_matches_annotation | PASS |  |
| positive_windows_contain_no_other_wakeword_like_annotation | PASS |  |
| positive_label_sources_are_explicit | PASS | {'synthetic_ground_truth': 2186, 'human': 279, 'asr_candidate_reviewed': 98} |
| no_synthetic_recording_has_asr_labelled_windows | PASS |  |
| max_positive_windows_per_occurrence | PASS | max=3 |
| positive_windows_min_shift | PASS |  |
| negative_windows_outside_guarded_annotations | PASS | 42128 plain negatives |
| partial_fragment_windows_never_cover_complete_wakeword | PASS | 1915 fragment windows; max coverage 0.5 |
| partial_fragment_windows_only_from_verified_bounds | PASS |  |
| no_fragments_from_loose_clip_extent_bounds | PASS | excluded methods: ['human_clip_extent'] |
| clip_level_annotations_span_whole_file | PASS | 8 clip-level annotations |
| recording_relabels_applied_and_recorded | PASS | 5 relabelled recordings |
| relabelled_recordings_keep_manifest_values | PASS |  |
| relabelled_recordings_have_no_machine_annotations | PASS |  |
| relabelled_recording_windows_use_human_label | PASS | 163 windows |
| no_positive_windows_from_negative_labelled_recordings | PASS |  |
| partial_manifest_clips_are_negative | PASS |  |
| no_positive_without_annotation | PASS |  |
| no_negatives_from_unresolved_flagged_recordings | PASS | 0 flagged recordings |
| windows_generated_with_current_config | PASS |  |
