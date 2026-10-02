# Augmentation config checks

24/24 passed

| check | status | detail |
|---|---|---|
| cited_rpi_positive_peak_db_matches_measurements | PASS | n=62, median=-39.77 |
| cited_rpi_positive_floor_db_matches_measurements | PASS | n=62, median=-54.20 |
| cited_rpi_positive_peak_over_floor_db_matches_measurements | PASS | n=62, median=14.75 |
| cited_owner_verified_lowest_matches | PASS | 4.42 dB |
| cited_level_gap_matches | PASS | 28.9 dB |
| gain_target_within_measured_rpi_peak_range | PASS | target [-49.7, -18.8] vs measured [-59.4, -18.8] |
| noise_hard_min_not_below_owner_verified_audible | PASS | hard_min 4.5 vs lowest verified 4.42 |
| noise_target_within_measured_rpi_range | PASS |  |
| floor_target_within_measured_rpi_floor_range | PASS |  |
| gain_applies_to_both_labels_with_one_probability | PASS | negative_confusable, negative_general_speech, negative_media, negative_partial_wakeword, negative_silence_noise, positive_wakeword |
| mic_coloring_unconditional_on_label | PASS | p=0.5, apply_to=['all categories'] |
| mic_coloring_is_mild | PASS |  |
| mic_coloring_runs_first | PASS |  |
| noise_floor_unconditional_on_label | PASS |  |
| digital_silence_absent_from_real_device_audio | PASS | motivates device_noise_floor |
| noise_bank_train_split_only | PASS | 1257 windows |
| noise_bank_excludes_all_zero_recordings | PASS | excluded ['Manual-BG-RPI1', 'Manual-BG-RPI2'] |
| noise_bank_windows_are_vetted_real_background_negatives | PASS |  |
| validation_test_streaming_not_augmented | PASS |  |
| preprocessing_rate_matches_window_export | PASS |  |
| clipping_rejection_supported | PASS | 0 of 89 real recordings clipped |
| every_enabled_transform_has_probability_and_gap | PASS |  |
| gain_precedes_noise_in_composition | PASS |  |
| v1_archived | PASS |  |
