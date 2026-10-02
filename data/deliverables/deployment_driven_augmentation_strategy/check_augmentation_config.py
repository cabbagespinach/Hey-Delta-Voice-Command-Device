#!/usr/bin/env python3
"""
Check augmentation_config.json against the waveform measurements and the split.

Re-derives every measured number the config cites from
deployment_level_measurements.csv (not from the config itself), and checks that
parameters stay inside measured ranges, that level can never predict the label,
and that no validation/test/all-zero audio feeds the noise bank.
Writes augmentation_checks.md; exit code 1 on any failure.

Usage: python measure_deployment_levels.py && python check_augmentation_config.py
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEG = HERE / "../segmentation_windowing"
EPS = 0.05
POS, NEG = {"positive_wakeword"}, {"negative_general_speech", "negative_media", "negative_confusable",
                                   "negative_partial_wakeword", "negative_silence_noise"}
LISTENED = ["Manual-Fil-RPI19", "Manual-Fil-RPI20", "Manual-Fil-RPI21"]
rows = []


def check(name, ok, detail=""):
    rows.append((name, "PASS" if ok else "FAIL", detail))


def main():
    cfg = json.loads((HERE / "augmentation_config.json").read_text())
    d = pd.read_csv(HERE / "deployment_level_measurements.csv")
    bank = pd.read_csv(HERE / "real_noise_bank.csv")
    inv = pd.read_csv(SEG / "outputs/recording_inventory.csv").set_index("file_id")
    win = pd.read_csv(SEG / "outputs/windows.csv").set_index("window_id")
    seg_cfg = json.loads((SEG / "window_config.json").read_text())
    T = {t["name"]: t for t in cfg["enabled_transforms"]}
    mv = cfg["evidence_scope"]["measured_values"]

    rpi = d[d.group.eq("real RPI positive")]
    q = lambda s, p: float(s.quantile(p))
    # Measured values cited in the config match the measurements.
    for key, col in [("rpi_positive_peak_db", "peak_db"), ("rpi_positive_floor_db", "floor_db"),
                     ("rpi_positive_peak_over_floor_db", "peak_over_floor_db")]:
        s = rpi[col]
        ok = mv[key]["n"] == len(s) and all(abs(mv[key][k] - v) <= EPS for k, v in
                                            dict(min=s.min(), p10=q(s, .1), median=s.median(),
                                                 p90=q(s, .9), max=s.max()).items())
        check(f"cited_{key}_matches_measurements", ok, f"n={len(s)}, median={s.median():.2f}")
    low = d[d.file_id.isin(LISTENED)].peak_over_floor_db.min()
    check("cited_owner_verified_lowest_matches", abs(mv["owner_verified_audible_lowest_peak_over_floor_db"] - low) <= EPS,
          f"{low:.2f} dB")
    tts = d[d.group.eq("synthetic TTS positive")].peak_db.median()
    check("cited_level_gap_matches", abs(mv["level_gap_tts_vs_rpi_db"] - (tts - rpi.peak_db.median())) <= 0.1,
          f"{tts - rpi.peak_db.median():.1f} dB")

    # Parameters inside measured ranges.
    g = T["device_level_gain"]["parameters"]["target_peak_db"]
    check("gain_target_within_measured_rpi_peak_range",
          rpi.peak_db.min() - EPS <= g["min"] < g["max"] <= rpi.peak_db.max() + EPS,
          f"target [{g['min']}, {g['max']}] vs measured [{rpi.peak_db.min():.1f}, {rpi.peak_db.max():.1f}]")
    n = T["background_noise_mixing"]["parameters"]["target_peak_over_floor_db"]
    check("noise_hard_min_not_below_owner_verified_audible", n["hard_min"] >= low - 1e-9 and n["hard_min"] <= n["min"],
          f"hard_min {n['hard_min']} vs lowest verified {low:.2f}")
    check("noise_target_within_measured_rpi_range",
          rpi.peak_over_floor_db.min() - EPS <= n["min"] < n["max"] <= rpi.peak_over_floor_db.max() + EPS)
    f = T["device_noise_floor"]["parameters"]["target_floor_db"]
    check("floor_target_within_measured_rpi_floor_range",
          rpi.floor_db.min() - EPS <= f["min"] < f["max"] <= rpi.floor_db.max() + EPS)

    # Level cannot predict the label.
    cats = set(T["device_level_gain"]["apply_to"])
    check("gain_applies_to_both_labels_with_one_probability",
          bool(cats & POS) and bool(cats & NEG) and isinstance(T["device_level_gain"]["probability"], (int, float)),
          ", ".join(sorted(cats)))
    mc = T["random_mic_coloring"]
    check("mic_coloring_unconditional_on_label", mc["apply_to"] == ["all categories"] and
          isinstance(mc["probability"], (int, float)) and 0 < mc["probability"] < 1,
          f"p={mc['probability']}, apply_to={mc['apply_to']}")
    check("mic_coloring_is_mild", all(abs(mc["parameters"][b]["gain_db"][k]) <= 6
                                      for b in ("low_shelf", "high_shelf", "peak") for k in ("min", "max")))
    check("mic_coloring_runs_first", cfg["composition_rules"]["order"][1] == "random_mic_coloring")
    nf = T["device_noise_floor"]
    check("noise_floor_unconditional_on_label", nf["apply_to"] == ["all categories"] and nf["probability"] == 1.0)
    check("digital_silence_absent_from_real_device_audio",
          int((rpi.floor_db < -100).sum()) == 0, "motivates device_noise_floor")

    # Noise bank provenance.
    check("noise_bank_train_split_only", bank.split.eq("train").all() and
          bank.file_id.map(inv.split).eq("train").all(), f"{len(bank)} windows")
    zero = set(d[d.nonzero_fraction == 0].file_id)
    check("noise_bank_excludes_all_zero_recordings", not bank.file_id.isin(zero).any(), f"excluded {sorted(zero)}")
    ok = bank.window_id.isin(win.index).all() and \
        win.loc[bank.window_id, "window_type"].eq("negative_real_background").all() and \
        win.loc[bank.window_id, "label"].eq("negative").all()
    check("noise_bank_windows_are_vetted_real_background_negatives", bool(ok))

    # Split / preprocessing / clipping rules.
    si = cfg["split_inheritance"]
    check("validation_test_streaming_not_augmented",
          "No augmentation" in si["validation"] and "No augmentation" in si["test"] and "Never" in si["streaming_eval_holdout"])
    check("preprocessing_rate_matches_window_export",
          cfg["preprocessing"]["resample_all_to_hz"] == seg_cfg["window"]["output_sample_rate"])
    real = d[d.source.eq("manual_recording")]
    check("clipping_rejection_supported", int((real.clipped_fraction > 0).sum()) == 0,
          f"{int((real.clipped_fraction > 0).sum())} of {len(real)} real recordings clipped")
    check("every_enabled_transform_has_probability_and_gap",
          all(isinstance(t.get("probability"), (int, float)) and t.get("deployment_gap") for t in T.values()))
    check("gain_precedes_noise_in_composition",
          cfg["composition_rules"]["order"].index("device_level_gain") <
          cfg["composition_rules"]["order"].index("background_noise_mixing"))
    check("v1_archived", (HERE / "archive/v1_2026-09-25/augmentation_config.json").exists())

    lines = ["# Augmentation config checks", "", f"{sum(r[1] == 'PASS' for r in rows)}/{len(rows)} passed", "",
             "| check | status | detail |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    (HERE / "augmentation_checks.md").write_text("\n".join(lines) + "\n")
    for a, b, c in rows:
        print(f"{b}  {a}  {c}")
    failed = [r for r in rows if r[1] == "FAIL"]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
