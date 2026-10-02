#!/usr/bin/env python3
"""
Waveform measurements that the v1 strategy listed as unknown: level, signal
over background, clipping, sample rate, device click, and per-device spectrum.

Inputs (read-only):
  ../segmentation_windowing/outputs/recording_inventory.csv   split, label, device source, header SR
  ../segmentation_windowing/outputs/resolved_annotations.csv  located wakeword spans
  ../segmentation_windowing/outputs/windows.csv               vetted real-background windows
  the audio files themselves

Outputs:
  deployment_level_measurements.csv   one row per measured file
  deployment_level_summary.json       group statistics cited by the strategy
  real_noise_bank.csv                 train-split real background windows usable as noise

Metric definitions (all on a 150-4000 Hz band-pass, 20 ms frames, dB re full scale):
  peak_db            98th percentile frame level (speech-dominated frames)
  floor_db           20th percentile frame level (background-dominated frames)
  peak_over_floor_db peak_db - floor_db. Defined for every file, including files where
                     the wakeword is not located, so all groups are comparable. It is
                     NOT a conventional RMS SNR; the augmentation config uses the same
                     definition so targets and measurements stay comparable.
  span_level_db      90th percentile frame level inside a located wakeword span
  span_over_floor_db span_level_db - 20th percentile frame level outside all spans
Manual recordings skip the first 80 ms (device start click, measured separately).

Usage: python measure_deployment_levels.py
"""
from pathlib import Path
import json, re
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfilt

HERE = Path(__file__).resolve().parent
ROOT = (HERE / "../../..").resolve()
SEG = HERE / "../segmentation_windowing/outputs"
FRAME, BAND, SKIP_MANUAL = 0.02, (150.0, 4000.0), 0.08
SEED, N_PER_NEG_CATEGORY = 20260925, 150
DEVICE_RX = re.compile(r"^Manual-[^-]+-([A-Za-z]+?)\d*$")
OCTAVES = [125, 250, 500, 1000, 2000, 4000]


def frames_db(y, sr):
    n = int(FRAME * sr)
    k = len(y) // n
    if k == 0:
        return np.array([-120.0])
    return 20 * np.log10(np.sqrt((y[:k * n].reshape(k, n) ** 2).mean(1)) + 1e-9)


def octave_levels(y, sr):
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / sr)
    out = {}
    for c in OCTAVES:
        m = (f >= c / np.sqrt(2)) & (f < c * np.sqrt(2))
        out[f"oct_{c}_db"] = 10 * np.log10(spec[m].sum() + 1e-20) if m.any() else np.nan
    tot = np.nansum([10 ** (v / 10) for v in out.values()])
    return {k: v - 10 * np.log10(tot) for k, v in out.items()}  # relative spectrum shape


def measure(fp, spans, manual):
    y, sr = sf.read(str(ROOT / fp), always_2d=True, dtype="float32")
    y = y.mean(1)
    r = dict(header_sr=sr, duration_sec=len(y) / sr,
             clipped_fraction=float((np.abs(y) >= 0.999).mean()),
             nonzero_fraction=float((y != 0).mean()),
             fullband_rms_dbfs=float(20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-12)))
    if manual:
        head = frames_db(y[: int(0.06 * sr)], sr).max()
        body = np.median(frames_db(y[int(SKIP_MANUAL * sr):], sr))
        r["start_click_over_median_db"] = float(head - body)
    skip = int(SKIP_MANUAL * sr) if manual else 0
    hi = min(BAND[1], 0.45 * sr)
    yb = sosfilt(butter(4, [BAND[0], hi], "bandpass", fs=sr, output="sos"), y)
    db = frames_db(yb[skip:], sr)
    r["peak_db"], r["floor_db"] = float(np.percentile(db, 98)), float(np.percentile(db, 20))
    r["peak_over_floor_db"] = r["peak_db"] - r["floor_db"]
    if spans:
        t = skip / sr + (np.arange(len(db)) + 0.5) * FRAME
        inside = np.zeros(len(db), bool)
        for s, e in spans:
            inside |= (t >= s) & (t <= e)
        if inside.sum() >= 3 and (~inside).sum() >= 3:
            r["span_level_db"] = float(np.percentile(db[inside], 90))
            r["span_over_floor_db"] = r["span_level_db"] - float(np.percentile(db[~inside], 20))
        if manual and inside.any():
            idx = np.where(inside)[0]
            n = int(FRAME * sr)
            seg = yb[skip:][idx[0] * n:(idx[-1] + 1) * n]
            if len(seg) > n:
                r.update(octave_levels(seg, sr))
    return r


def group_of(row):
    if row.source == "manual_recording":
        dev = (DEVICE_RX.match(row.file_id) or [None, "unknown"])[1]
        lab = row.label
        if row.label_origin == "human_relabel":
            lab = f"{row.label} (relabelled)"
        return f"real {dev} {lab}"
    if row.source == "tts_synth":
        return "synthetic TTS positive"
    return f"synthetic {row.canonical_category}"


def main():
    inv = pd.read_csv(SEG / "recording_inventory.csv")
    ann = pd.read_csv(SEG / "resolved_annotations.csv")
    win = pd.read_csv(SEG / "windows.csv")
    live = ann[ann.status.ne("removed") & ann.match_type.eq("complete")]
    located = live[live.boundary_method.ne("human_clip_extent")]
    spans = located.groupby("file_id")[["start_sec", "end_sec"]].apply(lambda g: list(map(tuple, g.values))).to_dict()
    clip_level = set(live[live.boundary_method.eq("human_clip_extent")].file_id)

    real = inv[inv.source == "manual_recording"]
    tts = inv[inv.source == "tts_synth"]
    neg = (inv[(inv.annotation_mode == "generation_label") & (inv.label == "negative")]
           .groupby("canonical_category").sample(n=N_PER_NEG_CATEGORY, random_state=SEED))
    rows = []
    for r in pd.concat([real, tts, neg]).itertuples(index=False):
        m = measure(r.filepath, spans.get(r.file_id), r.source == "manual_recording")
        wake = ("located" if r.file_id in spans else
                "owner_confirmed_unlocated" if r.file_id in clip_level else
                "n/a" if r.label != "positive" else "unresolved")
        rows.append(dict(file_id=r.file_id, split=r.split, label=r.label, source=r.source,
                         group=group_of(r), wakeword=wake, **m))
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "deployment_level_measurements.csv", index=False)

    def stats(x):
        x = x.dropna()
        if x.empty:
            return None
        return {k: round(float(v), 2) for k, v in
                dict(n=len(x), min=x.min(), p10=x.quantile(.1), median=x.median(),
                     p90=x.quantile(.9), max=x.max()).items()}
    groups = {}
    for (g, w), s in d.groupby(["group", "wakeword"]):
        groups[f"{g} | wakeword={w}"] = {c: stats(s[c]) for c in
                                         ["fullband_rms_dbfs", "peak_db", "floor_db", "peak_over_floor_db",
                                          "span_level_db", "span_over_floor_db", "start_click_over_median_db"]
                                         if c in s and stats(s[c])}
    listened = ["Manual-Fil-RPI19", "Manual-Fil-RPI20", "Manual-Fil-RPI21"]
    rpi_pos = d[d.group.eq("real RPI positive")]
    spec = d[d.source.eq("manual_recording") & d.filter(like="oct_").notna().all(axis=1)]
    spec_cols = [c for c in d.columns if c.startswith("oct_")]
    summary = dict(
        definitions=__doc__.split("Metric definitions")[1].split("Usage")[0].strip(),
        groups=groups,
        owner_verified_audible_low_signal=d[d.file_id.isin(listened)][
            ["file_id", "split", "peak_db", "peak_over_floor_db"]].round(2).to_dict("records"),
        rpi_positive_peak_db_range=stats(rpi_pos.peak_db),
        rpi_positive_peak_over_floor_range=stats(rpi_pos.peak_over_floor_db),
        rpi_positive_floor_db_range=stats(rpi_pos.floor_db),
        level_gap_db=dict(
            tts_positive_peak_median=round(float(d[d.group.eq("synthetic TTS positive")].peak_db.median()), 2),
            rpi_positive_peak_median=round(float(rpi_pos.peak_db.median()), 2)),
        clipping=dict(files_with_any_clipped_sample=int((d.clipped_fraction > 0).sum()),
                      max_clipped_fraction=float(d.clipped_fraction.max())),
        header_sample_rates=d.groupby("group").header_sr.agg(lambda s: sorted(set(map(int, s)))).to_dict(),
        speech_spectrum_shape_by_device_db={
            dev: spec[spec.file_id.str.contains(f"-{dev}")][spec_cols].median().round(1).to_dict()
            for dev in ["RPI", "Macmic", "Phonemic"] if spec.file_id.str.contains(f"-{dev}").any()},
    )
    # Real background noise bank: train-split real-background windows only. They
    # already exclude every wakeword-like annotation plus a 0.5 s guard.
    # All-zero recordings (the device captured nothing) are not device noise.
    all_zero = set(d[d.nonzero_fraction == 0].file_id)
    bank = win[(win.window_type == "negative_real_background") & (win.split == "train")
               & ~win.file_id.isin(all_zero)]
    bank = bank[["window_id", "file_id", "filepath", "split", "src_start_sec", "src_end_sec"]]
    bank.to_csv(HERE / "real_noise_bank.csv", index=False)
    summary["real_noise_bank"] = dict(windows=len(bank), files=sorted(bank.file_id.unique()),
                                      excluded_non_train_files=sorted(
                                          win[(win.window_type == "negative_real_background") &
                                              (win.split != "train")].file_id.unique()),
                                      excluded_all_zero_files=sorted(all_zero))
    m = d[d.source == "manual_recording"]
    summary["all_zero_recordings"] = m[m.nonzero_fraction == 0][["file_id", "split", "label"]].to_dict("records")
    summary["start_click_over_10db"] = {g: f"{int((s.start_click_over_median_db > 10).sum())}/{len(s)}"
                                        for g, s in m.groupby("group")}
    summary["digital_silence_floor"] = {g: f"{int((s.floor_db < -100).sum())}/{len(s)}"
                                        for g, s in d.groupby("group")}
    summary["clipped_files_by_group"] = d[d.clipped_fraction > 0].groupby("group").size().to_dict()
    (HERE / "deployment_level_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"measured {len(d)} files; noise bank {len(bank)} windows")


if __name__ == "__main__":
    main()
