#!/usr/bin/env python3
"""
Training-time augmentation for the command classifier: make synthetic and public-dataset clips sound like the Pi
(owner-approved 2026-09-30). Reuses the wakeword project's DeploymentAugmenter
(../dataloading/augmentation.py, config ../deployment_driven_augmentation_strategy/augmentation_config.json v4):

    random mic colouring -> level to the measured RPI speech range -> room echo (MIT IR bank, 132 rooms)
    -> real background from the owner's rooms at the measured RPI speech-over-background range -> device noise floor

Differences from the wakeword setup:
- Every transform applies to every class with the same probability, so no condition (level, noise, echo, colour)
  can hint at a class. The owner's real Pi recordings are augmented too, but never get room echo (they already
  have a real room), and background mixing skips them when they are already as noisy as the target.
- Background comes from the owner's room recordings reused as `unknown_room` (data/commands_unknown_reuse), TRAIN
  split only, with every command word / "hey" / "delta" already cut out. The windows of one recording are joined
  in time order into one stream, so a background is rarely tiled.

    from augment_commands import CommandAugmenter
    aug = CommandAugmenter()                     # loads banks once
    y, applied = aug(x, rng, owner_recording=False)

Check / preview (CPU only):  python augment_commands.py --preview 40
  writes data/commands_aug_preview/*.wav and compares level statistics (speech peak, background floor,
  peak over floor) of the owner's recordings with synthetic / public clips before and after augmentation.
"""
from pathlib import Path
import argparse, sys

import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "dataloading"))
import augmentation as wk                                              # noqa: E402

SR = 16000
REUSE = ROOT / "data/commands_unknown_reuse"
RIR_BANK = ROOT / "data/rir/rir_bank.csv"


def _load16(path):
    y, sr = sf.read(str(path), dtype="float64", always_2d=True)
    y = y.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        y = resample_poly(y, SR, sr)
    return y


class _AllClasses(wk.DeploymentAugmenter):
    def _applies(self, name, window_type):              # same transforms for every class
        return True


def _bed_streams(label):
    """TRAIN-split reused clips' source recordings of one kind (unknown_speech: FLEURS + MUSAN speech; unknown_music:
    MUSAN music), each source file whole, joined into one int16 stream (shared copy-on-write by the loader workers)."""
    m = pd.read_csv(REUSE / "manifest.csv")
    files = sorted(set(m[(m.label == label) & (m.source_split == "train")].origin_file))
    y = np.concatenate([_load16(ROOT / f) for f in files])
    return (np.clip(y, -1, 1) * 32767).astype(np.int16)


class CommandAugmenter:
    def __init__(self, owner_extra_background: dict | None = None, background_talk: dict | None = None):
        self.extra = owner_extra_background            # see dataloader_config.json "owner_extra_background"
        self.talk = background_talk                    # see dataloader_config.json "background_talk"
        if self.talk:
            self.beds = {"talk": _bed_streams("unknown_speech"), "music": _bed_streams("unknown_music")}
        m = pd.read_csv(REUSE / "manifest.csv")
        room = m[(m.label == "unknown_room") & (m.source_split == "train")].sort_values(["origin_file", "offset_sec"])
        bank = {f"{Path(f).stem}@stream": np.concatenate([_load16(REUSE / x) for x in g.file])
                for f, g in room.groupby("origin_file")}
        rirs = []
        for r in pd.read_csv(RIR_BANK).itertuples(index=False):
            h = _load16(ROOT / r.filepath)
            rirs.append(dict(id=r.id, wave=h / np.abs(h).max(), rt60_s=float(r.rt60_s)))
        self.aug = _AllClasses(SR, bank, rir_bank=rirs)
        self.bank_minutes = sum(len(v) for v in bank.values()) / SR / 60

    def __call__(self, x, rng, owner_recording=False):
        meta = dict(window_type="command", parent_source="manual_recording" if owner_recording else "other",
                    file_id="RPI" if owner_recording else "")
        y, applied = self.aug(x, rng, meta)
        if owner_recording and self.extra and rng.random() < self.extra["probability"]:
            # a second real room layer at its own recorded level (the bank is owner-room Pi audio), so the owner's
            # voice is also heard over room sounds; every owner class gets it alike
            noise, wid = self.aug._bank_noise(len(y), rng)
            g = float(rng.uniform(*self.extra["gain_db"]))
            y = y + (noise * 10 ** (g / 20)).astype(np.float32)
            m = np.abs(y).max() if len(y) else 0.0
            if m > 0.999:
                y *= 0.999 / m
            applied["owner_extra_background"] = dict(source=f"real:{wid}", gain_db=round(g, 2))
        return y, applied

    def background_talk(self, y, rng, n_full):
        """TV / radio under the capture (added 2026-10-01 after the Pi TV session): talk (or music) from a TRAIN-split
        bed at `speech_over_bed_db` below the clip's speech peak, optionally through a room response. With
        `full_window_probability` the capture first runs on to n_full samples (the model's 5 s window) with its own
        room sound, and the bed plays to the end, like a capture that only stopped at its time limit. Every class
        alike (unknown_silent + bed = TV alone, still unknown). Returns (y, applied) or (y, None) when not applied."""
        t = self.talk
        if rng.random() >= t["probability"] or len(y) == 0:
            return y, None
        pk = wk.peak_db(y, SR)
        if pk <= wk.SILENT_DB:
            return y, None
        applied = {}
        if len(y) < n_full and rng.random() < t["full_window_probability"]:
            room, _ = self.aug._bank_noise(n_full - len(y), rng)
            tail = y[-int(0.3 * SR):]
            g = wk.floor_db(tail, SR) - wk.floor_db(room, SR) if len(tail) >= int(0.1 * SR) else -60.0
            y = np.concatenate([y, (room * 10 ** (g / 20)).astype(np.float32)])
            applied["full_window"] = True
        kind = "talk" if rng.random() < t["talk_share"] else "music"
        s = self.beds[kind]
        off = int(rng.integers(0, len(s) - len(y)))
        bed = s[off:off + len(y)].astype(np.float64) / 32767
        if self.aug.rirs and rng.random() < t["rir_probability"]:
            from scipy.signal import fftconvolve
            r = self.aug.rirs[int(rng.integers(len(self.aug.rirs)))]["wave"]
            bed = fftconvolve(bed, r)[:len(y)]
        bpk = wk.peak_db(bed, SR)
        if bpk <= wk.SILENT_DB:
            return y, None
        margin = float(rng.uniform(*t["speech_over_bed_db"]))
        y = y + (bed * 10 ** ((pk - margin - bpk) / 20)).astype(np.float32)
        m = np.abs(y).max()
        if m > 0.999:
            y *= 0.999 / m
        applied.update(kind=kind, speech_over_bed_db=round(margin, 1))
        return y.astype(np.float32), applied


def levels(y):
    pk, fl = wk.peak_db(y, SR), wk.floor_db(y, SR)
    return pk, fl, pk - fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=int, default=40, help="augmented examples to write for listening")
    ap.add_argument("--stats", type=int, default=300, help="clips per group for the level comparison")
    a = ap.parse_args()
    rng = np.random.default_rng(20260930)
    aug = CommandAugmenter()
    print(f"background bank: {len(aug.aug.bank_ids)} owner room recordings, {aug.bank_minutes:.1f} min (train only); "
          f"{len(aug.aug.rirs)} room responses")

    rec = HERE.parent / "model/deploy/recordings"
    groups = {
        "owner Pi recordings": [(rec / f, True) for f in pd.read_csv(rec / "manifest.csv").file],
        "synthetic": [(ROOT / "data/commands_synthetic" / f, False) for f in pd.read_csv(ROOT / "data/commands_synthetic/manifest.csv").file],
        "public datasets": [(ROOT / "data/commands_real_web" / f, False) for f in pd.read_csv(ROOT / "data/commands_real_web/manifest.csv").file],
    }
    out = ROOT / "data/commands_aug_preview"
    out.mkdir(parents=True, exist_ok=True)
    stats, n_prev = [], 0
    for g, files in groups.items():
        pick = rng.choice(len(files), size=min(a.stats, len(files)), replace=False)
        for i in pick:
            p, own = files[i]
            y = _load16(p)
            stats.append((g, "before", *levels(y)))
            ya, applied = aug(y, rng, owner_recording=own)
            stats.append((g, "after", *levels(ya)))
            if g != "owner Pi recordings" and n_prev < a.preview:
                sf.write(str(out / f"{n_prev:02d}_{p.stem}__orig.wav"), y.astype(np.float32), SR, subtype="PCM_16")
                sf.write(str(out / f"{n_prev:02d}_{p.stem}__aug.wav"), ya, SR, subtype="PCM_16")
                n_prev += 1
    s = pd.DataFrame(stats, columns=["group", "when", "speech_peak_db", "background_db", "peak_over_bg_db"])
    t = s.groupby(["group", "when"]).median().round(1)
    t = t.join(s.groupby(["group", "when"]).quantile(0.1).round(1), rsuffix="_p10")
    t = t.join(s.groupby(["group", "when"]).quantile(0.9).round(1), rsuffix="_p90")
    print("\nmedian (p10 / p90) per group; the target is to match the owner's Pi recordings 'before':")
    for (g, w), r in t.iterrows():
        print(f"  {g:22} {w:6}  speech peak {r.speech_peak_db:6.1f} ({r.speech_peak_db_p10:6.1f}/{r.speech_peak_db_p90:6.1f})"
              f"   background {r.background_db:6.1f} ({r.background_db_p10:6.1f}/{r.background_db_p90:6.1f})"
              f"   peak-over-bg {r.peak_over_bg_db:5.1f} ({r.peak_over_bg_db_p10:5.1f}/{r.peak_over_bg_db_p90:5.1f})")
    s.to_csv(out / "level_stats.csv", index=False)
    print(f"\npreview: {n_prev} pairs (__orig / __aug) in {out}")


if __name__ == "__main__":
    main()
