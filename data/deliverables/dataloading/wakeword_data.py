#!/usr/bin/env python3
"""
Source-aware Dataset / DataLoader layer for "Hey Delta" training.

    windows.csv (+ recording inventory, normalized manifest)   -> metadata per window
    SplitGuard                                                  -> source-level split boundaries, enforced
    WakewordWindowDataset                                       -> load source audio, [train] augment, preprocess
    BucketQuotaSampler                                          -> training epoch mix (hard vs easy negatives)
    build_dataloaders()                                         -> DataLoaders for train / validation / test

Every example carries its metadata (window id, file, time span, label, category,
source, recording, source group, speaker, split, augmentation applied), so any
batch element can be traced back to, and regenerated from, its source audio.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import json, math, sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "preprocessing"))
import wakeword_preprocessing as wp          # noqa: E402
import augmentation as aug                   # noqa: E402
import reproducibility as rep                # noqa: E402

DEFAULT_CONFIG = HERE / "dataloader_config.json"
TRAIN, EVAL_ONLY_SPLIT = "train", "streaming_eval_holdout"
SPLITS = (TRAIN, "validation", "test", EVAL_ONLY_SPLIT)


class SplitBoundaryError(RuntimeError):
    """Raised when data would cross a source-level split boundary."""


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
def load_config(path=DEFAULT_CONFIG) -> dict:
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    cfg["_inputs"] = {k: (path.parent / v).resolve() for k, v in cfg["inputs"].items()}
    return cfg


# ----------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------
EXTERNAL_SOURCE_PREFIX = "ext_"      # real human audio from public corpora (MSWC, FLEURS, MUSAN)


def _device(file_id: str, source: str) -> str:
    if str(source).startswith(EXTERNAL_SOURCE_PREFIX):
        return "external"
    if source != "manual_recording":
        return "synthetic"
    for d in ("RPI", "Macmic", "Phonemic"):
        if f"-{d}" in file_id:
            return d
    return "unknown"


def load_windows(cfg: dict) -> pd.DataFrame:
    """windows.csv joined with recording-level and manifest-level metadata."""
    I = cfg["_inputs"]
    w = pd.read_csv(I["windows_csv"])
    inv = pd.read_csv(I["recording_inventory"])[
        ["file_id", "category", "canonical_category", "source", "label_origin", "manifest_label",
         "manifest_category", "split_origin"]]
    nm = pd.read_csv(I["normalized_manifest"])[
        ["filepath", "speaker_id_normalized", "session_id", "voice", "provenance_type", "transcription"]]
    w = w.merge(inv, on="file_id", how="left", validate="many_to_one")
    w = w.merge(nm, on="filepath", how="left", validate="many_to_one")
    # Files added after dataset_split was frozen (manual recordings, public corpora, the 2026-09-29 TTS
    # voices) are not in its normalized manifest: take their speaker from the window itself (segmentation
    # copies it from manifest.csv), which is the id their split group was built from.
    w["speaker_id_normalized"] = w.speaker_id_normalized.fillna(w.speaker_id)
    if w.source.isna().any():
        raise ValueError(f"{int(w.source.isna().sum())} windows have no recording_inventory row")
    w["device"] = [_device(f, s) for f, s in zip(w.file_id, w.source)]
    w["eval_subset"] = np.select([w.device == "RPI", w.device == "synthetic", w.device == "external"],
                                 ["real_device_rpi", "synthetic", "real_external"], "real_other_device")
    w["label_int"] = (w.label == "positive").astype(int)
    return w


def _clean(v):
    if isinstance(v, (float, np.floating)) and math.isnan(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


# ----------------------------------------------------------------------------
# Split boundaries
# ----------------------------------------------------------------------------
class SplitGuard:
    """Source-level split enforcement.

    The frozen split is dataset_split/split_assignments.csv plus
    segmentation_windowing/outputs/split_extension.csv (source_group -> split).
    """

    def __init__(self, cfg: dict):
        I = cfg["_inputs"]
        self.frozen = {}
        for p in (I["split_assignments"], I["split_extension"]):
            d = pd.read_csv(p)
            clash = {g for g, s in zip(d.source_group, d.split) if self.frozen.get(g, s) != s}
            if clash:
                raise SplitBoundaryError(f"split files disagree for source groups {sorted(clash)[:5]}")
            self.frozen.update(zip(d.source_group, d.split))
        self.inventory = pd.read_csv(I["recording_inventory"]).set_index("filepath")

    def check_windows(self, w: pd.DataFrame):
        """Every window sits in its source group's frozen split; no source group or
        audio file appears in two splits; eval-only windows stay in the eval-only split."""
        errors = []
        main = w[~w.is_eval_only.astype(bool)]
        frozen = main.source_group.map(self.frozen)
        if frozen.isna().any():
            errors.append(f"{int(frozen.isna().sum())} windows have a source_group missing from the frozen split")
        bad = main[frozen.notna() & (frozen != main.split)]
        if len(bad):
            errors.append(f"{len(bad)} windows disagree with the frozen split, e.g. {bad.window_id.iloc[0]} "
                          f"({bad.split.iloc[0]} vs frozen {self.frozen[bad.source_group.iloc[0]]})")
        ev = w[w.is_eval_only.astype(bool)]
        if (ev.split != EVAL_ONLY_SPLIT).any():
            errors.append("eval-only windows outside the streaming_eval_holdout split")
        if len(ev) and ev.source_group.isin([g for g, s in self.frozen.items()]).any():
            errors.append("a streaming_eval source group also appears in the train/validation/test split")
        for col in ("source_group", "filepath"):
            n = w.groupby(col).split.nunique()
            if (n > 1).any():
                errors.append(f"{int((n > 1).sum())} {col} values span more than one split, e.g. {n[n > 1].index[0]}")
        if errors:
            raise SplitBoundaryError("; ".join(errors))

    def check_rows(self, rows: pd.DataFrame, split: str):
        if split not in SPLITS:
            raise ValueError(f"unknown split {split}")
        if (rows.split != split).any():
            raise SplitBoundaryError(f"dataset for {split} received rows from {sorted(set(rows.split) - {split})}")
        if split != EVAL_ONLY_SPLIT:
            f = rows.source_group.map(self.frozen)
            if (f != split).any():
                raise SplitBoundaryError(f"{int((f != split).sum())} {split} rows belong to source groups frozen elsewhere")

    def check_noise_bank(self, bank: pd.DataFrame):
        """Augmentation noise may only come from train-split source groups."""
        if (bank.split != TRAIN).any():
            raise SplitBoundaryError("noise bank lists non-train windows")
        groups = bank.filepath.map(self.inventory.source_group)
        if groups.isna().any() or (groups.map(self.frozen) != TRAIN).any():
            raise SplitBoundaryError("noise bank audio comes from a non-train source group")


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
META_COLUMNS = [
    "window_id", "split", "is_eval_only", "filepath", "file_id", "start_sec", "end_sec",
    "src_start_sec", "src_end_sec", "pad_left_sec", "pad_right_sec", "label", "label_int", "window_type",
    "label_source", "annotation_id", "wakeword_start_in_window_sec", "wakeword_end_in_window_sec",
    "wakeword_coverage", "category", "canonical_category", "manifest_label", "manifest_category",
    "label_origin", "source", "parent_source", "provenance_type", "source_id", "recording_id", "source_group",
    "speaker_id", "speaker_id_normalized", "session_id", "voice", "transcription", "device", "eval_subset",
    "recording_duration_sec", "source_recording_offset_sec", "split_origin", "config_hash",
]


class WakewordWindowDataset(Dataset):
    """One split's windows -> (features [1, n_mels, n_frames], label float, metadata dict).

    Keys: an int index (fixed preprocessing), or (index, epoch, draw) from
    BucketQuotaSampler. With an augmenter (train only), the augmentation RNG is
    keyed by (seed, epoch, draw, window_id): stochastic across draws and epochs,
    reproducible for a given key, independent of worker count.
    """

    def __init__(self, split: str, windows: pd.DataFrame, guard: SplitGuard,
                 preprocessor: wp.WakewordPreprocessor, augmenter=None, seed: int = 0, cache_files: int = 256):
        if augmenter is not None and split != TRAIN:
            raise ValueError(f"augmentation is training-only; refused for split {split!r}")
        rows = windows[windows.split == split].reset_index(drop=True)
        guard.check_rows(rows, split)
        self.split, self.rows = split, rows
        self.pre, self.cfg = preprocessor, preprocessor.cfg
        self.augmenter, self.seed = augmenter, seed
        self.meta = [{k: _clean(v) for k, v in r.items()} for r in rows[META_COLUMNS].to_dict("records")]
        self._cache, self._cache_size = OrderedDict(), cache_files

    def __len__(self):
        return len(self.rows)

    def _wave(self, filepath: str) -> torch.Tensor:
        if filepath in self._cache:
            self._cache.move_to_end(filepath)
            return self._cache[filepath]
        w = wp.load_waveform(wp.PROJECT_ROOT / filepath, self.cfg)
        self._cache[filepath] = w
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return w

    def load(self, key):
        """(framed waveform [num_samples], metadata) for a key. Call inside rep.single_thread()
        (as __getitem__ does) for results that do not depend on the process's thread count."""
        i, epoch, draw = (key, 0, key) if isinstance(key, (int, np.integer)) else key
        m = dict(self.meta[int(i)])
        wave = self._wave(m["filepath"])
        sr, N = self.cfg.sample_rate, self.cfg.num_samples
        start = int(round(m["start_sec"] * sr))
        a, b = max(0, start), min(wave.numel(), start + N)
        excerpt = wave[a:b]
        applied = {}
        frame = wp.frame_at(excerpt, start - a, self.cfg, pad_key=m["window_id"])
        if self.augmenter is not None:
            # Augment the whole framed window, padding included (2026-09-29). Augmenting only the excerpt
            # left the padded flanks of short clips at the fixed pad-noise floor while the word itself got
            # background, so "speech with quiet flanks" became a positive cue: most TTS positives are ~1 s
            # clips (0.68 s of padding per window on average), negatives are mostly continuous audio.
            # Validation/test are never augmented, so their windows are unchanged.
            y, applied = self.augmenter(frame.numpy(), rep.example_rng(self.seed, epoch, draw, m["window_id"]), m)
            frame = torch.from_numpy(y)
        m.update(index=int(i), epoch=int(epoch) if self.augmenter is not None else None,
                 draw=int(draw) if self.augmenter is not None else None,
                 augmented=any(not (isinstance(v, dict) and "skipped" in v) for v in applied.values()),
                 augmentation=json.dumps(applied, sort_keys=True) if applied else "")
        return frame, m

    def __getitem__(self, key):
        with rep.single_thread():
            frame, m = self.load(key)
            x = self.pre(frame)
        return x, torch.tensor(float(m["label_int"])), m


def collate(batch) -> dict:
    return {"features": torch.stack([b[0] for b in batch]),
            "labels": torch.stack([b[1] for b in batch]),
            "meta": [b[2] for b in batch]}


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------
SOURCE_KINDS = ("real", "synthetic", "external")


def source_kind(rows: pd.DataFrame) -> np.ndarray:
    """'real' for our own device/microphone recordings, 'external' for public-corpus human audio
    (source 'ext_*'), 'synthetic' for everything generated."""
    src = rows.source.astype(str)
    return np.select([src == "manual_recording", src.str.startswith(EXTERNAL_SOURCE_PREFIX)],
                     ["real", "external"], "synthetic")


class BucketQuotaSampler(Sampler):
    """Training epoch = epoch_size draws; bucket b gets round(share_b * epoch_size) of them.

    Within a bucket, weight(window) = 1 / n_windows(source_group) ** alpha.
    Yields (index, epoch, draw_position). Call set_epoch(e) before each epoch.
    """

    def __init__(self, rows: pd.DataFrame, sampling: dict, seed: int):
        self.seed, self.epoch = seed, 0
        self.replacement = bool(sampling.get("replacement", True))
        alpha = float(sampling["source_group_exponent"])
        buckets = sampling["buckets"]
        total = sum(b["share"] for b in buckets.values())
        if abs(total - 1) > 1e-6:
            raise ValueError(f"bucket shares sum to {total}, not 1")
        # A bucket owns (window_type, source_kind) pairs; "sources" defaults to both kinds.
        owner = {}
        for name, b in buckets.items():
            for t in b["window_types"]:
                for k in b.get("sources", SOURCE_KINDS):
                    if (t, k) in owner:
                        raise ValueError(f"({t}, {k}) is in buckets {owner[(t, k)]} and {name}")
                    owner[(t, k)] = name
        keys = list(zip(rows.window_type, source_kind(rows)))
        missing = set(keys) - set(owner)
        if missing:
            raise ValueError(f"(window type, source) pairs without a sampling bucket: {sorted(missing)}")
        self.bucket_of = np.array([owner[k] for k in keys])
        es = sampling.get("epoch_size", "train_size")
        self.epoch_size = len(rows) if es == "train_size" else int(es)
        # largest-remainder rounding: counts sum exactly to epoch_size, deterministically
        raw = {n: b["share"] * self.epoch_size for n, b in buckets.items()}
        counts = {n: int(math.floor(v)) for n, v in raw.items()}
        for n in sorted(raw, key=lambda n: (-(raw[n] - counts[n]), n))[: self.epoch_size - sum(counts.values())]:
            counts[n] += 1
        self.buckets = {}
        groups = rows.source_group.to_numpy()
        for name, b in buckets.items():
            idx = np.where(self.bucket_of == name)[0]
            if len(idx) == 0:
                raise ValueError(f"bucket {name} has no training windows")
            if not self.replacement and counts[name] > len(idx):
                raise ValueError(f"bucket {name} needs {counts[name]} draws but has {len(idx)} windows")
            size_in_bucket = pd.Series(groups[idx]).map(pd.Series(groups[idx]).value_counts()).to_numpy()
            wgt = 1.0 / size_in_bucket ** alpha
            self.buckets[name] = dict(role=b["role"], share=b["share"], count=counts[name], index=idx,
                                      weight=torch.tensor(wgt / wgt.sum(), dtype=torch.float64))
        self.rows = rows

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.epoch_size

    def epoch_indices(self, epoch: int | None = None) -> np.ndarray:
        e = self.epoch if epoch is None else epoch
        g = torch.Generator().manual_seed(rep.stable_seed("sampler", self.seed, e))
        parts = []
        for name in sorted(self.buckets):
            b = self.buckets[name]
            pick = torch.multinomial(b["weight"], b["count"], replacement=self.replacement, generator=g)
            parts.append(b["index"][pick.numpy()])
        idx = np.concatenate(parts)
        return idx[torch.randperm(len(idx), generator=g).numpy()]

    def __iter__(self):
        e = self.epoch
        for pos, i in enumerate(self.epoch_indices(e)):
            yield (int(i), e, pos)


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
def build_datasets(cfg_path=DEFAULT_CONFIG, splits=None, seed=None, augment=None):
    cfg = load_config(cfg_path)
    I = cfg["_inputs"]
    seed = cfg["seed"] if seed is None else seed
    splits = [TRAIN] + cfg["evaluation"]["splits"] if splits is None else list(splits)
    windows = load_windows(cfg)
    guard = SplitGuard(cfg)
    guard.check_windows(windows)
    pcfg = wp.PreprocessConfig.from_json(I["preprocessing_config"])
    pre = wp.WakewordPreprocessor(pcfg, I["normalization_stats"])
    use_aug = cfg["augmentation"]["enabled_for_train"] if augment is None else augment
    augmenter = None
    if TRAIN in splits and use_aug:
        bank_rows = pd.read_csv(I["real_noise_bank"])
        guard.check_noise_bank(bank_rows)
        N = pcfg.num_samples

        def load_window(r):
            w = wp.load_waveform(wp.PROJECT_ROOT / r.filepath, pcfg)
            a = int(round(r.src_start_sec * pcfg.sample_rate))
            return w[a:a + N].numpy().astype(np.float32)

        bank = {r.window_id: load_window(r) for r in bank_rows.itertuples(index=False)}
        rirs = None
        if cfg["augmentation"].get("rir_bank"):
            # Curated bank (id, filepath, rt60_s). Each response is scaled so its direct path has
            # unit amplitude, so convolution adds reverberation without changing the direct level.
            rir_rows = pd.read_csv(wp.PROJECT_ROOT / cfg["augmentation"]["rir_bank"])
            rirs = []
            for r in rir_rows.itertuples(index=False):
                h = wp.load_waveform(wp.PROJECT_ROOT / r.filepath, pcfg).numpy().astype(np.float64)
                rirs.append(dict(id=r.id, wave=h / np.abs(h).max(), rt60_s=float(r.rt60_s)))
        augmenter = aug.DeploymentAugmenter(pcfg.sample_rate, bank, I["augmentation_config"], rirs)
    cache = cfg["loader"]["file_cache_per_worker"]
    ds = {s: WakewordWindowDataset(s, windows, guard, pre, augmenter if s == TRAIN else None, seed, cache)
          for s in splits}
    return cfg, ds


def build_dataloaders(cfg_path=DEFAULT_CONFIG, splits=None, batch_size=None, num_workers=None,
                      seed=None, augment=None) -> dict:
    """{split: DataLoader}. Train uses BucketQuotaSampler (loader.sampler.set_epoch(e) each
    epoch); other splits iterate every window once, in file order, unaugmented."""
    cfg, datasets = build_datasets(cfg_path, splits, seed, augment)
    seed = cfg["seed"] if seed is None else seed
    L = cfg["loader"]
    bs = L["batch_size"] if batch_size is None else batch_size
    nw = L["num_workers"] if num_workers is None else num_workers
    common = dict(batch_size=bs, num_workers=nw, collate_fn=collate, pin_memory=L["pin_memory"],
                  worker_init_fn=rep.worker_init_fn if nw > 0 else None,
                  persistent_workers=bool(L["persistent_workers"]) and nw > 0,
                  prefetch_factor=L["prefetch_factor"] if nw > 0 else None)
    loaders = {}
    for split, ds in datasets.items():
        if split == TRAIN:
            sampler = BucketQuotaSampler(ds.rows, cfg["sampling"], seed)
            loaders[split] = DataLoader(ds, sampler=sampler, drop_last=L["drop_last_train"],
                                        generator=rep.loader_generator(seed), **common)
        else:
            loaders[split] = DataLoader(ds, shuffle=False, drop_last=False, **common)
    return loaders


def set_epoch(loaders: dict, epoch: int):
    s = loaders.get(TRAIN)
    if s is not None:
        s.sampler.set_epoch(epoch)
