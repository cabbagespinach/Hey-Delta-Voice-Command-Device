#!/usr/bin/env python3
"""
Dataset and loaders for the command classifier (config: dataloader_config.json).

    import command_data as cd
    cfg = cd.load_config()
    train = cd.TrainDraws(cfg)             # weighted draws per epoch; train.set_epoch(e) before each epoch
    val = cd.EvalClips(cfg, "validation")  # every clip once, unaugmented, fixed format alignment
    loader = cd.make_loader(train, cfg)    # batches of (wave [B, 80000], class [B], row [B])

Each item is a 5 s waveform (command_preprocessing.frame_command) plus its class index; features are computed by
the caller (on the GPU, command_preprocessing.CommandPreprocessor), so the loader workers only read, align and augment.

Randomness: every draw has its own seed derived from (config seed, epoch, draw index), so an epoch is identical for
any worker count or order.
"""
from __future__ import annotations

from pathlib import Path
import hashlib, json, os, sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
CC = HERE.parent
sys.path.insert(0, str(CC))
sys.path.insert(0, str(CC / "preprocessing"))
import command_preprocessing as cp                                        # noqa: E402

ROOT = cp.PROJECT_ROOT
# COMMAND_LOADER_CONFIG: another loader config (schema-B run, 2026-10-01: clip list + its own "label_map")
DEFAULT_CONFIG = Path(os.environ.get("COMMAND_LOADER_CONFIG", HERE / "dataloader_config.json"))
SR = 16000


def load_config(path=DEFAULT_CONFIG) -> dict:
    return json.loads(Path(path).read_text())


LABEL_MAP = ROOT / load_config().get("label_map", "data/deliverables/model/deploy/label_map.csv")


def classes() -> list:
    """Class names in index order: the command classes in label_map order, then `unknown` last."""
    lm = pd.read_csv(LABEL_MAP)
    out = [c for c in dict.fromkeys(lm["class"]) if c != "unknown"]
    return out + ["unknown"]


def seed_of(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:15], 16)


def source_group(dataset: str, cls: str) -> str:
    """Sampling group of a clip (see dataloader_config.json)."""
    classmate = dataset.startswith("classmate_") or dataset == "other_speakers"   # schema-B run, 2026-10-01
    if cls == "unknown":
        if dataset == "web_gsc":
            return "gsc"                                # 100k single words: own group so they cannot flood `web`
        if classmate:
            return "classmates"
        if dataset.startswith("fragments_"):
            return "fragments"
        if dataset == "owner_recordings":
            return "owner"
        if dataset.startswith("reuse_"):
            return dataset
        if dataset.startswith("web_"):
            return "web"
        return "synthetic"
    if dataset == "owner_recordings":
        return "owner"
    if dataset.startswith("web_"):
        return "web"
    if dataset.startswith("variants_"):
        return "variants"
    if classmate:
        return "classmates"                             # classmates' recordings and synthetic sets (2026-10-01)                               # other wordings (2026-10-01), own weight
    return "synthetic"                                  # synthetic_*, converted_owner


def read_wave(path) -> np.ndarray:
    y, sr = sf.read(str(ROOT / path), dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != SR:
        y = cp.wp.resample(torch.from_numpy(y), sr, cp.load_config()).numpy()
    return y


class _Common:
    def __init__(self, cfg: dict, split: str):
        self.cfg, self.split = cfg, split
        self.pcfg = cp.load_config()
        self.classes = classes()
        self.cidx = {c: i for i, c in enumerate(self.classes)}
        a = pd.read_csv(ROOT / cfg["clips_csv"])
        self.rows = a[a.split == split].reset_index(drop=True)
        self.rows["group"] = [source_group(d, c) for d, c in zip(self.rows.dataset, self.rows["class"])]
        self.rows["needs_tail"] = self.rows.dataset.map(
            lambda d: any(d.startswith(p) for p in cfg["format_alignment"]["datasets_without_delta_tail"]))
        self._tails = None
        self._aug = None

    # prefixes ('Delta' tail + chime gap + pause), TRAIN split only, whatever split is being served
    def tails(self):
        if self._tails is None:
            a = pd.read_csv(ROOT / self.cfg["clips_csv"])
            tr = a[a.split == "train"]
            bank = []
            chk = pd.read_csv(CC.parent / "model/deploy/recordings/_check/check.csv")
            own = chk[chk.cmd_start.notna() & (chk.cmd_start >= 0.35)]
            own_train = set(tr[tr.dataset == "owner_recordings"].path)
            for r in own.itertuples(index=False):
                p = "data/deliverables/model/deploy/recordings/" + r.file
                if p in own_train:
                    bank.append((p, r.cmd_start - 0.05))
            syn = pd.read_csv(ROOT / "data/commands_synthetic/manifest.csv")
            syn["path"] = "data/commands_synthetic/" + syn.file
            syn = syn[syn.path.isin(set(tr.path)) & syn.delta_tail_sec.notna()]
            syn = syn.sample(min(600, len(syn)), random_state=self.cfg["seed"])
            for r in syn.itertuples(index=False):
                bank.append((r.path, float(r.delta_tail_sec) + float(r.pause_sec)))
            self._tails = [read_wave(p)[:int(e * SR)] for p, e in bank]
            self._tails = [t for t in self._tails if len(t) >= int(0.2 * SR)]
        return self._tails

    def augmenter(self):
        if self._aug is None:
            from augment_commands import CommandAugmenter
            self._aug = CommandAugmenter(self.cfg.get("owner_extra_background"), self.cfg.get("background_talk"))
        return self._aug

    def build(self, row, rng, augment: bool):
        y = read_wave(row.path)
        if row.needs_tail and rng.random() < self.cfg["format_alignment"]["probability"]:
            t = self.tails()
            y = np.concatenate([t[int(rng.integers(len(t)))], y])
        y = y[:self.pcfg.num_samples]
        if augment:
            owner = row.dataset == "owner_recordings" or row.dataset == "fragments_owner_recordings"
            y, _ = self.augmenter()(y, rng, owner_recording=owner)
            if self.cfg.get("background_talk"):
                y, _ = self.augmenter().background_talk(y, rng, self.pcfg.num_samples)
        w = cp.frame_command(torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)), self.pcfg,
                             pad_key=f"{row.path}|{rng.integers(1 << 30)}")
        return w


class TrainDraws(_Common, torch.utils.data.Dataset):
    """Weighted draws for one epoch: unknown_share of draws are unknown (by unknown_source_weights), the rest pick a
    command class uniformly, then a source group by command_source_weights, then a clip uniformly."""

    def __init__(self, cfg: dict, split="train", augment=None, exclude_groups=()):
        super().__init__(cfg, split)
        self.augment = cfg["augment"] if augment is None else augment
        if exclude_groups:                          # group names or dataset-name prefixes, e.g. for the probe that
            drop = self.rows.group.isin(exclude_groups) | self.rows.dataset.map(   # trains without the owner's data
                lambda d: any(d.startswith(x) for x in exclude_groups))
            self.rows = self.rows[~drop].reset_index(drop=True)
        self.by = {k: g.index.to_numpy() for k, g in self.rows.groupby(["class", "group"])}
        self.cmd_classes = [c for c in self.classes if c != "unknown" and any(k[0] == c for k in self.by)]
        self.epoch = 0
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        rng = np.random.default_rng(seed_of(self.cfg["seed"], "epoch", epoch))
        cw, uw = self.cfg["command_source_weights"], self.cfg["unknown_source_weights"]
        draws = []
        for _ in range(self.cfg["epoch_draws"]):
            if rng.random() < self.cfg["unknown_share"]:
                gs = [g for g in uw if ("unknown", g) in self.by]
                p = np.array([uw[g] for g in gs])
                g = gs[rng.choice(len(gs), p=p / p.sum())]
                idx = self.by[("unknown", g)]
            else:
                c = self.cmd_classes[rng.integers(len(self.cmd_classes))]
                gs = [g for g in cw if (c, g) in self.by]
                p = np.array([cw[g] for g in gs])
                g = gs[rng.choice(len(gs), p=p / p.sum())]
                idx = self.by[(c, g)]
            draws.append(int(idx[rng.integers(len(idx))]))
        self.draws = draws

    def __len__(self):
        return len(self.draws)

    def __getitem__(self, i):
        row = self.rows.iloc[self.draws[i]]
        rng = np.random.default_rng(seed_of(self.cfg["seed"], self.epoch, i))
        return self.build(row, rng, self.augment), self.cidx[row["class"]], self.draws[i]


class EvalClips(_Common, torch.utils.data.Dataset):
    """Every clip of a split once, unaugmented; format alignment fixed per clip."""

    def __init__(self, cfg: dict, split: str, rows: pd.DataFrame | None = None):
        super().__init__(cfg, split)
        if rows is not None:
            self.rows = rows.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        rng = np.random.default_rng(seed_of(self.cfg["seed"], "eval", row.path))
        return self.build(row, rng, False), self.cidx[row["class"]], i


def _worker_init(_):
    torch.set_num_threads(1)


def make_loader(ds, cfg: dict, shuffle=False, num_workers=None, batch_size=None):
    """Workers are re-forked every epoch (not persistent), so they see the draws of the current set_epoch(). The
    prefix bank and the augmenter's banks are loaded here, before the fork, and shared copy-on-write."""
    ds.tails()
    if getattr(ds, "augment", False):
        ds.augmenter()
    nw = cfg["num_workers"] if num_workers is None else num_workers
    return torch.utils.data.DataLoader(ds, batch_size=batch_size or cfg["batch_size"], shuffle=shuffle,
                                       num_workers=nw, worker_init_fn=_worker_init, pin_memory=True,
                                       persistent_workers=False, prefetch_factor=4 if nw > 0 else None)
