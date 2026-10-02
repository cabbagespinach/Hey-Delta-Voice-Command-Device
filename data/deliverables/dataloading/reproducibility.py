#!/usr/bin/env python3
"""
Reproducibility utilities.

Randomness in the data layer is keyed, not sequential: every random draw
(epoch sampling, per-example augmentation) derives its seed from stable
identifiers (base seed, epoch, draw position, window id). The result does not
depend on num_workers, worker scheduling, or which examples were loaded
before, so a run can be reproduced exactly and any single example regenerated
in isolation from its metadata.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib, os, random

import numpy as np
import torch


def stable_seed(*parts) -> int:
    """Process-independent 63-bit seed from arbitrary parts (Python's hash() is salted per process)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def seed_everything(seed: int, deterministic_torch: bool = True):
    """Seed Python, NumPy and torch for the main process (model init, etc.)."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id: int):
    """DataLoader worker init: seed global RNGs from the loader's base seed and pin
    one intra-op thread per worker (avoids oversubscription; keeps float
    reductions in a fixed order)."""
    base = torch.initial_seed()
    random.seed(base + worker_id)
    np.random.seed((base + worker_id) % 2**32)
    torch.set_num_threads(1)


@contextmanager
def single_thread():
    """Run torch ops with one intra-op thread, then restore the previous setting.

    Multi-threaded reductions (resampling convolution, STFT, mel projection) sum
    in an order that depends on the thread count, which changes the last float
    bits. Pinning one thread makes an example's features identical whether it is
    loaded in the main process (num_workers=0) or in any worker.
    """
    prev = torch.get_num_threads()
    if prev == 1:
        yield
        return
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


def loader_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def example_rng(seed: int, epoch: int, draw: int, window_id: str) -> np.random.Generator:
    """RNG for one augmented example. Keyed by draw position, so a window drawn
    twice in one epoch gets two independent augmentations."""
    return np.random.default_rng(stable_seed("augment", seed, epoch, draw, window_id))
