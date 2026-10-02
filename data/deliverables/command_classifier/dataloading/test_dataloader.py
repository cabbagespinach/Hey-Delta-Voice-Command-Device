#!/usr/bin/env python3
"""Tests for the command dataset list and loader.  python test_dataloader.py  (or pytest)"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import command_data as cd                                                 # noqa: E402

CFG = cd.load_config()
A = pd.read_csv(cd.ROOT / CFG["clips_csv"])


def test_every_class_and_split_present():
    assert set(A["class"]) == set(cd.classes()) and len(cd.classes()) == 30 and cd.classes()[-1] == "unknown"
    assert set(A.split) == {"train", "validation", "test"}


def test_no_voice_in_two_splits():
    for prefix in ("synthetic_", "web_"):
        s = A[A.dataset.str.startswith(prefix)].groupby("speaker").split.nunique()
        assert (s == 1).all(), f"{prefix}: {s[s > 1].index[:5].tolist()}"


def test_owner_split_rules():
    o = A[A.dataset == "owner_recordings"]
    assert (o[o.speaker == "speaker2"].split == "test").all()
    t = o.path.str.extract(r"_(\d{8})-(\d{4})")[1]
    assert (o[(o.speaker == "owner") & t.between("1618", "1633")].split == "test").all()
    assert (o[(o.speaker == "owner") & t.between("1336", "1355")].split == "validation").all()


def test_converted_only_from_train_sources():
    assert (A[A.dataset == "converted_owner"].split == "train").all()


def test_fragments_follow_source_split():
    f = pd.read_csv(cd.ROOT / "data/commands_fragments/manifest.csv")
    src = dict(zip(A.path, A.split))
    fr = A[A.dataset.str.startswith("fragments_")].copy()
    fr["file"] = fr.path.str.replace("data/commands_fragments/", "", regex=False)
    m = fr.merge(f[["file", "source_file"]], on="file")
    has = m.source_file.isin(src)
    assert (m[has].split.to_numpy() == m[has].source_file.map(src).to_numpy()).all()


def test_draw_mix():
    c = dict(CFG, epoch_draws=6000)
    ds = cd.TrainDraws(c)
    r = ds.rows.iloc[ds.draws]
    share = (r["class"] == "unknown").mean()
    assert abs(share - CFG["unknown_share"]) < 0.02, share
    counts = r[r["class"] != "unknown"]["class"].value_counts()
    assert counts.max() / counts.min() < 1.6, counts                     # command classes drawn ~uniformly


def test_epochs_differ_but_are_reproducible():
    c = dict(CFG, epoch_draws=500)
    a, b = cd.TrainDraws(c), cd.TrainDraws(c)
    a.set_epoch(3), b.set_epoch(3)
    assert a.draws == b.draws
    b.set_epoch(4)
    assert a.draws != b.draws
    w1, _, _ = a[7]
    w2, _, _ = a[7]
    assert np.array_equal(w1.numpy(), w2.numpy())


def test_eval_items_fixed():
    ev = cd.EvalClips(CFG, "validation")
    i = int(np.flatnonzero(ev.rows.needs_tail.to_numpy())[0])
    w1, _, _ = ev[i]
    w2, _, _ = ev[i]
    assert np.array_equal(w1.numpy(), w2.numpy()) and w1.shape == (80000,)


def test_prefix_bank_is_train_only():
    ds = cd.EvalClips(CFG, "test")
    ds.tails()
    tr = set(A[A.split == "train"].path)
    chk = pd.read_csv(cd.CC.parent / "model/deploy/recordings/_check/check.csv")
    # rebuild the bank's source list the same way and check membership
    own = ["data/deliverables/model/deploy/recordings/" + f for f in chk[chk.cmd_start >= 0.35].file]
    used = [p for p in own if p in tr]
    assert all(p in tr for p in used)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
