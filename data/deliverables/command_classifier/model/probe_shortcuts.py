#!/usr/bin/env python3
"""
Small-scale test, part 1: can the class be predicted from things that are NOT the spoken words? (2026-09-30)

    python probe_shortcuts.py            # CPU only; writes runs/probes/shortcuts.json

Like the wakeword project's source-confound diagnostic: a simple classifier (logistic regression) is trained on the
TRAIN split with deliberately limited inputs and scored on VALIDATION. If an input that carries no words predicts the
command, a model can take the same shortcut and fail on real use.

Inputs (unaugmented clips, the loader's format alignment applied, 5 s front end, log-mel in dB):
  prefix_only    mean + std of each mel bin over the first 0.25 s (the 'Delta' tail / pre-roll; no command yet)
  channel_only   mean + std of each mel bin over the whole clip (overall colour and level: microphone, room,
                 voice; the order of sounds, i.e. the words, is thrown away)
  length_only    clip duration before padding, and the fraction of the 5 s window above the noise floor
Targets:
  command class, on command clips only (29 classes; chance ~3.4% macro accuracy)
  unknown vs command (reported as ROC AUC; 0.5 = no information)
Scores are reported per source group; the owner group is the most telling, since all its clips share one voice,
microphone and room, so any class signal there comes from session or recording habits, not from the channel.
"""
from pathlib import Path
import json, sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dataloading"))
import command_data as cd                                                 # noqa: E402
import command_preprocessing as cp                                        # noqa: E402

PER_CLASS_GROUP = 120        # train clips per (class, group) at most, so no source dominates the fit


def features(ds, idx, ext):
    F = {"prefix_only": [], "channel_only": [], "length_only": []}
    for i in idx:
        row = ds.rows.iloc[i]
        rng = np.random.default_rng(cd.seed_of(ds.cfg["seed"], "probe", row.path))
        y = cd.read_wave(row.path)
        if row.needs_tail:
            t = ds.tails()
            y = np.concatenate([t[int(rng.integers(len(t)))], y])
        n = min(len(y), ds.pcfg.num_samples)
        w = cp.frame_command(torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)), ds.pcfg, pad_key=row.path)
        with torch.no_grad():
            x = ext(w).numpy()                                             # [40, 497]
        F["prefix_only"].append(np.r_[x[:, :25].mean(1), x[:, :25].std(1)])
        F["channel_only"].append(np.r_[x.mean(1), x.std(1)])
        e = 20 * np.log10(np.sqrt((w.numpy()[:n // 320 * 320].reshape(-1, 320) ** 2).mean(1)) + 1e-9)
        F["length_only"].append(np.r_[n / cd.SR, (e > -45).mean()])
    return {k: np.array(v) for k, v in F.items()}


def main():
    cfg = cd.load_config()
    tr = cd.EvalClips(cfg, "train")
    va = cd.EvalClips(cfg, "validation")
    ext = cp.FeatureExtractor(cp.load_config())
    tri = (tr.rows.groupby(["class", "group"], group_keys=False)
           .apply(lambda g: g.sample(min(PER_CLASS_GROUP, len(g)), random_state=0)).index.to_numpy())
    print(f"train sample {len(tri)} clips, validation {len(va.rows)} clips", flush=True)
    Ftr, Fva = features(tr, tri, ext), features(va, np.arange(len(va.rows)), ext)
    rtr, rva = tr.rows.iloc[tri].reset_index(drop=True), va.rows
    res = {}
    for name in Ftr:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"))
        cm_tr, cm_va = rtr["class"] != "unknown", rva["class"] != "unknown"
        clf.fit(Ftr[name][cm_tr], rtr["class"][cm_tr])
        pred = clf.predict(Fva[name][cm_va])
        v = rva[cm_va].assign(ok=pred == rva["class"][cm_va].to_numpy())
        acc = {g: round(float(s.groupby("class").ok.mean().mean()), 3) for g, s in v.groupby("group")}
        u = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        u.fit(Ftr[name], rtr["class"] == "unknown")
        s = u.predict_proba(Fva[name])[:, 1]
        auc_all = round(float(roc_auc_score(rva["class"] == "unknown", s)), 3)
        own = (rva.group == "owner").to_numpy()
        auc_own = (round(float(roc_auc_score(rva["class"][own] == "unknown", s[own])), 3)
                   if (rva["class"][own] == "unknown").nunique() == 2 else None)
        res[name] = dict(command_macro_acc_by_group=acc, chance=round(1 / 29, 3), unknown_auc_all=auc_all,
                         unknown_auc_owner=auc_own)
        print(f"{name:13} command macro-acc by group {acc} (chance {1 / 29:.3f}) | unknown-vs-command AUC "
              f"all {auc_all}, owner-only {auc_own}", flush=True)
    out = HERE / "runs/probes"
    out.mkdir(parents=True, exist_ok=True)
    (out / "shortcuts.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
