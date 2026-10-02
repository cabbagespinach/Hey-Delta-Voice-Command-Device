#!/usr/bin/env python3
"""
Isolated-clip evaluation on a fixed, deterministic window set.

The isolated validation set is every window of the validation split, loaded through
the dataloading layer with fixed preprocessing (no augmentation, no sampling). Its
exact membership is frozen in isolated_set/isolated_validation_manifest.csv together
with fingerprints of the preprocessing configuration and normalization statistics.
Later runs verify the manifest instead of rewriting it, so a changed windows.csv or
preprocessing setting cannot silently change the evaluation set.

Reports per negative category (window_type), per real/synthetic subset, and chooses
the operating threshold (validation only): the lowest threshold whose validation
false-positive rate is <= detection.threshold_policy.target_fpr.
"""
from pathlib import Path
import hashlib, json, sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dataloading"))
import detection as det             # noqa: E402
import wakeword_data as wd          # noqa: E402
import reproducibility as rep       # noqa: E402

CFG = json.loads((HERE / "eval_config.json").read_text())
MANIFEST_COLS = ["window_id", "split", "filepath", "file_id", "start_sec", "end_sec", "label", "window_type",
                 "eval_subset", "device", "source", "source_group", "category"]


def manifest_path(split: str) -> Path:
    return HERE / f"isolated_set/isolated_{split}_manifest.csv"


def fingerprint(manifest: pd.DataFrame, pre) -> dict:
    stats = Path(wd.load_config()["_inputs"]["normalization_stats"])
    return dict(n_windows=len(manifest), n_positive=int((manifest.label == "positive").sum()),
                manifest_sha256=hashlib.sha256(manifest.to_csv(index=False).encode()).hexdigest(),
                preprocessing_feature_hash=pre.cfg.feature_hash(),
                normalization_stats_sha256=hashlib.sha256(stats.read_bytes()).hexdigest())


def load_isolated_set(split: str = "validation"):
    """(dataset, manifest). Freezes the manifest on first use; afterwards verifies it."""
    _, ds = wd.build_datasets(splits=[split], augment=False)
    d = ds[split]
    manifest = pd.DataFrame(d.meta)[MANIFEST_COLS]
    fp = fingerprint(manifest, d.pre)
    path = manifest_path(split)
    fp_path = path.with_suffix(".json")
    if path.exists():
        frozen = json.loads(fp_path.read_text())
        if frozen != fp:
            diff = {k: (frozen.get(k), fp[k]) for k in fp if frozen.get(k) != fp[k]}
            raise RuntimeError(f"isolated {split} set differs from its frozen manifest: {diff}. "
                               "Delete the manifest deliberately to re-freeze.")
    else:
        path.parent.mkdir(exist_ok=True)
        manifest.to_csv(path, index=False)
        fp_path.write_text(json.dumps(fp, indent=2) + "\n")
    return d, manifest


def score_dataset(d, scorer, batch_size=256, num_workers=8) -> np.ndarray:
    loader = torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                         collate_fn=wd.collate,
                                         worker_init_fn=rep.worker_init_fn if num_workers else None)
    out = []
    with rep.single_thread():
        for b in loader:
            out.append(scorer(b["features"]))
    return np.concatenate(out)


def policy_groups(neg: pd.DataFrame, pol: dict) -> pd.Series:
    """Negative group each window counts in under the threshold policy (NaN = excluded)."""
    g = neg.window_type.copy()
    if pol["mode"] == "validation_fpr":
        return pd.Series("all", index=neg.index)
    g = g.where(~neg.eval_subset.isin(pol.get("exclude_subsets", [])))
    small = g.value_counts()
    small = small[small < pol.get("min_category_n", 0)].index
    return g.where(~g.isin(small), "other (pooled small categories)")


def choose_threshold(neg: pd.DataFrame, pol: dict):
    """Operating threshold under eval_config detection.threshold_policy, plus the per-group thresholds."""
    groups = policy_groups(neg, pol)
    per = {k: det.threshold_at_fpr(neg.score[groups == k], pol["target_fpr"]) for k in groups.dropna().unique()}
    return max(per.values()), dict(mode=pol["mode"], per_group_threshold=per,
                                   excluded_windows=int(groups.isna().sum()))


def evaluate(scorer, split: str = "validation", threshold: float | None = None):
    d, manifest = load_isolated_set(split)
    scores = score_dataset(d, scorer)
    s = manifest.assign(score=scores, y=(manifest.label == "positive").astype(int))
    neg, pos = s[s.y == 0], s[s.y == 1]
    pol = CFG["detection"]["threshold_policy"]
    chosen_here = threshold is None
    if chosen_here:
        if split != "validation":
            raise ValueError("the operating threshold may only be chosen on validation")
        threshold, policy_detail = choose_threshold(neg, pol)
    s["predicted"] = (s.score >= threshold).astype(int)
    neg, pos = s[s.y == 0], s[s.y == 1]

    def rate_row(g, positive):
        k = int(g.predicted.sum())
        r, lo, hi = det.proportion_ci(k, len(g))
        return dict(n=len(g), **({"detected": k, "detection_rate": r} if positive else {"false_positives": k, "fpr": r}),
                    ci95=[lo, hi], score_mean=float(g.score.mean()), score_p99=float(g.score.quantile(0.99)),
                    score_max=float(g.score.max()))

    res = dict(
        split=split, n_windows=len(s), n_positive=len(pos), n_negative=len(neg),
        roc_auc=det.roc_auc(s.y, s.score), average_precision=det.average_precision(s.y, s.score),
        threshold=float(threshold),
        threshold_source=((f"chosen on isolated validation: lowest threshold with FPR <= {pol['target_fpr']} in every "
                           f"negative category of deployment-like audio (excluding {', '.join(pol.get('exclude_subsets', []))})"
                           if pol["mode"] == "per_category_fpr" else
                           f"chosen on isolated validation: lowest threshold with pooled FPR <= {pol['target_fpr']}")
                          if chosen_here else "supplied (chosen on isolated validation)"),
        threshold_policy=policy_detail if chosen_here else None,
        positives=rate_row(pos, True), negatives=rate_row(neg, False),
        negatives_by_subset={k: rate_row(g, False) for k, g in neg.groupby("eval_subset")},
        positives_by_subset={k: rate_row(g, True) for k, g in pos.groupby("eval_subset")},
        negatives_by_category={k: rate_row(g, False) for k, g in neg.groupby("window_type")},
        negatives_by_category_and_subset={f"{a} | {b}": rate_row(g, False)
                                          for (a, b), g in neg.groupby(["window_type", "eval_subset"])},
        low_signal_rpi={r.file_id: dict(window_id=r.window_id, score=float(r.score), detected=bool(r.predicted))
                        for r in pos[pos.file_id.isin(["Manual-Fil-RPI19", "Manual-Fil-RPI20", "Manual-Fil-RPI21"])]
                        .itertuples(index=False)},
        fingerprint=json.loads(manifest_path(split).with_suffix(".json").read_text()),
    )
    grid = np.unique(np.quantile(s.score, np.linspace(0, 1, CFG["detection"]["sweep_points"])))
    res["sweep"] = [dict(threshold=float(t), detection_rate=float((pos.score >= t).mean()),
                         fpr=float((neg.score >= t).mean())) for t in grid]
    return res, s
