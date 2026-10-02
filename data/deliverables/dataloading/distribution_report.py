#!/usr/bin/env python3
"""
Class and negative-category balance before and after training sampling.

Writes distribution_report.md and distribution_report.json:
  1. natural distribution of every split (label, role, window type, eval subset)
  2. train before sampling vs. after (exact quota counts + an empirical check over
     several epochs): class share, hard/easy negative share, per-category share of
     negatives, repeat factor, window coverage, source-group concentration
  3. augmentation application rates measured on a sample of real training draws

Usage: python distribution_report.py [--epochs 5] [--aug-sample 1000]
"""
from pathlib import Path
import argparse, datetime, json, sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wakeword_data as wd
import reproducibility as rep


def role_of(sampling):
    return {t: (name, b["role"]) for name, b in sampling["buckets"].items() for t in b["window_types"]}


def effective_groups(groups: pd.Series, weights=None) -> float:
    """1 / sum(p_g^2): how many equally-sized source groups the distribution is worth."""
    p = (pd.Series(weights if weights is not None else np.ones(len(groups)), index=groups.values)
         .groupby(level=0).sum())
    p = p / p.sum()
    return float(1 / (p ** 2).sum())


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--aug-sample", type=int, default=1000)
    args = ap.parse_args()

    cfg = wd.load_config()
    samp = cfg["sampling"]
    roles = role_of(samp)
    w = wd.load_windows(cfg)
    w["bucket"] = w.window_type.map(lambda t: roles.get(t, ("eval_only", "eval_only"))[0])
    w["role"] = w.window_type.map(lambda t: roles.get(t, ("eval_only", "eval_only"))[1])
    out = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               seed=cfg["seed"], sampling=samp)

    # 1. natural distribution per split -------------------------------------------------
    nat = {}
    for split, g in w.groupby("split"):
        nat[split] = dict(
            n=int(len(g)),
            label=g.label.value_counts().to_dict(),
            role=g.role.value_counts().to_dict(),
            window_type=g.window_type.value_counts().to_dict(),
            positives_by_eval_subset=g[g.label == "positive"].eval_subset.value_counts().to_dict(),
            source_groups=int(g.source_group.nunique()))
    out["natural"] = nat

    # 2. train before vs after sampling ----------------------------------------------------
    tr = w[w.split == "train"].reset_index(drop=True)
    sampler = wd.BucketQuotaSampler(tr, samp, cfg["seed"])
    N = len(sampler)
    rows = []
    for name, b in sampler.buckets.items():
        idx = b["index"]
        grp = tr.source_group.iloc[idx]
        rows.append(dict(bucket=name, role=b["role"], window_types=", ".join(samp["buckets"][name]["window_types"]),
                         windows=len(idx), before_share=len(idx) / len(tr), after_share=b["count"] / N,
                         draws_per_epoch=b["count"], repeat_factor=b["count"] / len(idx),
                         source_groups=int(grp.nunique()),
                         eff_groups_before=effective_groups(grp),
                         eff_groups_after=effective_groups(grp, b["weight"].numpy()),
                         top_group=grp.value_counts().index[0],
                         top_group_share_before=float(grp.value_counts(normalize=True).iloc[0]),
                         top_group_share_after=float(pd.Series(b["weight"].numpy(), index=grp.values)
                                                     .groupby(level=0).sum().max())))
    bt = pd.DataFrame(rows).sort_values(["role", "bucket"]).reset_index(drop=True)

    # empirical: several epochs
    emp, coverage = [], []
    for e in range(args.epochs):
        idx = sampler.epoch_indices(e)
        emp.append(pd.Series(sampler.bucket_of[idx]).value_counts(normalize=True))
        coverage.append(pd.Series(sampler.bucket_of[np.unique(idx)]).value_counts() /
                        pd.Series(sampler.bucket_of).value_counts())
    emp = pd.concat(emp, axis=1).fillna(0)
    cov = pd.concat(coverage, axis=1).fillna(0)
    bt["after_share_empirical_mean"] = bt.bucket.map(emp.mean(axis=1))
    bt["unique_windows_seen_per_epoch"] = bt.bucket.map(cov.mean(axis=1))

    def summary(share_col):
        s = bt.groupby("role")[share_col].sum()
        neg = bt[bt.role != "positive"]
        within = (neg.set_index("bucket")[share_col] / neg[share_col].sum()).to_dict()
        return dict(positive=float(s.get("positive", 0)), hard_negative=float(s.get("hard_negative", 0)),
                    easy_negative=float(s.get("easy_negative", 0)),
                    hard_to_easy_ratio=float(s.get("hard_negative", 0) / s.get("easy_negative", 1)),
                    negative_category_share_within_negatives=within)

    out["train_before"] = summary("before_share")
    out["train_after"] = summary("after_share")
    # real (manual recording) vs synthetic source share within each class, before and after sampling
    kind = wd.source_kind(tr)
    exp_real = {"positive": 0.0, "negative": 0.0}
    exp_all = {"positive": 0.0, "negative": 0.0}
    for name, b in sampler.buckets.items():
        lab = "positive" if b["role"] == "positive" else "negative"
        real_w = float(b["weight"].numpy()[kind[b["index"]] == "real"].sum())
        exp_real[lab] += b["count"] * real_w
        exp_all[lab] += b["count"]
    out["real_source_share"] = {lab: dict(before=float((kind[tr.label.to_numpy() == lab] == "real").mean()),
                                          after=exp_real[lab] / exp_all[lab]) for lab in ("positive", "negative")}
    out["train_buckets"] = bt.to_dict("records")
    out["epoch_size"] = N

    # 3. augmentation rates on real draws ---------------------------------------------------
    _, ds = wd.build_datasets(splits=["train"])
    d = ds["train"]
    keys = list(sampler)[: args.aug_sample]
    rec = []
    with rep.single_thread():
        for k in keys:
            _, m = d.load(k)
            a = json.loads(m["augmentation"] or "{}")
            rec.append(dict(bucket=sampler.bucket_of[k[0]], **{t: (t in a and "skipped" not in a[t])
                            for t in ("device_level_gain", "rir_reverberation", "background_noise_mixing",
                                      "device_noise_floor")}, any=bool(a)))
    ar = pd.DataFrame(rec)
    aug_rates = ar.groupby("bucket").mean(numeric_only=True)
    aug_rates.loc["ALL"] = ar.mean(numeric_only=True)
    out["augmentation_rates_on_sample"] = dict(n_draws=len(ar), rates=aug_rates.round(3).to_dict("index"))

    (HERE / "distribution_report.json").write_text(json.dumps(out, indent=2, default=float) + "\n")

    # markdown ---------------------------------------------------------------------------
    L = [f"# Dataset / DataLoader distribution report",
         "",
         f"Generated {out['created_at']} by `distribution_report.py` (seed {cfg['seed']}, "
         f"source_group_exponent {samp['source_group_exponent']}, epoch size {N}). Machine-readable: `distribution_report.json`.",
         "",
         "## 1. Natural distribution per split (no sampling)",
         "",
         "| Split | Windows | Positive | Hard negative | Easy negative | Eval-only | Source groups | Positives: RPI / other real / synthetic |",
         "|---|---:|---:|---:|---:|---:|---:|---|"]
    for split in ("train", "validation", "test", "streaming_eval_holdout"):
        n = nat[split]
        r, ps = n["role"], n["positives_by_eval_subset"]
        L.append(f"| {split} | {n['n']:,} | {r.get('positive', 0)} ({pct(r.get('positive', 0) / n['n'])}) | "
                 f"{r.get('hard_negative', 0)} ({pct(r.get('hard_negative', 0) / n['n'])}) | "
                 f"{r.get('easy_negative', 0)} ({pct(r.get('easy_negative', 0) / n['n'])}) | {r.get('eval_only', 0)} | "
                 f"{n['source_groups']:,} | {ps.get('real_device_rpi', 0)} / {ps.get('real_other_device', 0)} / {ps.get('synthetic', 0)} |")
    L += ["", "Validation and test are loaded unsampled and unaugmented, so metrics see this natural mix.",
          "Streaming-eval positives/negatives come from continuous synthetic streams (`negative_stream_background`).",
          "",
          "## 2. Training: before vs after sampling",
          "",
          "| | Before (natural) | After (quota) |",
          "|---|---:|---:|"]
    b, a = out["train_before"], out["train_after"]
    for k, lab in [("positive", "Positive"), ("hard_negative", "Hard negative"), ("easy_negative", "Easy negative")]:
        L.append(f"| {lab} | {pct(b[k])} | {pct(a[k])} |")
    L.append(f"| Hard : easy negative ratio | {b['hard_to_easy_ratio']:.2f} | {a['hard_to_easy_ratio']:.2f} |")
    rs = out["real_source_share"]
    for lab in ("positive", "negative"):
        L.append(f"| Real-recording share of {lab}s | {pct(rs[lab]['before'])} | {pct(rs[lab]['after'])} |")
    L += ["", "The real-recording share per class shows how strongly *source* could predict the label "
          "(see `evaluation/diagnostics/source_confound_report.md`); ideally it is equal for both classes."]
    L += ["", "**Negative categories as a share of all negatives:**", "",
          "| Bucket | Role | Before | After |", "|---|---|---:|---:|"]
    for name in bt[bt.role != "positive"].bucket:
        L.append(f"| {name} | {bt.set_index('bucket').role[name]} | "
                 f"{pct(b['negative_category_share_within_negatives'][name])} | "
                 f"{pct(a['negative_category_share_within_negatives'][name])} |")
    L += ["", "**Per bucket:**", "",
          "| Bucket | Windows | Before | After (quota) | After (empirical, "
          f"{args.epochs} epochs) | Draws/epoch | Repeat factor | Windows seen/epoch | Source groups | "
          "Effective groups before → after | Largest group share before → after |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in bt.itertuples(index=False):
        L.append(f"| {r.bucket} | {r.windows:,} | {pct(r.before_share)} | {pct(r.after_share)} | "
                 f"{pct(r.after_share_empirical_mean)} | {r.draws_per_epoch:,} | {r.repeat_factor:.2f}× | "
                 f"{pct(r.unique_windows_seen_per_epoch)} | {r.source_groups} | "
                 f"{r.eff_groups_before:.1f} → {r.eff_groups_after:.1f} | "
                 f"{pct(r.top_group_share_before)} → {pct(r.top_group_share_after)} (`{r.top_group}`) |")
    L += ["",
          "- **Repeat factor** = draws per epoch / windows in the bucket. Above 1, windows are drawn several times per "
          "epoch; each draw gets its own augmentation seed, so repeats are different augmented examples.",
          "- **Effective groups** = 1 / Σ p²ᵍ over source groups, i.e. how many equally weighted recordings the bucket "
          "is worth. Square-root source-group weighting raises it by limiting long recordings.",
          "",
          f"## 3. Augmentation applied on {len(ar):,} sampled training draws",
          "",
          "| Bucket | Gain | Reverb | Noise mixing | Device noise floor | Any |",
          "|---|---:|---:|---:|---:|---:|"]
    for name, r in aug_rates.iterrows():
        L.append(f"| {name} | {pct(r.device_level_gain)} | {pct(r.rir_reverberation)} | "
                 f"{pct(r.background_noise_mixing)} | {pct(r.device_noise_floor)} | {pct(r['any'])} |")
    T = {t["name"]: t for t in json.loads(Path(cfg["_inputs"]["augmentation_config"]).read_text())["enabled_transforms"]}
    L += ["",
          f"Configured probabilities (augmentation config): gain {T['device_level_gain']['probability']}, reverb "
          f"{T['rir_reverberation']['probability']} (disabled: no RIR bank), noise mixing "
          f"{T['background_noise_mixing']['probability']} (only when the example is cleaner than the target; noisier "
          f"real recordings are skipped), device noise floor {T['device_noise_floor']['probability']} when the excerpt "
          "contains ≥ 20 ms of exact zeros. Each transform applies only to the categories listed in the augmentation "
          "config (e.g. no gain or mixing for real device background; no mixing for silence/noise clips), so rates "
          "differ by bucket."]
    (HERE / "distribution_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
