#!/usr/bin/env python3
"""
Train a BC-ResNet "Hey Delta" model on the train split (recipe: train_config.json).

    python train.py --model bcresnet3 --out runs/bcresnet3
    python train.py --model bcresnet3 --out runs/bcresnet3_hn --mine-from runs/bcresnet3/best.pt

Data comes only through the dataloading layer: bucket-quota sampling and training-only augmentation
(augmentation_config.json v4). Each epoch the model scores the frozen isolated validation set and the
best epoch (train_config.json "selection") is kept as best.pt; last.pt is the final epoch. The test split
and the streaming sets are never read.

Outputs in --out: best.pt, last.pt, train_log.json (per-epoch loss and validation metrics, config,
data fingerprints), and for a mining run mined_negatives.csv.
"""
from pathlib import Path
import argparse, datetime, json, math, sys, time

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
DELIV = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DELIV / "evaluation"))
sys.path.insert(0, str(DELIV / "dataloading"))
import bc_resnet as bcr                 # noqa: E402
import wakeword_data as wd              # noqa: E402
import reproducibility as rep           # noqa: E402
import isolated_eval as iso             # noqa: E402
import detection as det                 # noqa: E402

CFG = json.loads((HERE / "train_config.json").read_text())


# ----------------------------------------------------------------------------
# Augmentation on features
# ----------------------------------------------------------------------------
def spec_augment(x: torch.Tensor, spec: dict, g: torch.Generator) -> torch.Tensor:
    """In-place-free SpecAugment on a batch [B, 1, F, T] (normalised features; masked value 0)."""
    B, _, F, T = x.shape
    dev = x.device
    keep = torch.ones(B, 1, F, T, device=dev)
    apply = torch.rand(B, generator=g) < spec["probability"]
    for _ in range(spec["freq_masks"]):
        w = torch.randint(0, spec["max_freq_bins"] + 1, (B,), generator=g)
        f0 = (torch.rand(B, generator=g) * (F - w + 1).float()).long()
        idx = torch.arange(F).view(1, F)
        m = (idx >= f0.view(B, 1)) & (idx < (f0 + w).view(B, 1)) & apply.view(B, 1)
        keep = keep * (~m).to(dev).view(B, 1, F, 1).float()
    for _ in range(spec["time_masks"]):
        w = torch.randint(0, spec["max_time_frames"] + 1, (B,), generator=g)
        t0 = (torch.rand(B, generator=g) * (T - w + 1).float()).long()
        idx = torch.arange(T).view(1, T)
        m = (idx >= t0.view(B, 1)) & (idx < (t0 + w).view(B, 1)) & apply.view(B, 1)
        keep = keep * (~m).to(dev).view(B, 1, 1, T).float()
    return x * keep


# ----------------------------------------------------------------------------
# Sampling with mined hard negatives
# ----------------------------------------------------------------------------
class MinedSampler(wd.BucketQuotaSampler):
    """Bucket-quota epochs where a fixed share of draws is replaced by draws from mined train negatives."""

    def __init__(self, rows, sampling, seed, mined_index: np.ndarray, share: float):
        super().__init__(rows, sampling, seed)
        self.mined, self.share = np.asarray(mined_index, dtype=np.int64), float(share)

    def epoch_indices(self, epoch=None):
        idx = super().epoch_indices(epoch).copy()
        e = self.epoch if epoch is None else epoch
        g = np.random.default_rng(rep.stable_seed("mined", self.seed, e))
        k = int(round(self.share * len(idx)))
        pos = g.choice(len(idx), size=k, replace=False)
        idx[pos] = g.choice(self.mined, size=k, replace=True)
        return idx


def build_train_loader(seed, bs, nw, mined=None):
    cfg, ds = wd.build_datasets(splits=["train"], seed=seed)
    d = ds["train"]
    L = cfg["loader"]
    if mined is None:
        sampler = wd.BucketQuotaSampler(d.rows, cfg["sampling"], seed)
    else:
        sampler = MinedSampler(d.rows, cfg["sampling"], seed, mined, CFG["hard_negative_mining"]["share_of_draws"])
    loader = torch.utils.data.DataLoader(
        d, sampler=sampler, batch_size=bs, num_workers=nw, collate_fn=wd.collate, pin_memory=True,
        drop_last=L["drop_last_train"], worker_init_fn=rep.worker_init_fn, persistent_workers=True,
        prefetch_factor=L["prefetch_factor"], generator=rep.loader_generator(seed))
    return d, sampler, loader


# ----------------------------------------------------------------------------
# Validation (frozen isolated set) and selection
# ----------------------------------------------------------------------------
def features_of(d, nw) -> torch.Tensor:
    loader = torch.utils.data.DataLoader(d, batch_size=512, shuffle=False, num_workers=nw, collate_fn=wd.collate,
                                         worker_init_fn=rep.worker_init_fn)
    return torch.cat([b["features"] for b in loader])


@torch.no_grad()
def scores_of(model, feats, dev, bs=1024) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(feats[i:i + bs].to(dev))).double().cpu() for i in range(0, len(feats), bs)]
    return torch.cat(out).numpy()


def validation_metrics(manifest: pd.DataFrame, scores: np.ndarray) -> dict:
    s = manifest.assign(score=scores, y=(manifest.label == "positive").astype(int))
    neg, pos = s[s.y == 0], s[s.y == 1]
    thr, _ = iso.choose_threshold(neg, iso.CFG["detection"]["threshold_policy"])
    det_by = {k: float((g.score >= thr).mean()) for k, g in pos.groupby("eval_subset")}
    sel = float(np.mean([det_by.get("real_device_rpi", 0.0), det_by.get("synthetic", 0.0)]))
    return dict(threshold=thr, selection=sel, detection_by_subset=det_by, auc=det.roc_auc(s.y, s.score),
                fpr_by_category={k: float((g.score >= thr).mean()) for k, g in neg.groupby("window_type")},
                fpr_external=float((neg[neg.eval_subset == "real_external"].score >= thr).mean()))


# ----------------------------------------------------------------------------
# Hard-negative mining (train negatives only)
# ----------------------------------------------------------------------------
def mine(model, dev, nw, out: Path) -> np.ndarray:
    _, ds = wd.build_datasets(splits=["train"], augment=False)
    d = ds["train"]
    neg = np.where(d.rows.label.to_numpy() == "negative")[0]
    sub = torch.utils.data.Subset(d, neg.tolist())
    loader = torch.utils.data.DataLoader(sub, batch_size=512, shuffle=False, num_workers=nw, collate_fn=wd.collate,
                                         worker_init_fn=rep.worker_init_fn)
    model.eval()
    sc = []
    with torch.no_grad():
        for b in loader:
            sc.append(torch.sigmoid(model(b["features"].to(dev))).cpu())
    sc = torch.cat(sc).numpy()
    H = CFG["hard_negative_mining"]
    k = min(H["max_windows"], int(math.ceil(H["top_fraction_of_train_negatives"] * len(neg))))
    top = np.argsort(-sc)[:k]
    rows = d.rows.iloc[neg[top]][["window_id", "file_id", "window_type", "source"]].assign(score=sc[top])
    rows.to_csv(out / "mined_negatives.csv", index=False)
    print(f"[mine] {k} of {len(neg)} train negatives, score >= {sc[top].min():.3f}; by type: "
          f"{rows.window_type.value_counts().to_dict()}", flush=True)
    return neg[top]


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(bcr.MODELS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mine-from", default=None, help="checkpoint of a round-1 model: enables hard-negative mining")
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = CFG["seed"]
    rep.seed_everything(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    mined = None
    if args.mine_from:
        m0 = bcr.MODELS[args.model]().to(dev)
        m0.load_state_dict(torch.load(args.mine_from, map_location=dev, weights_only=True))
        mined = mine(m0, dev, CFG["num_workers"], out)

    vd, vman = iso.load_isolated_set("validation")                # verifies the frozen v-set
    vfeat = features_of(vd, CFG["num_workers"])
    d, sampler, loader = build_train_loader(seed, CFG["batch_size"], CFG["num_workers"], mined)

    model = bcr.MODELS[args.model]().to(dev)
    O = CFG["optimizer"]
    opt = torch.optim.AdamW(model.parameters(), lr=O["lr"], weight_decay=O["weight_decay"])
    steps = args.epochs * len(loader)
    warm = CFG["schedule"]["warmup_epochs"] * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    lossf = nn.BCEWithLogitsLoss()
    ls = CFG["label_smoothing"]
    g = torch.Generator().manual_seed(rep.stable_seed("specaug", seed))

    log, best, t0 = [], None, time.time()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        tot, n = 0.0, 0
        for b in loader:
            x = spec_augment(b["features"].to(dev, non_blocking=True), CFG["spec_augment"], g)
            y = b["labels"].to(dev) * (1 - ls) + ls / 2
            loss = lossf(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        vm = validation_metrics(vman, scores_of(model, vfeat, dev))
        row = dict(epoch=epoch, train_loss=round(tot / n, 5), lr=opt.param_groups[0]["lr"],
                   seconds=round(time.time() - t0, 1), **vm)
        log.append(row)
        is_best = best is None or (vm["selection"], vm["auc"]) > (best["selection"], best["auc"])
        if is_best:
            best = row
            torch.save(model.state_dict(), out / "best.pt")
        print(f"epoch {epoch:2d} loss {row['train_loss']:.4f} | val AUC {vm['auc']:.4f} thr {vm['threshold']:.3f} "
              f"RPI {vm['detection_by_subset'].get('real_device_rpi', 0):.3f} "
              f"syn {vm['detection_by_subset'].get('synthetic', 0):.3f}{' *' if is_best else ''} | {row['seconds']:.0f}s",
              flush=True)
    torch.save(model.state_dict(), out / "last.pt")
    (out / "train_log.json").write_text(json.dumps(dict(
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        model=args.model, parameters=bcr.count_params(model), config=CFG, mined_from=args.mine_from,
        mined_windows=None if mined is None else len(mined), best_epoch=best["epoch"], best=best, epochs=log,
        validation_fingerprint=json.loads(iso.manifest_path("validation").with_suffix(".json").read_text())),
        indent=2, default=float) + "\n")
    print(f"best epoch {best['epoch']}: selection {best['selection']:.3f}, AUC {best['auc']:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
