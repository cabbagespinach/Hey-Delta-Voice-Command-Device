#!/usr/bin/env python3
"""
Train the BC-ResNet command classifier (config: train_config.json; data: ../dataloading).

    CUDA_VISIBLE_DEVICES=<one gpu> python train_commands.py --tau 3 --out runs/bcresnet3
    # small-scale probe (short run, optionally without some sources):
    CUDA_VISIBLE_DEVICES=<gpu> python train_commands.py --tau 1 --epochs 8 --epoch-draws 6000 --out runs/probe_all
    CUDA_VISIBLE_DEVICES=<gpu> python train_commands.py --tau 1 --epochs 8 --epoch-draws 6000 \
        --exclude-groups owner --out runs/probe_no_owner

Every epoch the model scores the whole validation split (unaugmented). Selection score = mean of
  macro accuracy on command clips of each source group (owner recordings, public datasets, synthetic voices)
  and 1 - the rate at which validation `unknown` clips are taken for a command (argmax, no confidence cutoff),
so no single source (the public datasets are 60% of the clips) decides which epoch is kept. Test data is never read.
Writes <out>/best.pt, last.pt, history.json and val_predictions.csv (probabilities of the best epoch).
"""
from pathlib import Path
import argparse, json, math, sys, time

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dataloading"))
import command_data as cd                                                 # noqa: E402
from command_model import CommandNet                                       # noqa: E402


def spec_augment(x, g, p):
    """SpecAugment on normalised features [B, 1, F, T] (masked value 0 = the train mean)."""
    if p["probability"] <= 0:
        return x
    x = x.clone()
    B, _, F, T = x.shape
    for b in range(B):
        if torch.rand(1, generator=g).item() >= p["probability"]:
            continue
        for _ in range(p["freq_masks"]):
            w = int(torch.randint(0, p["max_freq_bins"] + 1, (1,), generator=g))
            f0 = int(torch.randint(0, F - w + 1, (1,), generator=g))
            x[b, :, f0:f0 + w, :] = 0
        for _ in range(p["time_masks"]):
            w = int(torch.randint(0, p["max_time_frames"] + 1, (1,), generator=g))
            t0 = int(torch.randint(0, T - w + 1, (1,), generator=g))
            x[b, :, :, t0:t0 + w] = 0
    return x


@torch.no_grad()
def predict(model, ds, cfg, device):
    model.eval()
    out, rows = [], []
    for w, y, i in cd.make_loader(ds, cfg, batch_size=128):
        out.append(torch.softmax(model(w.to(device, non_blocking=True)), 1).float().cpu())
        rows.append(i)
    P = torch.cat(out).numpy()
    idx = torch.cat(rows).numpy()
    order = np.argsort(idx)
    return P[order]


# classmates: schema-B run (2026-10-01); a group without validation clips is skipped (nan)
GROUPS = ("owner", "web", "synthetic", "classmates")


def metrics(P, rows: pd.DataFrame, classes: list) -> dict:
    """Macro accuracy on commands per source group, unknown false-action rate (argmax)."""
    y = rows["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    pred = P.argmax(1)
    unk = len(classes) - 1
    r = rows.assign(ok=(pred == y), pred=pred)
    out = {}
    for g in GROUPS:
        s = r[(r.group == g) & (r["class"] != "unknown")]
        out[f"{g}_macro_acc"] = float(s.groupby("class").ok.mean().mean()) if len(s) else float("nan")
    u = r[r["class"] == "unknown"]
    out["unknown_false_action"] = float((u.pred != unk).mean()) if len(u) else float("nan")
    out["unknown_false_action_by_group"] = {g: round(float((s.pred != unk).mean()), 4) for g, s in u.groupby("group")}
    parts = [out[f"{g}_macro_acc"] for g in GROUPS if not math.isnan(out[f"{g}_macro_acc"])]
    out["selection_score"] = float(np.mean(parts + [1 - out["unknown_false_action"]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "train_config.json"))
    ap.add_argument("--tau", type=float, default=3)
    ap.add_argument("--arch", default="bcresnet", choices=["bcresnet", "dscnn"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--epoch-draws", type=int)
    ap.add_argument("--exclude-groups", nargs="*", default=[])
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tc = json.loads(Path(a.config).read_text())
    cfg = cd.load_config()
    if a.epoch_draws:
        cfg["epoch_draws"] = a.epoch_draws
    epochs = a.epochs or tc["epochs"]
    out = Path(a.out)
    if not out.is_absolute():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(tc["seed"])
    device = "cuda"

    train = cd.TrainDraws(cfg, augment=not a.no_augment, exclude_groups=a.exclude_groups)
    val = cd.EvalClips(cfg, "validation")
    classes = train.classes
    model = CommandNet(len(classes), a.tau, arch=a.arch).to(device)
    opt = torch.optim.AdamW(model.net.parameters(), lr=tc["optimizer"]["lr"], weight_decay=tc["optimizer"]["weight_decay"])
    steps_per_epoch = math.ceil(len(train) / cfg["batch_size"])
    total, warm = epochs * steps_per_epoch, tc["schedule"]["warmup_epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    lossf = nn.CrossEntropyLoss(label_smoothing=tc["label_smoothing"])
    g = torch.Generator().manual_seed(tc["seed"])
    hist, best = [], -1
    json.dump(dict(args=vars(a), train_config=tc, loader_config=cfg, classes=classes,
                   params=sum(p.numel() for p in model.net.parameters())), open(out / "run_info.json", "w"), indent=1)
    print(f"{len(train.rows)} train clips ({len(train)} draws/epoch), {len(val)} validation clips, "
          f"{sum(p.numel() for p in model.net.parameters()):,} parameters", flush=True)
    for ep in range(epochs):
        t0 = time.time()
        train.set_epoch(ep)
        model.train()
        tl, n = 0.0, 0
        for w, y, _ in cd.make_loader(train, cfg):
            w, y = w.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(w, feature_hook=lambda x: spec_augment(x, g, tc["spec_augment"]))
            loss = lossf(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(y)
            n += len(y)
        P = predict(model, val, cfg, device)
        m = metrics(P, val.rows, classes)
        m.update(epoch=ep, train_loss=round(tl / n, 4), minutes=round((time.time() - t0) / 60, 2))
        hist.append(m)
        if m["selection_score"] > best:
            best = m["selection_score"]
            torch.save(dict(state_dict=model.state_dict(), classes=classes, tau=a.tau, arch=a.arch, epoch=ep, metrics=m),
                       out / "best.pt")
            vp = val.rows[["path", "class", "dataset", "group", "speaker"]].copy()
            for i, c in enumerate(classes):
                vp[f"p_{c}"] = P[:, i]
            vp.to_csv(out / "val_predictions.csv", index=False)
        torch.save(dict(state_dict=model.state_dict(), classes=classes, tau=a.tau, arch=a.arch, epoch=ep, metrics=m), out / "last.pt")
        json.dump(hist, open(out / "history.json", "w"), indent=1)
        print(f"ep {ep:2d} loss {m['train_loss']:.3f} | owner {m['owner_macro_acc']:.3f} web {m['web_macro_acc']:.3f} "
              f"synth {m['synthetic_macro_acc']:.3f} unk-false {m['unknown_false_action']:.3f} | "
              f"score {m['selection_score']:.3f}{' *' if m['selection_score'] >= best else ''} ({m['minutes']} min)",
              flush=True)
    print(f"best selection score {best:.3f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
