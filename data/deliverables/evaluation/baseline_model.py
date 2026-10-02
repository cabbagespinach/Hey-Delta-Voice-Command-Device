#!/usr/bin/env python3
"""
Pipeline-check baseline: a small CNN, trained on the train split only.

This is NOT the project's wakeword model. It exists so the evaluators produce real,
non-degenerate numbers and so the training DataLoader is exercised end to end
(bucket-quota sampling, training-only augmentation). No validation, test or
streaming data is read during training. Replace it with the trained model: the
evaluators accept any torch module mapping [B, 1, 40, 147] features to [B] logits.

Usage: python baseline_model.py   -> baseline/baseline_cnn.pt, baseline/train_log.json
"""
from pathlib import Path
import datetime, json, sys, time

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dataloading"))
import wakeword_data as wd          # noqa: E402
import reproducibility as rep       # noqa: E402

OUT = HERE / "baseline"


class SmallKwsCnn(nn.Module):
    def __init__(self, n_mels: int = 40):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64),
                                      nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(64 * 2, 1)

    def forward(self, x):
        h = self.features(x)                                     # [B, 64, mels/8, frames/8]
        h = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.head(h).squeeze(-1)


def load_baseline(path=OUT / "baseline_cnn.pt") -> nn.Module:
    m = SmallKwsCnn()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()


def main():
    cfg = json.loads((HERE / "eval_config.json").read_text())["baseline"]
    rep.seed_everything(cfg["seed"])
    OUT.mkdir(exist_ok=True)
    loaders = wd.build_dataloaders(splits=["train"], batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                                   seed=cfg["seed"])
    train = loaders["train"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallKwsCnn().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * len(train))
    lossf = nn.BCEWithLogitsLoss()
    log, t0 = [], time.time()
    for epoch in range(cfg["epochs"]):
        wd.set_epoch(loaders, epoch)
        model.train()
        tot, n, correct = 0.0, 0, 0
        for b in train:
            x, y = b["features"].to(dev), b["labels"].to(dev)
            logits = model(x)
            loss = lossf(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
            correct += ((logits > 0).float() == y).sum().item()
        log.append(dict(epoch=epoch, train_loss=round(tot / n, 5), train_acc=round(correct / n, 4),
                        seconds=round(time.time() - t0, 1)))
        print(log[-1])
    torch.save(model.cpu().state_dict(), OUT / "baseline_cnn.pt")
    (OUT / "train_log.json").write_text(json.dumps(dict(
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        purpose="pipeline-check baseline, not the project model",
        data="train split only, dataloading layer (bucket-quota sampling, training-only augmentation)",
        config=cfg, parameters=sum(p.numel() for p in model.parameters()), epochs=log), indent=2) + "\n")


if __name__ == "__main__":
    main()
