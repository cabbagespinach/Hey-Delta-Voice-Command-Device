#!/usr/bin/env python3
"""
Phase 4a: validation-chosen confidence cutoffs and the held-out TEST results of the trained command classifiers.

    CUDA_VISIBLE_DEVICES=<one gpu> python evaluate_commands.py ../model/runs/bcresnet3 ../model/runs/bcresnet6

Decision rule on the device: take the most likely class; if it is a command but its probability is below a cutoff,
answer `unknown` ("didn't catch that"). Two cutoffs are chosen on VALIDATION (owner request: compare both):
  cautious   the lowest cutoff at which at most 2% of validation `unknown` clips trigger a command
  balanced   the same at 5%
The rate is the mean over unknown source groups (owner prompts, owner room, MUSAN, FLEURS, MSWC, public requests,
synthetic, cut-off fragments), so the large groups cannot hide a weak one.

TEST results (never used for any choice), per model and cutoff (argmax, cautious, balanced):
  per source group: owner (test session 16:18-16:33), speaker2 (never heard in training), public datasets, synthetic
  for command clips: correct / wrong command (the harmful error) / rejected as unknown (asks again)
  for unknown clips: how often each kind triggers a command
Writes results/test_results.json, results/test_summary.md, results/<model>_test_predictions.csv, and per-class
tables for the owner and speaker2 clips.
"""
from pathlib import Path
import json, os, sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
CC = HERE.parent
sys.path.insert(0, str(CC / "model"))
sys.path.insert(0, str(CC / "dataloading"))
import command_data as cd                                                 # noqa: E402
from command_model import CommandNet                                       # noqa: E402
from train_commands import predict                                         # noqa: E402

TARGETS = {"cautious": 0.02, "balanced": 0.05}
OUT = Path(os.environ.get("COMMAND_RESULTS_DIR", HERE / "results"))   # schema-B run: its own folder


def decide(P, classes, cutoff):
    unk = len(classes) - 1
    pred = P.argmax(1)
    if cutoff is not None:
        pred = np.where((pred != unk) & (P.max(1) < cutoff), unk, pred)
    return pred


def unknown_rate(pred, rows, unk):
    u = rows["class"].to_numpy() == "unknown"
    by = pd.Series(pred[u] != unk).groupby(rows.group.to_numpy()[u]).mean()
    return float(by.mean()), by.round(4).to_dict()


def choose_cutoff(P, rows, classes, target):
    unk = len(classes) - 1
    for c in np.round(np.arange(0.0, 1.0001, 0.005), 3):
        r, _ = unknown_rate(decide(P, classes, c), rows, unk)
        if r <= target:
            return float(c)
    return 1.0


def summarise(pred, rows, classes):
    unk = len(classes) - 1
    y = rows["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    r = rows.assign(pred=pred, y=y)
    r["who"] = np.where(r.group == "owner", np.where(r.speaker == "speaker2", "speaker2", "owner"), r.group)
    out = {}
    for who, s in r[r["class"] != "unknown"].groupby("who"):
        per = s.groupby("class").apply(lambda g: pd.Series(dict(correct=(g.pred == g.y).mean(),
                                                                 wrong=((g.pred != g.y) & (g.pred != unk)).mean(),
                                                                 rejected=(g.pred == unk).mean())),
                                       include_groups=False)
        out[who] = dict(clips=int(len(s)), classes=int(s["class"].nunique()),
                        correct=round(float(per.correct.mean()), 4), wrong_command=round(float(per.wrong.mean()), 4),
                        rejected=round(float(per.rejected.mean()), 4))
    u = r[r["class"] == "unknown"]
    out["unknown_false_action_by_group"] = {g: round(float((s.pred != unk).mean()), 4) for g, s in u.groupby("group")}
    out["unknown_false_action_mean"] = round(float(np.mean(list(out["unknown_false_action_by_group"].values()))), 4)
    return out, r


def main():
    runs = [Path(x).resolve() for x in sys.argv[1:]]
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = cd.load_config()
    test = cd.EvalClips(cfg, "test")
    classes = cd.classes()
    res = {}
    for run in runs:
        name = run.name
        ck = torch.load(run / "best.pt", map_location="cpu")
        assert ck["classes"] == classes
        model = CommandNet(len(classes), ck["tau"], arch=ck.get("arch", "bcresnet")).to("cuda")
        model.load_state_dict(ck["state_dict"])
        vp = pd.read_csv(run / "val_predictions.csv")
        Pv = vp[[f"p_{c}" for c in classes]].to_numpy()
        cut = {k: choose_cutoff(Pv, vp, classes, t) for k, t in TARGETS.items()}
        Pt = predict(model, test, cfg, "cuda")
        res[name] = dict(tau=ck["tau"], best_epoch=ck["epoch"], validation=ck["metrics"], cutoffs=cut,
                         params=int(sum(p.numel() for p in model.net.parameters())), test={})
        tp = test.rows[["path", "class", "dataset", "group", "speaker"]].copy()
        for i, c in enumerate(classes):
            tp[f"p_{c}"] = Pt[:, i]
        tp.to_csv(OUT / f"{name}_test_predictions.csv", index=False)
        for rule, c in [("argmax", None)] + list(cut.items()):
            s, r = summarise(decide(Pt, classes, c), test.rows, classes)
            res[name]["test"][rule] = s
            if rule != "argmax":
                for who in ("owner", "speaker2"):
                    x = r[(r.who == who) & (r["class"] != "unknown")]
                    t = x.groupby("class").apply(lambda g: pd.Series(dict(
                        clips=len(g), correct=(g.pred == g.y).mean(),
                        wrong=((g.pred != g.y) & (g.pred != len(classes) - 1)).mean(),
                        predicted_as=", ".join(sorted({classes[p] for p in g.pred[g.pred != g.y]})))),
                        include_groups=False)
                    t.to_csv(OUT / f"{name}_{rule}_{who}_per_class.csv")
        print(f"{name}: cutoffs {cut}", flush=True)
    (OUT / "test_results.json").write_text(json.dumps(res, indent=1))

    L = ["# Command classifier: test results", "",
         "Cutoffs chosen on validation; test clips were never used for any choice. Command rates are macro averages over "
         "classes: **correct**, **wrong command** (the harmful error), **rejected** (answered 'unknown' = asks again).", ""]
    for name, r in res.items():
        L += [f"## {name} ({r['params']:,} parameters, best epoch {r['best_epoch']})", "",
              f"Cutoffs: cautious {r['cutoffs']['cautious']:.3f}, balanced {r['cutoffs']['balanced']:.3f}", "",
              "| rule | who | clips | correct | wrong command | rejected |", "|---|---|---|---|---|---|"]
        for rule, s in r["test"].items():
            for who in ("owner", "speaker2", "web", "synthetic", "classmates"):
                if who in s:
                    x = s[who]
                    L.append(f"| {rule} | {who} | {x['clips']} | {x['correct']:.1%} | {x['wrong_command']:.1%} | "
                             f"{x['rejected']:.1%} |")
        L += ["", "Unknown clips that trigger a command (test):", "", "| rule | " +
              " | ".join(r["test"]["argmax"]["unknown_false_action_by_group"]) + " | mean |",
              "|---|" + "---|" * (len(r["test"]["argmax"]["unknown_false_action_by_group"]) + 1)]
        for rule, s in r["test"].items():
            v = s["unknown_false_action_by_group"]
            L.append(f"| {rule} | " + " | ".join(f"{x:.1%}" for x in v.values()) +
                     f" | {s['unknown_false_action_mean']:.1%} |")
        L.append("")
    (OUT / "test_summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
