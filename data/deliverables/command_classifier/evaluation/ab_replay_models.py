#!/usr/bin/env python3
"""
Re-score the owner's Pi A/B captures (model/ab_kit/ab_results, 2026-10-01; never used for training) with several
exported models, exactly as the Pi does (ONNX, the kit's scoring). CPU only.

    python ab_replay_models.py bcresnet6 bcresnet6_ownernoise bcresnet6_improved

Models come from model/export/<name>/command_config.json. Cutoffs are each model's own VALIDATION-chosen ones.
Writes results/ab_replay_<names>.md (per condition: correct / WRONG / asks again; reworded commands split into
wordings now in training vs HELD_OUT wordings never trained) and results/ab_replay_<names>.csv (per capture).
"""
from pathlib import Path
import csv, sys

HERE = Path(__file__).resolve().parent
CC = HERE.parent
KIT = CC / "model/ab_kit"
sys.path.insert(0, str(KIT)), sys.path.insert(0, str(CC))
import command_ab as ab                                                 # noqa: E402
from command_pi import CommandClassifier                               # noqa: E402
from generate_phrasing_variants import HELD_OUT, TRAIN                  # noqa: E402

CONDITIONS = [("trained wording, quiet (you, talker #2, your voice message)",
               lambda r: r["session"] in ("20261001-123944", "20261001-125647", "20261001-125842")),
              ("TV talk show", lambda r: r["session"] == "20261001-131935"),
              ("reworded commands, aircon", lambda r: r["session"] == "20261001-132809"),
              ("  ... wordings now in training", lambda r: r["session"] == "20261001-132809" and r["wording"] == "trained now"),
              ("  ... HELD-OUT wordings (never trained)", lambda r: r["session"] == "20261001-132809" and r["wording"] == "held out"),
              ("  ... original trained wording", lambda r: r["session"] == "20261001-132809" and r["wording"] == "original"),
              ("voice message, talker #3", lambda r: r["session"] == "20261001-134101"),
              ("ALL", lambda r: True)]


def wording(label, phrasing):
    norm = lambda s: s.lower().strip(" ?.!")
    if any(norm(phrasing) == norm(w) for w in HELD_OUT.values()):
        return "held out"
    if any(norm(phrasing) == norm(w) for ws in TRAIN.values() for w in ws):
        return "trained now"
    return "original"


def main():
    names = sys.argv[1:]
    models = [(n, CommandClassifier(CC / "model/export" / n / "command_config.json", rule="argmax")) for n in names]
    rows = []
    for res in sorted((KIT / "ab_results").glob("2026*/results.csv")):
        for r in csv.DictReader(open(res)):
            out = ab.classify(models, ab.read_wav(res.parent / r["file"]))
            row = ab.row_for(models, out, r["label"], r["expected"] or None, r["phrasing"], r["file"],
                             dict(session=res.parent.name, note=r["note"], end_reason=r["end_reason"]))
            row["wording"] = wording(r["label"], r["phrasing"])
            rows.append(row)
        print(res.parent.name, "done", flush=True)
    tag = "_".join(names)
    with open(HERE / f"results/ab_replay_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(), w.writerows(rows)
    md = ["# Pi A/B captures re-scored (balanced rule, each model's validation cutoffs)", "",
          "correct / WRONG (wrong command) / asks again, over command captures; `non-cmd` = non-commands that made it act.", "",
          "| condition | n | " + " | ".join(names) + " | non-cmd -> action |", "|---" * (3 + len(names)) + "|"]
    for title, keep in CONDITIONS:
        sel = [r for r in rows if keep(r)]
        cmd = [r for r in sel if r["expected"] != "unknown"]
        cells = []
        for n in names:
            o = [r[f"{n}_balanced_outcome"] for r in cmd]
            cells.append(f"{o.count('correct')} / {o.count('WRONG')} / {o.count('asks again')}")
        fa = " ".join(f"{sum(r[f'{n}_balanced_outcome'] == 'false action' for r in sel)}" for n in names)
        md.append(f"| {title} | {len(cmd)} | " + " | ".join(cells) + f" | {fa} |")
    md += ["", "## Every capture where the models differ (balanced)", "",
           "| session | said | expected | " + " | ".join(names) + " |", "|---" * (3 + len(names)) + "|"]
    for r in rows:
        got = [r[f"{n}_balanced"] for n in names]
        if len(set(got)) > 1:
            md.append(f"| {r['session'][-6:]} | {r['phrasing']} | {r['expected']} | " + " | ".join(got) + " |")
    (HERE / f"results/ab_replay_{tag}.md").write_text("\n".join(md))
    print("\n".join(md[:16]))


if __name__ == "__main__":
    main()
