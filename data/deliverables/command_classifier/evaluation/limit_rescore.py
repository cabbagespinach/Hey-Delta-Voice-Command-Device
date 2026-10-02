#!/usr/bin/env python3
"""
Shorter listener time limit (2026-10-01): re-score the Pi A/B captures as if `max_command_sec` had been shorter.
CPU only, no training; uses the ONNX models of the A/B kit exactly as on the Pi.

A capture is [0.3 s pre-roll][wait][speech ...]. The listener stops `max_command_sec` after speech STARTS, so a shorter
limit is simulated by cutting each saved capture at (speech start + limit):
- max_length captures: speech start = duration - 6.0 s exactly (that is how they ended),
- other captures: speech start = first moment after 0.55 s (pre-roll + chime ignore) where the 100 ms smoothed level
  stays >= 6 dB above the capture's 20th-percentile level for 0.15 s (the listener's rule; its background estimate
  comes from before the wake-up, which is not saved, so the capture's own quiet parts stand in for it).
Captures already shorter than the cut are unchanged.

    python limit_rescore.py [--limits 2.5 3 3.5 4 4.5 6]
Writes results/limit_rescore.csv (per capture x limit x model) and results/limit_rescore.md (summary).
"""
from pathlib import Path
import argparse, csv, sys

import numpy as np

KIT = Path(__file__).resolve().parents[1] / "model/ab_kit"
sys.path.insert(0, str(KIT))
import command_ab as ab                                                 # noqa: E402

SR, FRAME = 16000, 320
OUT = Path(__file__).resolve().parent / "results"


def speech_start(x, dur, reason):
    if reason == "max_length":
        return max(0.0, dur - 6.0)
    n = len(x) // FRAME
    db = 10 * np.log10((x[:n * FRAME].reshape(n, FRAME) ** 2).mean(1) + 1e-18)
    sm = 10 * np.log10(np.convolve(10 ** (db / 10), np.ones(5) / 5, mode="full")[:n] + 1e-18)
    thr = np.percentile(db, 20) + 6.0
    run = 0
    for i in range(int(0.55 * SR / FRAME), n):
        run = run + 1 if sm[i] >= thr else 0
        if run * FRAME / SR >= 0.15:
            return (i - run + 1) * FRAME / SR
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limits", type=float, nargs="+", default=[2.5, 3.0, 3.5, 4.0, 4.5, 6.0])
    ap.add_argument("--sessions", default=str(KIT / "ab_results"))
    a = ap.parse_args()
    models = ab.load_models(threads=1)
    rows = []
    for res in sorted(Path(a.sessions).glob("2026*/results.csv")):
        sess = res.parent
        note = None
        for r in csv.DictReader(open(res)):
            x = ab.read_wav(sess / r["file"])
            dur = len(x) / SR
            st = speech_start(x, dur, r["end_reason"])
            note = r["note"]
            for lim in a.limits:
                cut = len(x) if st is None else min(len(x), int((st + lim) * SR))
                y = x[:cut]
                out = ab.classify(models, y)
                for name, _ in models:
                    got = out[name]["balanced"][0]
                    rows.append(dict(session=sess.name, note=note, file=r["file"], expected=r["expected"],
                                     phrasing=r["phrasing"], end_reason=r["end_reason"], duration=round(dur, 2),
                                     speech_start=None if st is None else round(st, 2), limit=lim,
                                     cut_sec=round(cut / SR, 2), model=name, balanced=got,
                                     argmax=out[name]["argmax"][0], prob=round(out[name]["argmax"][1], 4),
                                     outcome=ab.outcome(r["expected"], got)))
        print(f"{sess.name}: done", flush=True)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "limit_rescore.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(), w.writerows(rows)

    lines = ["# Shorter listener time limit: Pi A/B captures re-scored (balanced rule)", "",
             "Cut = speech start + limit (6.0 = today's setting). `cut` = mean capture length the model then gets.", ""]
    names = [n for n, _ in models]
    groups = [("all sessions", lambda r: True),
              ("TV talk show session", lambda r: "talk show" in r["note"]),
              ("captures that hit the 6 s limit", lambda r: r["end_reason"] == "max_length"),
              ("all other captures", lambda r: r["end_reason"] != "max_length")]
    for title, keep in groups:
        lines += [f"## {title}", "", "| limit | cut (s) | " + " | ".join(f"{n} correct / wrong / asks" for n in names)
                  + " | non-cmd -> action |", "|---" * (3 + len(names)) + "|"]
        for lim in a.limits:
            sel = [r for r in rows if r["limit"] == lim and keep(r)]
            cmd = [r for r in sel if r["expected"] != "unknown"]
            cells = []
            for n in names:
                c = [r["outcome"] for r in cmd if r["model"] == n]
                cells.append(f"{c.count('correct')}/{len(c)}  {c.count('WRONG')}  {c.count('asks again')}")
            fa = [r for r in sel if r["expected"] == "unknown" and r["outcome"] == "false action"]
            cut = np.mean([r["cut_sec"] for r in sel]) if sel else 0
            lines.append(f"| {lim} | {cut:.1f} | " + " | ".join(cells) + f" | {len(fa)} |")
        lines.append("")
    (OUT / "limit_rescore.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
