#!/usr/bin/env python3
"""
Command models side by side on the Raspberry Pi: every "Hey Delta" capture goes to BOTH models (same audio), and
each answer is scored against the command you were asked to say.

    python3 command_ab.py check                                   # each model vs the server's scores
    python3 command_ab.py live --prompts commands.txt --note "..." # prompted live test (Enter: d / s / q, see below)
    python3 command_ab.py live --prompts phrasings.txt            # same, with other ways of saying each command
    python3 command_ab.py live                                    # no prompts: free talk, answers shown, not scored
    python3 command_ab.py replay ab_results/<session>              # re-score saved captures (or a demo_record folder)
    python3 command_ab.py devices                                 # list microphones

Prompt file lines: "label | words to say" (a label may appear on several lines with different words). Labels map to
classes through label_map.csv (skip -> next, louder -> volume_up, unknown_* -> unknown). While live, type then Enter:
    d   delete the last capture (you said the wrong thing; it is asked again)
    s   skip the current prompt
    q   quit and print the report

Output: ab_results/<date-time>/<label>/<label>_<time>.wav (the capture) and results.csv (one row per capture: what
you were asked, each model's answer and probability under every rule). The report at the end compares the models.
"""
from pathlib import Path
import argparse, csv, datetime, json, random, sys, threading, time, wave

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from command_pi import CommandClassifier                                   # noqa: E402

RULES = ["argmax", "cautious", "balanced"]
SR = 16000


def load_models(threads=1):
    cfg = json.loads((HERE / "models.json").read_text())
    return [(m["name"], CommandClassifier(HERE / "models" / m["config"], rule="argmax", threads=threads))
            for m in cfg["models"]]


def label_classes():
    with open(HERE / "label_map.csv") as f:
        return {r["label"]: r["class"] for r in csv.DictReader(f)}


def read_prompts(path):
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lab, _, text = (x.strip() for x in line.partition("|"))
            out.append((lab, text or lab))
    return out


def write_wav(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def read_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2, f"{path}: need 16 kHz 16-bit WAV"
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    return x.mean(1).astype(np.float32) / 32768.0


def classify(models, audio):
    """Both models on the same audio -> {name: {"probs", "ms", rule: (label, prob)}}."""
    out = {}
    for name, clf in models:
        t0 = time.perf_counter()
        p = clf.probs(audio)
        ms = (time.perf_counter() - t0) * 1000
        k = int(np.argmax(p))
        top = clf.classes[k]
        r = {"probs": p, "ms": ms, "argmax": (top, float(p[k]))}
        for rule in ("cautious", "balanced"):
            c = clf.cfg["cutoffs"][rule]
            r[rule] = ("unknown" if top != "unknown" and p[k] < c else top, float(p[k]))
        out[name] = r
    return out


def outcome(want, got):
    if want is None:
        return ""
    if got == want:
        return "correct"
    if want == "unknown":
        return "false action"                  # a non-command made it do something
    return "asks again" if got == "unknown" else "WRONG"


MARK = {"correct": "ok ", "asks again": "?  ", "WRONG": "XX ", "false action": "XX ", "": ""}


def row_for(models, res, label, want, phrasing, file, extra):
    row = dict(file=file, label=label, expected=want or "", phrasing=phrasing, **extra)
    for name, _ in models:
        for rule in RULES:
            got, p = res[name][rule]
            row[f"{name}_{rule}"] = got
            row[f"{name}_{rule}_outcome"] = outcome(want, got)
        row[f"{name}_prob"] = round(res[name]["argmax"][1], 4)
        row[f"{name}_ms"] = round(res[name]["ms"], 1)
    return row


def show(models, res, want, rule):
    for name, _ in models:
        got, p = res[name][rule]
        o = outcome(want, got)
        alt = "  ".join(f"{r}: {res[name][r][0]}" for r in RULES if r != rule and res[name][r][0] != got)
        print(f"      {MARK[o]}{name:22s} {got:20s} {p:.2f}  {o:12s} {('(' + alt + ')') if alt else ''}")


def report(rows, models, out=None):
    rows = [r for r in rows if r["expected"]]
    lines = [f"\n=== Report: {len(rows)} scored captures "
             f"({sum(r['expected'] != 'unknown' for r in rows)} commands, "
             f"{sum(r['expected'] == 'unknown' for r in rows)} non-commands) ==="]
    if not rows:
        lines.append("(nothing scored: use --prompts to know what was said)")
    cmd = [r for r in rows if r["expected"] != "unknown"]
    unk = [r for r in rows if r["expected"] == "unknown"]
    for rule in RULES:
        lines.append(f"\nrule {rule}:")
        lines.append(f"  {'model':22s} {'correct':>8s} {'WRONG':>8s} {'asks again':>11s} {'non-cmd -> action':>18s}")
        for name, _ in models:
            o = [r[f"{name}_{rule}_outcome"] for r in cmd]
            u = [r[f"{name}_{rule}_outcome"] for r in unk]
            pct = lambda xs, k: f"{(xs.count(k) / len(xs) * 100):.0f}%" if xs else "-"
            lines.append(f"  {name:22s} {pct(o, 'correct'):>8s} {pct(o, 'WRONG'):>8s} {pct(o, 'asks again'):>11s} "
                         f"{pct(u, 'false action'):>18s}")
    if len(models) == 2 and cmd:
        (a, _), (b, _) = models
        for rule in ("balanced", "cautious"):
            ok_a = [r[f"{a}_{rule}_outcome"] == "correct" for r in cmd]
            ok_b = [r[f"{b}_{rule}_outcome"] == "correct" for r in cmd]
            only_a = sum(x and not y for x, y in zip(ok_a, ok_b))
            only_b = sum(y and not x for x, y in zip(ok_a, ok_b))
            lines.append(f"\n{rule}: only {a} right on {only_a} command(s); only {b} right on {only_b}")
        diffs = [r for r in cmd if r[f"{a}_balanced_outcome"] != r[f"{b}_balanced_outcome"]]
        if diffs:
            lines.append("captures where they disagree (balanced):")
            for r in diffs[:25]:
                lines.append(f"  {r['phrasing'][:38]:38s} {a}: {r[f'{a}_balanced']:16s} {b}: {r[f'{b}_balanced']}")
    by = {}
    for r in cmd:
        by.setdefault(r["phrasing"], []).append(r)
    if len(by) > 1:
        lines.append("\nper phrasing (balanced, correct):")
        for ph, rs in sorted(by.items()):
            lines.append(f"  {ph[:40]:40s} n={len(rs):2d}  " + "  ".join(
                f"{name}: {sum(r[f'{name}_balanced_outcome'] == 'correct' for r in rs)}/{len(rs)}" for name, _ in models))
    ms = {name: [float(r[f"{name}_ms"]) for r in rows if r.get(f"{name}_ms")] for name, _ in models}
    lines.append("\ntime per capture: " + "  ".join(f"{n}: {np.median(v):.0f} ms" for n, v in ms.items() if v))
    text = "\n".join(lines)
    print(text)
    if out:
        Path(out).write_text(text + "\n")
        print(f"\n(report saved to {out})")


def cmd_check(models):
    ok = True
    for name, clf in models:
        r = np.load(HERE / "models" / f"{name}_reference.npz")
        p = np.stack([clf.probs(x) for x in r["audio"]])
        d = np.abs(p - r["probs"]).max()
        same = int((p.argmax(1) == r["probs"].argmax(1)).sum())
        good = d < 1e-3 and same == len(p)
        ok &= good
        print(f"{'OK  ' if good else 'FAIL'} {name:22s} max prob difference {d:.1e}, same answer {same}/{len(p)}")
    print("all models match the server" if ok else "a model does NOT match the server: tell Claude")


def cmd_live(models, a):
    sys.path.insert(0, str(HERE / "listener"))
    from heydelta_listener import HeyDeltaListener, Settings
    cls = label_classes()
    prompts = read_prompts(a.prompts) if a.prompts else []
    out = Path(a.out) / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rows, order, state = [], [], dict(cur=None, last=None)
    fields = None

    def next_prompt():
        if not prompts:
            return
        if not order:
            order.extend(random.sample(prompts, len(prompts)) if a.order == "random" else prompts)
        state["cur"] = order.pop(0)
        print(f"\n>>> say: \"Hey Delta, {state['cur'][1]}\"      [{state['cur'][0]}]")

    def save_rows():
        with open(out / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file"])
            w.writeheader(), w.writerows(rows)

    def on_command(cap):
        label, phrasing = state["cur"] if state["cur"] else ("unlabeled", "")
        want = cls.get(label) if prompts else None
        if cap.audio is None:
            print("    (nothing captured - say it again)")
            return
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        path = out / label / f"{label}_{stamp}.wav"
        write_wav(path, cap.audio)
        res = classify(models, cap.audio)
        rows.append(row_for(models, res, label, want, phrasing, str(path.relative_to(out)),
                            dict(speaker=a.speaker, duration_sec=round(cap.duration_sec, 2), end_reason=cap.reason,
                                 note=a.note, time=stamp)))
        save_rows()
        state["last"] = (path, state["cur"])
        print(f"    capture {cap.duration_sec:.1f} s ({cap.reason})   rule shown: {a.rule}")
        show(models, res, want, a.rule)
        next_prompt()

    listener = HeyDeltaListener(on_command=on_command, device=a.device, config_path=HERE / "listener/deploy_config.json",
                                settings=Settings(chimes=not a.no_chimes))

    def keys():
        for line in sys.stdin:
            k = line.strip().lower()
            if k == "q":
                listener.stop()
                return
            if k == "s" and prompts:
                print("    skipped")
                next_prompt()
            elif k == "d" and state["last"]:
                path, prompt = state["last"]
                path.unlink(missing_ok=True)
                rows[:] = [r for r in rows if r["file"] != str(path.relative_to(out))]
                save_rows()
                state["last"] = None
                print("    deleted the last capture - say it again")
                if prompt:
                    order.insert(0, state["cur"])
                    state["cur"] = prompt
                    print(f"\n>>> say: \"Hey Delta, {prompt[1]}\"      [{prompt[0]}]")

    threading.Thread(target=keys, daemon=True).start()
    print(f"models: {', '.join(n for n, _ in models)}   results in {out}")
    if not prompts:
        print("No --prompts: answers are shown but not scored. Use --prompts commands.txt to score them.")
    next_prompt()
    try:
        listener.run(minutes=a.minutes)
    except KeyboardInterrupt:
        pass
    report(rows, models, out / "report.txt")


def cmd_replay(models, a):
    """Re-score saved captures: an ab_results session (results.csv) or a demo_record folder (manifest.csv)."""
    d = Path(a.folder)
    cls = label_classes()
    src = d / "results.csv" if (d / "results.csv").exists() else d / "manifest.csv"
    assert src.exists(), f"no results.csv or manifest.csv in {d}"
    with open(src) as f:
        items = [(r["file"], r["label"], r.get("phrasing", "") or r["label"]) for r in csv.DictReader(f)]
    rows = []
    for file, label, phrasing in items:
        if not (d / file).exists():
            continue
        res = classify(models, read_wav(d / file))
        rows.append(row_for(models, res, label, cls.get(label), phrasing, file, {}))
    print(f"{len(rows)} captures from {src}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report(rows, models, out / f"replay_{d.resolve().name}_{datetime.datetime.now():%Y%m%d-%H%M%S}.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "live", "replay", "devices"])
    ap.add_argument("folder", nargs="?", help="replay: an ab_results/<session> or demo_record recordings folder")
    ap.add_argument("--prompts", help="commands.txt or phrasings.txt (label | words to say)")
    ap.add_argument("--order", choices=["cycle", "random"], default="random")
    ap.add_argument("--rule", choices=RULES, default="balanced", help="which rule to print live (all are saved)")
    ap.add_argument("--speaker", default="owner")
    ap.add_argument("--note", default="", help="saved with every capture (room, mic, noise...)")
    ap.add_argument("--device", type=int, default=None, help="microphone number (see: devices)")
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--no-chimes", action="store_true")
    ap.add_argument("--out", default=str(HERE / "ab_results"))
    a = ap.parse_args()
    if a.mode == "devices":
        import sounddevice as sd
        print(sd.query_devices())
        return
    models = load_models()
    {"check": lambda: cmd_check(models), "live": lambda: cmd_live(models, a),
     "replay": lambda: cmd_replay(models, a)}[a.mode]()


if __name__ == "__main__":
    main()
