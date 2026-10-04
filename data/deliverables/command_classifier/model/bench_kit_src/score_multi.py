#!/usr/bin/env python3
"""
Score every model of a multi-model benchmark run (vcm_bench_assistant.py --also), on the laptop.

    cd vcm-benchmark                                        # the benchmark folder (needs its `--rescore`)
    scp <user>@<pi-address>:~/vcm_bench_kit/bench_others.jsonl runs/<date-time>/
    python <kit>/score_multi.py runs/<date-time>            # default variants, see VARIANTS
    python <kit>/score_multi.py runs/<date-time> --variants hf_plus:balanced,hf_only:argmax

For each variant (model:rule) it copies the run to runs/<date-time>__<model>_<rule>/, puts that model's answer
(from bench_others.jsonl, joined by "seq") in place of the official one in trials.jsonl, and lets the benchmark
score the copy (`benchmark.py --rescore`): a full report.md per variant. Then it writes
runs/<date-time>/multi_model_report.md: one table for all variants + the trials where they disagree.

What is per model: every classification number, the slot numbers, inference time (infer_ms), FLOPs/parameters.
What is shared (the same capture for all): the wake word (missed wake = no answer for every model), response latency,
CPU / RAM / temperature of the Pi (the process runs all the models).
The original run folder is never changed (except the added multi_model_report.md).
"""
from pathlib import Path
import argparse, json, shutil, subprocess, sys

KIT = Path(__file__).resolve().parent
VARIANTS = "hf_plus:cautious,hf_plus:balanced,hf_only:cautious,hf_only:balanced"
SHORT = {"hf_plus": "bcresnet6_hf_plus", "hf_only": "bcresnet6_hf_only", "hf_ourlabels": "bcresnet6_hf_ourlabels"}


def pct(v):
    return "-" if v is None or v != v else f"{100 * v:.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the benchmark's run folder, e.g. runs/20261003-101500")
    ap.add_argument("--others", help="bench_others.jsonl copied from the Pi (default: <run>/bench_others.jsonl)")
    ap.add_argument("--variants", default=VARIANTS, help=f"model:rule list (default {VARIANTS})")
    ap.add_argument("--bench", help="the vcm-benchmark folder (default: the run folder's grandparent)")
    a = ap.parse_args()
    run = Path(a.run).resolve()
    bench = Path(a.bench).resolve() if a.bench else run.parent.parent
    if not (bench / "benchmark.py").exists():
        sys.exit(f"benchmark.py not found in {bench}; pass --bench <vcm-benchmark folder>")
    sys.path.insert(0, str(bench))
    from vcmbench import schema as S                                       # the benchmark's own label rules
    others_path = Path(a.others) if a.others else run / "bench_others.jsonl"
    if not others_path.exists():
        sys.exit(f"{others_path} not found: copy it from the Pi (scp <user>@<pi>:~/vcm_bench_kit/bench_others.jsonl "
                 f"{run}/)")
    others = {}
    for line in others_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            others[o["seq"]] = o["models"]
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    trials = [json.loads(l) for l in (run / "trials.jsonl").read_text(encoding="utf-8").splitlines() if l]
    aliases = S.build_alias_table(cfg.get("aliases"))
    order = S.id_orders()["manifest"]
    old_metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    official = None
    for t in trials:
        if t.get("pred_raw", "").startswith("{"):
            r = json.loads(t["pred_raw"])
            official = official or (r.get("model"), r.get("rule"))

    variants, rows, answers = [], [], {}
    for v in a.variants.split(","):
        short, rule = v.strip().split(":")
        model = SHORT.get(short, short)
        out = run.parent / f"{run.name}__{short}_{rule}"
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(run, out, ignore=shutil.ignore_patterns("audio", "*__*", "multi_model_report.md"))
        missing, same, changed = 0, 0, 0
        new = []
        for t in trials:
            t = dict(t)
            raw = t.get("pred_raw", "")
            if raw.startswith("{") and "seq" in json.loads(raw):
                seq = json.loads(raw)["seq"]
                ans = others.get(seq, {}).get(model)
                if ans is None:
                    missing += 1
                else:
                    intent, slot, _, variation = S.resolve_prediction(ans[rule], "", "", aliases, order)
                    changed += intent != t["pred_intent"] or slot != t["pred_slot"]
                    same += intent == t["pred_intent"] and slot == t["pred_slot"]
                    t.update(pred_intent=intent, pred_slot=slot, pred_variation=variation, pred_variation_id=None,
                             infer_ms=ans["infer_ms"],
                             pred_raw=json.dumps(dict(intent=ans[rule], model=model, rule=rule, seq=seq,
                                                      confidence=ans["confidence"], infer_ms=ans["infer_ms"])))
            new.append(t)
        if missing:
            print(f"WARNING {short}:{rule}: {missing} answer(s) missing in {others_path.name}; "
                  "those trials keep the official model's answer")
        (out / "trials.jsonl").write_text("".join(json.dumps(t) + "\n" for t in new), encoding="utf-8")
        m = dict(old_metrics)
        onnx = next((KIT / "models" / model).glob("*.onnx"), None)
        if onnx is not None:
            from vcmbench.flops import onnx_profile, format_si
            prof = onnx_profile(str(onnx))
            m["model"] = dict(prof, path=f"vcm_bench_kit/models/{model}/{onnx.name}", flops_si=f"{format_si(prof['flops'])}FLOP" if prof.get("flops") else None)
        (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        cfg2 = dict(cfg, student=f"{cfg.get('student', '')} [{model}, {rule}, same captures as {run.name}]")
        (out / "config.json").write_text(json.dumps(cfg2, indent=2), encoding="utf-8")
        res = subprocess.run([sys.executable, "benchmark.py", "--rescore", str(out)], cwd=bench,
                             capture_output=True, text=True)
        if res.returncode:
            sys.exit(f"--rescore failed for {out}:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
        mm = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        name = f"{short} {rule}" + (" (official)" if (model, rule) == official else "")
        variants.append(name)
        answers[name] = {t["order"]: t["pred_intent"] + (f" {t['pred_slot']}" if t["pred_slot"] else "")
                         for t in new}
        il, cl, bd, pi = mm["intent_level"], mm["command_level"], mm.get("breakdowns") or {}, mm["pi"]
        voice = {k: (bd.get(k) or {}).get("accuracy") for k in ("real voice", "synthetic voice")}
        rows.append([name, pct(il["accuracy"]), pct(cl["accuracy"]), pct(il["macro_f1"]),
                     pct(il["false_accept_rate"]), pct(il["false_reject_rate"]), pct(il["misfire_rate"]),
                     pct(voice["real voice"]), pct(voice["synthetic voice"]),
                     f"{pi['infer_ms'].get('mean', float('nan')):.1f}", str(out.relative_to(run.parent))])
        print(f"{name}: intent accuracy {pct(il['accuracy'])}  ({changed} trials differ from the official answer)")

    head = ["variant", "intent acc", "command acc", "F1 (macro)", "false accept (OOS fired)",
            "false reject", "misfire (wrong cmd)", "real voice", "synthetic voice", "infer ms", "full report"]
    lines = [f"# Multi-model benchmark: {run.name}", "",
             f"Same captures for every variant (one benchmark run; official model {official[0]} {official[1]}). "
             "Classification and inference time are per model; wake word, response latency and Pi CPU/RAM are "
             "shared (see each full report).", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    by_order = {t["order"]: t for t in trials}
    dis = [o for o in sorted(by_order) if len({answers[v][o] for v in variants}) > 1]
    lines += ["", f"## Trials where the variants disagree ({len(dis)})", "",
              "| # | said | truth | " + " | ".join(variants) + " |", "|" + "---|" * (3 + len(variants))]
    for o in dis:
        t = by_order[o]
        truth = t["true_intent"] + (f" {t['true_slot']}" if t.get("true_slot") else "")
        lines.append(f"| {o} | {t.get('transcript', '')} | {truth} | "
                     + " | ".join(answers[v][o] for v in variants) + " |")
    (run / "multi_model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:6 + len(rows)]))
    print(f"\nsaved {run / 'multi_model_report.md'}")


if __name__ == "__main__":
    main()
