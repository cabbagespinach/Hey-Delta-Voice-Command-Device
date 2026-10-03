#!/usr/bin/env python3
"""
Side-by-side version of the class benchmark report (airimonda/vcm-benchmark, report.md): the same sections, row
names and number formats, one column per benchmarked model. Reads each run's metrics.json as written by the
benchmark (or by `benchmark.py --rescore`).

    python make_benchmark_comparison.py            # -> benchmark_comparison.md

_fmt / _ci / _table are copied from vcmbench/report.py (vcm-benchmark ab39857) so every number is printed exactly
as in the per-run reports.
"""
from pathlib import Path
import json, math

HERE = Path(__file__).resolve().parent
OUT = HERE / "benchmark_comparison.md"
# (column header, run folder holding metrics.json, how the run was scored)
COLUMNS = [
    ("hf_plus cautious (Oct 2)", "20261002-135144__rescored_ab39857", "live run 20261002-135144"),
    ("hf_plus balanced (Oct 3)", "20261003-075251__hf_plus_balanced", "same captures, re-scored"),
    ("hf_plus cautious (Oct 3)", "20261003-075251__hf_plus_cautious", "same captures, re-scored"),
    ("hf_only balanced (Oct 3)", "20261003-075251__hf_only_balanced", "same captures, re-scored"),
    ("hf_only cautious (Oct 3)", "20261003-075251__hf_only_cautious", "same captures, re-scored"),
]
MODELS = {"hf_plus": "bcresnet6_hf_plus: class HF dataset + our data",
          "hf_only": "bcresnet6_hf_only: class HF dataset only"}


def _fmt(v, pct=False, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if pct:
        return f"{100 * v:.1f}%"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _ci(ci):
    return "-" if ci is None or any(math.isnan(x) for x in ci) else f"[{100*ci[0]:.0f}-{100*ci[1]:.0f}%]"


def _table(rows):
    w = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |"
    return [line(rows[0]), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [line(r) for r in rows[1:]]


def side(label_rows, ms, first="metric"):
    """One row per (label, f(metrics)), one column per run."""
    rows = [[first] + [h for h, _ in ms]]
    for label, f in label_rows:
        rows.append([label] + [f(m) for _, m in ms])
    return _table(rows) + [""]


def main():
    ms = [(h, json.loads((HERE / d / "metrics.json").read_text())) for h, d, _ in COLUMNS]
    out = ["# VCM benchmark - side by side - Hey Delta (Raspberry Pi 5)", ""]
    out += ["Each column is one model and cutoff rule, scored by the class benchmark "
            "(github.com/airimonda/vcm-benchmark, scoring of commit ab39857). Rows, names and number formats are "
            "the benchmark's own report.md; the full per-run reports are linked below.", ""]
    rows = [["column", "how it was run", "full report"]]
    for (h, d, how) in COLUMNS:
        rows.append([h, how, f"[{d}/report.md]({d}/report.md)"])
    out += _table(rows) + [""]
    out += ["Models (all BC-ResNet-6, 474,512 parameters with the built-in front end, same 31 commands + "
            "'not a command'):", ""] + [f"- {v}" for v in MODELS.values()] + [""]
    out += ["Rules: `cautious` / `balanced` = the model's two validated confidence cutoffs (below the cutoff it "
            "answers out of scope); cutoffs: hf_plus 0.61 / 0.34, hf_only 0.89 / 0.67.", ""]
    m0 = ms[-1][1]["meta"]
    oct3 = ("- Oct 3: one benchmark run in which the Pi ran several command models on every capture (they "
            "answered one after the other). Each Oct 3 column is one model's answers to these **same captures**, "
            "scored with the benchmark's own `--rescore`; they are not separate live runs. Classification, slots, "
            "inference time and model size are per column; the wake word, response latency (time to the Pi's "
            "first answer) and Pi CPU / RAM / temperature are those of the one run.")
    out += ["How the runs were made:", "",
            f"- Both days: wake word **{m0.get('wake_word')}**, the class holdout pinned to Hugging Face revision "
            f"da92a79 (202 trials with the wake word + 16 without), shuffle seed {m0.get('seed')}, connection "
            f"{m0.get('mode')}, laptop speaker about 1 m from the Pi.",
            oct3,
            "- Oct 2: a separate run (hf_plus, cautious only), re-scored with the same benchmark version so its "
            "report has the same sections; its answers and numbers are unchanged "
            "([original report](20261002-135144/report.md)). Different day, so room and microphone setup can "
            "differ from Oct 3.", ""]

    # ---- At a glance (overall group)
    o = lambda m: m["breakdowns"]["overall"]
    cnt = lambda v, key, k, n: f"{_fmt(v.get(key), True)} ({v.get(k)}/{v.get(n)})" if v.get(n) else "-"
    glance = [
        ("intent accuracy (19)", lambda m: _fmt(o(m).get("accuracy"), True)),
        ("command accuracy (93)", lambda m: _fmt(o(m).get("command_accuracy"), True)),
        ("false accept (out of scope fired)", lambda m: cnt(o(m), "false_accept_rate", "false_accepts", "n_out_of_scope")),
        ("false reject (command ignored)", lambda m: _fmt(o(m).get("false_reject_rate"), True)),
        ("false wake (no wake word, fired)", lambda m: cnt(o(m), "false_wake_rate", "false_wakes", "n_no_wake")),
        ("slot exact", lambda m: _fmt(o(m).get("slot_exact_rate"), True)),
        ("latency p95", lambda m: f"{_fmt(o(m).get('latency_p95'), nd=2)} s" if o(m).get("latency_p95") is not None else "-"),
    ]
    out += ["## At a glance", ""] + side(glance, ms, "")

    # ---- Classification: one table per label level
    out += ["# Detailed metrics", "", "## Classification", ""]
    keys = [("accuracy", "accuracy"), ("balanced_accuracy", "balanced accuracy"),
            ("macro_precision", "precision (macro)"), ("macro_recall", "recall (macro)"),
            ("macro_f1", "F1 (macro)"), ("macro_f2", "F2 (macro)"),
            ("false_accept_rate", "false accept rate (OOS fired)"),
            ("false_reject_rate", "false reject rate (in-scope silent/rejected)"),
            ("misfire_rate", "misfire rate (wrong command fired)")]
    fw = lambda m: (f"{_fmt(m['false_wake']['false_wake_rate'], True)} {_ci(m['false_wake']['false_wake_ci95'])} "
                    f"({m['false_wake']['false_wakes']}/{m['false_wake']['n']})")
    for lvl, title in (("intent_level", "19 intents (+reject)"), ("command_level", "93 commands (+reject)")):
        lines = [(label, (lambda k: lambda m: _fmt(m[lvl][k], True))(k)) for k, label in keys]
        lines += [("accuracy 95% CI", lambda m: _ci(m[lvl]["accuracy_ci95"]))]
        if lvl == "intent_level":
            lines += [("false accept 95% CI", lambda m: f"{_ci(m[lvl]['false_accept_ci95'])} "
                                                        f"({m[lvl]['false_accepts']}/{m[lvl]['n_out_of_scope']})")]
        else:
            lines += [("false accept 95% CI", lambda m: _ci(m[lvl]["false_accept_ci95"]))]
        lines += [("false wake rate (command without wake word fired)", fw)]
        out += [f"### {title}", ""] + side(lines, ms)
    p = lambda m: m["pipeline"]
    out += ["### Responses", ""] + side([
        ("fired a command", lambda m: _fmt(p(m)["response_rate"], True)),
        ("no response", lambda m: p(m)["no_response"]),
        ("extra fires", lambda m: p(m)["extra_fires"]),
        ("wake detect rate", lambda m: _fmt(p(m)["wake_detect_rate"], True))], ms)

    # ---- Overall vs real vs synthetic voices: one table per group
    out += ["## Overall vs real vs synthetic voices", "",
            "Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 "
            "out-of-scope clips are all real recordings (none are synthetic), so there is no false accept "
            "rate for synthetic voices.", ""]
    rate = lambda v, key, k, n: f"{_fmt(v.get(key), True)} ({v.get(k)}/{v.get(n)})" if v.get(n) else "-"
    blines = [
        ("clips (with wake word)", lambda v: v["n"]),
        ("**19 intents** accuracy", lambda v: f"{_fmt(v.get('accuracy'), True)} {_ci(v.get('accuracy_ci95'))}"),
        ("balanced accuracy", lambda v: _fmt(v.get("balanced_accuracy"), True)),
        ("F1 (macro)", lambda v: _fmt(v.get("macro_f1"), True)),
        ("F2 (macro)", lambda v: _fmt(v.get("macro_f2"), True)),
        ("false accept rate", lambda v: rate(v, "false_accept_rate", "false_accepts", "n_out_of_scope")),
        ("false reject rate", lambda v: _fmt(v.get("false_reject_rate"), True)),
        ("misfire rate", lambda v: _fmt(v.get("misfire_rate"), True)),
        ("**93 commands** accuracy", lambda v: _fmt(v.get("command_accuracy"), True)),
        ("balanced accuracy", lambda v: _fmt(v.get("command_balanced_accuracy"), True)),
        ("F1 (macro)", lambda v: _fmt(v.get("command_macro_f1"), True)),
        ("F2 (macro)", lambda v: _fmt(v.get("command_macro_f2"), True)),
        ("misfire rate", lambda v: _fmt(v.get("command_misfire_rate"), True)),
        ("slot exact (intent right)", lambda v: f"{_fmt(v.get('slot_exact_rate'), True)} (n={v.get('n_slot')})"
                                                if v.get("n_slot") else "-"),
        ("latency p50 / p95", lambda v: f"{_fmt(v.get('latency_p50'), nd=2)} / {_fmt(v.get('latency_p95'), nd=2)} s"
                                        if v.get("latency_p50") is not None else "-"),
        ("false wake rate (no wake word)", lambda v: rate(v, "false_wake_rate", "false_wakes", "n_no_wake")),
    ]
    for g in ("overall", "real voice", "synthetic voice"):
        lines = [(label, (lambda f: lambda m: f(m["breakdowns"][g]) if m["breakdowns"][g]["n"]
                          or label.startswith("false wake") else "-")(f)) for label, f in blines]
        out += [f"### {g}", ""] + side(lines, ms)

    # ---- Slots: one table per measure
    out += ["## Slot values (slotted intents, intent right)", "",
            "abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); "
            "rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised "
            "edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.", ""]
    intents = sorted({k for _, m in ms for k in m["slots"]} - {"ALL"}) + ["ALL"]   # union: a column can lack one
    sv = lambda m, k: m["slots"].get(k) or {}
    for title, f in [("n", lambda v: v.get("n", "-")),
                     ("exact", lambda v: _fmt(v.get("exact_rate"), True)),
                     ("mean abs error", lambda v: f"{_fmt(v.get('mean_abs_error'), nd=1)} {v.get('unit') or ''}".strip()
                                                  if v else "-"),
                     ("mean rel error", lambda v: _fmt(v.get("mean_rel_error"))),
                     ("phonetic dist", lambda v: _fmt(v.get("mean_phonetic_dist"))),
                     ("char dist", lambda v: _fmt(v.get("mean_char_dist")))]:
        lines = [(k, (lambda k: lambda m: f(sv(m, k)))(k)) for k in intents]
        out += [f"### {title}", ""] + side(lines, ms, "intent")

    # ---- Raspberry Pi
    s = ms[-1][1]["pi_specs"]
    pk = s.get("packages") or {}
    out += ["## Raspberry Pi", "",
            f"- **{s.get('model') or '?'}**, {s.get('cores')} cores "
            f"{s.get('cpu_model') or ''} up to {s.get('max_freq_mhz')} MHz, RAM {s.get('ram_mb')} MB, "
            f"{s.get('os')}, kernel {s.get('kernel')}, Python {s.get('python')}",
            "- packages: " + (", ".join(f"{k} {v}" for k, v in pk.items() if v) or "-"), "",
            "mean / p95 / max. Oct 3 columns share one run: only inference time (and FLOPs / size) is per model; "
            "the Oct 3 process ran several models (one after the other), so its RAM and the whole-Pi CPU are "
            "higher than in a one-model run.", ""]

    def d(k, unit="", nd=1):
        return lambda m: (f"{_fmt(m['pi'][k].get('mean'), nd=nd)} / {_fmt(m['pi'][k].get('p95'), nd=nd)} / "
                          f"{_fmt(m['pi'][k].get('max'), nd=nd)} {unit}") if m["pi"][k].get("n") else "-"
    lat = lambda m: m["pi"]["response_latency_s"]
    mp = lambda m: m.get("model") or {}
    out += side([
        ("response latency (command end -> Pi output)", d("response_latency_s", "s", 3)),
        ("latency p50 / p99", lambda m: f"{_fmt(lat(m).get('p50'))} / {_fmt(lat(m).get('p99'))} s" if lat(m).get("n") else "-"),
        ("inference time (Pi-reported)", d("infer_ms", "ms")),
        ("real-time factor (infer / audio window)", d("rtf", "", 3)),
        ("CPU temperature", d("temp_c", "C")),
        ("CPU use, whole Pi", d("cpu_pct_system", "%")),
        ("CPU use, your runtime process", d("cpu_pct_process", "%")),
        ("RAM (RSS), your runtime process", d("rss_mb_process", "MB")),
        ("RAM used, whole Pi", d("mem_used_mb_system", "MB")),
        ("CPU clock", d("freq_mhz", "MHz", 0)),
        ("load average (1 min)", d("load1", "", 2)),
        ("runtime CPU-seconds per second of speech", lambda m: _fmt(m["pi"].get("cpu_seconds_per_speech_second"))),
        ("runtime CPU share of wall time", lambda m: _fmt(m["pi"].get("process_cpu_share_of_wall"), True)),
        ("throttling flags seen", lambda m: ", ".join(m["pi"]["throttled_flags_seen"]) or "none"),
        ("test wall time", lambda m: f"{m['pi']['test_wall_time_s'] / 60:.1f} min"),
        ("model parameters", lambda m: f"{mp(m).get('params'):,}" if mp(m) else "-"),
        ("model size", lambda m: f"{mp(m).get('size_mb'):.2f} MB" if mp(m) else "-"),
        ("model FLOPs per inference", lambda m: mp(m).get("flops_si", "-")),
        ("effective GFLOP/s (FLOPs / mean infer time)",
         lambda m: f"{m['pi']['effective_gflops_per_s']:.2f}" if m["pi"].get("effective_gflops_per_s") else "-"),
    ], ms)

    # ---- Confusions: per column
    out += ["## Most frequent confusions", ""]
    for h, m in ms:
        out += [f"### {h}", ""]
        for lvl in ("intent_level", "command_level"):
            conf = m[lvl]["confusions"]
            out += [f"**{lvl.replace('_', ' ')}:** " +
                    ("; ".join(f"{t} -> {pr} ({n})" for (t, pr), n in conf[:10]) or "none"), ""]

    # ---- Per-intent: one table per score
    out += ["## Per-intent scores", ""]
    classes = sorted({k for _, m in ms for k in m["intent_level"]["per_class"]})
    pc = lambda m, c: m["intent_level"]["per_class"].get(c) or {}
    out += ["### n", ""] + side([(c, (lambda c: lambda m: pc(m, c).get("support", "-"))(c)) for c in classes], ms, "class")
    for k, title in (("precision", "precision"), ("recall", "recall"), ("f1", "F1"), ("f2", "F2")):
        out += [f"### {title}", ""] + side([(c, (lambda c: lambda m: _fmt(pc(m, c).get(k), True))(c))
                                            for c in classes], ms, "class")
    out += ["Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. "
            "Command level: a prediction matches a variation when intent and slot are right (the Pi does "
            "not predict the wording); wrong predictions count against the first variation of their "
            "(intent, slot). Macro scores average over classes present in the holdout. False accept rate "
            "rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake "
            "rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); "
            "any command the Pi fires for them is a false wake. These trials are not part of the "
            "19/93 scores.", ""]
    OUT.write_text("\n".join(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
