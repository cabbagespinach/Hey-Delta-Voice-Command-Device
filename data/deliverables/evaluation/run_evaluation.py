#!/usr/bin/env python3
"""
Run isolated + streaming evaluation for a model and write the metrics report.

  python run_evaluation.py                                  # pipeline-check baseline
  python run_evaluation.py --checkpoint m.pt --model-class mymodule:MyNet

The model must map features [B, 1, 40, 147] to logits [B]. Steps:
  1. isolated validation (fixed manifest) -> operating threshold (validation only)
  2. verify the frozen streaming set (checksums, independence)
  3. score every stream through the streaming front end, detect, match, attribute
  4. results/: metrics_report.md, results.json and CSV tables
"""
from pathlib import Path
import argparse, datetime, functools, importlib, json, sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import detection as det           # noqa: E402
import isolated_eval as iso       # noqa: E402
import streaming_eval as se       # noqa: E402
import baseline_model as bm       # noqa: E402

OUT = HERE / "results"
sys.path.insert(0, str(HERE.parent / "model"))          # project models (bc_resnet) for --model-class


def load_model(checkpoint, model_class):
    if model_class is None:
        return bm.load_baseline(checkpoint)
    mod, cls = model_class.split(":")
    m = getattr(importlib.import_module(mod), cls)()
    m.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return m.eval()


def pct(x, d=1):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def ci(v, d=1, scale=100, suffix="%"):
    return f"[{scale * v[0]:.{d}f}, {scale * v[1]:.{d}f}]{suffix}"


def _s(x):
    """seconds to 2 decimals; 'n/a' when nothing was detected (no latency to measure)."""
    return "n/a" if x is None else f"{x:.2f}"


def report(model_desc, I, S, verification):
    pos_counts = ", ".join(f"{k} {v['n']}" for k, v in I["positives_by_subset"].items())
    L = ["# Evaluation report", "",
         f"Generated {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')} by `run_evaluation.py`.",
         f"Model: **{model_desc}**.", ""]
    L += ["## Data and independence", "",
          "| Set | Audio | Real / synthetic | Used for |", "|---|---|---|---|",
          f"| Isolated validation | {I['n_windows']} fixed windows (`isolated_set/isolated_validation_manifest.csv`) | "
          f"positives: {', '.join(f'{k} {v['n']}' for k, v in I['positives_by_subset'].items())} | metrics + choosing the threshold |",
          f"| Streaming Set A (phase-1 holdout) | {S['by_set']['A_phase1_holdout']['streams']} streams, "
          f"{S['by_set']['A_phase1_holdout']['exposure_hours'] + 0:.2f} h scored exposure | synthetic | metrics only |",
          f"| Streaming Set B (composed for evaluation) | {S['by_set']['B_composed']['streams']} streams, "
          f"{S['by_set']['B_composed']['exposure_hours']:.2f} h scored exposure | synthetic (new Piper TTS + procedural audio) | metrics only |",
          "",
          f"- Streaming set verified: {verification['streams']} files match their frozen sha256; no stream id, file or source "
          "group appears in train, validation, test or the isolated set (`streaming_eval.verify_set`).",
          "- Set B text is eval-only (no sentence, confusable or carrier phrase equals a training transcription), and "
          "wakewords are new stochastic syntheses off the training speed/pitch grid; see `streaming_set/build_log.json` "
          "for the near-duplicate check against existing positives.",
          "- **Streaming evaluation contains no real device audio.** Real RPI performance is measured only by isolated "
          "validation (subset `real_device_rpi`).", ""]
    L += ["## Operating point", "",
          f"Threshold **{I['threshold']:.4f}** — {I['threshold_source']}. Streaming data never influenced it. "
          f"Detection: fire when {S['detection_config'].get('k_of_n', [1, 1])[0]} of the last "
          f"{S['detection_config'].get('k_of_n', [1, 1])[1]} window scores reach it, "
          f"{S['detection_config']['smoothing_windows']}-window smoothing, "
          f"{S['detection_config']['refractory_sec']} s refractory, hit zone [wake start − "
          f"{S['detection_config']['hit_zone']['before_start_sec']} s, wake end + {S['detection_config']['hit_zone']['after_end_sec']} s]; "
          "sliding 1.5 s windows every 100 ms via `StreamingPreprocessor`.", ""]
    if I.get("threshold_policy"):
        L += ["Per-category thresholds (deployment-like validation negatives; the operating threshold is the largest): "
              + ", ".join(f"{k} {v:.4f}" for k, v in sorted(I["threshold_policy"]["per_group_threshold"].items(),
                                                            key=lambda kv: -kv[1])) + ".", ""]
    L += ["## Isolated validation", "",
          f"ROC AUC **{I['roc_auc']:.4f}**, average precision **{I['average_precision']:.4f}**. At the operating threshold: "
          f"detection rate **{pct(I['positives']['detection_rate'])}** {ci(I['positives']['ci95'])} "
          f"({I['positives']['detected']}/{I['positives']['n']}), false-positive rate **{pct(I['negatives']['fpr'], 2)}** "
          f"({I['negatives']['false_positives']}/{I['negatives']['n']}).", "",
          "**Positives by source:**", "", "| Subset | n | Detected | Rate (95% CI) |", "|---|---:|---:|---|"]
    for k, v in I["positives_by_subset"].items():
        L.append(f"| {k} | {v['n']} | {v['detected']} | {pct(v['detection_rate'])} {ci(v['ci95'])} |")
    if I["low_signal_rpi"]:
        L += ["", "Owner-verified low-signal RPI positives in this split: " + ", ".join(
            f"{k} score {v['score']:.3f} ({'detected' if v['detected'] else 'missed'})" for k, v in I["low_signal_rpi"].items()) + "."]
    L += ["", "**Negatives by category:**", "",
          "| Category | n | False positives | FPR (95% CI) | Score p99 | Score max |", "|---|---:|---:|---|---:|---:|"]
    for k, v in sorted(I["negatives_by_category"].items(), key=lambda kv: -kv[1]["fpr"]):
        L.append(f"| {k} | {v['n']} | {v['false_positives']} | {pct(v['fpr'], 2)} {ci(v['ci95'], 2)} | "
                 f"{v['score_p99']:.3f} | {v['score_max']:.3f} |")
    L += ["", "**Negatives by source:**", "", "| Subset | n | False positives | FPR (95% CI) |", "|---|---:|---:|---|"]
    for k, v in I.get("negatives_by_subset", {}).items():
        L.append(f"| {k} | {v['n']} | {v['false_positives']} | {pct(v['fpr'], 2)} {ci(v['ci95'], 2)} |")
    L += ["", "`real_external` = public-corpus negatives (MSWC, FLEURS, MUSAN, Hey Snips): reported, never used to "
          "choose the threshold. Per category and source: `negatives_by_category_and_subset` in `results.json`.", ""]
    o = S["overall"]
    L += ["## Streaming (always listening)", "",
          f"Overall: **{o['fa_per_hour']:.2f} false accepts/hour** {ci(o['fa_per_hour_ci95'], 2, 1, '')} "
          f"({o['false_accepts']} in {o['exposure_hours']:.2f} h of wakeword-free exposure); detection rate "
          f"**{pct(o['detection_rate'])}** {ci(o['detection_rate_ci95'])} ({o['detected']}/{o['wakewords']}, "
          f"{o['misses']} misses); median latency **{_s(o['latency_sec']['median'])} s** (p90 {_s(o['latency_sec']['p90'])} s) "
          f"after the end of the wakeword; {o['duplicate_triggers']} duplicate triggers.", "",
          *([f"With the plain rule (fire on any single window at the same threshold): "
             f"{S['plain_rule']['fa_per_hour']:.2f} false accepts/hour, detection {pct(S['plain_rule']['detection_rate'])}.", ""]
            if "plain_rule" in S else []),
          "- **FA/hour** = false accepts ÷ hours during which a false accept was possible (stream time after the first "
          "full window, minus every wakeword hit zone). Intervals are exact Poisson 95% CIs.",
          "- **Latency** = trigger time (end of the triggering window) − wakeword end; negative means it fired before "
          "the word ended.", "",
          "| Set / condition | Hours | FA | FA/h (95% CI) | Wakewords | Detected | Detection rate (95% CI) | Median latency |",
          "|---|---:|---:|---|---:|---:|---|---:|"]
    for group, title in ((S["by_set"], "set"), (S["by_condition"], "condition")):
        for k, v in group.items():
            lat = v["latency_sec"]["median"]
            L.append(f"| {title}: {k} | {v['exposure_hours']:.2f} | {v['false_accepts']} | {v['fa_per_hour']:.2f} "
                     f"{ci(v['fa_per_hour_ci95'], 1, 1, '')} | {v['wakewords']} | {v['detected']} | "
                     f"{pct(v['detection_rate'])} {ci(v['detection_rate_ci95']) if v['wakewords'] else ''} | "
                     f"{'–' if lat is None else f'{lat:.2f} s'} |")
    L += ["", "### False accepts by category", "",
          "Category = the non-wakeword event overlapping the triggering 1.5 s window the most; otherwise the stream's "
          "background condition. Full table: `fa_by_category.csv`; every false accept: `streaming_false_accepts.csv`.", "",
          "| Category | False accepts | Share | Events of this type | Events that caused an FA (95% CI) | FA/h in that background |",
          "|---|---:|---:|---:|---|---|"]
    for c in S["false_accepts_by_category"]:
        if c["category"].startswith("background:"):
            L.append(f"| {c['category']} | {c['false_accepts']} | {pct(c['share_of_false_accepts'])} | – | – | "
                     f"{c['fa_per_hour_in_condition']:.2f} {ci(c['fa_per_hour_in_condition_ci95'], 1, 1, '')} |")
        else:
            L.append(f"| {c['category']} | {c['false_accepts']} | {pct(c['share_of_false_accepts'])} | "
                     f"{c['events_of_this_type']} | {pct(c['fa_per_event'])} {ci(c['fa_per_event_ci95'])} | – |")
    L += ["", "### Detection by wakeword type, level and voice", "", "| Breakdown | Group | n | Detection rate (95% CI) |",
          "|---|---|---:|---|"]
    for key, title in (("detection_by_wake_type", "type"), ("detection_by_level", "level (peak over floor)"),
                       ("detection_by_voice", "voice")):
        for k, v in S[key].items():
            L.append(f"| {title} | {k} | {v['n']} | {pct(v['rate'])} [{100 * v['lo']:.1f}, {100 * v['hi']:.1f}]% |")
    p = S["wakeword_position"]
    L += ["", "### Wakeword positions", "",
          f"Relative start positions span {p['relative_start_min']:.2f}–{p['relative_start_max']:.2f} of the stream "
          f"(quartiles {', '.join(f'{q:.2f}' for q in p['relative_start_quartiles'])}). By set: "
          + "; ".join(f"{k} {a:.2f}–{b:.2f}" for k, (a, b) in p["by_set"].items())
          + ". Set B places events uniformly; Set A (phase-1) restricted wakewords to the middle 20–80 %. "
            "With a 100 ms hop the wakeword also lands at every offset inside the sliding windows.", "",
          "### Threshold sweep", "",
          "`streaming_sweep.csv` (FA/h vs miss rate per set) and the `sweep` list in `results.json` (isolated "
          "detection rate vs FPR) give the full trade-off, independent of the chosen operating point.", "",
          "## Caveats", "",
          "- **Set A ground truth includes TTS silence padding** (phase-1 insertion metadata, used verbatim; several spans are "
          "1.87 s long). Its wakeword 'end' is late, so Set A latencies are biased negative and not comparable with "
          "Set B, whose spans are energy-trimmed. Set A detection and false-accept counts are unaffected: the hit zone "
          "still starts at the wakeword start.",
          "- **Levels differ between sets.** Set A is peak-normalised (loud, like phase-1 synthetic data); Set B is at "
          "the measured device level (floor −58 to −51 dB, events 5–26 dB above the floor). The louder "
          "`environmental_noise`, `noisy_speech` and `mixed` floors (−50 to −38 dB) are an assumption, not a measurement.",
          "- **Small isolated positive counts** (validation: " + pos_counts + ") give wide intervals; they are "
          "reported with every rate.",
          "- **Latency** is measured from the end of the triggering 1.5 s window, i.e. the earliest moment a streaming "
          "detector could fire, excluding model compute time.", ""]
    return "\n".join(L) + "\n"


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(bm.OUT / "baseline_cnn.pt"))
    ap.add_argument("--model-class", default=None, help="module:Class; default = pipeline-check baseline")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--out", default=str(OUT), help="results directory (default: results/)")
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    factory = functools.partial(load_model, args.checkpoint, args.model_class)
    model_desc = ("pipeline-check baseline CNN (`baseline_model.py`, train split only) — not the project model"
                  if args.model_class is None else f"{args.model_class} from {args.checkpoint}")

    scorer = det.TorchScorer(factory())
    I, iso_scores = iso.evaluate(scorer, "validation")
    thr = I["threshold"]
    print(f"[isolated] AUC {I['roc_auc']:.4f}, threshold {thr:.4f}")

    streams, events = se.load_set()
    verification = se.verify_set(streams, [iso.manifest_path("validation")])
    scores = se.score_streams(streams, factory, args.workers)
    np.savez_compressed(OUT / "stream_scores.npz", **{f"{k}__t": v[0] for k, v in scores.items()},
                        **{f"{k}__s": v[1] for k, v in scores.items()})
    S, per_stream, dets, fas, trig = se.evaluate(streams, events, scores, thr)
    if se.CFG["detection"].get("k_of_n", [1, 1]) != [1, 1]:
        plain, *_ = se.evaluate(streams, events, scores, thr, k_of_n=(1, 1))
        S["plain_rule"] = dict(fa_per_hour=plain["overall"]["fa_per_hour"],
                               detection_rate=plain["overall"]["detection_rate"],
                               by_set={k: dict(fa_per_hour=v["fa_per_hour"], detection_rate=v["detection_rate"])
                                       for k, v in plain["by_set"].items()})
    print(f"[streaming] {S['overall']['fa_per_hour']:.2f} FA/h, detection {S['overall']['detection_rate']:.3f}")

    if not args.no_sweep:
        allscores = np.concatenate([s for _, s in scores.values()])
        grid = np.unique(np.concatenate([np.quantile(allscores, np.linspace(0.5, 1, 40)), [thr]]))
        se.sweep(streams, events, scores, grid).to_csv(OUT / "streaming_sweep.csv", index=False)

    iso_scores.to_csv(OUT / "isolated_scores.csv", index=False)
    iso_scores[(iso_scores.y == 0) & (iso_scores.predicted == 1)].sort_values("score", ascending=False) \
        .to_csv(OUT / "isolated_false_positives.csv", index=False)
    dets.to_csv(OUT / "streaming_detections.csv", index=False)
    fas.to_csv(OUT / "streaming_false_accepts.csv", index=False)
    per_stream.to_csv(OUT / "streaming_per_stream.csv", index=False)
    trig.to_csv(OUT / "streaming_triggers.csv", index=False)
    pd.DataFrame(S["false_accepts_by_category"]).to_csv(OUT / "fa_by_category.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(dict(model=model_desc, isolated=I, streaming=S,
                                                      streaming_set_verification=verification),
                                                 indent=2, default=float) + "\n")
    (OUT / "metrics_report.md").write_text(report(model_desc, I, S, verification))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
