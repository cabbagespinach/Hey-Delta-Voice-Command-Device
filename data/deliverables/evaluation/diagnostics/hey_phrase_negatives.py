#!/usr/bin/env python3
"""
Real "Hey + word" false-accept check on the Qualcomm Keyword Speech Dataset (evaluation only).

4,270 clips from 50 speakers saying "Hey Android", "Hey Snapdragon", "Hi Galaxy" or "Hi Lumina".
Every clip is a negative for "Hey Delta" and shares its "Hey/Hi + word" shape. The dataset's licence
(internal research only; clause 1(ii) forbids incorporating it into another data set) keeps it out of
manifest.csv and out of training. It is read in place from external_raw/qualcomm/ and scored here.

Each clip is framed the way the evaluator frames short clips: one 1.5 s window centred on the clip
(preprocessing pads the rest), or, for clips longer than a window, sliding windows every 100 ms with the
clip score = the maximum window score. Clips are scored twice:
  as recorded     - the corpus's own level (close-talk microphone)
  device level    - speech-band peak moved to the RPI median (-39.8 dB) plus a pink floor at the RPI
                    median floor (-54.2 dB), i.e. the recording condition the confound diagnostic probes.
The threshold is the operating threshold in results/results.json (chosen on isolated validation only).

Usage: python diagnostics/hey_phrase_negatives.py [--model-class module:Class --checkpoint path]
Writes diagnostics/hey_phrase_negatives_report.md and .json.
"""
from pathlib import Path
import argparse, datetime, json, sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(HERE))
import detection as det               # noqa: E402
import run_evaluation as re_          # noqa: E402
import reproducibility as rep         # noqa: E402
import source_confound as sc          # noqa: E402

CORPUS = det.ROOT / "external_raw/qualcomm/qualcomm_keyword_speech_dataset"
SR = 16000


def clip_frames(x, cfg, key):
    """Windows covering a clip: one centred window if it fits, else every 100 ms."""
    t = torch.from_numpy(x.astype(np.float32))
    N = cfg.num_samples
    if len(x) <= N:
        return [det.wp.frame_at(t, -((N - len(x)) // 2), cfg, pad_key=key)]
    return [det.wp.frame_at(t, s, cfg, pad_key=key) for s in det.wp.sliding_starts(len(x), cfg)]


def score_clips(clips, pre, scorer):
    frames, owner = [], []
    for k, (key, x) in enumerate(clips):
        f = clip_frames(x, pre.cfg, key)
        frames += f
        owner += [k] * len(f)
    s = []
    with rep.single_thread():
        for a in range(0, len(frames), 512):
            s.append(scorer(pre(torch.stack(frames[a:a + 512]))))
    s = np.concatenate(s)
    return pd.Series(s).groupby(np.array(owner)).max().to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(EVAL / "baseline/baseline_cnn.pt"))
    ap.add_argument("--model-class", default=None)
    ap.add_argument("--results", default=str(EVAL / "results"), help="run_evaluation output dir (threshold source)")
    ap.add_argument("--out", default=str(HERE), help="where to write this diagnostic's outputs")
    args = ap.parse_args()
    scorer = det.TorchScorer(re_.load_model(args.checkpoint, args.model_class))
    thr = json.loads((Path(args.results) / "results.json").read_text())["isolated"]["threshold"]
    OUTD = Path(args.out)
    OUTD.mkdir(parents=True, exist_ok=True)
    pre = det.wp.WakewordPreprocessor.from_files()

    files = sorted(CORPUS.glob("*/*/*.wav"))
    meta = pd.DataFrame(dict(path=files, keyword=[f.parent.parent.name for f in files],
                             speaker=[f.parent.name for f in files]))
    raw = [det.wp.load_waveform(f, pre.cfg).numpy().astype(np.float64) for f in files]
    keys = [f.stem for f in files]
    meta["score_as_recorded"] = score_clips(list(zip(keys, raw)), pre, scorer)
    dev = [sc.add_floor(sc.to_peak(x, sc.DEVICE_PEAK), sc.DEVICE_FLOOR, k) for k, x in zip(keys, raw)]
    meta["score_device_level"] = score_clips(list(zip(keys, dev)), pre, scorer)

    def summ(g):
        out = {}
        for c in ("as_recorded", "device_level"):
            s = g[f"score_{c}"]
            fa = int((s >= thr).sum())
            out[c] = dict(n=len(s), false_accepts=fa, rate=fa / len(s), ci95=list(det.proportion_ci(fa, len(s))[1:]),
                          speakers_with_fa=int(g.loc[s >= thr, "speaker"].nunique()),
                          mean_score=float(s.mean()), p99=float(s.quantile(0.99)))
        return out

    res = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               model=args.model_class or "pipeline-check baseline", checkpoint=args.checkpoint, threshold=thr,
               corpus="Qualcomm Keyword Speech Dataset (evaluation only; not in manifest.csv)",
               overall=summ(meta), by_keyword={k: summ(g) for k, g in meta.groupby("keyword")})
    (OUTD / "hey_phrase_negatives.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    meta.drop(columns="path").assign(clip=[f.name for f in files]).to_csv(OUTD / "hey_phrase_negatives_scores.csv",
                                                                          index=False)
    L = ["# Real \"Hey + word\" negatives (Qualcomm Keyword Speech Dataset)", "",
         f"Generated {res['created_at']} by `hey_phrase_negatives.py`. Model: {res['model']}. Threshold {thr:.4f} "
         "(from isolated validation). Evaluation only: the licence keeps this corpus out of the dataset.", "",
         "| Keyword | Clips | False accepts, as recorded | Rate | False accepts, device level + floor | Rate | Speakers with an FA (device level) |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in list(res["by_keyword"].items()) + [("**all**", res["overall"])]:
        a, d = v["as_recorded"], v["device_level"]
        L.append(f"| {k} | {a['n']} | {a['false_accepts']} | {100 * a['rate']:.2f}% {re_.ci(a['ci95'], 2)} | "
                 f"{d['false_accepts']} | {100 * d['rate']:.2f}% {re_.ci(d['ci95'], 2)} | {d['speakers_with_fa']} / 50 |")
    top = meta.sort_values("score_device_level", ascending=False).head(10)
    L += ["", "Highest-scoring clips (device level):", "", "| Clip | Keyword | Speaker | As recorded | Device level |",
          "|---|---|---|---:|---:|"]
    L += [f"| {r.path.name} | {r.keyword} | {r.speaker} | {r.score_as_recorded:.3f} | {r.score_device_level:.3f} |"
          for r in top.itertuples(index=False)]
    (OUTD / "hey_phrase_negatives_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
