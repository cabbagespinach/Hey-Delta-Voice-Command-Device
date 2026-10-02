#!/usr/bin/env python3
"""
Voice conversion of the owner's real Pi recordings into other voices (owner-approved TEST, 2026-09-30).

Keeps each recording's timing, pauses, hesitations, "Delta" tail and room sound, and replaces only the voice
(Chatterbox VC, MIT; runs in the SEPARATE environment envs/chatterbox, like clone_fleurs_voices.py):

    CUDA_VISIBLE_DEVICES=<one gpu> ../../../envs/chatterbox/bin/python convert_owner_voices.py --targets 10

Sources: owner recordings Whisper confirmed as complete (recordings/_check/check.csv) of commands with >= --min-words
words (3). Test 2026-09-30 (2 clips per label x 4 voices): one- and two-word commands often came out garbled
("Skip" -> "The skill"), longer ones kept their words in 88-100% of clips; the owner chose 3+ word commands only,
kept only where Whisper hears the same words (check_converted.py, strict).
Targets: --targets reference voices from the cloned-voice stage (FLEURS fil_ph speakers, alternating genders).
Output: data/commands_vc/<label>/<label>__vc-<voice>__<source stem>.wav (16 kHz) + index.csv. Checking (Whisper) and
listening come next; nothing here is used for training until the owner approves the test.
Chatterbox adds its inaudible Perth watermark to every converted clip (as for the cloned clips).
"""
from pathlib import Path
import argparse, json

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REC = HERE.parent / "model/deploy/recordings"
STAGE = ROOT / "external_raw/commands_clone_stage"
OUT = ROOT / "data/commands_vc"
SEED = 20260930


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=0, help="source clips per label (0 = all confirmed ones)")
    ap.add_argument("--min-words", type=int, default=3,
                    help="only labels whose prompt has at least this many words (one/two-word commands garble)")
    ap.add_argument("--targets", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--extra-targets", type=int, default=0,
                    help="(2026-10-01) ADD this many target voices: cloned voices already in the TRAIN split by the voice "
                         "hash (not the forced fleursfil00-09), alternating genders; only train-split sources; appends to "
                         "index.csv and skips clips already converted")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from chatterbox.vc import ChatterboxVC
    vc = ChatterboxVC.from_pretrained(device="cuda")

    chk = pd.read_csv(REC / "_check/check.csv")
    words = {}
    for line in (HERE.parent / "model/deploy/commands.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            lab, _, txt = (x.strip() for x in line.partition("|"))
            words[lab] = len(txt.split())
    src = chk[(chk.verdict == "complete") & (chk.label.map(words).fillna(0) >= a.min_words)]
    src = src.sample(frac=1, random_state=SEED)
    if a.per_class:
        src = src.groupby("label").head(a.per_class)
    src = src.sort_values("label")
    idx = json.loads((STAGE / "index.json").read_text())
    males = [e for e in idx if e["gender"] == "MALE"]
    females = [e for e in idx if e["gender"] == "FEMALE"]
    targets = [(females if i % 2 == 0 else males)[i // 2] for i in range(a.targets)]
    old = pd.read_csv(OUT / "index.csv") if a.extra_targets and (OUT / "index.csv").exists() else None
    if a.extra_targets:
        from build_commands_all import voice_split, VC_TARGETS, owner_split
        have = set(old.target_voice) if old is not None else set()
        pool = [e for e in idx if e["voice"] not in have | VC_TARGETS and voice_split(e["voice"]) == "train"]
        pf = [e for e in pool if e["gender"] == "FEMALE"]
        pm = [e for e in pool if e["gender"] == "MALE"]
        targets = [(pf if i % 2 == 0 else pm)[i // 2] for i in range(a.extra_targets)]
        man = pd.read_csv(REC / "manifest.csv").set_index("file")
        src = src[[owner_split(man.at[f, "speaker"], man.at[f, "recorded_at"]) == "train" for f in src.file]]
    print(f"{len(src)} source clips x {len(targets)} target voices "
          f"({', '.join(t['voice'] for t in targets)})", flush=True)

    rows = []
    for t in targets:
        vc.set_target_voice(str(ROOT / t["reference"]))
        for r in src.itertuples(index=False):
            f = f"{r.label}/{r.label}__vc-{t['voice']}__{Path(r.file).stem}.wav"
            if a.extra_targets and (OUT / f).exists():                      # resume after a disconnect
                y, _ = sf.read(str(OUT / f), dtype="float32")
                rows.append(dict(file=f, label=r.label, source_file=r.file, target_voice=t["voice"],
                                 target_gender=t["gender"], target_reference=t["reference"],
                                 source_duration_sec=r.duration_sec, duration_sec=round(len(y) / 16000, 2)))
                continue
            with torch.inference_mode():
                wav = vc.generate(str(REC / r.file))
            y = wav.squeeze(0).cpu().numpy().astype(np.float32)
            y = resample_poly(y, 16000, vc.sr).astype(np.float32)          # 24 kHz -> 16 kHz like the recordings
            f = f"{r.label}/{r.label}__vc-{t['voice']}__{Path(r.file).stem}.wav"
            (OUT / r.label).mkdir(parents=True, exist_ok=True)
            sf.write(str(OUT / f), y, 16000, subtype="PCM_16")
            rows.append(dict(file=f, label=r.label, source_file=r.file, target_voice=t["voice"],
                             target_gender=t["gender"], target_reference=t["reference"],
                             source_duration_sec=r.duration_sec, duration_sec=round(len(y) / 16000, 2)))
        print(f"  {t['voice']} done", flush=True)
    new = pd.DataFrame(rows)
    if old is not None:
        new = pd.concat([old, new[~new.file.isin(set(old.file))]], ignore_index=True)
    new.to_csv(OUT / "index.csv", index=False)
    print(f"wrote {len(rows)} clips -> {OUT}")


if __name__ == "__main__":
    main()
