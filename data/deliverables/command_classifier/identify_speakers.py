#!/usr/bin/env python3
"""
Which of the two people speaks in each owner recording (owner request 2026-09-30: split by speaker).

Runs in envs/chatterbox (its voice encoder is the speaker-embedding model; CPU is enough):

    CUDA_VISIBLE_DEVICES="" ../../../envs/chatterbox/bin/python identify_speakers.py

1. For every recording with speech, keep only the command words (Whisper word times in recordings/_check/check.csv,
   padded 0.1 s), so the "Delta" tail, chimes and pauses don't blur the voice.
2. Voiceprint: Chatterbox VoiceEncoder embedding (256 numbers per clip, L2-normalised).
3. Two groups: agglomerative clustering (average linkage, cosine distance) into 2 clusters.
4. A second, independent clue: median pitch (F0, pYIN 70-400 Hz) of the command words.
5. Confidence per clip: how much closer it is to its own group's centre than to the other's (margin). Low-margin
   clips are listed for the owner to listen to.
Clips without usable speech (unknown_silent, empty) get no speaker; they are split by session only.

Writes recordings/_check/speakers.csv (file, label, session, cluster, margin, f0_hz, n_speech_sec) and prints a
summary plus clips to listen to. Nothing is relabelled until the owner confirms which cluster is which person.
"""
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
REC = HERE.parent / "model/deploy/recordings"
SR = 16000


def main():
    torch.set_num_threads(4)
    from chatterbox.models.voice_encoder import VoiceEncoder
    ve = VoiceEncoder()
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    ve.load_state_dict(load_file(hf_hub_download("ResembleAI/chatterbox", "ve.safetensors")))   # cached locally
    ve.eval()

    chk = pd.read_csv(REC / "_check/check.csv")
    man = pd.read_csv(REC / "manifest.csv")
    t = pd.to_datetime(man.recorded_at.astype(str).str[:15], format="%Y%m%d-%H%M%S")
    man = man.assign(t=t).sort_values("t")
    man["session"] = (man.t.diff().dt.total_seconds() > 300).cumsum()
    sess = dict(zip(man.file, man.session))

    rows, embs = [], []
    for r in chk.itertuples(index=False):
        row = dict(file=r.file, label=r.label, session=sess.get(r.file))
        if pd.isna(r.cmd_start) or pd.isna(r.cmd_end) or r.cmd_end - r.cmd_start < 0.3:
            rows.append(row)
            continue
        y, _ = sf.read(str(REC / r.file), dtype="float32")
        a, b = max(0, int((r.cmd_start - 0.1) * SR)), min(len(y), int((r.cmd_end + 0.1) * SR))
        seg = y[a:b]
        e = np.asarray(ve.embeds_from_wavs([seg], sample_rate=SR)).reshape(-1)
        f0, vf, _ = librosa.pyin(seg, fmin=70, fmax=400, sr=SR, frame_length=1024)
        row.update(f0_hz=float(np.nanmedian(f0[vf])) if vf.any() else np.nan, n_speech_sec=round(len(seg) / SR, 2),
                   emb_idx=len(embs))
        embs.append(e / np.linalg.norm(e))
        rows.append(row)
    E = np.stack(embs)
    np.save(REC / "_check/speaker_embeddings.npy", E)
    from sklearn.cluster import AgglomerativeClustering
    lab = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average").fit_predict(E)
    cent = np.stack([E[lab == k].mean(0) for k in (0, 1)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    sim = E @ cent.T
    d = pd.DataFrame(rows)
    has = d.emb_idx.notna()
    idx = d.loc[has, "emb_idx"].astype(int).to_numpy()
    d.loc[has, "cluster"] = lab[idx]
    d.loc[has, "margin"] = np.round(sim[idx, lab[idx]] - sim[idx, 1 - lab[idx]], 3)
    d = d.drop(columns="emb_idx")
    d.to_csv(REC / "_check/speakers.csv", index=False)

    print(f"{has.sum()} clips with speech embedded; {(~has).sum()} without usable speech\n")
    print("cluster sizes, median pitch, and similarity between the two cluster centres "
          f"({float(cent[0] @ cent[1]):.2f}; 1 = identical voices):")
    print(d[has].groupby("cluster").agg(clips=("file", "size"), median_f0_hz=("f0_hz", "median"),
                                        median_margin=("margin", "median")).round(2).to_string())
    print("\nclips per session and cluster:")
    print(pd.crosstab(d.session, d.cluster.fillna(-1).astype(int)).rename(columns={-1: "no speech"}).to_string())
    print("\nlowest-confidence clips (listen to these):")
    print(d[has].nsmallest(10, "margin")[["file", "cluster", "margin", "f0_hz"]].to_string(index=False))
    for k in (0, 1):
        print(f"\nmost typical clips of cluster {k}:")
        print(d[d.cluster == k].nlargest(4, "margin")[["file", "margin", "f0_hz"]].to_string(index=False))


if __name__ == "__main__":
    main()
