#!/usr/bin/env python3
"""
Filipino-accented synthetic speakers by zero-shot voice cloning (owner-approved 2026-09-30).

Runs in the SEPARATE environment envs/chatterbox (Chatterbox TTS, MIT licence; it needs torch 2.6, which the project
environment must not be downgraded to):

    CUDA_VISIBLE_DEVICES=<one gpu> ../../../envs/chatterbox/bin/python clone_fleurs_voices.py --voices 60

1. Reference voices: FLEURS fil_ph utterances (Filipino speakers, CC-BY 4.0; external_raw/fleurs) of 6-12 s.
   FLEURS has no speaker ids, so every candidate is embedded with Chatterbox's own voice encoder and --voices
   references are picked by farthest-point sampling: clearly different-sounding voices, not the same speaker
   cloned many times.
2. Each reference voice says "Hey Delta." and every command (spoken forms from generate_synthetic_commands.py) plus
   3 off-list sentences, with a random expressiveness (exaggeration 0.3-0.7) per voice. Re-running after the
   command list changed synthesizes only the missing commands for finished voices and drops removed ones.
3. Raw utterances go to external_raw/commands_clone_stage/<voice>/<key>.wav with index.json. Assembly into
   capture-like clips and the Whisper check are done by generate_synthetic_commands.py --tier clone (project env).

Chatterbox adds an inaudible "Perth" watermark to its output. It is applied to every cloned clip of every class, so it
cannot mark a class.
"""
from pathlib import Path
import argparse, json, random, sys

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FLEURS = ROOT / "external_raw/fleurs/data/fil_ph"
STAGE = ROOT / "external_raw/commands_clone_stage"
SEED = 20260930
sys.path.insert(0, str(HERE))


def spoken_texts():
    import csv
    from command_texts import SPOKEN, UNKNOWN_SENTENCES
    rows = list(csv.DictReader(open(HERE.parent / "model/deploy/label_map.csv")))
    cmds = {r["label"]: SPOKEN.get(r["label"], r["words"]) for r in rows if not r["label"].startswith("unknown")}
    return cmds, UNKNOWN_SENTENCES


def command_texts(cmds):
    texts = {}
    for lab, t in cmds.items():
        # Chatterbox is unstable on 1-2 word texts ("Stop" came out as a garble): 3 takes, the assembly keeps
        # the first one Whisper confirms.
        if len(t.split()) <= 2:
            texts.update({f"{lab}#t{n}": t for n in (1, 2, 3)})
        else:
            texts[lab] = t
    return texts


def update_voice(model, entry, cmds):
    """A finished voice after the command list changed (e.g. set_alarm_730am -> set_alarm_7am, 2026-09-30):
    synthesize only the commands it lacks (same reference and expressiveness) and drop commands no longer listed."""
    want = command_texts(cmds)
    stale = [k for k in entry["texts"] if k != "__delta__" and not k.startswith("unknown_other") and k not in want]
    for k in stale:
        (STAGE / entry["files"][k]).unlink(missing_ok=True)
        del entry["texts"][k], entry["files"][k]
    missing = {k: t for k, t in want.items() if k not in entry["texts"]}
    for key, text in missing.items():
        with torch.inference_mode():
            wav = model.generate(text, audio_prompt_path=str(ROOT / entry["reference"]),
                                 exaggeration=entry["exaggeration"], cfg_weight=0.5)
        rel = f"{entry['voice']}/{key.replace('#', '_')}.wav"
        sf.write(str(STAGE / rel), wav.squeeze(0).cpu().numpy(), model.sr)
        entry["texts"][key], entry["files"][key] = text, rel
    return stale, list(missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", type=int, default=60)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (shared server: keep it small)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from chatterbox.tts import ChatterboxTTS
    rng = random.Random(SEED)
    model = ChatterboxTTS.from_pretrained(device="cuda")

    # 1. candidate references: FLEURS fil_ph, 6-12 s
    cands = []
    for split in ("train", "dev", "test"):
        tsv = FLEURS / f"{split}.tsv"
        for line in open(tsv, encoding="utf-8"):
            c = line.rstrip("\n").split("\t")
            p = FLEURS / "audio" / split / c[1]
            if p.exists() and 6.0 <= int(c[5]) / 16000 <= 12.0:
                cands.append((p, c[6]))
    rng.shuffle(cands)
    cands = cands[:600]
    emb = []
    for p, _ in cands:
        y, sr = sf.read(str(p), dtype="float32")
        e = model.ve.embeds_from_wavs([y], sample_rate=sr)          # speaker embedding (Chatterbox voice encoder)
        emb.append(np.asarray(e).reshape(-1))
    emb = np.stack(emb)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    chosen = [0]
    d = 1 - emb @ emb[0]
    for _ in range(a.voices - 1):                                    # farthest-point sampling
        k = int(np.argmax(d))
        chosen.append(k)
        d = np.minimum(d, 1 - emb @ emb[k])
    refs = [cands[k] for k in chosen]
    print(f"{len(cands)} candidate references; picked {len(refs)} "
          f"(genders: {sum(g == 'MALE' for _, g in refs)} male / {sum(g == 'FEMALE' for _, g in refs)} female)")

    # 2. synthesize
    cmds, unknown = spoken_texts()
    STAGE.mkdir(parents=True, exist_ok=True)
    # Resume: voices already in index.json with every file on disk are kept and skipped. The reference choice is
    # deterministic (seeded), so a restart picks the same voices in the same order.
    done = {}
    if (STAGE / "index.json").exists():
        for e in json.loads((STAGE / "index.json").read_text()):
            if all((STAGE / f).exists() for f in e["files"].values()):
                done[e["voice"]] = e
    index = []
    for vi, (ref, gender) in enumerate(refs):
        voice = f"fleursfil{vi:02d}_{gender[:1].lower()}"
        ex = round(rng.uniform(0.3, 0.7), 2)
        if voice in done and done[voice]["reference"] == str(ref.relative_to(ROOT)):
            stale, added = update_voice(model, done[voice], cmds)
            index.append(done[voice])
            if stale or added:
                (STAGE / "index.json").write_text(json.dumps(index + [e for v, e in done.items()
                                                                      if v not in {x["voice"] for x in index}], indent=1))
            print(f"voice {vi + 1}/{len(refs)} {voice} already done" +
                  (f"; dropped {stale}, added {added}" if stale or added else ", skipped"), flush=True)
            continue
        texts = {"__delta__": "Hey Delta."}
        texts.update(command_texts(cmds))
        for k, s in enumerate(rng.sample(unknown, 3)):
            texts[f"unknown_other#{k}"] = s
        out = STAGE / voice
        out.mkdir(exist_ok=True)
        for key, text in texts.items():
            with torch.inference_mode():
                wav = model.generate(text, audio_prompt_path=str(ref), exaggeration=ex, cfg_weight=0.5)
            sf.write(str(out / f"{key.replace('#', '_')}.wav"), wav.squeeze(0).cpu().numpy(), model.sr)
        index.append(dict(voice=voice, reference=str(ref.relative_to(ROOT)), gender=gender, exaggeration=ex,
                          texts={k: v for k, v in texts.items()},
                          files={k: f"{voice}/{k.replace('#', '_')}.wav" for k in texts}))
        (STAGE / "index.json").write_text(json.dumps(index, indent=1))
        print(f"voice {vi + 1}/{len(refs)} {voice} done", flush=True)


if __name__ == "__main__":
    main()
