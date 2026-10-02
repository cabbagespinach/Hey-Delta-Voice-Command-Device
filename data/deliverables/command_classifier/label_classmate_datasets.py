#!/usr/bin/env python3
"""
Label the datasets downloaded from the classmates' list into the schema-B classes (owner, 2026-10-01).

    python label_classmate_datasets.py [--workers 8]
    # -> data/commands_classmates/<dataset>/<class>/*.wav  (16 kHz mono 16-bit)
    #    data/commands_classmates/manifest.csv   same columns as data/commands_schema_b.csv
    #    data/commands_classmates/dropped.csv    every clip left out, with the reason

Sources (external_raw/, see download_class_datasets.sh). Zips are read directly (no extracted copy on disk).
| dataset                | source                                              | labels from |
|------------------------|-----------------------------------------------------|-------------|
| classmate_optionb      | Classmate A's voice-cloned Option-B set (github markandrian30/AI231, MEX2/Data) | transcript = schema wording; FLAGGED/ (failed his QA) left out |
| web_gsc                | Google Speech Commands v0.02                        | "stop" -> STOP, every other word -> unknown; _background_noise_ -> unknown |
| web_snips              | Snips SLU (smart-lights, smart-speaker)             | transcript: schema wording or an accepted phrasing; borderline -> dropped |
| classmate_syntts       | SynTTS-Commands English (Free-ST + VoxCeleb voices) | command folder (table below) |
| web_multisensor        | Classmate F's VCM: Multi-Sensor "next song" (KU Leuven)    | NEXT |
| classmate_vcm_real/_piper | Classmate F's VCM: SET_TEMPERATURE recordings / Piper   | transcript value (18/22/26 -> class, other values -> unknown) |
VCM rows from Fluent Speech Commands, SLURP and Timers and Such are NOT used: we already have those corpora
(data/commands_real_web), labelled by our own rules; their GSC-noise SILENCE clips duplicate web_gsc.

Owner rules (2026-10-01, "strict"): a clip counts for a class only if it says the schema wording or very clearly the
same request; the wrong slot value is an `unknown` near-miss ("timer for 5 minutes" is not TIMER_1MIN); borderline
sentences are dropped, not guessed ("I'd like to listen to <artist>", "Mute", "Max volume", "Set volume to 50%",
lights in a named room). Splits: the source's own split where it has one, else train (provisional: the class has not
fixed its benchmark split). GSC validation/test unknowns are capped at 2,000 each (seeded) to keep the per-epoch
validation pass short; the rest of them are listed in dropped.csv.
"""
from pathlib import Path
from multiprocessing import Pool
import argparse, ast, io, os, re, sys, wave, zipfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = ROOT / "external_raw"
OUT = ROOT / "data/commands_classmates"
sys.path.insert(0, str(HERE))
from generate_synthetic_commands import norm, to16k                       # noqa: E402
from prepare_real_commands import PHRASE2LABEL                            # noqa: E402
from schema_b import OLD_TO_NEW                                           # noqa: E402

SEED = 20261001
LM = pd.read_csv(HERE.parent / "model/deploy/label_map_schema_b.csv")
EXACT = {norm(w): c for c, ws in zip(LM["class"], LM.wordings) for w in ws.split(" | ")}
# phrasings already accepted for the public datasets (prepare_real_commands.PHRASES) in schema-B classes
ACCEPTED = {**{p: OLD_TO_NEW.get(l, l) for p, l in PHRASE2LABEL.items()}, **EXACT}
SPLIT = {"train": "train", "val": "validation", "valid": "validation", "validation": "validation", "dev": "validation",
         "test": "test"}
LICENSE = {
    "classmate_optionb": "no licence stated (classmate Classmate A, github markandrian30/AI231; voices cloned from LibriSpeech CC BY 4.0 refs)",
    "web_gsc": "CC BY 4.0 (Warden 2018, arXiv:1804.03209)",
    "web_snips": "Snips SLU research datasets terms (HF mirror card: MIT); Saade et al. 2018 arXiv:1810.12735",
    "classmate_syntts": "MIT (SynTTS-Commands, arXiv:2511.07821)",
    "web_multisensor": "CC BY 4.0 (Multi-Sensor Voice Command Dataset, doi:10.48804/IEKKVZ)",
    "classmate_vcm_real": "no licence stated (classmate Classmate F's own recordings, VCM)",
    "classmate_vcm_piper": "Piper TTS output (classmate Classmate F, VCM); voice licences per Piper voice",
}
SYNTTS = {  # SynTTS English command -> schema class (None = dropped: volume-related, borderline)
    "Pause": "PAUSE", "Skip_song": "NEXT", "Next_track": "NEXT", "Volume_up": "VOLUME_UP", "Volume_down": "VOLUME_DOWN",
    "Mute": None, "Max_volume": None, "Set_volume_to_50%": None,
}   # every other SynTTS command (wake words, Play, Resume, previous/repeat track, answer/decline call...) -> unknown


def write16(dst: Path, y: np.ndarray):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())


def convert_bytes(data: bytes, dst: Path):
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    write16(dst, to16k(y.mean(1), sr))


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def row(dst, label, cls, dataset, speaker, split, real, text, rule):
    return dict(path=rel(dst), label=label, **{"class": cls}, dataset=dataset, speaker=speaker, split=split, real=real,
                license=LICENSE[dataset], old_class=label, text=text, rule=rule)


# ---------------------------------------------------------------- Classmate A's Option-B set (already 16 kHz PCM16: links)
def optionb():
    base = RAW / "class_optionb/AI231/MEX2/Data"
    m = pd.read_csv(base / "manifest.csv")
    keep, drop = [], []
    for r in m.itertuples(index=False):
        cls = EXACT.get(norm(r.transcript))
        src = base / r.path
        if cls is None or not src.exists():
            drop.append(dict(source=rel(src), dataset="classmate_optionb", label=r.label, text=r.transcript,
                             reason="missing file" if cls else "transcript is not a schema wording"))
            continue
        dst = OUT / "classmate_optionb" / cls / Path(r.path).name
        link(src, dst)
        keep.append(row(dst, r.label, cls, "classmate_optionb", f"optionb-{r.speaker}", SPLIT[r.split], False,
                        r.transcript, f"schema wording ({r.variant_id})"))
    n_flag = sum(1 for _ in (base / "FLAGGED").rglob("*.wav"))
    drop.append(dict(source=rel(base / "FLAGGED"), dataset="classmate_optionb", label="", text="",
                     reason=f"{n_flag} clips failed Classmate A's own QA (Whisper score < 0.80): not used"))
    print(f"optionb: {len(keep)} kept, {n_flag} flagged left out", flush=True)
    return keep, drop


# ---------------------------------------------------------------- Google Speech Commands v0.02 (16 kHz PCM16: links)
def gsc():
    base = RAW / "google_speech_commands/v0.02"
    val = set((base / "validation_list.txt").read_text().split())
    test = set((base / "testing_list.txt").read_text().split())
    rows = []
    for w in sorted(p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for f in sorted((base / w).glob("*.wav")):
            k = f"{w}/{f.name}"
            rows.append(dict(src=f, word=w, split="test" if k in test else "validation" if k in val else "train",
                             speaker=f"gsc-{f.name.split('_')[0]}"))
    d = pd.DataFrame(rows)
    d["cls"] = np.where(d.word == "stop", "STOP", "unknown")
    keep_idx = []
    for sp in ("validation", "test"):                     # cap eval unknowns; every 'stop' clip stays
        u = d[(d.split == sp) & (d.cls == "unknown")]
        keep_idx += list(u.sample(min(2000, len(u)), random_state=SEED).index)
    capped = d[(d.split != "train") & (d.cls == "unknown") & ~d.index.isin(keep_idx)]
    d = d.drop(capped.index)
    keep = []
    for r in d.itertuples(index=False):
        dst = OUT / "web_gsc" / r.cls / f"{r.word}__{r.src.name}"
        link(r.src, dst)
        keep.append(row(dst, r.word, r.cls, "web_gsc", r.speaker, r.split, True, r.word,
                        "schema wording 'Stop'" if r.cls == "STOP" else "other single word (near-miss)"))
    for f in sorted((base / "_background_noise_").glob("*.wav")):
        dst = OUT / "web_gsc" / "unknown" / f"noise__{f.name}"
        link(f, dst)
        keep.append(row(dst, "_background_noise_", "unknown", "web_gsc", "gsc-noise", "train", True, "",
                        "background noise (long recording)"))
    drop = [dict(source=rel(r.src), dataset="web_gsc", label=r.word, text=r.word,
                 reason=f"{r.split} unknown beyond the 2,000 cap") for r in capped.itertuples(index=False)]
    print(f"gsc: {len(keep)} kept ({(d.cls == 'STOP').sum()} STOP), {len(drop)} eval unknowns over the cap", flush=True)
    return keep, drop


# ---------------------------------------------------------------- Snips SLU (parquet with wav bytes)
COLOR = re.compile(r"(please )?(change|switch|set|make|turn) (the )?(color|colour|light|lights) (color |colour )?(to )?"
                   r"(?P<v>[a-z]+)( please)?")
BRIGHT = re.compile(r"(please )?(set|adjust|change) (the )?brightness (level )?to (?P<v>[a-z ]+?)( percent)?( please)?")
COLOR_WORDS = {"red", "blue", "green", "pink", "yellow", "purple", "orange", "white", "violet", "magenta", "cyan",
               "teal", "turquoise", "indigo", "brown", "gold", "black", "lavender", "lilac", "beige", "grey", "gray"}
COLORS = {"red": "COLOR_RED", "blue": "COLOR_BLUE", "green": "COLOR_GREEN"}
BRIGHTS = {"twenty": "BRIGHTNESS_20", "sixty": "BRIGHTNESS_60", "one hundred": "BRIGHTNESS_100", "hundred": "BRIGHTNESS_100"}


def snips_class(text):
    t = norm(text)
    if t in ACCEPTED:
        return ACCEPTED[t], "schema wording / accepted phrasing"
    m = COLOR.fullmatch(t)
    if m and m["v"] in COLOR_WORDS:                       # "make the lights brighter" is not a colour
        return (COLORS[m["v"]], "colour request, schema value") if m["v"] in COLORS else \
            ("unknown", "colour request, other colour (near-miss)")
    m = BRIGHT.fullmatch(t)
    if m:
        return (BRIGHTS[m["v"]], "brightness request, schema value") if m["v"] in BRIGHTS else \
            ("unknown", "brightness request, other value (near-miss)")
    return None, "borderline paraphrase (strict rule): dropped"


def snips():
    import pyarrow.parquet as pq
    keep, drop = [], []
    for f in sorted((RAW / "snips_slu").glob("*.parquet")):
        for r in pq.read_table(f).to_pylist():
            cls, rule = snips_class(r["text"])
            w = r["worker"] if isinstance(r["worker"], dict) else ast.literal_eval(str(r["worker"]))
            name = f"snips_{r['ID']:05d}_{r['distance']}.wav"
            if cls is None:
                drop.append(dict(source=f"{f.name}#{r['ID']}/{r['distance']}", dataset="web_snips", label=r["source"],
                                 text=r["text"], reason=rule))
                continue
            dst = OUT / "web_snips" / cls / name
            convert_bytes(r["audio"]["bytes"], dst)
            keep.append(row(dst, r["source"], cls, "web_snips", f"snips-{w['id']}", "train", True, r["text"], rule))
    print(f"snips: {len(keep)} kept, {len(drop)} dropped", flush=True)
    return keep, drop


# ---------------------------------------------------------------- SynTTS-Commands English (zips, 24 kHz float)
def _syntts_chunk(job):
    zpath, members, subset, cmd, cls = job
    out = []
    with zipfile.ZipFile(zpath) as z:
        for m in members:
            stem = Path(m).stem
            dst = OUT / "classmate_syntts" / cls / f"{subset}__{stem}.wav"
            try:
                if not dst.exists():
                    convert_bytes(z.read(m), dst)
            except Exception as e:                       # unreadable member: reported, not fatal
                out.append(("drop", m, str(e)[:80]))
                continue
            spk = stem[: -len(cmd) - 1] if stem.endswith("_" + cmd) else stem
            out.append(("keep", rel(dst), f"syntts-{subset}-{spk}"))
    return zpath, cmd, cls, out


def syntts(workers):
    jobs, drop = [], []
    for subset in ("Free_ST_English", "VoxCeleb12_English"):
        for zp in sorted((RAW / "syntts_commands" / subset).glob("*.zip")):
            cmd = zp.stem
            cls = SYNTTS.get(cmd, "unknown")
            with zipfile.ZipFile(zp) as z:
                mem = [m for m in z.namelist() if m.lower().endswith(".wav")]
            if cls is None:
                drop.append(dict(source=rel(zp), dataset="classmate_syntts", label=cmd, text=cmd.replace("_", " "),
                                 reason=f"{len(mem)} clips: volume-related, borderline (strict rule)"))
                continue
            for i in range(0, len(mem), 1500):
                jobs.append((str(zp), mem[i:i + 1500], subset.split("_")[0].lower(), cmd, cls))
    keep = []
    with Pool(workers) as pool:
        for zpath, cmd, cls, out in pool.imap_unordered(_syntts_chunk, jobs):
            for kind, a, b in out:
                if kind == "keep":
                    keep.append(dict(path=a, label=cmd, **{"class": cls}, dataset="classmate_syntts", speaker=b,
                                     split="train", real=False, license=LICENSE["classmate_syntts"], old_class=cmd,
                                     text=cmd.replace("_", " "),
                                     rule="schema command" if cls != "unknown" else "other command (near-miss)"))
                else:
                    drop.append(dict(source=f"{zpath}#{a}", dataset="classmate_syntts", label=cmd, text="",
                                     reason=f"unreadable: {b}"))
            print(f"  syntts {Path(zpath).parent.name}/{cmd}: +{len(out)}", flush=True)
    print(f"syntts: {len(keep)} kept, {len(drop)} dropped rows", flush=True)
    return keep, drop


# ---------------------------------------------------------------- Classmate F's VCM (zip; only the parts we do not have)
TEMP_WORD = re.compile(r"\b(temperature|thermostat|temp)\b")
TEMP_VAL = re.compile(r"\b(?P<v>[a-z]+(?: [a-z]+)?) degrees\b")
TEMPS = {"eighteen": "TEMPERATURE_18", "twenty two": "TEMPERATURE_22", "twenty six": "TEMPERATURE_26"}


def temp_class(text):
    t = norm(text)
    m = TEMP_VAL.search(t)
    v = m["v"] if m else ""
    v = " ".join(v.split()[-2:]) if v.split()[-2:-1] == ["twenty"] else v.split()[-1] if v else ""
    if v not in TEMPS:
        return "unknown", "temperature request, other value (near-miss)"
    if TEMP_WORD.search(t):
        return TEMPS[v], "temperature request, schema value"
    return None, "'set it to N degrees' without saying temperature: borderline (strict rule)"


def vcm():
    zp = RAW / "class_vcm/VCM.zip"
    keep, drop = [], []
    with zipfile.ZipFile(zp) as z:
        a = pd.read_csv(io.BytesIO(z.read("VCM/VCM_MASTER/manifests/all.csv")))
        for r in a.itertuples(index=False):
            ds = r.original_dataset
            if ds == "Multi-Sensor":
                cls, rule, dataset, real = "NEXT", "schema wording 'Next song'", "web_multisensor", True
            elif ds in ("SET_TEMPERATURE_REAL", "SET_TEMPERATURE_SYNTHETIC"):
                cls, rule = temp_class(r.transcript)
                dataset = "classmate_vcm_real" if ds.endswith("REAL") else "classmate_vcm_piper"
                real = ds.endswith("REAL")
            else:
                drop.append(dict(source=f"VCM.zip#{r.filepath}", dataset="classmate_vcm", label=r.label,
                                 text=r.transcript, reason=f"{ds}: corpus already in our data (own labels)"))
                continue
            if cls is None:
                drop.append(dict(source=f"VCM.zip#{r.filepath}", dataset=dataset, label=r.label, text=r.transcript,
                                 reason=rule))
                continue
            dst = OUT / dataset / cls / (Path(r.filepath).stem + ".wav")
            convert_bytes(z.read("VCM/VCM_MASTER/" + r.filepath), dst)
            spk = {"web_multisensor": "ms", "classmate_vcm_real": "vcm", "classmate_vcm_piper": "vcm-piper"}[dataset]
            keep.append(row(dst, r.label, cls, dataset, f"{spk}-{r.speaker}", SPLIT[r.split], real, r.transcript, rule))
    print(f"vcm: {len(keep)} kept, {len(drop)} not used", flush=True)
    return keep, drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    keep, drop = [], []
    for step in (optionb, gsc, snips, vcm, lambda: syntts(a.workers)):
        k, d = step()
        keep += k
        drop += d
    man = pd.DataFrame(keep)
    assert set(man["class"]) <= set(LM["class"]) | {"unknown"}, set(man["class"]) - set(LM["class"])
    assert man.path.is_unique
    man.to_csv(OUT / "manifest.csv", index=False)
    pd.DataFrame(drop).to_csv(OUT / "dropped.csv", index=False)
    t = pd.crosstab(man["class"], man.dataset).reindex(list(LM["class"]) + ["unknown"]).fillna(0).astype(int)
    t["total"] = t.sum(1)
    print(t.to_string())
    print(pd.crosstab(man.dataset, man.split).to_string())
    print(f"manifest {len(man)} clips -> {OUT / 'manifest.csv'}; dropped/not used {len(drop)} rows")


if __name__ == "__main__":
    main()
