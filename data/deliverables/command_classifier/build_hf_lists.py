#!/usr/bin/env python3
"""
Clip lists for the class-benchmark comparison (owner, 2026-10-02): the class's Hugging Face dataset
(airimonda/ai231-me2-voice-commands, data/commands_hf/manifest.csv from hf_extract.py) alone, and with our data added.

    python build_hf_lists.py     # -> data/commands_hf_only.csv, data/commands_hf_plus.csv, data/commands_hf_excluded.csv

Both lists, same columns as data/commands_schema_b_all.csv:
  train       hf_only: HF train.   hf_plus: HF train + our extra clips (our train split) after the filters below.
  validation  identical in both: our validation clips after the filters below, never anything from HF (HF has no
              validation split; owner rule: validation must NOT come from the test set).
  test        identical in both: HF test (the class benchmark).  holdout: HF holdout (kept for the Pi live test).
HF labels -> schema-B classes: command + slot value (TEMPERATURE + "22 degrees" -> TEMPERATURE_22); OUT_OF_SCOPE and
"<COMMAND> (other slot value)" -> unknown.
Filters on our data (owner-approved notes 1-4):
  1. every speaker who is in HF test or holdout is removed from our side (Classmate A's voices, Classmate1 = HF S-CM1,
     Classmate F's S4/S5, FSC / Timers and Such / Snips / Multi-Sensor / SLURP / Common Voice (MSWC) test speakers). SLURP
     speakers come from SLURP's metadata.json, MSWC speakers from external_raw/mswc_en/selection.csv, cut-off
     fragments take the speaker of the clip they were cut from;
  2. clips HF already has in train are not added again (owner recordings, Classmate2, Classmate A's sampled clips,
     FSC / Timers and Such files);
  3. validation: additionally no speaker who is in HF train (except the owner and speaker2, who are the deployment
     users) and no clip that is in HF train;
  4. our test-split clips are not used (the test set is HF test).
HF owner / speaker2 clips point to the owner's original recordings (identical audio, same file names) so the
recipe treats them exactly as before (owner background augmentation, 'Delta' prefix bank).
"""
from pathlib import Path
import json, re

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
D = ROOT / "data"
LM = pd.read_csv(HERE.parent / "model/deploy/label_map_schema_b.csv")
CLASSMATE1 = "S-CM1"           # HF speaker id of our Classmate1 (audio-matched 46 of 50 clips, 2026-10-02)
CLASSMATE2 = "S-CM2"           # HF speaker id of our Classmate2 (audio-matched 49 of 49)
OWNER = {"S-OWNER": "owner", "S-OWNER_speaker2": "speaker2"}
DS = {"group_synthetic": "classmate_hf_optionb", "real_voice": "classmate_hf_real", "xela_SET_TEMPERATURE_REAL": "classmate_hf_vcm",
      "xela_Multi-Sensor": "web_hf_multisensor", "FluentSpeechCommands": "web_hf_fsc", "SLURP": "web_hf_slurp",
      "SNIPS": "web_hf_snips", "TimersAndSuch": "web_hf_tas", "CommonVoice_en": "web_hf_commonvoice",
      "SpeechCommands_v2": "web_hf_gsc"}
SPK = {"group_synthetic": "optionb-", "FluentSpeechCommands": "fsc-", "TimersAndSuch": "tas-", "SNIPS": "snips-",
       "xela_Multi-Sensor": "ms-", "xela_SET_TEMPERATURE_REAL": "vcm-", "SLURP": "slurpspk-", "CommonVoice_en": "cv-",
       "real_voice": "hf-", "SpeechCommands_v2": "gsc-"}


def hf_rows(own_paths):
    h = pd.read_csv(D / "commands_hf/manifest.csv", low_memory=False)
    val2cls = {(l, v.lower()): c for c, l, v in zip(LM["class"], LM.label, LM.value.fillna(""))}
    fixed = set(LM[LM.type == "fixed"]["class"])
    cls = []
    for r in h.itertuples(index=False):
        if r.command in fixed:
            cls.append(r.command)
        elif isinstance(r.slot_value, str) and (r.command, r.slot_value.lower()) in val2cls:
            cls.append(val2cls[(r.command, r.slot_value.lower())])
        else:
            cls.append("unknown")                  # OUT_OF_SCOPE, other slot values
    h["class"] = cls
    sid = h.speaker_id.fillna("none").astype(str)
    h["speaker"] = [OWNER.get(s, SPK[src] + s) for s, src in zip(sid, h.source)]
    h["dataset"] = h.source.map(DS)
    stem = h.file.str.replace(r"^audio/[a-z]+_\d+_real_voice_", "", regex=True)
    own = sid.isin(OWNER)
    mapped = stem.map(own_paths)
    h.loc[own & mapped.notna(), "path"] = mapped[own & mapped.notna()]
    h.loc[own, "dataset"] = "owner_recordings"
    out = pd.DataFrame(dict(path=h.path, label=h.bucket, **{"class": h["class"]}, dataset=h.dataset, speaker=h.speaker,
                            split=h.split, real=h.is_synthetic == 0, license="see HF dataset card (per source)",
                            old_class=h.command, text=h.transcript, rule="HF label (command + slot value)"))
    print(f"HF: {len(out)} clips; owner clips pointing to our original files: {int((own & mapped.notna()).sum())} of {int(own.sum())}")
    return out, h


def true_speakers(a):
    """Speaker ids comparable with the HF ones (SLURP user ids, Common Voice client ids, fragment sources)."""
    sp = a.speaker.copy()
    meta = json.loads((ROOT / "external_raw/commands/slurp/metadata.json").read_text())
    usr = {Path(f).stem: r["usrid"] for v in meta.values() for f, r in v["recordings"].items()}
    sel = pd.read_csv(ROOT / "external_raw/mswc_en/selection.csv")
    cv = {Path(l).stem.replace("common_voice_en_", ""): s for l, s in zip(sel.LINK, sel.SPEAKER)}
    by_path = dict(zip(a.path, a.speaker))
    for i, r in a.iterrows():
        p = r.path
        if r.speaker.startswith("frag-"):          # a cut-off clip: the speaker of its source clip
            p = r.speaker[len("frag-"):]
            sp[i] = by_path.get(p, r.speaker)
        m = re.search(r"slurp__(audio-[-\d]+?(?:-headset)?)(?:__frag)?\.wav$", p)
        if m:
            sp[i] = "slurpspk-" + usr.get(m.group(1), "unknown-" + m.group(1))
        elif r.dataset == "reuse_mswc":
            k = r.speaker.replace("mswc-", "")
            sp[i] = "cv-" + cv[k] if k in cv else r.speaker
    return sp


def main():
    a = pd.read_csv(D / "commands_schema_b_all.csv")
    own_rec = a[a.dataset == "owner_recordings"]
    own_paths = dict(zip(own_rec.path.str.split("/").str[-1].str.replace(".wav", "", regex=False) + ".wav", own_rec.path))
    hf, h = hf_rows(own_paths)
    a["true_speaker"] = true_speakers(a)
    hf_eval_spk = set(hf[hf.split.isin(["test", "holdout"])].speaker) | {"classmate1"}
    hf_train_spk = set(hf[hf.split == "train"].speaker) | {"classmate2"}
    # clips HF already has in train (by original file name)
    tr = h[h.split == "train"]
    base = tr.file.str.replace(r"^audio/train_\d+_", "", regex=True).str.replace(".wav", "", regex=False)
    dup = set(base[tr.source == "group_synthetic"].str.replace("group_synthetic_", "", regex=False))
    dup |= set("fsc__" + base[tr.source == "FluentSpeechCommands"].str.replace("FluentSpeechCommands_", "", regex=False))
    dup |= set("tas__" + base[tr.source == "TimersAndSuch"].str.replace("TimersAndSuch_", "", regex=False))
    dup |= set(base[tr.source == "real_voice"].str.replace("real_voice_", "", regex=False))
    stem = a.path.str.split("/").str[-1].str.replace(".wav", "", regex=False)
    a["why"] = ""
    a.loc[a.true_speaker.isin(hf_eval_spk) | a.speaker.isin(hf_eval_spk), "why"] = "speaker in HF test/holdout"
    a.loc[(a.why == "") & (stem.isin(dup) | (a.speaker == "classmate2")), "why"] = "already in HF train"
    a.loc[(a.why == "") & (a.split == "test"), "why"] = "our test split (not used; test = HF test)"
    users = {"owner", "speaker2", "owner-room"}
    a.loc[(a.why == "") & (a.split == "validation") & (a.true_speaker.isin(hf_train_spk) | a.speaker.isin(hf_train_spk))
          & ~a.speaker.isin(users), "why"] = "validation speaker also in HF train"
    keep = a[a.why == ""].drop(columns=["why", "true_speaker"])
    ours_train, val = keep[keep.split == "train"], keep[keep.split == "validation"]
    assert not (set(val.path) & set(hf.path)) and not (set(ours_train.path) & set(hf.path))
    hf_only = pd.concat([hf, val], ignore_index=True)
    hf_plus = pd.concat([hf, ours_train, val], ignore_index=True)
    for name, t in (("hf_only", hf_only), ("hf_plus", hf_plus)):
        assert t.path.is_unique, t[t.path.duplicated()].head()
        missing = [p for p in t.path if not (ROOT / p).exists()]
        assert not missing, f"{name}: {len(missing)} missing, e.g. {missing[:3]}"
        t.to_csv(D / f"commands_{name}.csv", index=False)
        print(f"== {name}: {len(t)} clips"); print(pd.crosstab(t.dataset, t.split).to_string())
        sp = {s: set(t[t.split == s].speaker) for s in ("train", "validation", "test")}
        print("  speakers shared train/test:", len(sp["train"] & sp["test"]), " validation/test:", len(sp["validation"] & sp["test"]))
    a[a.why != ""][["path", "dataset", "speaker", "true_speaker", "split", "why"]].to_csv(D / "commands_hf_excluded.csv", index=False)
    print(a[a.why != ""].groupby(["why", "dataset"]).size().to_string())
    v = val[val["class"] == "unknown"]
    import sys
    sys.path.insert(0, str(HERE / "dataloading"))
    import command_data as cd
    print("validation unknown clips per group:", pd.Series([cd.source_group(d, "unknown") for d in v.dataset]).value_counts().to_dict())
    print("validation command clips per group:", pd.Series([cd.source_group(d, c) for d, c in zip(val.dataset, val["class"]) if c != "unknown"]).value_counts().to_dict())


if __name__ == "__main__":
    main()
