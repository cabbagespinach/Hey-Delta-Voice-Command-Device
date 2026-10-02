#!/usr/bin/env python3
"""
One list of every command-classifier clip, with class, source, speaker, licence and split (decisions of 2026-09-30).

    python build_commands_all.py            # -> data/commands_all.csv (+ a summary)

Splits:
- owner recordings: test = every `speaker2` clip + owner session 16:18-16:33; validation = owner 13:36-13:55;
  train = the rest. Clips Whisper found empty or cut off are left out, except for the unknown prompts (an empty
  `unknown_silent` is what it should be).
- public datasets (FSC, SLURP, Timers and Such): their own speaker-disjoint splits (valid/devel -> validation).
- reused unknowns (data/commands_unknown_reuse): the wakeword project's split of each source recording.
- synthetic voices: each voice is in exactly one split, 80/10/10 by a stable hash of its name. The 10 cloned voices
  used as voice-conversion targets (fleursfil00-09) are forced into train, since converted training clips carry them.
- voice-converted clips: only strict-pass clips whose SOURCE recording is in train; the others are left out.
- cut-off fragments (unknown_partial): the split of the clip they were cut from.
- other speakers' recordings (data/commands_other_speakers): one split per speaker (OTHER_SPEAKER_SPLIT).
Paths are relative to the project root. `real` is True for recorded (not synthesized or converted) audio.
"""
from pathlib import Path
import hashlib, re

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
D = ROOT / "data"
REC = HERE.parent / "model/deploy/recordings"
VC_TARGETS = {f"fleursfil{i:02d}_{g}" for i in range(10) for g in "mf"}
SPLIT_NAME = {"train": "train", "valid": "validation", "validation": "validation", "devel": "validation", "dev": "validation",
              "test": "test"}


# other speakers (data/commands_other_speakers): all four in train (owner decision 2026-10-01)
OTHER_SPEAKER_SPLIT = {"classmate1": "train", "classmate2": "train", "classmate3": "train", "classmate4": "train"}


def voice_split(voice):
    if voice in VC_TARGETS:
        return "train"
    h = int(hashlib.md5(voice.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else "validation" if h < 90 else "test"


def owner_split(speaker, recorded_at):
    hm = str(recorded_at)[9:13]
    if speaker == "speaker2" or "1618" <= hm <= "1633":
        return "test"
    if "1336" <= hm <= "1355":
        return "validation"
    return "train"


def collect(phrasing_test_exclusions=True):
    """Every clip of every source as one table (old 31-label set). phrasing_test_exclusions=False keeps the other
    speakers' clips that say a held-out wording of the phrasing test (schema B has no such test)."""
    lm = pd.read_csv(HERE.parent / "model/deploy/label_map.csv")
    cls = dict(zip(lm.label, lm["class"]))
    to_class = lambda l: cls.get(l, "unknown" if str(l).startswith("unknown") else None)
    parts = []

    # 1. owner recordings
    m = pd.read_csv(REC / "manifest.csv")
    chk = pd.read_csv(REC / "_check/check.csv")[["file", "verdict"]]
    m = m.merge(chk, on="file", how="left")
    bad = m.verdict.isin(["empty", "cut_off"]) & ~m.label.str.startswith("unknown")
    print(f"owner recordings: {len(m)}; left out {bad.sum()} empty / cut-off command clips")
    m = m[~bad]
    parts.append(pd.DataFrame(dict(
        path="data/deliverables/model/deploy/recordings/" + m.file, label=m.label, dataset="owner_recordings",
        speaker=m.speaker, split=[owner_split(s, t) for s, t in zip(m.speaker, m.recorded_at)], real=True,
        license="own data")))

    # 2. public datasets
    w = pd.read_csv(D / "commands_real_web/manifest.csv")
    parts.append(pd.DataFrame(dict(path="data/commands_real_web/" + w.file, label=w.label, dataset="web_" + w.source,
                                   speaker=w.speaker, split=w.source_split.map(SPLIT_NAME), real=True, license=w.license)))

    # 3. reused unknowns
    u = pd.read_csv(D / "commands_unknown_reuse/manifest.csv")
    parts.append(pd.DataFrame(dict(path="data/commands_unknown_reuse/" + u.file, label=u.label,
                                   dataset="reuse_" + u.source, speaker=u.speaker,
                                   split=u.source_split.map(SPLIT_NAME), real=u.source.isin(["owner", "fleurs", "musan", "mswc"]),
                                   license=u.license)))

    # 4. synthetic voices
    s = pd.read_csv(D / "commands_synthetic/manifest.csv")
    lic = {"ph": "edge-tts output (Microsoft neural voices)", "clone": "Chatterbox (MIT) clones of FLEURS CC-BY 4.0 speakers",
           "piper": "Piper voices (per-voice model licences)"}
    parts.append(pd.DataFrame(dict(path="data/commands_synthetic/" + s.file, label=s.label, dataset="synthetic_" + s.tier,
                                   speaker=s.voice, split=s.voice.map(voice_split), real=False, license=s.tier.map(lic))))
    syn_split = dict(zip("data/commands_synthetic/" + s.file, s.voice.map(voice_split)))

    # 5. voice-converted owner recordings
    v = pd.read_csv(D / "commands_vc/check.csv")
    osplit = dict(zip(m.file, [owner_split(a, b) for a, b in zip(m.speaker, m.recorded_at)]))
    v = v[v.kept & (v.source_file.map(osplit) == "train")]
    parts.append(pd.DataFrame(dict(path="data/commands_vc/" + v.file, label=v.label, dataset="converted_owner",
                                   speaker="vc-" + v.target_voice, split="train", real=False,
                                   license="Chatterbox VC (MIT) of own recordings")))

    # 6. cut-off fragments
    fp = D / "commands_fragments/manifest.csv"
    if fp.exists():
        f = pd.read_csv(fp)
        web_split = dict(zip("data/commands_real_web/" + w.file, w.source_split.map(SPLIT_NAME)))

        def fsplit(src):
            if src.startswith("data/deliverables/model/deploy/recordings/"):
                return osplit.get(src.split("recordings/", 1)[1])
            return syn_split.get(src) or web_split.get(src)
        f["split"] = f.source_file.map(fsplit)
        parts.append(pd.DataFrame(dict(path="data/commands_fragments/" + f.file, label="unknown_partial",
                                       dataset="fragments_" + f.source_dataset, speaker="frag-" + f.source_file,
                                       split=f.split, real=f.source_dataset.str.match("owner|web"),
                                       license="derived from source clip")))
    else:
        print("(no fragments yet)")

    # 7. other wordings (generate_phrasing_variants.py, 2026-10-01): Whisper-confirmed clips; split = the voice's split
    #    (train / validation voices only, so no test voice is ever heard)
    vp = D / "commands_variants_train/manifest.csv"
    if vp.exists():
        q = pd.read_csv(vp)
        q = q[q.get("whisper_ok", False) == True]                           # noqa: E712
        lic = {"ph": "edge-tts output (Microsoft neural voices)", "piper": "Piper voices (per-voice model licences)",
               "clone": "Chatterbox (MIT) clones of FLEURS CC-BY 4.0 speakers"}
        parts.append(pd.DataFrame(dict(path=q.path, label=q.label, dataset="variants_" + q.tier, speaker=q.voice,
                                       split=q.split, real=False, license=q.tier.map(lic))))
        print(f"other wordings: {len(q)} Whisper-confirmed clips")

    # 8. other speakers' recordings (owner upload 2026-10-01, prepare_other_speakers.py): labelled clips only; one split
    #    per speaker; a TRAIN clip saying a never-trained wording of the phrasing test is left out (it would be trained on)
    op = D / "commands_other_speakers/manifest.csv"
    if op.exists():
        o = pd.read_csv(op)
        o["split"] = o.speaker.map(OTHER_SPEAKER_SPLIT)
        o = o[o.keep & ~(o.held_out_wording & (o.split == "train") & phrasing_test_exclusions)]
        parts.append(pd.DataFrame(dict(path=o.file, label=o.label, dataset="other_speakers", speaker=o.speaker,
                                       split=o.split, real=True,
                                       license="recorded by classmates for this project (owner-provided)")))
        print(f"other speakers: {len(o)} clips")

    a = pd.concat(parts, ignore_index=True)
    a["class"] = a.label.map(to_class)
    assert a["class"].notna().all(), a[a["class"].isna()].label.unique()
    assert a.split.isin(["train", "validation", "test"]).all(), a[~a.split.isin(["train", "validation", "test"])].head()
    missing = [p for p in a.path if not (ROOT / p).exists()]
    assert not missing, f"{len(missing)} missing files, e.g. {missing[:3]}"
    return a[["path", "label", "class", "dataset", "speaker", "split", "real", "license"]]


def main():
    a = collect()
    a.to_csv(D / "commands_all.csv", index=False)

    print(f"\n{len(a)} clips, {a['class'].nunique()} classes -> {D / 'commands_all.csv'}")
    print(pd.crosstab(a.dataset, a.split, margins=True).to_string())
    t = pd.crosstab(a["class"], a.split)
    print("\nclips per class and split:\n" + t.to_string())
    real_test = a[(a.split == "test") & (a.dataset == "owner_recordings")]
    print(f"\nreal owner-recording test clips: {len(real_test)} "
          f"({(real_test.speaker == 'speaker2').sum()} speaker2)")


if __name__ == "__main__":
    main()
