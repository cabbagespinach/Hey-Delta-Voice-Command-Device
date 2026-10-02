#!/usr/bin/env python3
"""
Inventory of every command dataset we have, the owner's and the classmates' (owner, 2026-10-01).

    python make_dataset_inventory.py     # -> data/command_dataset/dataset_inventory.csv

One row per dataset: who it comes from, what it is, licence, citation / DOI, clip counts per split from the schema-B
training list (data/commands_schema_b_all.csv), classes covered, speakers, and whether it is used for training. Rows
for datasets we have or listed but do not train on say why. DOIs: arXiv papers carry the standard arXiv DOI
(10.48550/arXiv.<id>); "none found" = no DOI known for it (cite the paper / URL).
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
D = ROOT / "data"

OWNER, CLASS = "owner (me)", "classmates' list"
# dataset -> (origin, contributor, description, citation, doi)
META = {
    "owner_recordings": (OWNER, "owner + speaker2", "own Pi/laptop recordings of the commands", "own data", "none (own data; Zenodo deposit would give one)"),
    "reuse_owner": (OWNER, "owner", "own room recordings (TV, talk, silence) used as unknown", "own data", "none (own data)"),
    "converted_owner": (OWNER, "owner", "own recordings voice-converted to other voices (Chatterbox VC)", "own data; Chatterbox (Resemble AI, MIT)", "none"),
    "synthetic_piper": (OWNER, "owner", "Piper TTS voices saying the commands", "Piper (rhasspy), per-voice licences", "none"),
    "synthetic_ph": (OWNER, "owner", "Philippine-accent neural TTS (edge-tts)", "Microsoft neural voices via edge-tts", "none"),
    "synthetic_clone": (OWNER, "owner", "Chatterbox voice clones of FLEURS speakers", "Chatterbox (MIT); FLEURS arXiv:2205.12446", "10.48550/arXiv.2205.12446"),
    "variants_piper": (OWNER, "owner", "other wordings of each command, Piper voices", "Piper (rhasspy)", "none"),
    "variants_ph": (OWNER, "owner", "other wordings, Philippine-accent TTS", "edge-tts", "none"),
    "variants_clone": (OWNER, "owner", "other wordings, Chatterbox clones", "Chatterbox (MIT)", "none"),
    "fragments_owner_recordings": (OWNER, "owner", "cut-off commands (unknown) from own recordings", "own data", "none"),
    "fragments_synthetic_piper": (OWNER, "owner", "cut-off commands from Piper clips", "derived", "none"),
    "fragments_synthetic_ph": (OWNER, "owner", "cut-off commands from edge-tts clips", "derived", "none"),
    "fragments_synthetic_clone": (OWNER, "owner", "cut-off commands from Chatterbox clips", "derived", "none"),
    "fragments_web_slurp": (CLASS, "owner (from SLURP)", "cut-off SLURP commands", "Bastianelli et al. 2020, SLURP, arXiv:2011.13205", "10.48550/arXiv.2011.13205"),
    "fragments_web_tas": (CLASS, "owner (from Timers and Such)", "cut-off Timers and Such commands", "Lugosch et al. 2021, arXiv:2104.01604", "10.5281/zenodo.4623772"),
    "reuse_musan": (OWNER, "owner", "MUSAN noise / music / speech as unknown", "Snyder et al. 2015, arXiv:1510.08484", "10.48550/arXiv.1510.08484"),
    "reuse_fleurs": (OWNER, "owner", "FLEURS Filipino/English speech as unknown", "Conneau et al. 2022, arXiv:2205.12446", "10.48550/arXiv.2205.12446"),
    "reuse_mswc": (OWNER, "owner", "MSWC single words (Common Voice audio) as unknown", "Mazumder et al. 2021, NeurIPS Datasets & Benchmarks", "none found"),
    "web_fsc": (CLASS, "Classmate B (listed); downloaded by owner", "Fluent Speech Commands", "Lugosch et al. 2019, arXiv:1904.03670", "10.48550/arXiv.1904.03670"),
    "web_slurp": (CLASS, "Classmate A (listed); downloaded by owner", "SLURP real recordings", "Bastianelli et al. 2020, arXiv:2011.13205", "10.48550/arXiv.2011.13205"),
    "web_tas": (CLASS, "Classmate E (listed); downloaded by owner", "Timers and Such v1.0, real part", "Lugosch et al. 2021, arXiv:2104.01604", "10.5281/zenodo.4623772"),
    "other_speakers": (CLASS, "classmates (4 speakers, owner-provided)", "classmates' own recordings of the schema", "classmates' recordings", "none"),
    "classmate_optionb": (CLASS, "Classmate A", "voice-cloned Option-B set (100 speakers, clean + noisy)", "github markandrian30/AI231 (commit 5b23a95)", "none"),
    "web_gsc": (CLASS, "Classmate B", "Google Speech Commands v0.02 ('stop' + other words, noise)", "Warden 2018, arXiv:1804.03209", "10.48550/arXiv.1804.03209"),
    "web_snips": (CLASS, "Classmate E", "Snips SLU smart-lights / smart-speaker", "Saade et al. 2018, arXiv:1810.12735", "10.48550/arXiv.1810.12735"),
    "classmate_syntts": (CLASS, "Classmate C", "SynTTS-Commands English (CosyVoice2)", "SynTTS-Commands (github lugan113), arXiv:2511.07821", "10.48550/arXiv.2511.07821"),
    "web_multisensor": (CLASS, "Classmate F (VCM)", "Multi-Sensor Voice Command Dataset, 'next song' x 4 sensors", "Rusci, Van hamme, Tuytelaars, KU Leuven RDR", "10.48804/IEKKVZ"),
    "classmate_vcm_real": (CLASS, "Classmate F (VCM)", "classmate's own SET_TEMPERATURE recordings", "classmate recordings", "none"),
    "classmate_vcm_piper": (CLASS, "Classmate F (VCM)", "classmate's Piper SET_TEMPERATURE clips", "Piper (rhasspy)", "none"),
}
NOT_USED = [  # dataset, origin, contributor, description, licence, citation, doi, reason
    ("kaggle_synthetic_speech_commands", CLASS, "Classmate B", "Kaggle synthetic speech commands (jbuchner)", "see Kaggle page", "kaggle.com/datasets/jbuchner/synthetic-speech-commands-dataset", "none found", "not downloaded: needs kaggle.json (owner: train without it)"),
    ("common_voice_kaggle", CLASS, "Classmate D", "Common Voice (Kaggle mirror)", "CC0", "Ardila et al. 2020, arXiv:1912.06670", "10.48550/arXiv.1912.06670", "not downloaded: needs kaggle.json; MSWC words (Common Voice audio) already used as unknown"),
    ("mlend_spoken_numerals", CLASS, "Classmate D", "MLEnd spoken numerals", "see Kaggle page", "kaggle.com/datasets/jesusrequena/mlend-spoken-numerals", "none found", "not downloaded: needs kaggle.json"),
    ("fsc_kaggle_mirror", CLASS, "Classmate B", "Fluent Speech Commands (Kaggle mirror)", "FSC Public License", "Lugosch et al. 2019", "10.48550/arXiv.1904.03670", "same corpus as web_fsc (already used)"),
    ("vcm_fsc_slurp_tas", CLASS, "Classmate F (VCM)", "VCM copies of FSC / SLURP / Timers and Such clips", "source licences (FSC forbids sharing audio)", "see web_fsc / web_slurp / web_tas", "", "duplicates of corpora we already have, labelled by our own rules"),
    ("vcm_gsc_silence", CLASS, "Classmate F (VCM)", "VCM SILENCE cut from GSC background noise", "CC BY 4.0", "Warden 2018", "10.48550/arXiv.1804.03209", "duplicates web_gsc _background_noise_"),
    ("classmate_optionb_flagged", CLASS, "Classmate A", "Option-B clips that failed Classmate A's Whisper QA", "none stated", "github markandrian30/AI231", "none", "failed the contributor's own quality check"),
    ("qualcomm_keyword_speech", OWNER, "owner", "Qualcomm Keyword Speech Dataset (wakeword evaluation)", "internal research only, no incorporation", "Qualcomm Technologies 2019", "none found", "licence forbids incorporating; wakeword evaluation only"),
    ("hey_snips", OWNER, "owner", "Sonos 'Hey Snips' keyword spotting v1", "academic research only", "Coucke et al. 2019, ICASSP", "10.1109/ICASSP.2019.8683474 (verify)", "wakeword data, not commands"),
    ("mit_ir", OWNER, "owner", "MIT acoustical reverberation survey (270 IRs)", "CC BY 4.0", "Traer & McDermott 2016, PNAS", "10.1073/pnas.1612524113 (verify)", "used as room-echo augmentation, not as clips"),
]


def main():
    a = pd.read_csv(D / "commands_schema_b_all.csv")
    drop = pd.read_csv(D / "commands_classmates/dropped.csv")
    rows = []
    for ds, g in a.groupby("dataset", sort=False):
        origin, who, desc, cite, doi = META[ds]
        sp = g.split.value_counts()
        cmd = g[g["class"] != "unknown"]
        nd = drop[drop.dataset == ds]
        rows.append(dict(dataset=ds, origin=origin, contributor=who, description=desc,
                         real_or_synthetic="real" if g.real.astype(str).str.lower().eq("true").all() else
                         "synthetic" if g.real.astype(str).str.lower().eq("false").all() else "mixed",
                         license=g.license.iloc[0], citation=cite, doi=doi, clips=len(g),
                         train=int(sp.get("train", 0)), validation=int(sp.get("validation", 0)), test=int(sp.get("test", 0)),
                         command_clips=len(cmd), unknown_clips=int((g["class"] == "unknown").sum()),
                         command_classes=cmd["class"].nunique(), speakers=g.speaker.nunique(),
                         used_in_training="yes (train split)" if sp.get("train", 0) else "evaluation only",
                         notes=f"{len(nd)} clips dropped/not used (see data/commands_classmates/dropped.csv)" if len(nd) else ""))
    for ds, origin, who, desc, lic, cite, doi, why in NOT_USED:
        rows.append(dict(dataset=ds, origin=origin, contributor=who, description=desc, license=lic, citation=cite, doi=doi,
                         clips=0, used_in_training="no", notes=why))
    t = pd.DataFrame(rows)
    out = D / "command_dataset/dataset_inventory.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    print(t[["dataset", "origin", "clips", "train", "validation", "test", "command_classes", "speakers",
             "used_in_training"]].to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
