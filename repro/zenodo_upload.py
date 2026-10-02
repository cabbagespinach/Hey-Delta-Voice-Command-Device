#!/usr/bin/env python3
"""
OWNER-SIDE: the restricted-access Zenodo dataset record (owner decisions 2026-10-02).

    python repro/zenodo_upload.py pack      # tar files of the shared data folder -> STAGE (temporary)
    python repro/zenodo_upload.py draft     # create the draft record + metadata; reserves the DOI (not published)
    python repro/zenodo_upload.py upload    # upload the tar files (resumable: skips files already uploaded)
    python repro/zenodo_upload.py publish   # ONLY after the owner's OK: makes the record (and DOI) permanent

Token: ~/.zenodo_token (the owner's). Record state: repro/zenodo_record.json (id, reserved DOI, links; no token).
Content = the shared HPC data folder (repro/make_shared_folder.py) minus what the owner left out: Classmate A's and Classmate F's
sets (cited by their public links), the class Hugging Face set (cited by revision). Never included anyway: Hey Snips,
Qualcomm, Fluent Speech Commands, classmates' own recordings. Edge-TTS clips included (owner: restricted).
"""
from pathlib import Path
import json, os, subprocess, sys

import requests

REPO = Path(__file__).resolve().parents[1]
SHARED = Path("/home/arvir.jane.redondo/AI231_ME2_reproduce")
STAGE = Path("/mnt/jfs_hpc/home/arvir.jane.redondo/zenodo_staging")
API = "https://zenodo.org/api"
STATE = REPO / "repro/zenodo_record.json"
LEAVE_OUT = ["data/commands_hf", "data/commands_classmates/classmate_optionb", "data/commands_classmates/classmate_vcm_real",
             "data/commands_classmates/classmate_vcm_piper"]
PARTS = {  # tar name -> folders (relative to the shared folder)
    "wakeword_positives": ["data/positives"],
    "wakeword_negatives_confusable": ["data/negatives_confusable"],
    "wakeword_negatives_general": ["data/negatives_general"],
    "wakeword_negatives_media": ["data/negatives_media"],
    "wakeword_negatives_partial": ["data/negatives_partial"],
    "wakeword_negatives_silence": ["data/negatives_silence"],
    "wakeword_rir": ["data/rir"],
    "wakeword_streaming_eval": ["data/streaming_eval", "data/streaming_eval_composed"],
    "owner_command_recordings": ["data/deliverables/model/deploy/recordings"],
    "commands_synthetic": ["data/commands_synthetic"],
    "commands_variants": ["data/commands_variants_train", "data/commands_variants_test"],
    "commands_voice_converted": ["data/commands_vc"],
    "commands_fragments_and_unknown": ["data/commands_fragments", "data/commands_unknown_reuse"],
    "commands_public_slurp_tas": ["data/commands_real_web"],
    "commands_google_speech_commands": ["data/commands_classmates/web_gsc"],
    "commands_syntts": ["data/commands_classmates/classmate_syntts"],
    "commands_snips_multisensor": ["data/commands_classmates/web_snips", "data/commands_classmates/web_multisensor"],
}
DESCRIPTION = """
<p>Audio for the AI231 ME2 project <b>Hey Delta</b> (wakeword + voice-command classifier for a Raspberry Pi 5):
the project owner's own recordings, synthetic voices made for the project, and prepared clips of openly licensed
public datasets, exactly as used for training and evaluation. Code, clip lists, splits, logs, model weights and
reproduction instructions: <a href="https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device">github.com/cabbagespinach/Hey-Delta-Voice-Command-Device</a>
(REPRODUCE.md). Each tar file holds folders under <code>data/</code>; extract them in the repository root.</p>
<p><b>Not included</b> (obtain from their owners; the repository's setup_data.sh prints how): Sonos "Hey Snips",
Qualcomm Keyword Speech Dataset, Fluent Speech Commands, classmates' own recordings, the classmates' voice-cloned
"Option B" set and VCM collection (public: github.com/markandrian30/AI231, MEX2/Data; Google Drive link in the
repository), and the class Hugging Face dataset (airimonda/ai231-me2-voice-commands, revision a90b8d1).</p>
<p><b>Sources and licences of the included prepared clips:</b> Google Speech Commands v0.02 (CC BY 4.0), MUSAN
(CC BY 4.0), FLEURS (CC BY 4.0), Multilingual Spoken Words Corpus (CC BY 4.0), MIT reverberation survey impulse
responses (CC BY 4.0), SLURP (CC BY-NC 4.0), Timers and Such (CC0), Snips SLU, SynTTS-Commands (MIT), Multi-Sensor
Voice Command Dataset (CC BY 4.0, doi:10.48804/IEKKVZ); synthetic voices: Piper, Chatterbox (MIT), Microsoft Edge
neural voices (edge-tts, academic use only). Each source keeps its own licence; full list with citations in the
repository's external_raw/README.md.</p>
"""
CONDITIONS = ("Access is granted for academic, non-commercial use: verifying or reproducing the AI231 ME2 'Hey "
              "Delta' project and related coursework or research. The files contain the author's own voice "
              "recordings: do not redistribute them. Each included public dataset keeps its own licence (see the "
              "description). Please state your name, affiliation and purpose in the request, and cite this record.")


def token():
    return (Path.home() / ".zenodo_token").read_text().strip()


def H():
    return {"Authorization": f"Bearer {token()}"}


def pack():
    STAGE.mkdir(mode=0o700, exist_ok=True)
    os.chmod(STAGE, 0o700)
    for name, dirs in PARTS.items():
        out = STAGE / f"{name}.tar"
        if out.exists():
            continue
        ex = [f"--exclude={x}" for x in LEAVE_OUT]
        subprocess.run(["tar", "-cf", str(out) + ".part", *ex, *dirs], cwd=SHARED, check=True)
        os.rename(str(out) + ".part", out)
        print(f"{out.name}: {out.stat().st_size / 1e9:.2f} GB", flush=True)
    print(f"total {sum(p.stat().st_size for p in STAGE.glob('*.tar')) / 1e9:.1f} GB in {STAGE}")


def draft():
    if STATE.exists():
        print("draft exists:", json.loads(STATE.read_text())["doi"]); return
    r = requests.post(f"{API}/deposit/depositions", headers=H(), json={}, timeout=60)
    r.raise_for_status()
    d = r.json()
    meta = dict(
        title="Hey Delta: wakeword and voice-command audio (AI231 ME2 project data)",
        upload_type="dataset", description=DESCRIPTION, access_right="restricted", access_conditions=CONDITIONS,
        creators=[{"name": "Redondo, Arvir Jane R."}], prereserve_doi=True,
        keywords=["keyword spotting", "wake word", "voice commands", "Raspberry Pi", "BC-ResNet", "Filipino English"],
        related_identifiers=[{"identifier": "https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device",
                              "relation": "isSupplementTo", "resource_type": "software"}],
        notes="Restricted access. The owner approves requests (e.g. course instructors) for academic use.")
    r = requests.put(f"{API}/deposit/depositions/{d['id']}", headers=H(), json={"metadata": meta}, timeout=60)
    r.raise_for_status()
    d = r.json()
    state = dict(id=d["id"], doi=d["metadata"]["prereserve_doi"]["doi"], bucket=d["links"]["bucket"],
                 html=d["links"]["html"], record_url=f"https://zenodo.org/records/{d['id']}", published=False)
    STATE.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2))


def upload():
    s = json.loads(STATE.read_text())
    have = {f["filename"]: f["filesize"] for f in
            requests.get(f"{API}/deposit/depositions/{s['id']}/files", headers=H(), timeout=60).json()}
    for p in sorted(STAGE.glob("*.tar")):
        if have.get(p.name) == p.stat().st_size:
            print(f"{p.name}: already uploaded"); continue
        with open(p, "rb") as f:
            r = requests.put(f"{s['bucket']}/{p.name}", headers=H(), data=f, timeout=None)
        r.raise_for_status()
        print(f"{p.name}: uploaded {p.stat().st_size / 1e9:.2f} GB", flush=True)
    print("upload done; draft NOT published:", s["html"])


def publish():
    s = json.loads(STATE.read_text())
    r = requests.post(f"{API}/deposit/depositions/{s['id']}/actions/publish", headers=H(), timeout=120)
    r.raise_for_status()
    s["published"] = True
    STATE.write_text(json.dumps(s, indent=2) + "\n")
    print("published:", s["record_url"], "doi:", s["doi"])


if __name__ == "__main__":
    {"pack": pack, "draft": draft, "upload": upload, "publish": publish}[sys.argv[1]]()
