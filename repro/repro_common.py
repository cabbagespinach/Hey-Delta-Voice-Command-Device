"""
Shared definitions for the reproduction tools (owner + professor, 2026-10-02).

What is shared with the professor on this HPC server, and what each person must obtain under their own licence.
"""
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SHARED = Path("/home/arvir.jane.redondo/AI231_ME2_reproduce")

# audio folders the pipelines read (relative to the repository root)
DATA_DIRS = [
    # wakeword
    "data/positives", "data/negatives_confusable", "data/negatives_general", "data/negatives_media",
    "data/negatives_partial", "data/negatives_silence", "data/noise", "data/rir", "data/streaming_eval",
    "data/streaming_eval_composed",
    # command classifier
    "data/commands_synthetic", "data/commands_variants_train", "data/commands_variants_test", "data/commands_vc",
    "data/commands_fragments", "data/commands_unknown_reuse", "data/commands_real_web", "data/commands_classmates",
    "data/commands_hf",
    # the owner's command recordings (audio only; the folder's CSVs are in git)
    "data/deliverables/model/deploy/recordings",
]
# folders the professor's clone must hold as real folders (files linked one by one), because restore_optional.py
# writes the clips of a dataset he obtained himself into them
WRITABLE_DIRS = ["data/negatives_general", "data/negatives_confusable", "data/commands_real_web"]

# datasets that may not be redistributed: never put in the shared folder; reproduce.sh runs without them and says so
OPTIONAL = {
    "hey_snips": dict(
        name="Sonos \"Hey Snips\" keyword-spotting dataset v1",
        used_for="wakeword training: 3,000 clips as negatives (\"not Hey Delta\"); also in the frozen validation/test sets",
        files=re.compile(r"/Ext-HeySnips-[^/]+\.wav$"),
        raw="external_raw/hey_snips/hey_snips_research_6k_en_train_eval_clean_ter/audio_files",
        licence="academic research only; redistribution of modified clips not allowed",
        how=["Request access through the form linked from https://github.com/sonos/keyword-spotting-research-datasets",
             "Extract it so that this folder exists:",
             "  external_raw/hey_snips/hey_snips_research_6k_en_train_eval_clean_ter/audio_files/",
             "Then run:  bash setup_data.sh --server   (rebuilds exactly the 3,000 clips we used)"],
        without="the wakeword is trained without these negatives and its validation/test sets lose those clips "
                "(re-frozen); results will be close to, not identical with, ours"),
    "qualcomm": dict(
        name="Qualcomm Keyword Speech Dataset",
        used_for="wakeword evaluation only: false wake-ups on other wake words (\"Hey Android\", \"Hi Galaxy\", ...)",
        files=None,
        raw="external_raw/qualcomm/qualcomm_keyword_speech_dataset",
        licence="internal research only; may not be incorporated into another dataset",
        how=["Download it from https://www.qualcomm.com/developer/software/keyword-speech-dataset (accept the licence)",
             "Extract it so that these folders exist:",
             "  external_raw/qualcomm/qualcomm_keyword_speech_dataset/{hey_android,hey_snapdragon,hi_galaxy,hi_lumina}/",
             "No conversion is needed; the evaluation reads it in place."],
        without="the Qualcomm false-wake-up check is skipped and marked as skipped in the results"),
    "fsc": dict(
        name="Fluent Speech Commands",
        used_for="command classifier training/validation/test clips (our public-recordings group)",
        files=re.compile(r"^data/commands_real_web/.*/fsc__[^/]+\.wav$"),
        raw="external_raw/commands/fsc/fluent_speech_commands_dataset",
        licence="academic, non-commercial; the audio may not be shared",
        how=["Request it from Fluent.ai (Fluent Speech Commands dataset request form)",
             "Extract it so that this folder exists (with data/ and wavs/ inside):",
             "  external_raw/commands/fsc/fluent_speech_commands_dataset/",
             "Then run:  bash setup_data.sh --server   (rebuilds exactly the FSC clips we used)",
             "Note: the class Hugging Face dataset contains 455 FSC clips; those are used either way."],
        without="the command models are trained and tested without our FSC clips; results will be close, not identical"),
    "classmate_recordings": dict(
        name="Classmates' own recordings (4 speakers)",
        used_for="command classifier training (158 clips in the schema-B list, 59 in hf_plus)",
        files=re.compile(r"^data/commands_other_speakers/"),
        raw=None,
        licence="the speakers' own voices: shared only with their consent",
        how=["Ask the project owner; shared only if the classmates agree.",
             "Two of the four speakers are also in the class Hugging Face dataset, which is used either way."],
        without="those 158 / 59 clips are left out of training"),
}
NEVER_SHARE = re.compile(r"qualcomm|sonos|hey_snips|/Ext-HeySnips-|/fsc__|commands_other_speakers|recordings_otherspeakers",
                         re.I)
AUDIO = re.compile(r"\.(wav|flac|mp3|opus|ogg)$", re.I)
