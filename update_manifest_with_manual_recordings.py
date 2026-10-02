"""
Scan the data folders referenced in manifest.csv for new recordings whose
filename starts with "Manual", and append rows for the ones not already
present in the manifest.

USAGE:
    Set BASE_DIR below to the root folder that contains "data/..."
    (i.e. the folder such that BASE_DIR / <filepath from manifest> is a
    real file on disk). Then run:

        python update_manifest_with_manual_recordings.py

WHAT GETS FILLED IN AUTOMATICALLY:
    - filepath        : relative path (matches manifest convention)
    - category         : the folder name the file was found in
    - label            : looked up from existing category -> label mapping
                          in the manifest (e.g. "positives" -> "positive")
    - duration_sec     : read directly from the audio file
    - sample_rate      : read directly from the audio file
    - recording_id     : filename without extension
    - source           : "manual_recording"
    - created_at        : current UTC timestamp, ISO format (matches manifest)

WHAT IS LEFT BLANK (not applicable / not derivable from the file itself):
    - source_id, speaker_id, voice, speed, pitch, transcription,
      edit_distance, parent_filepath
    These are TTS-synthesis fields or need human input; fill them in by
    hand afterward if needed, or extend FILENAME_PARSER below if your
    "Manual.*" filenames actually encode any of them.
"""

import os
import glob
from datetime import datetime, timezone

import pandas as pd
import soundfile as sf

MANIFEST_PATH = "manifest.csv"          # existing manifest to update
BASE_DIR = ""                              # <-- SET THIS: root folder containing "data/"
OUTPUT_PATH = "manifest_updated.csv"    # written separately; won't clobber original
FILENAME_PREFIX = "Manual"
AUDIO_EXTENSIONS = (".wav", ".m4a", ".mp3", ".flac")


def get_audio_info(full_path):
    """Read duration and sample rate directly from the audio file."""
    try:
        info = sf.info(full_path)
        return info.duration, info.samplerate
    except Exception as e:
        print(f"  WARNING: could not read audio info for {full_path}: {e}")
        return None, None


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    existing_filepaths = set(manifest["filepath"])

    # Derive category -> label mapping from the manifest itself, so we
    # never have to guess or hardcode it.
    category_to_label = (
        manifest.groupby("category")["label"].unique().apply(lambda x: x[0]).to_dict()
    )
    categories = list(category_to_label.keys())

    new_rows = []
    for category in categories:
        folder = os.path.join(BASE_DIR, "data", category)
        if not os.path.isdir(folder):
            print(f"Skipping missing folder: {folder}")
            continue

        for full_path in sorted(glob.glob(os.path.join(folder, f"{FILENAME_PREFIX}*"))):
            filename = os.path.basename(full_path)
            if not filename.lower().endswith(AUDIO_EXTENSIONS):
                continue

            rel_path = os.path.relpath(full_path, BASE_DIR).replace("\\", "/")
            if rel_path in existing_filepaths:
                continue  # already in manifest, skip

            duration_sec, sample_rate = get_audio_info(full_path)
            recording_id = os.path.splitext(filename)[0]

            new_rows.append({
                "filepath": rel_path,
                "label": category_to_label[category],
                "category": category,
                "source": "manual_recording",
                "source_id": None,
                "speaker_id": None,
                "recording_id": recording_id,
                "duration_sec": duration_sec,
                "sample_rate": sample_rate,
                "voice": None,
                "speed": None,
                "pitch": None,
                "transcription": None,
                "edit_distance": None,
                "parent_filepath": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

    if not new_rows:
        print("No new 'Manual*' recordings found that aren't already in the manifest.")
        return

    new_df = pd.DataFrame(new_rows)
    updated = pd.concat([manifest, new_df], ignore_index=True)
    updated.to_csv(OUTPUT_PATH, index=False)

    print(f"Added {len(new_rows)} new recording(s):")
    for row in new_rows:
        print(f"  {row['filepath']}  (label={row['label']}, "
              f"duration={row['duration_sec']}, sr={row['sample_rate']})")
    print(f"\nUpdated manifest written to: {OUTPUT_PATH}")
    print("(original manifest left untouched — review before replacing it)")


if __name__ == "__main__":
    main()
