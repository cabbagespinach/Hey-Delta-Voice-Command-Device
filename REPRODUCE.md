# Reproducing this project on the AI231 HPC server

Three commands, run under your own account on the same server as the project owner (a GPU is needed for training):

```bash
git clone <this repository> ai231-me2 && cd ai231-me2
bash setup_data.sh --server        # links the shared audio, checks the datasets that cannot be shared
tmux new -s repro "bash reproduce.sh"     # trains, evaluates and exports every model again (~6 h on one GPU)
```

Pick a free GPU with `GPU=<index> bash reproduce.sh`. One part only: `bash reproduce.sh wakeword` (~45 min),
`commands` (~2.5 h) or `hf` (~2.5 h). `SMOKE=1 bash reproduce.sh` runs everything at toy size in a few minutes to
check the setup.

The result is `repro_outputs/REPRODUCTION_REPORT.md`: the reported numbers next to the reproduced ones. Reproduced
runs are written under new names (`*_repro`); the reported runs, checkpoints and results in this repository are
never overwritten.

## What is reproduced

| Part | Model | Training data | Evaluated on | Reported results |
|---|---|---|---|---|
| `wakeword` | BC-ResNet-6 "Hey Delta" (round 1c) | `data/deliverables/segmentation_windowing/outputs/windows.csv` (train split) | frozen isolated validation + test sets, streaming sets | `data/deliverables/model/README.md`, `model/runs/bcresnet6/` |
| `commands` | schema-B command classifier: BC-ResNet-6 and the DS-CNN baseline of comparable size | `data/commands_schema_b_all.csv` | its test split, incl. unseen speakers | `command_classifier/evaluation/results_schema_b/` |
| `hf` | BC-ResNet-6 on the class Hugging Face dataset only vs HF + our data | `data/commands_hf_only.csv`, `data/commands_hf_plus.csv` | the class benchmark (HF test split) | `command_classifier/evaluation/results_hf_compare.md` |

Each part retrains from the prepared data with the same code, configuration and seeds, recomputes the normalisation
statistics (and checks them against the reported file), evaluates exactly as reported, and exports the ONNX model
used on the Raspberry Pi 5. Pi latency: `data/deliverables/model/pi_kit/heydelta_pi.py bench` (wakeword) on the Pi.

Exact equality of every digit is not expected (GPU arithmetic and data-loading order differ slightly between
runs); with all datasets present the numbers agree within a few points.

## Where the data comes from

| Data | How you get it |
|---|---|
| The owner's recordings, synthetic voices, prepared clips of the public datasets, the class Hugging Face set | **Shared folder** `/home/arvir.jane.redondo/AI231_ME2_reproduce/` (read-only), linked by `setup_data.sh --server` |
| The same audio, for citation and for use outside this server | **Zenodo, restricted access: doi:[10.5281/zenodo.23093493](https://doi.org/10.5281/zenodo.23093493)**. Request access on the record page (academic use; the owner approves). Extract each tar file in the repository root. Classmate A's and Classmate F's sets and the class HF set are not in it (cited by their own links; the HF set is fetched by `hf_extract.py`). |
| Python environment | The owner's environment is used read-only by default (exact versions). Your own: `pip install -r repro/requirements-frozen.txt`, then `PY=/path/to/python bash setup_data.sh --server` |
| Code, configurations, clip lists, splits, logs, checkpoints, results | This repository |

### Datasets that cannot be shared

Their licences do not allow redistribution, so they are not in the shared folder. **The reproduction runs without
them**: `setup_data.sh` prints how to obtain each missing one, `reproduce.sh` leaves their clips out and says so
(`repro_outputs/missing_data.md`, and "reduced data" in the report). If you obtain one, put it where `setup_data.sh`
says and run `setup_data.sh --server` again: it rebuilds exactly the clips we used, so the reproduction becomes
exact.

| Dataset | Used for | Without it |
|---|---|---|
| Sonos "Hey Snips" (research only) | wakeword training negatives (3,000 clips) | wakeword trained without them; its validation/test sets are re-frozen without them |
| Qualcomm Keyword Speech Dataset (research only, no incorporation) | wakeword evaluation only: false wake-ups on other wake words | that check is skipped and marked as skipped |
| Fluent Speech Commands (academic, no sharing of audio) | command classifier clips (the 455 FSC clips inside the class HF set are used either way) | our FSC clips are left out |
| Classmates' own recordings (consent) | 158 / 59 command training clips | left out |

## Files

| File | Role |
|---|---|
| `setup_data.sh` | link the shared data, rebuild optional datasets you obtained, report missing ones |
| `reproduce.sh` | the whole reproduction (or one part), then the comparison report |
| `repro/` | the helpers: `link_data.py`, `optional_data.py` (check / restore / filter), `compare_results.py`, `make_shared_folder.py` (owner side), `requirements-frozen.txt` |
| `data/deliverables/command_classifier/reproduce_schema_b.sh`, `run_hf_compare.sh` | the command-classifier pipelines (also runnable on their own) |
| `external_raw/README.md` | every dataset with licence, citation and DOI |

Rebuilding the datasets themselves from raw downloads (TTS generation, Whisper segmentation, voice conversion) is
not part of `reproduce.sh`: it takes many GPU hours and the TTS/voice-conversion steps are not bit-exact. Their
scripts are in the repository (`generate_tts_voices_v2.py`, `data/deliverables/segmentation_windowing/`,
`data/deliverables/command_classifier/generate_*.py`, `label_classmate_datasets.py`, `hf_extract.py`).
