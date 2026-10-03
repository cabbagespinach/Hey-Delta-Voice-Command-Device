# Hey Delta: wakeword + voice-command classifier for a Raspberry Pi 5

AI231 ME2 coursework (2026). An offline voice-command device: a **"Hey Delta" wakeword** model listens continuously;
after it fires, a **command classifier** maps the next few seconds of audio to one of the class benchmark's 31
commands (schema "Option B", e.g. LIGHT_ON, TIMER_10S, COLOR_RED) or `unknown`. No speech recogniser runs on the
device. Both models are BC-ResNet networks running in ONNX Runtime on a Raspberry Pi 5 (4 GB).

| | Model | Result (held-out test) | Details |
|---|---|---|---|
| Wakeword | BC-ResNet-6 (186k parameters), threshold 0.7335, fires on 2 of 3 windows | real Pi positives 33/33, unseen synthetic voices 86%, worst negative category 0.16% false positives | `data/deliverables/model/README.md` |
| Commands | BC-ResNet-6 (192k parameters), schema B, 31 commands + `unknown` | your own / unseen speakers / public / classmates' data: 98–100% (argmax) | `data/deliverables/command_classifier/evaluation/results_schema_b/` |
| Baseline of comparable size | DS-CNN (195k parameters), same data and recipe | 62–94% (argmax): BC-ResNet better everywhere | same folder |
| Class benchmark | BC-ResNet-6 on the class Hugging Face set alone vs with our data | `data/deliverables/command_classifier/evaluation/results_hf_compare.md` | |
| Class benchmark on the Pi (airimonda/vcm-benchmark) | BC-ResNet-6, HF + our data (`bcresnet6_hf_plus`), cautious cutoff, live on the Raspberry Pi 5 | 202 holdout trials: intent accuracy 69.8%, false accept 6.2%, false wake 0%, inference 66.5 ms | `data/deliverables/evaluation/20261002-135144/report.md` |
| Class benchmark on the Pi, side by side (2026-10-02 / 03) | `bcresnet6_hf_plus` vs `bcresnet6_hf_only`, cautious and balanced cutoffs, same captures on 2026-10-03, + the 2026-10-02 run | intent accuracy (Oct 3): hf_plus balanced 84.7% / cautious 75.2%, hf_only balanced 50.5% / cautious 35.6%; false wake 0% | `data/deliverables/evaluation/benchmark_comparison.md` |

## Reproduce

See **[REPRODUCE.md](REPRODUCE.md)**: `git clone` → `bash setup_data.sh --server` → `bash reproduce.sh`, on the
course HPC server (one A100). Compute details, wall-clock times and seeds: **[COMPUTE.md](COMPUTE.md)**.

## Data

All datasets, with licence, citation and DOI: `external_raw/README.md`; per-dataset clip counts and splits:
`data/command_dataset/dataset_inventory.csv`. Audio is not in this repository: our own recordings and prepared clips
are on Zenodo (restricted access, DOI in REPRODUCE.md) and in a read-only folder on the HPC server; datasets whose
licences forbid redistribution (Sonos "Hey Snips", Qualcomm Keyword Speech, Fluent Speech Commands) must be obtained
from their owners (instructions printed by `setup_data.sh`). Classmates are named by neutral labels (Classmate A–F).

## Licences

- **Code:** MIT (`LICENSE`), Copyright (c) 2026 Arvir Jane R. Redondo.
- **Trained model weights** (checkpoints, ONNX): CC BY-NC 4.0 (`MODEL_LICENSE.md`): non-commercial, because some
  training data allows only non-commercial / research use.
- **Datasets:** each keeps its own licence (`external_raw/README.md`).

## Layout

| Path | What |
|---|---|
| `data/deliverables/` | wakeword pipeline: `dataset_split/`, `segmentation_windowing/`, `preprocessing/`, `dataloading/`, `deployment_driven_augmentation_strategy/`, `model/` (training, ONNX export, Pi kit, deployment module), `evaluation/` |
| `data/deliverables/command_classifier/` | command classifier: data building, `preprocessing/`, `dataloading/`, `model/`, `evaluation/`, pipelines `reproduce_schema_b.sh`, `run_hf_compare.sh` |
| `data/*.csv` | clip lists (training / validation / test membership of every clip) |
| `repro/`, `setup_data.sh`, `reproduce.sh` | reproduction tools |
