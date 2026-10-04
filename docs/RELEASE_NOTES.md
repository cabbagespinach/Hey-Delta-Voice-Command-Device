# v1.1 – Hey Delta: wakeword, commands, assistant, one-line Pi kits

Offline "Hey Delta" wakeword + voice-command classifier for the Raspberry Pi 5 (AI231 ME2, Arvir Jane R. Redondo),
now with a voice assistant that carries the commands out, one-line Pi installers, one-command training reproduction
and the public audio on Zenodo.

## Checklist

| # | Item | Where |
|---|---|---|
| 1 | Dataset location, DOI and access terms | **Class data:** [Hugging Face airimonda/ai231-me2-voice-commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands), [doi:10.57967/hf/10723](https://doi.org/10.57967/hf/10723). **My data:** public part [doi:10.5281/zenodo.23132588](https://doi.org/10.5281/zenodo.23132588), full set with real voices [doi:10.5281/zenodo.23093493](https://doi.org/10.5281/zenodo.23093493). Access terms: [REPRODUCE.md, "Where the data comes from"](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/docs/REPRODUCE.md#where-the-data-comes-from) |
| 2 | A100 cluster: node ID, number of GPUs, wall-clock times, seeds | [COMPUTE.md: node and GPUs](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/docs/COMPUTE.md), [wall-clock times](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/docs/COMPUTE.md#wall-clock-times-one-a100-each-from-the-run-logs), [seeds](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/docs/COMPUTE.md#seeds) |
| 3 | Model weights: release URL and licence | [this release](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/releases/tag/v1.1) (`AI231ME2RedondoModelWeights.zip`), licence CC BY-NC 4.0: [MODEL_LICENSE.md](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/MODEL_LICENSE.md) |
| 4 | Public repository, one-command reproduction | [repository](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device) (code licence MIT: [LICENSE](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/LICENSE)), [REPRODUCE.md](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/docs/REPRODUCE.md), [reproduce.sh](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/reproduce.sh). <br><br>  One-command deployment on the Pi, model only and model + assistant: see [Downloads](#downloads) below |
| 5 | Dataset licensed and citable (DOI) | [doi:10.5281/zenodo.23132588](https://doi.org/10.5281/zenodo.23132588) (CC BY-NC 4.0), [doi:10.5281/zenodo.23093493](https://doi.org/10.5281/zenodo.23093493) (restricted, custom terms), [doi:10.57967/hf/10723](https://doi.org/10.57967/hf/10723) (class data); every external dataset with licence and citation: [external_raw/README.md](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/external_raw/README.md), [REFERENCES.md](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/docs/REFERENCES.md) |
| 6 | Training logs and final checkpoints committed | wakeword: [runs/bcresnet6/](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/tree/v1.1/data/deliverables/model/runs/bcresnet6) (`train.out`, `train_log.json`, `best.pt`); commands: [runs/](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/tree/v1.1/data/deliverables/command_classifier/model/runs) (`bcresnet6_schema_b`, `dscnn_schema_b`, `bcresnet6_hf_plus`, `bcresnet6_hf_only`: `history.json`, `run_info.json`, `best.pt`, `last.pt`, plus each run's `.log`) |
| 7 | Pi 5 latency, reproducible with the posted script | script: [pi_latency.py](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/data/deliverables/evaluation/pi_latency.py); measured on the Pi with it: [report (2026-10-04)](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/main/data/deliverables/evaluation/pi_latency/Raspberry-Pi-5-Model-B-Rev-1-1_20261004-210212.md); all Pi numbers: [COMPUTE.md, "Raspberry Pi 5 latency"](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/docs/COMPUTE.md#raspberry-pi-5-latency) |
| 8 | Held-out test set, unseen speakers | wakeword: [test report](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/data/deliverables/model/runs/bcresnet6/test/test_report.md), [split config](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/data/deliverables/dataset_split/split_config.json); commands: [test summary](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/data/deliverables/command_classifier/evaluation/results_schema_b/test_summary.md), [unseen speakers](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/data/deliverables/command_classifier/evaluation/results_schema_b/unseen_speakers.md) |
| 9 | Baseline of comparable size compared | DS-CNN (195k parameters) vs BC-ResNet-6 (192k): [ABLATIONS.md, C1](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/docs/ABLATIONS.md#c1-bc-resnet-6-vs-ds-cnn-same-size-same-data), [test summary](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/data/deliverables/command_classifier/evaluation/results_schema_b/test_summary.md) |

## Downloads

| File | What | On the Pi |
|---|---|---|
| [`AI231ME2RedondoModelWeights.zip`](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/releases/download/v1.1/AI231ME2RedondoModelWeights.zip) | all trained weights: wakeword + 4 command classifiers, PyTorch (`best.pt`) and ONNX, configs, checksums | — |
| [`AI231ME2RedondoModelDeploy.zip`](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/releases/download/v1.1/AI231ME2RedondoModelDeploy.zip) | model deployment: "Hey Delta" → command → the recognised command printed | `unzip -o AI231ME2RedondoModelDeploy.zip && bash AI231ME2RedondoModelDeploy/install.sh` |
| [`AI231ME2RedondoAssistantDeploy.zip`](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/releases/download/v1.1/AI231ME2RedondoAssistantDeploy.zip) | model + assistant: the command is carried out (Tapo bulb, phone, weather, music, timers, alarms, reminders, simulated thermostat), spoken reply, browser dashboard | `unzip -o AI231ME2RedondoAssistantDeploy.zip && bash AI231ME2RedondoAssistantDeploy/install.sh` |

Both kits can also be built from a clone with one command: [docs/DEPLOY.md](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/docs/DEPLOY.md).
Weights licence: CC BY-NC 4.0 ([MODEL_LICENSE.md](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/MODEL_LICENSE.md)).

## Training reproduction (one command)

On the course HPC server, under your own account, with one GPU:

```bash
git clone https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device.git AI231ME2RedondoTrainRepro && cd AI231ME2RedondoTrainRepro
tmux new -s repro "bash reproduce.sh"
```

`reproduce.sh` links the shared audio by itself, retrains all 5 models with the same code, configuration and seeds,
evaluates them as reported, exports the ONNX models and writes `repro_outputs/REPRODUCTION_REPORT.md` (reported vs
reproduced numbers). About 6 hours; one part only: `wakeword` (~45 min), `commands` (~2.5 h) or `hf` (~2.5 h).
`SMOKE=1 bash reproduce.sh` checks the whole pipeline with tiny training (about 2.5 h). Details:
[docs/REPRODUCE.md](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/docs/REPRODUCE.md);
hardware, wall-clock times and seeds: [docs/COMPUTE.md](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/docs/COMPUTE.md).

## Datasets

| Data | Where | Access |
|---|---|---|
| Public part of our audio: every synthetic clip (Piper, Chatterbox) and every prepared clip of an openly licensed dataset | Zenodo, [doi:10.5281/zenodo.23132588](https://doi.org/10.5281/zenodo.23132588) | open, CC BY-NC 4.0 |
| Full set, including the real voice recordings | Zenodo, [doi:10.5281/zenodo.23093493](https://doi.org/10.5281/zenodo.23093493) | restricted: request access on the page (research and education only, no redistribution) |
| Class Hugging Face dataset (revision `da92a79`) | [airimonda/ai231-me2-voice-commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands), [doi:10.57967/hf/10723](https://doi.org/10.57967/hf/10723) | public, per-source research-only licence |
| External datasets (MUSAN, MSWC, FLEURS, MIT RIR, Google Speech Commands, SLURP, Timers and Such, SynTTS-Commands, ...) | sources, licences and citations: [external_raw/README.md](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/external_raw/README.md) | each under its own licence |
| On the course HPC server | read-only shared folder, linked by `setup_data.sh --server` ([docs/REPRODUCE.md](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/docs/REPRODUCE.md#where-the-data-comes-from)) | course accounts |

## Models

| Model | Parameters | Use |
|---|---|---|
| Wakeword BC-ResNet-6 (threshold 0.7335, fires on 2 of 3 windows) | 186k | deployed |
| Command classifier BC-ResNet-6, class Hugging Face data + our data (`hf_plus`, balanced rule) | 192k | deployed default |
| The same on the class data only (`hf_only`) | 192k | comparison |
| BC-ResNet-6 on our data only (schema B) | 192k | main experiment |
| DS-CNN baseline of comparable size (schema B) | 195k | baseline |

## Results

- **Wakeword (held-out test):** real Raspberry Pi "Hey Delta" 33/33 detected; unseen synthetic voices 86%; worst
  negative category 0.16% false positives ([test report](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/data/deliverables/model/runs/bcresnet6/test/test_report.md)).
- **Command classifier (schema B, held-out test):** 98–100% correct on the owner's test session and an unseen
  speaker, 98% on the classmates' test voices; DS-CNN of the same size 62–94%
  ([test summary](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/data/deliverables/command_classifier/evaluation/results_schema_b/test_summary.md),
  [unseen speakers](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/data/deliverables/command_classifier/evaluation/results_schema_b/unseen_speakers.md)).
- **Class benchmark (HF test split):** adding our data to the class set cuts wrong commands from 7.7% to 1.2% on the
  classmates' voices and raises public-recording accuracy from 53% to 64%
  ([comparison](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/data/deliverables/command_classifier/evaluation/results_hf_compare.md)).
- **Live on the Pi 5 (class benchmark, Oct 3, hf_plus balanced):** intent accuracy 84.7%, 14.0% of commands ignored; wakeword
  woke on 202/202 "Hey Delta" trials, 0/16 false wakes; command inference 57.7 ms mean per 5 s capture (real-time
  factor 0.012); response 1.45 s median; assistant process 12.3% CPU
  ([benchmark comparison](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/data/deliverables/evaluation/benchmark_comparison.md)).

What each change did: [docs/ABLATIONS.md](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/docs/ABLATIONS.md).
References: [docs/REFERENCES.md](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/docs/REFERENCES.md).

## New since v1.0

- Voice assistant ([data/deliverables/assistant/](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/tree/v1.1/data/deliverables/assistant)):
  real or simulated devices, offline Piper replies, live dashboard, start at boot.
- One-line install + run on the Pi for both deployment kits; deployed command models use the balanced rule.
- One-command training reproduction (it sets up the data itself).
- Audio on Zenodo: public part (open) and full set (restricted).
- Class benchmark runs on the Pi, ablation report, references, Pi latency tool, repository reorganised.

Code: MIT ([LICENSE](https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device/blob/v1.1/LICENSE)).
Model weights: CC BY-NC 4.0. Datasets: their own licences.
