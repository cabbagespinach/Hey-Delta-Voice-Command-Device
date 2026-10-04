# v1.0 – Hey Delta wakeword + command classifier

First public release of the AI231 ME2 project: an offline voice-command device for a Raspberry Pi 5. A "Hey Delta"
wakeword model wakes the device; a command classifier then maps the spoken command to one of the class benchmark's
31 commands (schema "Option B") or `unknown`.

## Model weights (licence: CC BY-NC 4.0, see ../MODEL_LICENSE.md)

| Model | Checkpoint (PyTorch) | ONNX (for the Pi) | Parameters |
|---|---|---|---|
| Wakeword BC-ResNet-6 (chosen model; threshold 0.7335, 2-of-3 rule) | `data/deliverables/model/runs/bcresnet6/best.pt` | `data/deliverables/model/export/final_bcresnet6/heydelta_fp32.onnx` | 186k |
| Command classifier BC-ResNet-6, schema B | `data/deliverables/command_classifier/model/runs/bcresnet6_schema_b/best.pt` | `data/deliverables/command_classifier/model/export/bcresnet6_schema_b/command_bcresnet6_schema_b.onnx` | 192k |
| Baseline of comparable size: DS-CNN, schema B | `.../command_classifier/model/runs/dscnn_schema_b/best.pt` | `.../model/export/dscnn_schema_b/command_dscnn_schema_b.onnx` | 195k |
| Class benchmark: HF data only | `.../command_classifier/model/runs/bcresnet6_hf_only/best.pt` | `.../model/export/bcresnet6_hf_only/command_bcresnet6_hf_only.onnx` | 192k |
| Class benchmark: HF + our data | `.../command_classifier/model/runs/bcresnet6_hf_plus/best.pt` | `.../model/export/bcresnet6_hf_plus/command_bcresnet6_hf_plus.onnx` | 192k |

Each ONNX file contains the front end (log-mel + normalisation) and the network: input = raw 16 kHz audio,
output = probabilities. Thresholds / cutoffs are in the `command_config.json` next to each command ONNX.

## Results (held-out test sets)

- **Wakeword:** real Raspberry Pi "Hey Delta" 33/33 detected; unseen synthetic voices 86%; worst negative category
  0.16% false positives (`data/deliverables/model/runs/bcresnet6/test/test_report.md`).
- **Command classifier (schema B):** 98–100% correct on the owner's test session and an unseen speaker, 98% on the
  classmates' test voices (best guess); DS-CNN of the same size 62–94% (`.../evaluation/results_schema_b/`).
- **Class benchmark (HF test split):** adding our data to the class Hugging Face set cuts wrong commands from 7.7%
  to 1.2% on the classmates' voices and raises public-recording accuracy from 53% to 64%
  (`.../evaluation/results_hf_compare.md`).
- **Ablations** (what each change did, one page): [ABLATIONS.md](ABLATIONS.md). **References:** [REFERENCES.md](REFERENCES.md).

## Reproduce

`git clone` → `bash reproduce.sh` (one command) on the course HPC server (docs/REPRODUCE.md;
hardware, wall-clock times and seeds in docs/COMPUTE.md). Audio: Zenodo, restricted access,
doi:10.5281/zenodo.23093493, plus a read-only folder on the server.

## Licences

Code: MIT (Copyright (c) 2026 Arvir Jane R. Redondo). Model weights: CC BY-NC 4.0. Datasets: their own licences
(external_raw/README.md).
