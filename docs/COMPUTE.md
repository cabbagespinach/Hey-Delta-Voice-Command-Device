# Compute: hardware, wall-clock times, seeds

All training and evaluation ran on the course HPC server; the deployment target is a Raspberry Pi 5 (4 GB).

| | |
|---|---|
| Cluster node | `ai-n002.hpc.coe.upd.edu.ph` (UP College of Engineering HPC) |
| GPUs on the node | 8 × NVIDIA A100-SXM4-40GB (shared with other users) |
| GPUs per run | **1** (pinned with `CUDA_VISIBLE_DEVICES`; at most 2 jobs at once, owner's rule for the shared server) |
| CPU | 2 × AMD EPYC 7742 (256 threads); 12 data-loader workers per training run |
| Software | Python 3.12.3, PyTorch 2.14.0 (CUDA 13.0, cuDNN 9.24), torchaudio 2.11.0, ONNX 1.23.0, ONNX Runtime 1.30.0, NVIDIA driver 580.159.03; full list `repro/requirements-frozen.txt` |

## Wall-clock times (one A100 each, from the run logs)

| Run | What | Wall-clock |
|---|---|---|
| `model/runs/bcresnet6` | wakeword BC-ResNet-6, 60 epochs (2026-09-29) | 26 min training (`train.out`: 1,583 s) |
| `command_classifier/model/runs/bcresnet6_schema_b` | command BC-ResNet-6, 60 epochs × 48,000 draws | 70 min (2026-10-01 23:31 → 10-02 00:41) |
| `command_classifier/model/runs/dscnn_schema_b` | DS-CNN baseline, same recipe | 70 min (00:41 → 01:51) |
| `command_classifier/model/runs/bcresnet6_hf_only` | class-benchmark model, HF data only | 64 min (07:47 → 08:51) |
| `command_classifier/model/runs/bcresnet6_hf_plus` | class-benchmark model, HF + our data | 82 min (08:51 → 10:13; the GPU was shared with smoke tests) |
| evaluation + ONNX export | per command model | 1–3 min |
| `reproduce.sh` (everything) | 5 models + evaluation | about 6 h |

## Seeds

| Stage | Seed | Where |
|---|---|---|
| Wakeword split (source-level, frozen) | 20260925 | `data/deliverables/dataset_split/split_config.json` |
| Wakeword data loading / augmentation | 20260928 | `data/deliverables/dataloading/dataloader_config.json` |
| Wakeword training | 20260929 | `data/deliverables/model/train_config.json` |
| External negatives (MSWC, FLEURS, MUSAN, Hey Snips) | 20260929 | `scripts/integrate_external_datasets.py` |
| Command training and data loading | 20260930 | `command_classifier/model/train_config.json`, `dataloading/dataloader_config*.json` |
| Command data generation (synthetic voices, fragments, reused unknowns, public datasets) | 20260930 | `command_classifier/generate_synthetic_commands.py`, `build_fragments.py`, `build_unknown_reuse.py`, `prepare_real_commands.py`, `convert_owner_voices.py`, `clone_fleurs_voices.py` |
| Other wordings, classmates' datasets | 20261001 | `generate_phrasing_variants.py`, `label_classmate_datasets.py` |

Every epoch's draws are seeded from (seed, epoch, draw index), so a run is the same for any number of loader workers.
GPU arithmetic is not bit-exact between runs, so a reproduction matches the reported numbers closely, not digit for digit.
