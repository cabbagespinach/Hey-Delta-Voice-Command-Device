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

## Raspberry Pi 5 latency

Deployment target: Raspberry Pi 5 (4 GB), ONNX Runtime on the CPU, **one thread** per model. Each ONNX file holds
the front end and the network, so the times are audio in -> probabilities out.

**Measured live on the Pi during the class benchmark** (Raspberry Pi 5 Model B, 4 GB, Debian 13, no throttling; command
model `bcresnet6_hf_plus`, wakeword listener running all the time; `data/deliverables/evaluation/benchmark_comparison.md`, raw samples in each run's `pi_metrics.csv`):

| | Oct 2 run (hf_plus only) | Oct 3 run (several models answered each capture; hf_plus column) |
|---|---|---|
| command inference per 5 s capture (mean / p95 / max) | 66.5 / 71.9 / 75.0 ms | 57.7 / 72.1 / 95.3 ms |
| real-time factor (inference / audio) | 0.013 / 0.014 / 0.015 | 0.012 / 0.014 / 0.019 |
| CPU use of the whole Pi, listener + model (mean / p95 / max) | 5.2 / 7.8 / 13.4 % | 10.8 / 26.9 / 35.9 % (several models per capture) |
| CPU temperature (mean / p95 / max) | 54.5 / 55.6 / 57.3 C | 54.6 / 56.2 / 57.9 C |
| response time, end of command -> Pi answer (p50 / p95) | 1.42 / 4.93 s | 1.45 / 3.25 s |
| assistant process CPU, share of wall time (wakeword listening all the time + command model) | 12.7 % | 12.3 % |
| assistant process RAM (mean / p95 / max) | 108.7 / 109.8 / 110.7 MB | 144.8 / 145.0 / 145.8 MB (several models loaded) |
| wakeword detection (trials with "Hey Delta" in which it woke up) | 98.5 % | 100 % (202 / 202) |
| false wakes (trials without the wake word) | 0 / 16 | 0 / 16 |

The response time is mostly the listener waiting for the end of speech, not the model. The process CPU is mostly the
wakeword listener (one 1.5 s window every 100 ms) plus audio capture; the wakeword model's own time per window on the
Pi is measured separately by `pi_latency.py` (below; on one server core: 4.8 ms per window).

Scripts (run on the Pi, from a clone of this repository):

- `python3 data/deliverables/evaluation/pi_latency.py`: times every deployed model on its own (wakeword per 1.5 s
  window, which runs every 100 ms, and each command model per 5 s capture) and writes
  `data/deliverables/evaluation/pi_latency/<device>_<date>.{json,md}`. Needs only `numpy` and `onnxruntime`.
- `data/deliverables/command_classifier/model/make_bench_kit.py` -> `vcm_bench_assistant.py`: the live pipeline used
  for the class benchmark (README in `bench_kit_src/README_BENCH.md`).

