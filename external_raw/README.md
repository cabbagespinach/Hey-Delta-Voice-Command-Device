# External datasets (downloaded 2026-09-29)

Raw downloads from public corpora, kept unmodified for provenance. The clips actually used are converted
and listed in `manifest.csv` by `../integrate_external_datasets.py` (sources `ext_*`, all **negative**;
no public corpus contains "Hey Delta"). RIRs are used through `data/rir/rir_bank.csv`
(`data/deliverables/deployment_driven_augmentation_strategy/build_rir_bank.py`).

| Folder | Dataset | Licence | Source | Used for |
|---|---|---|---|---|
| `mit_ir/` | MIT Acoustical Reverberation Scene Statistics Survey, 16 kHz copy (270 IRs) | CC-BY 4.0 | https://mcdermottlab.mit.edu/Reverb/IR_Survey.html via huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses | `data/rir/` bank (132 IRs with RT60 0.15–0.6 s) for `rir_reverberation` |
| `musan/` | MUSAN: music, speech and noise corpus (OpenSLR 17) | CC-BY 4.0 and US public domain; per-file attribution in each subfolder's `LICENSE` | https://www.openslr.org/17/ | `ext_musan_speech`, `ext_musan_music`, `ext_musan_noise` |
| `mswc_en/` | Multilingual Spoken Words Corpus, English: only the 7,323 clips in `selection.csv` (50 near-miss words, ≤ 300 per word, one per speaker), streamed from the shards by `stream_extract.py` | CC-BY 4.0 | https://mlcommons.org/datasets/multilingual-spoken-words/ via huggingface.co/datasets/MLCommons/ml_spoken_words | `ext_mswc` |
| `fleurs/` | FLEURS, `fil_ph` (all splits) and `en_us` (dev, test) | CC-BY 4.0 | huggingface.co/datasets/google/fleurs | `ext_fleurs` |
| `qualcomm/` | Qualcomm Keyword Speech Dataset ("Hey Android", "Hey Snapdragon", "Hi Galaxy", "Hi Lumina"; 4,270 clips, 50 speakers) | **Internal research only, non-commercial; clause 1(ii) forbids incorporating it into another data set without QTI's written authorization.** Copyright (c) 2019 Qualcomm Technologies, Inc. All rights reserved. Full terms: `qualcomm/qualcomm_keyword_speech_dataset/LICENSE.pdf` | https://www.qualcomm.com/developer/software/keyword-speech-dataset | **Not integrated** (pending the owner's decision on the licence) |
| `hey_snips/` | Sonos "Hey Snips" keyword spotting dataset v1 (Coucke et al. 2019), 96,396 clips, 6,240 speakers | **Academic/research use only, no commercial use; publication only of the unmodified dataset** (Sonos keyword-spotting-research-datasets terms) | Sonos request form; github.com/sonos/keyword-spotting-research-datasets | **Not integrated** (pending the owner's decision on the licence) |

Citations:
- Traer, J. & McDermott, J. H. (2016). Statistics of natural reverberation enable perceptual separation of sound and space. PNAS 113(48).
- Snyder, D., Chen, G. & Povey, D. (2015). MUSAN: A Music, Speech, and Noise Corpus. arXiv:1510.08484.
- Mazumder, M. et al. (2021). Multilingual Spoken Words Corpus. NeurIPS Datasets and Benchmarks.
- Conneau, A. et al. (2022). FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech. arXiv:2205.12446.
- Coucke, A. et al. (2019). Efficient keyword spotting using dilated convolutions and gating. ICASSP.

# Command datasets (added 2026-10-01, schema-B command classifier)

Full per-dataset inventory with clip counts and splits: `data/command_dataset/dataset_inventory.csv`
(built by `data/deliverables/command_classifier/make_dataset_inventory.py`). DOIs of arXiv papers are the standard
arXiv DOIs (10.48550/arXiv.<id>); entries marked "verify" should be checked against the publisher before citing.

| Folder | Dataset | Licence | Citation | DOI |
|---|---|---|---|---|
| `commands/fsc` | Fluent Speech Commands | FSC Public License: academic/non-commercial, **no sharing of the audio or derivatives** | Lugosch et al. (2019), Speech model pre-training for end-to-end SLU, Interspeech, arXiv:1904.03670 | 10.48550/arXiv.1904.03670 |
| `commands/slurp` | SLURP (real recordings) | CC BY-NC 4.0 | Bastianelli et al. (2020), SLURP: A Spoken Language Understanding Resource Package, EMNLP, arXiv:2011.13205 | 10.48550/arXiv.2011.13205 |
| `commands/timers_and_such` | Timers and Such v1.0 | CC0 1.0 | Lugosch et al. (2021), Timers and Such, NeurIPS Datasets & Benchmarks, arXiv:2104.01604 | 10.5281/zenodo.4623772 (dataset) |
| `google_speech_commands/` | Google Speech Commands v0.02 | CC BY 4.0 | Warden (2018), Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition, arXiv:1804.03209 | 10.48550/arXiv.1804.03209 |
| `snips_slu/` | Snips SLU v1.0 (HF mirror MWilinski/snips_slu_v1.0) | Snips research datasets terms (HF card: MIT) | Saade et al. (2018), Spoken Language Understanding on the Edge, arXiv:1810.12735 | 10.48550/arXiv.1810.12735 |
| `syntts_commands/` | SynTTS-Commands (English: Free-ST, VoxCeleb1&2 voices) | MIT | SynTTS-Commands, arXiv:2511.07821 (github lugan113/SynTTS-Commands-Official) | 10.48550/arXiv.2511.07821 |
| `class_vcm/` (Multi-Sensor part) | Multi-Sensor Voice Command Dataset (KU Leuven), via classmate Classmate F's VCM | CC BY 4.0 (GDPR: track downloads of derivatives) | Rusci, Van hamme, Tuytelaars, KU Leuven RDR | 10.48804/IEKKVZ |
| `class_vcm/` (temperature part) | classmate Classmate F's SET_TEMPERATURE recordings + Piper clips | none stated (ask the contributor) | classmate data | none |
| `class_optionb/` | classmate Classmate A's voice-cloned Option-B set (commit 5b23a95) | none stated (ask the contributor); reference voices LibriSpeech (CC BY 4.0) | github markandrian30/AI231 | none |
| (not downloaded) | Kaggle synthetic speech commands, Common Voice (Kaggle), MLEnd spoken numerals | see Kaggle pages; Common Voice CC0 | Common Voice: Ardila et al. (2020), arXiv:1912.06670 | 10.48550/arXiv.1912.06670 (Common Voice) |

Also cited above: MUSAN 10.48550/arXiv.1510.08484, FLEURS 10.48550/arXiv.2205.12446, MIT IR survey (PNAS 2016)
10.1073/pnas.1612524113 (verify), Hey Snips (ICASSP 2019) 10.1109/ICASSP.2019.8683474 (verify); MSWC (NeurIPS 2021
Datasets & Benchmarks): no DOI found. Own recordings and own synthetic data have no DOI (a Zenodo deposit would give one).
