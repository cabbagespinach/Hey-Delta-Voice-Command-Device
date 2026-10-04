# References

Papers, datasets and software this project uses. Licences and per-dataset clip counts:
[`external_raw/README.md`](../external_raw/README.md) and `data/command_dataset/dataset_inventory.csv`. DOIs of arXiv
papers are the standard arXiv DOIs (10.48550/arXiv.&lt;id&gt;). Entries marked *(verify)* should be checked against
the publisher before formal citation.

## This project

- Redondo, A. J. R. (2026). *Hey Delta: wakeword + voice-command classifier for a Raspberry Pi 5* (AI231 ME2).
  Code: https://github.com/cabbagespinach/Hey-Delta-Voice-Command-Device (MIT; model weights CC BY-NC 4.0).
- Audio used for training and evaluation (own recordings, synthetic voices, prepared public clips): Zenodo,
  restricted access, doi:[10.5281/zenodo.23093493](https://doi.org/10.5281/zenodo.23093493).

## Models and training methods

- **BC-ResNet** (wakeword and command models): Kim, B., Chang, S., Lee, J. & Sung, D. (2021). Broadcasted Residual
  Learning for Efficient Keyword Spotting. *Interspeech 2021*. arXiv:2106.04140, doi:10.48550/arXiv.2106.04140.
  Reference code: https://github.com/Qualcomm-AI-research/bcresnet
- **DS-CNN** (command baseline): Zhang, Y., Suda, N., Lai, L. & Chandra, V. (2017). Hello Edge: Keyword Spotting on
  Microcontrollers. arXiv:1711.07128, doi:10.48550/arXiv.1711.07128.
- **SpecAugment**: Park, D. S. et al. (2019). SpecAugment: A Simple Data Augmentation Method for Automatic Speech
  Recognition. *Interspeech 2019*. arXiv:1904.08779, doi:10.48550/arXiv.1904.08779.
- **AdamW**: Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*.
  arXiv:1711.05101, doi:10.48550/arXiv.1711.05101.
- **Cosine learning-rate schedule**: Loshchilov, I. & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm
  Restarts. *ICLR 2017*. arXiv:1608.03983, doi:10.48550/arXiv.1608.03983.
- **Label smoothing**: Szegedy, C. et al. (2016). Rethinking the Inception Architecture for Computer Vision. *CVPR
  2016*. arXiv:1512.00567, doi:10.48550/arXiv.1512.00567.
- **Whisper** (checking synthetic clips say the right words; also used by the class dataset's labelling): Radford, A.
  et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. *ICML 2023*. arXiv:2212.04356,
  doi:10.48550/arXiv.2212.04356. Run with faster-whisper (https://github.com/SYSTRAN/faster-whisper), large-v3.

## Datasets: wakeword

- **MIT reverberation survey** (room impulse responses): Traer, J. & McDermott, J. H. (2016). Statistics of natural
  reverberation enable perceptual separation of sound and space. *PNAS* 113(48). doi:10.1073/pnas.1612524113
  *(verify)*. CC BY 4.0.
- **MUSAN** (music, speech, noise): Snyder, D., Chen, G. & Povey, D. (2015). MUSAN: A Music, Speech, and Noise
  Corpus. arXiv:1510.08484, doi:10.48550/arXiv.1510.08484. CC BY 4.0 / US public domain.
- **Multilingual Spoken Words Corpus** (near-miss words): Mazumder, M. et al. (2021). Multilingual Spoken Words
  Corpus. *NeurIPS Datasets and Benchmarks*. No DOI found. CC BY 4.0.
- **FLEURS** (Filipino / English speech): Conneau, A. et al. (2022). FLEURS: Few-shot Learning Evaluation of Universal
  Representations of Speech. arXiv:2205.12446, doi:10.48550/arXiv.2205.12446. CC BY 4.0.
- **Hey Snips** (training negatives; obtain from Sonos): Coucke, A. et al. (2019). Efficient keyword spotting using
  dilated convolutions and gating. *ICASSP 2019*. doi:10.1109/ICASSP.2019.8683474 *(verify)*. Academic use only.
- **Qualcomm Keyword Speech Dataset** (evaluation only; obtain from Qualcomm): Qualcomm Technologies (2019).
  Internal research, non-commercial use only.

## Datasets: commands

- **Class dataset** (AI231 ME2 voice commands, schema "Option B"): Hugging Face
  [`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands),
  doi:[10.57967/hf/10723](https://doi.org/10.57967/hf/10723). Public; licence "per-source research-only" (research and
  education use; each source keeps its own licence, see the dataset's `LICENSE.md`). Revision da92a79 for the class
  benchmark; class benchmark tool: https://github.com/airimonda/vcm-benchmark.
- **Fluent Speech Commands**: Lugosch, L. et al. (2019). Speech Model Pre-training for End-to-End Spoken Language
  Understanding. *Interspeech 2019*. arXiv:1904.03670, doi:10.48550/arXiv.1904.03670. FSC licence: no sharing of the
  audio.
- **SLURP**: Bastianelli, E. et al. (2020). SLURP: A Spoken Language Understanding Resource Package. *EMNLP 2020*.
  arXiv:2011.13205, doi:10.48550/arXiv.2011.13205. CC BY-NC 4.0.
- **Timers and Such**: Lugosch, L. et al. (2021). Timers and Such: A Practical Benchmark for Spoken Language
  Understanding with Numbers. *NeurIPS Datasets and Benchmarks*. arXiv:2104.01604; dataset
  doi:10.5281/zenodo.4623772. CC0 1.0.
- **Google Speech Commands v0.02**: Warden, P. (2018). Speech Commands: A Dataset for Limited-Vocabulary Speech
  Recognition. arXiv:1804.03209, doi:10.48550/arXiv.1804.03209. CC BY 4.0.
- **Snips SLU**: Saade, A. et al. (2018). Spoken Language Understanding on the Edge. arXiv:1810.12735,
  doi:10.48550/arXiv.1810.12735.
- **SynTTS-Commands**: arXiv:2511.07821, doi:10.48550/arXiv.2511.07821
  (https://github.com/lugan113/SynTTS-Commands-Official). MIT.
- **Multi-Sensor Voice Command Dataset** (KU Leuven): Rusci, M., Van hamme, H. & Tuytelaars, T. KU Leuven RDR,
  doi:10.48804/IEKKVZ. CC BY 4.0.
- **Common Voice** (inside MSWC): Ardila, R. et al. (2020). Common Voice: A Massively-Multilingual Speech Corpus.
  arXiv:1912.06670, doi:10.48550/arXiv.1912.06670. CC0.
- **LibriSpeech** (reference voices in a classmate's voice-cloned set): Panayotov, V. et al. (2015). LibriSpeech: an
  ASR corpus based on public domain audio books. *ICASSP 2015*. CC BY 4.0.
- Classmates' own recordings: credited by neutral labels (Classmate A–F) in `external_raw/README.md`.

## Software and voices

- **Piper TTS** (synthetic training voices and the assistant's spoken replies): https://github.com/rhasspy/piper;
  voices from https://huggingface.co/rhasspy/piper-voices (per-voice licences in each model card; the assistant uses
  "Amy", en_US medium).
- **edge-tts** (Philippine-accent neural voices): https://github.com/rany2/edge-tts (Microsoft neural voices).
- **Chatterbox** (voice cloning and voice conversion; Resemble AI, MIT): https://github.com/resemble-ai/chatterbox
- **PyTorch / torchaudio** (training): Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance Deep
  Learning Library. *NeurIPS 2019*. arXiv:1912.01703.
- **ONNX Runtime** (inference on the Pi): https://onnxruntime.ai
- **librosa** (audio loading in the evaluation and speaker-check tools): McFee, B. et al. (2015). librosa: Audio and Music
  Signal Analysis in Python. *Proc. SciPy 2015*.
- **Assistant integrations**: Open-Meteo weather API (https://open-meteo.com, CC BY 4.0 data); python-kasa for the
  TP-Link Tapo bulb (https://github.com/python-kasa/python-kasa); BlueZ / obexd (Bluetooth MAP texts) and oFono
  (Bluetooth HFP calls) on Raspberry Pi OS.
