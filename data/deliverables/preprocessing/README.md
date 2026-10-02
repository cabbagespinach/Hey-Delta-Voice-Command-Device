# Preprocessing: one deterministic front end for training, evaluation and streaming

```
raw waveform (16 / 22.05 / 24 / 48 kHz, mono or stereo, WAV or MP3-in-.wav)
  → mono float32, resampled to 16 kHz                    load_waveform / StreamResampler
  → training-only waveform augmentation (caller opt-in)   WindowDataset(augment=…)
  → fixed-duration framing: 1.5 s = 24,000 samples        frame_at / frame_clip / StreamingPreprocessor
  → torchaudio MelSpectrogram (power)                     FeatureExtractor
  → dB, fixed reference and floor                         FeatureExtractor
  → per-mel-bin normalization, frozen train statistics    Normalizer
  → model-ready features [1, 40, 147]                     WakewordPreprocessor
```

| File | Role |
|---|---|
| `wakeword_preprocessing.py` | The module: `PreprocessConfig`, loading/resampling, framing, `FeatureExtractor`, `Normalizer`, `WakewordPreprocessor`, `WindowDataset`, `StreamingPreprocessor`, `StreamResampler` |
| `preprocessing_config.json` | The central configuration (every value, with its reason in the `_*` keys) |
| `compute_normalization_stats.py` | Training-statistics generation (train split only) |
| `normalization_stats.json` | Frozen statistics artifact |
| `test_preprocessing.py` | Unit and sanity tests |

## Usage

```python
import wakeword_preprocessing as wp
pre = wp.WakewordPreprocessor.from_files()          # config + frozen stats; refuses mismatched stats

# Training (augmentation must be requested explicitly, and only the train split accepts it)
train = wp.WindowDataset(pre, "train", augment=my_augment, seed=0)   # my_augment(wave, sr, generator, meta) -> wave
train.set_epoch(epoch)                                                # new, reproducible augmentation draw per epoch
val   = wp.WindowDataset(pre, "validation")                           # deterministic
x, y, window_id = val[0]                                              # x: [1, 40, 147], y: 0./1.

# Isolated clip
feats = pre.process_clip("some.wav")                                  # [K, 1, 40, 147]

# Streaming (any input rate; the device records at 48 kHz)
stream = wp.StreamingPreprocessor(pre, input_sample_rate=48000)
starts, feats = stream.push(chunk)                                    # every window completed by this chunk
```

Regenerate the statistics whenever `windows.csv` or a feature setting changes:

```
python compute_normalization_stats.py      # writes normalization_stats.json
python test_preprocessing.py               # or: pytest test_preprocessing.py
```

## Policy

### Sample rate, duration, framing

| Item | Policy | Why |
|---|---|---|
| Sample rate | **16 kHz** mono for every source and at inference | Set by the augmentation strategy and segmentation. Sources are 16/22.05/24/48 kHz on disk; one rate removes bandwidth as a class cue. |
| Resampler | `torchaudio.functional.resample`, sinc/Hann, width 6, roll-off 0.99, pinned in the config | Deterministic on CPU; `StreamResampler` applies the same kernel chunk by chunk. |
| Duration | **1.5 s = 24,000 samples** | `segmentation_windowing` `window.duration_sec`. |
| Temporal placement (dataset) | Taken from `windows.csv` `start_sec`, rounded to the nearest 16 kHz sample (≤ 0.4 samples) | Segmentation already placed every window. Positives are end-anchored with the wakeword ending 0.10 / 0.25 / 0.40 s before the window end. Preprocessing never moves them. |
| Temporal placement (streaming) | Windows start at 0, 0.1 s, 0.2 s, … of the stream; the newest window ends at the newest audio | Every utterance ends up within 50 ms of a trained end-anchored placement. |
| Temporal placement (isolated clip) | Shorter than 1.5 s: centred. Longer: error by default, or `long_clip_policy="sliding"` (streaming schedule plus an end-aligned tail window) | Same centring rule segmentation uses for short clips. A long clip is never cut silently because a cut could drop part of a wakeword. |
| Padding | Samples outside the source are filled with **deterministic Gaussian noise** (`pad_mode: noise`) at −50.6 dBFS RMS, seeded by the window id, **identically in train, validation and test**. Streaming never pads. `pad_mode: zeros` remains available. | Exact digital silence never occurs on the device. The level reproduces the measured median RPI floor (−54.2 dB on the deployment metric). See **Padding** below. |
| Truncation | None. A window is always exactly 24,000 samples cut from where its start says; positive windows that would not fit were already rejected by segmentation | Positives are never truncated. |

### Features

| Item | Value | Why |
|---|---|---|
| STFT | `n_fft` 512, `win_length` 400 (25 ms), `hop_length` 160 (10 ms), Hann, `center=False` | Standard keyword-spotting front end. `center=False` means frames use only samples inside the window, so no reflection padding at the edges. 147 frames per window. |
| Mel | 40 HTK bands, `f_min` 20 Hz, `f_max` 7,600 Hz, power 2 | `f_max` stays below the resampler roll-off near 8 kHz, whose shape differs slightly by source rate. |
| dB | `10·log10(max(power, 1e-10))`, reference 1.0, **no `top_db`** | A fixed floor of −100 dB. `top_db` clamps relative to each window's own maximum, which would remove absolute level and make a window depend on its loudest frame. |
| Normalization | `(x − mean[bin]) / std[bin]`, statistics frozen from the **train split only** | See below. |

### Normalization and amplitude

- Statistics are per mel bin, over every frame of every unaugmented train window (split `train`, never `streaming_eval_holdout`, `validation` or `test`). `compute_normalization_stats.py` also checks each window's `source_group` against the frozen split (`dataset_split/split_assignments.csv` plus `segmentation_windowing/outputs/split_extension.csv`) and refuses any disagreement.
- The artifact records the feature-config hash, a hash of `windows.csv`, and a hash of the exact train window ids used. `Normalizer.from_stats` refuses statistics made with a different feature config or from any split other than `train`.
- **No per-utterance normalization.** It isn't needed and would hurt:
  - The deployment device captures the wakeword about 29 dB below synthetic speech, and only 5–26 dB above its background. That level is information the model should see.
  - The augmentation strategy already covers level variation explicitly (`device_level_gain`, `background_noise_mixing`).
  - Per-window statistics would make a streaming window's features depend on whatever else falls in that window.
- **Amplitude is preserved.** There's no peak or RMS normalization, and the dB reference is fixed. Normalization is one global affine map per bin, so an input 6 dB louder stays 6.02 dB louder in every bin (tested).

### Augmentation contract

- Off unless the caller passes `augment=` to `WindowDataset`. Only `splits="train"` accepts it; validation, test and streaming evaluation raise.
- The callable gets the window's real-audio excerpt at 16 kHz and a `torch.Generator` seeded from `(seed, epoch, window_id)`. It must return the same length.
- It runs **before** framing, as the required flow specifies. Padding added by framing is therefore never augmented. It gets the same deterministic noise fill in every split instead (see Padding).
- The transforms themselves (`device_level_gain`, `rir_reverberation`, `background_noise_mixing`, `device_noise_floor`, in the order the augmentation config specifies) are defined by `../deployment_driven_augmentation_strategy/augmentation_config.json`. They plug into this hook but are not implemented here.

### Streaming

- `StreamingPreprocessor.push(chunk)` accepts chunks of any size at any input rate. It returns every window completed so far, each processed by the same `WakewordPreprocessor` call as an isolated window.
- At 16 kHz input, streaming features are **bit-identical** to framing the same audio offline (tested).
- At 48 kHz, `StreamResampler` resamples in whole resampling periods with enough real context on both sides that its output matches resampling the whole file. The remaining difference comes from floating-point summation order (see tests for the measured value). Added latency: a few samples of resampler context plus the 100 ms hop.
- Nothing is emitted before 1.5 s of audio has arrived, so streaming never pads.

## Determinism

- Everything after loading is a pure function of the waveform and the config, run on CPU in float32. The only randomness is caller-requested augmentation (explicitly seeded) and the padding fill (seeded by the window id, so the same window always gets the same fill).
- The statistics are accumulated in float64 in a fixed file and window order.
- GPU execution of the STFT may differ in the last bits. Compute validation and test features on CPU, or with `torch.use_deterministic_algorithms(True)`, if bitwise reproducibility matters.

## Known issues and decisions for the next stage

**Padding.** Measured on the current `windows.csv`:
- Padded windows by class: train 56% of positives vs 13% of negatives; validation 100% vs 12%; test 98% vs 12%. Streaming never pads.
- By share of audio time, padding is 5.7% of training audio. By window type, confusable negatives are the most padded (29% of their time), then positives (15%), then partial-wakeword negatives (3%). Other negative types have none.
- So padding is not a clean "positive" cue, because the hardest negatives carry it too. But zero padding is exact digital silence (−100 dB frames), which the device never produces.
- **Decision (2026-09-28): `pad_mode: noise` in every split.** Three options were considered:
  1. **Zeros everywhere.** Consistent across splits, but every split then contains a pattern the device cannot produce, and validation/test positives are almost all padded.
  2. **Deterministic noise everywhere (chosen).** Train, validation, test and the device all lack digital silence. Validation and test stay bit-deterministic, and the required augmentation → framing order is kept.
  3. **Train-only fill through augmentation.** Training would never see digital silence while almost every validation/test positive still contains it, which distorts recall estimates. (Until 2026-09-29 augmentation ran before framing and could not reach the padding. It now runs on the framed window during training; see the dataloading README.)
- The fill is white Gaussian noise at −50.6 dBFS RMS. Under the deployment metric this reproduces the measured median RPI floor, −54.2 dB (tested to ±0.5 dB). Each window's fill is seeded by its window id.
- Remaining limitation: white noise is not the device's real (coloured) background, so a weaker seam remains where real audio meets the fill.
- **Possible upgrade:** fill with real device background drawn deterministically from the *same split* (train background for train, validation background for validation, and so on). That's more realistic and keeps audio from crossing splits, but needs a per-split noise pool.
- Use the real-device and streaming-eval reports (the streaming-eval holdout contains no padding) to confirm the effect.

**Digital silence in the statistics.** With noise padding, **8.0%** of training frame values sit exactly at the −100 dB floor, down from 13.3% with zero padding. The rest is exact zeros *inside* source files (silent synthetic regions, the all-zero `Manual-BG-RPI2`). No padding choice can reach those. The augmentation strategy's `device_noise_floor` replaces them in training, and validation and test keep them, as that strategy specifies. The bimodal mix is why per-bin standard deviations are large (26–35 dB). `normalization_stats.json` records the floor fraction per bin (`diagnostic_fraction_at_db_floor`).

**Statistics use unaugmented training audio.** Training with augmentation shifts the training feature distribution (for example, `device_noise_floor` removes most of the remaining −100 dB frames), so augmented training batches will not be exactly zero-mean and unit-variance. That's intended: the statistics are a fixed affine map shared by every split and by inference. Computing them from random augmentation draws would make them non-deterministic.

**Measured results (2026-09-28, `pad_mode: noise`, feature hash `e697d07bc6531278`):**
- 6,012 train windows (5,744 negative, 268 positive) from 5,467 files, 883,764 frames.
- Validation (847 windows), test (844) and streaming-eval (4,642) windows are never read.
- Per-bin mean −33.2 to −14.2 dB, std 26.3 to 34.7 dB. With zero padding they were −37.7 to −18.1 and 30.0 to 39.7 dB. Computation takes about 7 minutes on CPU.
- 22/22 tests pass. The 48 kHz streaming vs offline difference is 3.8e-6 in normalized units.
