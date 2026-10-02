# Dataset / DataLoader layer

Connects the manifests and split (`dataset_split`), the windows (`segmentation_windowing`), the augmentation
strategy (`deployment_driven_augmentation_strategy`) and the shared front end (`preprocessing`) to model training.

| File | Deliverable |
|---|---|
| `wakeword_data.py` | Dataset (`WakewordWindowDataset`), split enforcement (`SplitGuard`), sampler (`BucketQuotaSampler`), `collate`, DataLoader factory (`build_dataloaders`) |
| `augmentation.py` | Training-only waveform augmentation implementing `augmentation_config.json` (`DeploymentAugmenter`) |
| `dataloader_config.json` | Sampling configuration plus loader settings (batch size, workers, seed, inputs) |
| `reproducibility.py` | Keyed seeding, worker init, single-thread context, `seed_everything` |
| `test_dataloader.py` | Dataset/DataLoader sanity tests (`python test_dataloader.py`, or pytest) |
| `distribution_report.py` → `distribution_report.md` / `.json` | Class and negative-category balance before and after sampling |

## Usage

```python
import wakeword_data as wd, reproducibility as rep

rep.seed_everything(20260928)                                   # model init etc.
loaders = wd.build_dataloaders(batch_size=64, num_workers=4)    # {"train", "validation", "test"}
for epoch in range(n_epochs):
    wd.set_epoch(loaders, epoch)                                 # new sampling + augmentation draw
    for batch in loaders["train"]:
        x, y = batch["features"], batch["labels"]                # [B, 1, 40, 147] float32, [B] in {0., 1.}
        meta = batch["meta"]                                     # list of B dicts (see Metadata)
    for batch in loaders["validation"]:                          # fixed, unaugmented, natural mix
        ...
```

`build_dataloaders(splits=[...], batch_size=, num_workers=, seed=, augment=)` overrides the config. `augment=False` gives
fixed preprocessing for training data too (for example, to measure training-set metrics).
`splits=["streaming_eval_holdout"]` loads the held-out streams (never augmented).

## What each example is

1. The window's source file is loaded (`preprocessing.load_waveform`: mono, 16 kHz, amplitude unchanged). Files are
   cached per worker.
2. The real-audio excerpt of the window (`windows.csv` `start_sec`, 1.5 s) is cut out.
3. It is framed to 24,000 samples. Padding gets the preprocessing's deterministic noise fill.
4. **Train only, if enabled:** the whole framed window, padding included, is augmented (below). Before 2026-09-29
   only the excerpt was augmented; see the note at the end of the augmentation section.
5. It goes through the shared `WakewordPreprocessor` (log-mel dB, frozen train-only normalization), giving `[1, 40, 147]`.

Validation and test (and the streaming holdout) skip step 4 and never see an augmenter: the Dataset raises if one is
passed for any split other than `train`.

## Metadata (every example, `batch["meta"]`)

| Group | Fields |
|---|---|
| Identity and location | `window_id`, `filepath`, `file_id`, `start_sec`, `end_sec`, `src_start_sec`, `src_end_sec`, `pad_left_sec`, `pad_right_sec`, `index` |
| Label | `label`, `label_int`, `window_type`, `label_source`, `annotation_id`, `wakeword_start_in_window_sec`, `wakeword_end_in_window_sec`, `wakeword_coverage`, `manifest_label`, `label_origin` |
| Category and source | `category`, `canonical_category`, `manifest_category`, `source`, `parent_source`, `provenance_type`, `device`, `eval_subset` (`real_device_rpi` / `real_other_device` / `synthetic`) |
| Grouping | `source_id`, `recording_id`, `source_group`, `speaker_id`, `speaker_id_normalized`, `session_id`, `voice`, `transcription` |
| Split | `split`, `split_origin`, `is_eval_only` |
| Augmentation | `epoch`, `draw`, `augmented`, `augmentation` (JSON of every transform applied and its sampled parameters) |

Where metadata is missing:
- **Speaker:** every synthetic speech window has a speaker (its TTS voice). Procedurally generated noise clips have none.
  Manual recordings have no speaker ID in any input (likely one speaker; see the segmentation README).
- **Session:** no session field exists in any input manifest, so `session_id` is always `None`.

An example can be regenerated from its metadata: `dataset[(index, epoch, draw)]` reproduces an augmented training
example exactly, and `(filepath, start_sec, window_id)` alone reproduces a validation or test example (tested).

## Split boundaries (enforced by code)

`SplitGuard` runs every time datasets are built and raises `SplitBoundaryError` if:
- a window's `source_group` is missing from the frozen split, or sits in a different split from the one frozen in
  `dataset_split/split_assignments.csv` + `segmentation_windowing/outputs/split_extension.csv`;
- any `source_group` or audio file appears in more than one split;
- a streaming-eval (eval-only) window is outside `streaming_eval_holdout`, or its source group also appears in
  train/validation/test;
- a dataset is given rows from another split, or rows whose source group is frozen elsewhere;
- the augmentation noise bank contains any window or audio file from a non-train source group.

Augmentation is refused for every split except `train`. Tests tamper with copies of the data to confirm each rule fires.

## Sampling strategy (`dataloader_config.json` → `sampling`)

**Quotas by bucket.** A training epoch is `epoch_size` draws (default: the number of training windows, 6,012). Each
bucket (a set of window types) receives a fixed share of the draws:

| Bucket | Role | Sources | Share | Natural share |
|---|---|---|---:|---:|
| positive_real | positive | real | 12.5% | 2.9% |
| positive_synthetic | positive | synthetic | 12.5% | 1.6% |
| confusable | hard negative | both | 20% | 16.3% |
| partial_wakeword | hard negative | both | 14% | 9.4% |
| general_speech | easy negative | both | 15% | 35.5% |
| media | easy negative | both | 8% | 16.3% |
| silence_noise | easy negative | both | 4% | 13.4% |
| device_background | easy negative | real | 14% | 4.7% |

**Source balance (v3, 2026-09-28).** A bucket can restrict itself to `real` (manual recordings) or `synthetic`
sources. `evaluation/diagnostics/source_confound.py` showed a model learning the recording condition instead of the
wakeword, because real audio was 64% of positives but 6% of negatives. The positive quota is now split evenly
between real and synthetic, and real device background rises to 14%. Real-recording share of draws is now 50% of
positives and 26% of negatives. Closing the remaining gap needs more real RPI negatives.

- Hard negatives are the near-misses a deployed detector must reject: confusable phrases (such as "A Delta")
  and fragments of the wakeword. They rise from 27% to 45% of all negatives. Easy negatives (general speech, media,
  synthetic noise, device background) fall from 70% to 41% of draws but remain represented.
- Real device background has a dedicated share because it is the only negative recorded on the deployment device.

**Source-group weighting inside a bucket.** A window's weight is `1 / n_windows(source_group, bucket) ^ 0.5`, so a long
recording contributes in proportion to the square root of its window count. For example, the largest real positive
source group (`Manual-Fil-Macmic1`) falls from 16.9% to 7.2% of real positive draws, and the effective number of real
positive source groups rises from 13.8 to 32.5.

**Draws are with replacement**, and each draw has its own augmentation seed, so repeated windows are different
augmented examples. Real positives are drawn about 4.4 times per epoch each. Synthetic positives, only 96 windows
from a handful of voices, are drawn about 7.8 times each. That repetition is the main overfitting risk this strategy
accepts. With v3 augmentation, about 90% of draws are level-adjusted and most are noise-mixed.

**Validation and test are not sampled:** they iterate every window once, in the natural class mix, so metrics are not
distorted by the training quotas.

See `distribution_report.md` for before/after balance, repeat factors, window coverage per epoch, source-group
concentration, and measured augmentation rates.

## Augmentation (`augmentation.py`)

The transforms follow `augmentation_config.json` in its composition order, with its probabilities, parameter ranges,
per-category `apply_to` lists and constraints:

| Transform | Probability | What it does | Status |
|---|---:|---|---|
| `random_mic_coloring` | 0.5 (v4) | Random mild microphone colour: low shelf (100–400 Hz), high shelf (2–6 kHz), each ±6 dB, and one peak (300–3000 Hz, ±4 dB), applied zero-phase in the FFT domain. Same probability for every category and label, so colour can't mark a class. Interim, until the RPI mic response is measured. | enabled (2026-09-29, owner-approved) |
| `device_level_gain` | 0.9 (v3) | Scales the excerpt so its speech-band peak (98th-percentile 20 ms frame, 150–4000 Hz) equals U(−49.7, −18.8) dB. Never above full scale. | enabled |
| `rir_reverberation` | 0.2 | Convolves with a curated RIR (direct path scaled to unit amplitude); skipped for real RPI recordings. | enabled (2026-09-29): `data/rir/rir_bank.csv`, 132 real MIT IR Survey responses with RT60 0.15–0.6 s (`build_rir_bank.py`) |
| `background_noise_mixing` | 0.9 (v3) | Adds one noise, 50% real device background (`real_noise_bank.csv`, train-only) and 50% synthetic (white, pink, hum, fan, clicks), scaled so peak-over-floor is U(5.4, 25.7) dB, never below 4.5 dB. Only applied when the example is cleaner than the target; skips are recorded. The level is corrected in closed loop and the 4.5 dB minimum is enforced on the result. Stationary noise lands within ±1 dB; non-stationary noise within about 3 dB below target, or cleaner. | enabled |
| `device_noise_floor` | 1.0 when the excerpt has ≥ 20 ms of exact zeros | Adds real device background over the whole excerpt at a floor of U(−58.2, −51.4) dB; skipped if mixing already produced a floor at least that high. | enabled |

- Levels use the strategy's own metric (`measure_deployment_levels.py`), and tests check the achieved levels.
- Mixing hits its target within about 1 dB on real draws.
- `data/noise/` is also empty, so the synthetic noises are generated procedurally from the example's RNG. The config
  names them, and they contain no audio from any split.
- Reverberation reads `augmentation.rir_bank` (`id, filepath, rt60_s`, paths relative to the project root). Set it to
  `null` to disable.
- **Source kinds (2026-09-29):** `real` (our device/microphone recordings), `external` (public-corpus human audio,
  sources `ext_*`: MSWC, FLEURS, MUSAN) and `synthetic`. The `external_confusable` bucket (8%) holds external
  confusables and partials: MSWC one-word clips and real "Hey Snips" phrases. Keeping them in their own bucket stops
  them from crowding out the synthetic phrase-level confusables. Evaluation reports external audio as its own subset,
  `real_external`.
- **Augmentation covers the padding (changed 2026-09-29).** Before, augmentation ran on the excerpt, before framing,
  so padding kept the fixed pad-noise floor while the word itself got background noise.
  - *The problem:* most TTS positives are ~1 s clips, with 0.68 s of padding per window on average; negatives are
    mostly continuous audio. "Speech with quiet flanks" became a positive cue.
  - *Measured with BC-ResNet-3:* synthetic validation positives fell from 89% to 37% detected once background covered
    the whole window. Real RPI positives, which have almost no padding, stayed at 100%. Wakewords inside continuous
    background in the streaming sets were mostly missed.
  - *The fix:* augmentation now runs on the framed window. Validation, test and streaming are never augmented, so
    they are unchanged.

## Reproducibility

- **Keyed randomness.** The epoch's draws are seeded by `(seed, epoch)`. Each example's augmentation is seeded by
  `(seed, epoch, draw position, window_id)`. Results therefore do not depend on `num_workers`, worker scheduling or
  load order. Tests check that training batches are identical with 0 and 2 workers.
- **Single-thread loading.** Each example is loaded and preprocessed with one torch intra-op thread
  (`reproducibility.single_thread`). Multi-threaded reductions sum in a thread-count-dependent order, and without
  this an example differed by up to 3e-7 between the main process and a worker.
- **Validation and test** outputs are bit-identical across runs and worker counts (tested).
- `seed_everything(seed)` covers the training loop's own randomness (model init, dropout).

## Performance notes

- Building the loaders takes about 30 s: metadata joins, the split check, and loading the 278-window noise bank.
- A 64-example training batch loads in about 1.2 s with 4 workers on this machine.
- `file_cache_per_worker` bounds the number of decoded files each worker keeps.
