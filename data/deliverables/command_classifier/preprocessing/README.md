# Phase 2: waveform processing and feature extraction (command classifier)

One front end for training, evaluation and deployment. It reuses the wakeword project's module
(`../../preprocessing/wakeword_preprocessing.py`): same features, a longer window.

```
capture waveform (any rate, mono/stereo)
  -> mono float32, 16 kHz                                  load_waveform
  -> first 5 s of the capture; shorter ones padded on the  frame_command
     right with noise at the measured Pi floor (-50.6 dBFS)
  -> log-mel [40, 497]: 25 ms Hann, 10 ms hop, 40 HTK mel  FeatureExtractor
     bins 20-7600 Hz, 10*log10 with a fixed floor
  -> per-mel-bin normalisation, frozen TRAIN statistics     Normalizer (normalization_stats.json)
```

| File | Role |
|---|---|
| `preprocessing_config.json` | Every value, with its reason in the `_*` keys |
| `command_preprocessing.py` | `frame_command`, `CommandPreprocessor` (features + normalisation, batched on the GPU during training) |
| `compute_normalization_stats.py` | Statistics from 4,000 augmented TRAINING draws (what the model is trained on); records the feature-config hash, and the Normalizer refuses mismatched statistics |
| `test_preprocessing.py` | Shapes, right-padding with noise, start kept, determinism, same feature definition as the wakeword, train-only statistics |

**Why 5 s, from the start.** A capture always starts at the wake-up. The owner's captures are 1.5–8.1 s (median 3.3 s),
so 5 s covers every complete capture of a command said within about 1.5 s of the chime. Longer captures (hitting the
maximum length, speech over a TV) are cut at 5 s, keeping the start, where the command is.

**Why noise padding, not zeros.** The device never produces exact digital silence. A run of zeros would tell the
model how long the capture was.
