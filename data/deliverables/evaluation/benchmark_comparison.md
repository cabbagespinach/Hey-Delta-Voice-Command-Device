# VCM benchmark - side by side - Hey Delta (Raspberry Pi 5)

Each column is one model and cutoff rule, scored by the class benchmark (github.com/airimonda/vcm-benchmark, scoring of commit ab39857). Rows, names and number formats are the benchmark's own report.md; the full per-run reports are linked below.

| column                   | how it was run           | full report                                                                                |
|--------------------------|--------------------------|--------------------------------------------------------------------------------------------|
| hf_plus cautious (Oct 2) | live run 20261002-135144 | [20261002-135144__rescored_ab39857/report.md](20261002-135144__rescored_ab39857/report.md) |
| hf_plus balanced (Oct 3) | same captures, re-scored | [20261003-075251__hf_plus_balanced/report.md](20261003-075251__hf_plus_balanced/report.md) |
| hf_plus cautious (Oct 3) | same captures, re-scored | [20261003-075251__hf_plus_cautious/report.md](20261003-075251__hf_plus_cautious/report.md) |
| hf_only balanced (Oct 3) | same captures, re-scored | [20261003-075251__hf_only_balanced/report.md](20261003-075251__hf_only_balanced/report.md) |
| hf_only cautious (Oct 3) | same captures, re-scored | [20261003-075251__hf_only_cautious/report.md](20261003-075251__hf_only_cautious/report.md) |

Models (all BC-ResNet-6, 474,512 parameters with the built-in front end, same 31 commands + 'not a command'):

- bcresnet6_hf_plus: class HF dataset + our data
- bcresnet6_hf_only: class HF dataset only

Rules: `cautious` / `balanced` = the model's two validated confidence cutoffs (below the cutoff it answers out of scope); cutoffs: hf_plus 0.61 / 0.34, hf_only 0.89 / 0.67.

How the runs were made:

- Both days: wake word **Hey Delta**, the class holdout pinned to Hugging Face revision da92a79 (202 trials with the wake word + 16 without), shuffle seed 79276, connection ssh, laptop speaker about 1 m from the Pi.
- Oct 3: one benchmark run in which the Pi ran several command models on every capture (they answered one after the other). Each Oct 3 column is one model's answers to these **same captures**, scored with the benchmark's own `--rescore`; they are not separate live runs. Classification, slots, inference time and model size are per column; the wake word, response latency (time to the Pi's first answer) and Pi CPU / RAM / temperature are those of the one run.
- Oct 2: a separate run (hf_plus, cautious only), re-scored with the same benchmark version so its report has the same sections; its answers and numbers are unchanged ([original report](20261002-135144/report.md)). Different day, so room and microphone setup can differ from Oct 3.

## At a glance

|                                   | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|-----------------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| intent accuracy (19)              | 69.8%                    | 84.7%                    | 75.2%                    | 50.5%                    | 35.6%                    |
| command accuracy (93)             | 69.3%                    | 84.2%                    | 75.2%                    | 50.5%                    | 35.6%                    |
| false accept (out of scope fired) | 6.2% (1/16)              | 6.2% (1/16)              | 0.0% (0/16)              | 6.2% (1/16)              | 0.0% (0/16)              |
| false reject (command ignored)    | 31.2%                    | 14.0%                    | 25.3%                    | 52.7%                    | 69.9%                    |
| false wake (no wake word, fired)  | 0.0% (0/16)              | 0.0% (0/16)              | 0.0% (0/16)              | 0.0% (0/16)              | 0.0% (0/16)              |
| slot exact                        | 98.7%                    | 98.9%                    | 100.0%                   | 100.0%                   | 100.0%                   |
| latency p95                       | 4.93 s                   | 3.25 s                   | 3.25 s                   | 3.25 s                   | 3.25 s                   |

# Detailed metrics

## Classification

### 19 intents (+reject)

| metric                                            | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|---------------------------------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| accuracy                                          | 69.8%                    | 84.7%                    | 75.2%                    | 50.5%                    | 35.6%                    |
| balanced accuracy                                 | 66.4%                    | 83.0%                    | 72.2%                    | 46.6%                    | 32.2%                    |
| precision (macro)                                 | 88.1%                    | 93.3%                    | 94.0%                    | 83.3%                    | 65.5%                    |
| recall (macro)                                    | 66.4%                    | 83.0%                    | 72.2%                    | 46.6%                    | 32.2%                    |
| F1 (macro)                                        | 72.6%                    | 85.9%                    | 77.9%                    | 54.6%                    | 38.1%                    |
| F2 (macro)                                        | 67.6%                    | 83.6%                    | 73.4%                    | 47.8%                    | 32.2%                    |
| false accept rate (OOS fired)                     | 6.2%                     | 6.2%                     | 0.0%                     | 6.2%                     | 0.0%                     |
| false reject rate (in-scope silent/rejected)      | 31.2%                    | 14.0%                    | 25.3%                    | 52.7%                    | 69.9%                    |
| misfire rate (wrong command fired)                | 1.1%                     | 2.2%                     | 1.6%                     | 0.5%                     | 0.0%                     |
| accuracy 95% CI                                   | [63-76%]                 | [79-89%]                 | [69-81%]                 | [44-57%]                 | [29-42%]                 |
| false accept 95% CI                               | [1-28%] (1/16)           | [1-28%] (1/16)           | [0-19%] (0/16)           | [1-28%] (1/16)           | [0-19%] (0/16)           |
| false wake rate (command without wake word fired) | 0.0% [0-19%] (0/16)      | 0.0% [0-19%] (0/16)      | 0.0% [0-19%] (0/16)      | 0.0% [0-19%] (0/16)      | 0.0% [0-19%] (0/16)      |

### 93 commands (+reject)

| metric                                            | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|---------------------------------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| accuracy                                          | 69.3%                    | 84.2%                    | 75.2%                    | 50.5%                    | 35.6%                    |
| balanced accuracy                                 | 67.5%                    | 83.4%                    | 73.4%                    | 47.3%                    | 30.9%                    |
| precision (macro)                                 | 88.9%                    | 94.0%                    | 90.7%                    | 72.7%                    | 50.1%                    |
| recall (macro)                                    | 67.5%                    | 83.4%                    | 73.4%                    | 47.3%                    | 30.9%                    |
| F1 (macro)                                        | 74.0%                    | 86.4%                    | 78.6%                    | 55.2%                    | 36.7%                    |
| F2 (macro)                                        | 69.4%                    | 84.2%                    | 74.9%                    | 49.6%                    | 32.4%                    |
| false accept rate (OOS fired)                     | 6.2%                     | 6.2%                     | 0.0%                     | 6.2%                     | 0.0%                     |
| false reject rate (in-scope silent/rejected)      | 31.2%                    | 14.0%                    | 25.3%                    | 52.7%                    | 69.9%                    |
| misfire rate (wrong command fired)                | 1.6%                     | 2.7%                     | 1.6%                     | 0.5%                     | 0.0%                     |
| accuracy 95% CI                                   | [63-75%]                 | [78-89%]                 | [69-81%]                 | [44-57%]                 | [29-42%]                 |
| false accept 95% CI                               | [1-28%]                  | [1-28%]                  | [0-19%]                  | [1-28%]                  | [0-19%]                  |
| false wake rate (command without wake word fired) | 0.0% [0-19%] (0/16)      | 0.0% [0-19%] (0/16)      | 0.0% [0-19%] (0/16)      | 0.0% [0-19%] (0/16)      | 0.0% [0-19%] (0/16)      |

### Responses

| metric           | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| fired a command  | 98.0%                    | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   |
| no response      | 4                        | 0                        | 0                        | 0                        | 0                        |
| extra fires      | 0                        | 0                        | 0                        | 0                        | 0                        |
| wake detect rate | 98.5%                    | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   |

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

### overall

| metric                         | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|--------------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| clips (with wake word)         | 202                      | 202                      | 202                      | 202                      | 202                      |
| **19 intents** accuracy        | 69.8% [63-76%]           | 84.7% [79-89%]           | 75.2% [69-81%]           | 50.5% [44-57%]           | 35.6% [29-42%]           |
| balanced accuracy              | 66.4%                    | 83.0%                    | 72.2%                    | 46.6%                    | 32.2%                    |
| F1 (macro)                     | 72.6%                    | 85.9%                    | 77.9%                    | 54.6%                    | 38.1%                    |
| F2 (macro)                     | 67.6%                    | 83.6%                    | 73.4%                    | 47.8%                    | 32.2%                    |
| false accept rate              | 6.2% (1/16)              | 6.2% (1/16)              | 0.0% (0/16)              | 6.2% (1/16)              | 0.0% (0/16)              |
| false reject rate              | 31.2%                    | 14.0%                    | 25.3%                    | 52.7%                    | 69.9%                    |
| misfire rate                   | 1.1%                     | 2.2%                     | 1.6%                     | 0.5%                     | 0.0%                     |
| **93 commands** accuracy       | 69.3%                    | 84.2%                    | 75.2%                    | 50.5%                    | 35.6%                    |
| balanced accuracy              | 67.5%                    | 83.4%                    | 73.4%                    | 47.3%                    | 30.9%                    |
| F1 (macro)                     | 74.0%                    | 86.4%                    | 78.6%                    | 55.2%                    | 36.7%                    |
| F2 (macro)                     | 69.4%                    | 84.2%                    | 74.9%                    | 49.6%                    | 32.4%                    |
| misfire rate                   | 1.6%                     | 2.7%                     | 1.6%                     | 0.5%                     | 0.0%                     |
| slot exact (intent right)      | 98.7% (n=78)             | 98.9% (n=93)             | 100.0% (n=83)            | 100.0% (n=55)            | 100.0% (n=35)            |
| latency p50 / p95              | 1.42 / 4.93 s            | 1.45 / 3.25 s            | 1.45 / 3.25 s            | 1.45 / 3.25 s            | 1.45 / 3.25 s            |
| false wake rate (no wake word) | 0.0% (0/16)              | 0.0% (0/16)              | 0.0% (0/16)              | 0.0% (0/16)              | 0.0% (0/16)              |

### real voice

| metric                         | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|--------------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| clips (with wake word)         | 96                       | 96                       | 96                       | 96                       | 96                       |
| **19 intents** accuracy        | 52.1% [42-62%]           | 74.0% [64-82%]           | 60.4% [50-70%]           | 28.1% [20-38%]           | 18.8% [12-28%]           |
| balanced accuracy              | 51.5%                    | 72.6%                    | 60.1%                    | 25.5%                    | 14.1%                    |
| F1 (macro)                     | 55.0%                    | 77.6%                    | 65.4%                    | 26.9%                    | 12.2%                    |
| F2 (macro)                     | 51.5%                    | 73.7%                    | 60.9%                    | 24.2%                    | 11.7%                    |
| false accept rate              | 0.0% (0/10)              | 0.0% (0/10)              | 0.0% (0/10)              | 0.0% (0/10)              | 0.0% (0/10)              |
| false reject rate              | 52.3%                    | 25.6%                    | 41.9%                    | 79.1%                    | 90.7%                    |
| misfire rate                   | 1.2%                     | 3.5%                     | 2.3%                     | 1.2%                     | 0.0%                     |
| **93 commands** accuracy       | 51.0%                    | 72.9%                    | 60.4%                    | 28.1%                    | 18.8%                    |
| balanced accuracy              | 46.0%                    | 70.1%                    | 56.3%                    | 20.7%                    | 10.3%                    |
| F1 (macro)                     | 44.8%                    | 69.1%                    | 55.2%                    | 19.4%                    | 9.4%                     |
| F2 (macro)                     | 45.2%                    | 69.6%                    | 55.6%                    | 19.8%                    | 9.6%                     |
| misfire rate                   | 2.3%                     | 4.7%                     | 2.3%                     | 1.2%                     | 0.0%                     |
| slot exact (intent right)      | 95.0% (n=20)             | 97.0% (n=33)             | 100.0% (n=24)            | 100.0% (n=8)             | 100.0% (n=4)             |
| latency p50 / p95              | 1.40 / 4.88 s            | 1.43 / 2.91 s            | 1.43 / 2.91 s            | 1.43 / 2.91 s            | 1.43 / 2.91 s            |
| false wake rate (no wake word) | 0.0% (0/7)               | 0.0% (0/7)               | 0.0% (0/7)               | 0.0% (0/7)               | 0.0% (0/7)               |

### synthetic voice

| metric                         | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|--------------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| clips (with wake word)         | 106                      | 106                      | 106                      | 106                      | 106                      |
| **19 intents** accuracy        | 85.8% [78-91%]           | 94.3% [88-97%]           | 88.7% [81-93%]           | 70.8% [61-79%]           | 50.9% [42-60%]           |
| balanced accuracy              | 79.2%                    | 92.0%                    | 82.3%                    | 65.7%                    | 48.8%                    |
| F1 (macro)                     | 81.1%                    | 92.1%                    | 83.8%                    | 69.2%                    | 52.2%                    |
| F2 (macro)                     | 79.4%                    | 91.8%                    | 82.3%                    | 65.9%                    | 48.1%                    |
| false accept rate              | 16.7% (1/6)              | 16.7% (1/6)              | 0.0% (0/6)               | 16.7% (1/6)              | 0.0% (0/6)               |
| false reject rate              | 13.0%                    | 4.0%                     | 11.0%                    | 30.0%                    | 52.0%                    |
| misfire rate                   | 1.0%                     | 1.0%                     | 1.0%                     | 0.0%                     | 0.0%                     |
| **93 commands** accuracy       | 85.8%                    | 94.3%                    | 88.7%                    | 70.8%                    | 50.9%                    |
| balanced accuracy              | 85.5%                    | 95.0%                    | 87.8%                    | 71.1%                    | 49.5%                    |
| F1 (macro)                     | 84.8%                    | 94.3%                    | 87.1%                    | 70.8%                    | 49.1%                    |
| F2 (macro)                     | 85.1%                    | 94.7%                    | 87.4%                    | 70.8%                    | 49.0%                    |
| misfire rate                   | 1.0%                     | 1.0%                     | 1.0%                     | 0.0%                     | 0.0%                     |
| slot exact (intent right)      | 100.0% (n=58)            | 100.0% (n=60)            | 100.0% (n=59)            | 100.0% (n=47)            | 100.0% (n=31)            |
| latency p50 / p95              | 1.45 / 4.98 s            | 1.46 / 3.60 s            | 1.46 / 3.60 s            | 1.46 / 3.60 s            | 1.46 / 3.60 s            |
| false wake rate (no wake word) | 0.0% (0/9)               | 0.0% (0/9)               | 0.0% (0/9)               | 0.0% (0/9)               | 0.0% (0/9)               |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

### n

| intent          | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|-----------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| ALARM           | 15                       | 15                       | 15                       | 11                       | 8                        |
| BRIGHTNESS      | 13                       | 15                       | 12                       | 9                        | 5                        |
| COLOR           | 17                       | 17                       | 17                       | 12                       | 6                        |
| CREATE_REMINDER | 10                       | 16                       | 12                       | 8                        | 5                        |
| TEMPERATURE     | 9                        | 15                       | 12                       | 6                        | 6                        |
| TIMER           | 14                       | 15                       | 15                       | 9                        | 5                        |
| ALL             | 78                       | 93                       | 83                       | 55                       | 35                       |

### exact

| intent          | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|-----------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| ALARM           | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   |
| BRIGHTNESS      | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   |
| COLOR           | 94.1%                    | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   |
| CREATE_REMINDER | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   |
| TEMPERATURE     | 100.0%                   | 93.3%                    | 100.0%                   | 100.0%                   | 100.0%                   |
| TIMER           | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   |
| ALL             | 98.7%                    | 98.9%                    | 100.0%                   | 100.0%                   | 100.0%                   |

### mean abs error

| intent          | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|-----------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| ALARM           | 0.0 min                  | 0.0 min                  | 0.0 min                  | 0.0 min                  | 0.0 min                  |
| BRIGHTNESS      | 0.0 %                    | 0.0 %                    | 0.0 %                    | 0.0 %                    | 0.0 %                    |
| COLOR           | -                        | -                        | -                        | -                        | -                        |
| CREATE_REMINDER | -                        | -                        | -                        | -                        | -                        |
| TEMPERATURE     | 0.0 deg                  | 0.3 deg                  | 0.0 deg                  | 0.0 deg                  | 0.0 deg                  |
| TIMER           | 0.0 s                    | 0.0 s                    | 0.0 s                    | 0.0 s                    | 0.0 s                    |
| ALL             | -                        | -                        | -                        | -                        | -                        |

### mean rel error

| intent          | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|-----------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| ALARM           | 0.000                    | 0.000                    | 0.000                    | 0.000                    | 0.000                    |
| BRIGHTNESS      | 0.000                    | 0.000                    | 0.000                    | 0.000                    | 0.000                    |
| COLOR           | -                        | -                        | -                        | -                        | -                        |
| CREATE_REMINDER | -                        | -                        | -                        | -                        | -                        |
| TEMPERATURE     | 0.000                    | 0.033                    | 0.000                    | 0.000                    | 0.000                    |
| TIMER           | 0.000                    | 0.000                    | 0.000                    | 0.000                    | 0.000                    |
| ALL             | 0.000                    | 0.008                    | 0.000                    | 0.000                    | 0.000                    |

### phonetic dist

| intent          | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|-----------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| ALARM           | 0.000                    | 0.000                    | 0.000                    | 0.000                    | 0.000                    |
| BRIGHTNESS      | 0.547                    | 0.543                    | 0.543                    | 0.562                    | 0.566                    |
| COLOR           | 0.039                    | 0.000                    | 0.000                    | 0.000                    | 0.000                    |
| CREATE_REMINDER | 0.000                    | 0.000                    | 0.000                    | 0.000                    | 0.000                    |
| TEMPERATURE     | 0.472                    | 0.509                    | 0.473                    | 0.505                    | 0.505                    |
| TIMER           | 0.357                    | 0.348                    | 0.348                    | 0.432                    | 0.467                    |
| ALL             | 0.218                    | 0.226                    | 0.210                    | 0.218                    | 0.234                    |

### char dist

| intent          | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|-----------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| ALARM           | 0.000                    | 0.000                    | 0.000                    | 0.000                    | 0.000                    |
| BRIGHTNESS      | 0.542                    | 0.539                    | 0.536                    | 0.567                    | 0.577                    |
| COLOR           | 0.035                    | 0.000                    | 0.000                    | 0.000                    | 0.000                    |
| CREATE_REMINDER | 0.000                    | 0.000                    | 0.000                    | 0.000                    | 0.000                    |
| TEMPERATURE     | 0.463                    | 0.485                    | 0.463                    | 0.472                    | 0.472                    |
| TIMER           | 0.408                    | 0.408                    | 0.408                    | 0.453                    | 0.475                    |
| ALL             | 0.225                    | 0.231                    | 0.218                    | 0.219                    | 0.231                    |

## Raspberry Pi

- **Raspberry Pi 5 Model B Rev 1.1**, 4 cores  up to 2400.0 MHz, RAM 4049.1 MB, Debian GNU/Linux 13 (trixie), kernel 6.18.50+rpt-rpi-2712, Python 3.13.5
- packages: numpy 2.2.4, sounddevice 0.5.6

mean / p95 / max. Oct 3 columns share one run: only inference time (and FLOPs / size) is per model; the Oct 3 process ran several models (one after the other), so its RAM and the whole-Pi CPU are higher than in a one-model run.

| metric                                      | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|---------------------------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| response latency (command end -> Pi output) | 1.783 / 4.935 / 5.569 s  | 1.616 / 3.252 / 5.670 s  | 1.616 / 3.252 / 5.670 s  | 1.616 / 3.252 / 5.670 s  | 1.616 / 3.252 / 5.670 s  |
| latency p50 / p99                           | 1.423 / 5.458 s          | 1.450 / 5.368 s          | 1.450 / 5.368 s          | 1.450 / 5.368 s          | 1.450 / 5.368 s          |
| inference time (Pi-reported)                | 66.5 / 71.9 / 75.0 ms    | 57.7 / 72.1 / 95.3 ms    | 57.7 / 72.1 / 95.3 ms    | 56.4 / 71.5 / 90.7 ms    | 56.4 / 71.5 / 90.7 ms    |
| real-time factor (infer / audio window)     | 0.013 / 0.014 / 0.015    | 0.012 / 0.014 / 0.019    | 0.012 / 0.014 / 0.019    | 0.011 / 0.014 / 0.018    | 0.011 / 0.014 / 0.018    |
| CPU temperature                             | 54.5 / 55.6 / 57.3 C     | 54.6 / 56.2 / 57.9 C     | 54.6 / 56.2 / 57.9 C     | 54.6 / 56.2 / 57.9 C     | 54.6 / 56.2 / 57.9 C     |
| CPU use, whole Pi                           | 5.2 / 7.8 / 13.4 %       | 10.8 / 26.9 / 35.9 %     | 10.8 / 26.9 / 35.9 %     | 10.8 / 26.9 / 35.9 %     | 10.8 / 26.9 / 35.9 %     |
| CPU use, your runtime process               | 12.8 / 18.9 / 22.8 %     | 12.3 / 18.9 / 25.8 %     | 12.3 / 18.9 / 25.8 %     | 12.3 / 18.9 / 25.8 %     | 12.3 / 18.9 / 25.8 %     |
| RAM (RSS), your runtime process             | 108.7 / 109.8 / 110.7 MB | 144.8 / 145.0 / 145.8 MB | 144.8 / 145.0 / 145.8 MB | 144.8 / 145.0 / 145.8 MB | 144.8 / 145.0 / 145.8 MB |
| RAM used, whole Pi                          | 648.4 / 653.9 / 666.6 MB | 601.6 / 625.5 / 665.4 MB | 601.6 / 625.5 / 665.4 MB | 601.6 / 625.5 / 665.4 MB | 601.6 / 625.5 / 665.4 MB |
| CPU clock                                   | 1850 / 2400 / 2400 MHz   | 2213 / 2400 / 2400 MHz   | 2213 / 2400 / 2400 MHz   | 2213 / 2400 / 2400 MHz   | 2213 / 2400 / 2400 MHz   |
| load average (1 min)                        | 0.23 / 0.43 / 0.79       | 0.60 / 0.81 / 1.18       | 0.60 / 0.81 / 1.18       | 0.60 / 0.81 / 1.18       | 0.60 / 0.81 / 1.18       |
| runtime CPU-seconds per second of speech    | 1.280                    | 1.223                    | 1.223                    | 1.223                    | 1.223                    |
| runtime CPU share of wall time              | 12.7%                    | 12.3%                    | 12.3%                    | 12.3%                    | 12.3%                    |
| throttling flags seen                       | none                     | none                     | none                     | none                     | none                     |
| test wall time                              | 60.5 min                 | 59.7 min                 | 59.7 min                 | 59.7 min                 | 59.7 min                 |
| model parameters                            | 474,512                  | 474,512                  | 474,512                  | 474,512                  | 474,512                  |
| model size                                  | 1.91 MB                  | 1.91 MB                  | 1.91 MB                  | 1.91 MB                  | 1.91 MB                  |
| model FLOPs per inference                   | 416 MFLOP                | 416 MFLOP                | 416 MFLOP                | 416 MFLOP                | 416 MFLOP                |
| effective GFLOP/s (FLOPs / mean infer time) | 6.26                     | 7.22                     | 7.22                     | 7.38                     | 7.38                     |

## Most frequent confusions

### hf_plus cautious (Oct 2)

**intent level:** TEMPERATURE -> REJECT (9); CREATE_REMINDER -> REJECT (8); PAUSE -> REJECT (6); BRIGHTNESS -> REJECT (5); TIMER -> REJECT (4); CALL -> REJECT (4); MESSAGE -> REJECT (3); ALARM -> REJECT (3); TIME -> REJECT (3); STOP -> REJECT (2)

**command level:** Temperature 22 degrees -> REJECT (2); Kill the lights -> REJECT (2); Pause -> REJECT (2); Pause audio -> REJECT (2); Pause song -> REJECT (2); Make a phone call -> REJECT (2); Remind me to Study -> REJECT (2); Stop playing -> REJECT (1); Set the temperature to 18 degrees -> REJECT (1); Timer 1 minute -> REJECT (1)

### hf_plus balanced (Oct 3)

**intent level:** BRIGHTNESS -> REJECT (3); TEMPERATURE -> REJECT (3); TIMER -> REJECT (3); ALARM -> REJECT (3); PAUSE -> REJECT (3); STOP -> PLAY_MUSIC (2); CREATE_REMINDER -> REJECT (2); VOLUME_DOWN -> VOLUME_UP (2); MESSAGE -> REJECT (2); TIME -> REJECT (2)

**command level:** Lower the volume -> Volume up (2); Pause song -> REJECT (2); Stop playing -> Play music (1); Adjust brightness to 20 percent -> REJECT (1); Reminder Study -> REJECT (1); Temperature 26 degrees -> REJECT (1); Timer 30 seconds -> REJECT (1); Temperature 18 degrees -> REJECT (1); Send my message -> REJECT (1); Adjust brightness to 100 percent -> REJECT (1)

### hf_plus cautious (Oct 3)

**intent level:** BRIGHTNESS -> REJECT (6); CREATE_REMINDER -> REJECT (6); TEMPERATURE -> REJECT (6); PAUSE -> REJECT (5); STOP -> REJECT (3); TIMER -> REJECT (3); MESSAGE -> REJECT (3); ALARM -> REJECT (3); VOLUME_DOWN -> VOLUME_UP (2); TIME -> REJECT (2)

**command level:** Stop playing -> REJECT (2); Reminder Study -> REJECT (2); Lower the volume -> Volume up (2); Pause -> REJECT (2); Pause song -> REJECT (2); Make a phone call -> REJECT (2); Adjust brightness to 20 percent -> REJECT (1); Reminder Exercise -> REJECT (1); Temperature 26 degrees -> REJECT (1); Timer 30 seconds -> REJECT (1)

### hf_only balanced (Oct 3)

**intent level:** TEMPERATURE -> REJECT (12); TIMER -> REJECT (9); BRIGHTNESS -> REJECT (9); CREATE_REMINDER -> REJECT (9); ALARM -> REJECT (7); CALL -> REJECT (6); COLOR -> REJECT (6); LIGHT_ON -> REJECT (6); WEATHER -> REJECT (5); MESSAGE -> REJECT (5)

**command level:** Make a call -> REJECT (2); Timer 1 minute -> REJECT (2); Weather -> REJECT (2); Alarm 9:00 PM -> REJECT (2); Lower the volume -> REJECT (2); Countdown for 1 minute -> REJECT (2); Temperature 22 degrees -> REJECT (2); Message -> REJECT (2); Turn on the lights -> REJECT (2); Send a message -> REJECT (2)

### hf_only cautious (Oct 3)

**intent level:** BRIGHTNESS -> REJECT (13); TIMER -> REJECT (13); CREATE_REMINDER -> REJECT (13); TEMPERATURE -> REJECT (12); COLOR -> REJECT (12); ALARM -> REJECT (10); CALL -> REJECT (6); WEATHER -> REJECT (6); MESSAGE -> REJECT (6); LIGHT_ON -> REJECT (6)

**command level:** Make a call -> REJECT (2); Adjust brightness to 100 percent -> REJECT (2); Set color to Blue -> REJECT (2); Timer 1 minute -> REJECT (2); Adjust brightness to 20 percent -> REJECT (2); Weather -> REJECT (2); Set an alarm for 8:00 AM -> REJECT (2); Reminder Study -> REJECT (2); Alarm 9:00 PM -> REJECT (2); Lower the volume -> REJECT (2)

## Per-intent scores

### n

| class           | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|-----------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| ALARM           | 18                       | 18                       | 18                       | 18                       | 18                       |
| BRIGHTNESS      | 18                       | 18                       | 18                       | 18                       | 18                       |
| CALL            | 6                        | 6                        | 6                        | 6                        | 6                        |
| COLOR           | 18                       | 18                       | 18                       | 18                       | 18                       |
| CREATE_REMINDER | 18                       | 18                       | 18                       | 18                       | 18                       |
| LIGHT_OFF       | 6                        | 6                        | 6                        | 6                        | 6                        |
| LIGHT_ON        | 6                        | 6                        | 6                        | 6                        | 6                        |
| LIST_REMINDERS  | 6                        | 6                        | 6                        | 6                        | 6                        |
| MESSAGE         | 6                        | 6                        | 6                        | 6                        | 6                        |
| NEXT            | 6                        | 6                        | 6                        | 6                        | 6                        |
| PAUSE           | 6                        | 6                        | 6                        | 6                        | 6                        |
| PLAY_MUSIC      | 6                        | 6                        | 6                        | 6                        | 6                        |
| REJECT          | 16                       | 16                       | 16                       | 16                       | 16                       |
| STOP            | 6                        | 6                        | 6                        | 6                        | 6                        |
| TEMPERATURE     | 18                       | 18                       | 18                       | 18                       | 18                       |
| TIME            | 6                        | 6                        | 6                        | 6                        | 6                        |
| TIMER           | 18                       | 18                       | 18                       | 18                       | 18                       |
| VOLUME_DOWN     | 6                        | 6                        | 6                        | 6                        | 6                        |
| VOLUME_UP       | 6                        | 6                        | 6                        | 6                        | 6                        |
| WEATHER         | 6                        | 6                        | 6                        | 6                        | 6                        |

### precision

| class           | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|-----------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| ALARM           | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   |
| BRIGHTNESS      | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   |
| CALL            | 100.0%                   | 100.0%                   | 100.0%                   | 0.0%                     | 0.0%                     |
| COLOR           | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   |
| CREATE_REMINDER | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   |
| LIGHT_OFF       | 75.0%                    | 85.7%                    | 100.0%                   | 66.7%                    | 0.0%                     |
| LIGHT_ON        | 100.0%                   | 100.0%                   | 100.0%                   | 0.0%                     | 0.0%                     |
| LIST_REMINDERS  | 100.0%                   | 100.0%                   | 100.0%                   | 85.7%                    | 100.0%                   |
| MESSAGE         | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   | 0.0%                     |
| NEXT            | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   |
| PAUSE           | 0.0%                     | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   |
| PLAY_MUSIC      | 83.3%                    | 71.4%                    | 83.3%                    | 100.0%                   | 100.0%                   |
| REJECT          | 20.5%                    | 36.6%                    | 25.4%                    | 13.3%                    | 11.0%                    |
| STOP            | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   |
| TEMPERATURE     | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   |
| TIME            | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   |
| TIMER           | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   |
| VOLUME_DOWN     | 83.3%                    | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   |
| VOLUME_UP       | 100.0%                   | 71.4%                    | 71.4%                    | 100.0%                   | 0.0%                     |
| WEATHER         | 100.0%                   | 100.0%                   | 100.0%                   | 100.0%                   | 0.0%                     |

### recall

| class           | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|-----------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| ALARM           | 83.3%                    | 83.3%                    | 83.3%                    | 61.1%                    | 44.4%                    |
| BRIGHTNESS      | 72.2%                    | 83.3%                    | 66.7%                    | 50.0%                    | 27.8%                    |
| CALL            | 33.3%                    | 83.3%                    | 66.7%                    | 0.0%                     | 0.0%                     |
| COLOR           | 94.4%                    | 94.4%                    | 94.4%                    | 66.7%                    | 33.3%                    |
| CREATE_REMINDER | 55.6%                    | 88.9%                    | 66.7%                    | 44.4%                    | 27.8%                    |
| LIGHT_OFF       | 50.0%                    | 100.0%                   | 66.7%                    | 33.3%                    | 0.0%                     |
| LIGHT_ON        | 83.3%                    | 100.0%                   | 83.3%                    | 0.0%                     | 0.0%                     |
| LIST_REMINDERS  | 83.3%                    | 100.0%                   | 83.3%                    | 100.0%                   | 100.0%                   |
| MESSAGE         | 50.0%                    | 66.7%                    | 50.0%                    | 16.7%                    | 0.0%                     |
| NEXT            | 83.3%                    | 100.0%                   | 83.3%                    | 83.3%                    | 50.0%                    |
| PAUSE           | 0.0%                     | 50.0%                    | 16.7%                    | 50.0%                    | 50.0%                    |
| PLAY_MUSIC      | 83.3%                    | 83.3%                    | 83.3%                    | 50.0%                    | 50.0%                    |
| REJECT          | 93.8%                    | 93.8%                    | 100.0%                   | 93.8%                    | 100.0%                   |
| STOP            | 50.0%                    | 50.0%                    | 33.3%                    | 50.0%                    | 33.3%                    |
| TEMPERATURE     | 50.0%                    | 83.3%                    | 66.7%                    | 33.3%                    | 33.3%                    |
| TIME            | 50.0%                    | 66.7%                    | 66.7%                    | 50.0%                    | 33.3%                    |
| TIMER           | 77.8%                    | 83.3%                    | 83.3%                    | 50.0%                    | 27.8%                    |
| VOLUME_DOWN     | 83.3%                    | 66.7%                    | 66.7%                    | 50.0%                    | 33.3%                    |
| VOLUME_UP       | 83.3%                    | 83.3%                    | 83.3%                    | 33.3%                    | 0.0%                     |
| WEATHER         | 66.7%                    | 100.0%                   | 100.0%                   | 16.7%                    | 0.0%                     |

### F1

| class           | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|-----------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| ALARM           | 90.9%                    | 90.9%                    | 90.9%                    | 75.9%                    | 61.5%                    |
| BRIGHTNESS      | 83.9%                    | 90.9%                    | 80.0%                    | 66.7%                    | 43.5%                    |
| CALL            | 50.0%                    | 90.9%                    | 80.0%                    | 0.0%                     | 0.0%                     |
| COLOR           | 97.1%                    | 97.1%                    | 97.1%                    | 80.0%                    | 50.0%                    |
| CREATE_REMINDER | 71.4%                    | 94.1%                    | 80.0%                    | 61.5%                    | 43.5%                    |
| LIGHT_OFF       | 60.0%                    | 92.3%                    | 80.0%                    | 44.4%                    | 0.0%                     |
| LIGHT_ON        | 90.9%                    | 100.0%                   | 90.9%                    | 0.0%                     | 0.0%                     |
| LIST_REMINDERS  | 90.9%                    | 100.0%                   | 90.9%                    | 92.3%                    | 100.0%                   |
| MESSAGE         | 66.7%                    | 80.0%                    | 66.7%                    | 28.6%                    | 0.0%                     |
| NEXT            | 90.9%                    | 100.0%                   | 90.9%                    | 90.9%                    | 66.7%                    |
| PAUSE           | 0.0%                     | 66.7%                    | 28.6%                    | 66.7%                    | 66.7%                    |
| PLAY_MUSIC      | 83.3%                    | 76.9%                    | 83.3%                    | 66.7%                    | 66.7%                    |
| REJECT          | 33.7%                    | 52.6%                    | 40.5%                    | 23.3%                    | 19.8%                    |
| STOP            | 66.7%                    | 66.7%                    | 50.0%                    | 66.7%                    | 50.0%                    |
| TEMPERATURE     | 66.7%                    | 90.9%                    | 80.0%                    | 50.0%                    | 50.0%                    |
| TIME            | 66.7%                    | 80.0%                    | 80.0%                    | 66.7%                    | 50.0%                    |
| TIMER           | 87.5%                    | 90.9%                    | 90.9%                    | 66.7%                    | 43.5%                    |
| VOLUME_DOWN     | 83.3%                    | 80.0%                    | 80.0%                    | 66.7%                    | 50.0%                    |
| VOLUME_UP       | 90.9%                    | 76.9%                    | 76.9%                    | 50.0%                    | 0.0%                     |
| WEATHER         | 80.0%                    | 100.0%                   | 100.0%                   | 28.6%                    | 0.0%                     |

### F2

| class           | hf_plus cautious (Oct 2) | hf_plus balanced (Oct 3) | hf_plus cautious (Oct 3) | hf_only balanced (Oct 3) | hf_only cautious (Oct 3) |
|-----------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| ALARM           | 86.2%                    | 86.2%                    | 86.2%                    | 66.3%                    | 50.0%                    |
| BRIGHTNESS      | 76.5%                    | 86.2%                    | 71.4%                    | 55.6%                    | 32.5%                    |
| CALL            | 38.5%                    | 86.2%                    | 71.4%                    | 0.0%                     | 0.0%                     |
| COLOR           | 95.5%                    | 95.5%                    | 95.5%                    | 71.4%                    | 38.5%                    |
| CREATE_REMINDER | 61.0%                    | 90.9%                    | 71.4%                    | 50.0%                    | 32.5%                    |
| LIGHT_OFF       | 53.6%                    | 96.8%                    | 71.4%                    | 37.0%                    | 0.0%                     |
| LIGHT_ON        | 86.2%                    | 100.0%                   | 86.2%                    | 0.0%                     | 0.0%                     |
| LIST_REMINDERS  | 86.2%                    | 100.0%                   | 86.2%                    | 96.8%                    | 100.0%                   |
| MESSAGE         | 55.6%                    | 71.4%                    | 55.6%                    | 20.0%                    | 0.0%                     |
| NEXT            | 86.2%                    | 100.0%                   | 86.2%                    | 86.2%                    | 55.6%                    |
| PAUSE           | 0.0%                     | 55.6%                    | 20.0%                    | 55.6%                    | 55.6%                    |
| PLAY_MUSIC      | 83.3%                    | 80.6%                    | 83.3%                    | 55.6%                    | 55.6%                    |
| REJECT          | 54.7%                    | 71.4%                    | 63.0%                    | 42.4%                    | 38.1%                    |
| STOP            | 55.6%                    | 55.6%                    | 38.5%                    | 55.6%                    | 38.5%                    |
| TEMPERATURE     | 55.6%                    | 86.2%                    | 71.4%                    | 38.5%                    | 38.5%                    |
| TIME            | 55.6%                    | 71.4%                    | 71.4%                    | 55.6%                    | 38.5%                    |
| TIMER           | 81.4%                    | 86.2%                    | 86.2%                    | 55.6%                    | 32.5%                    |
| VOLUME_DOWN     | 83.3%                    | 71.4%                    | 71.4%                    | 55.6%                    | 38.5%                    |
| VOLUME_UP       | 86.2%                    | 80.6%                    | 80.6%                    | 38.5%                    | 0.0%                     |
| WEATHER         | 71.4%                    | 100.0%                   | 100.0%                   | 20.0%                    | 0.0%                     |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
