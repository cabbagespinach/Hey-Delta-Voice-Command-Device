# VCM benchmark - Arvir Jane R. Redondo (hf_plus, cautious) - 20261002-135144

Wake word: **Hey Delta** - trials: 202 with the wake word + 16 without - shuffle seed: 79276 - connection: ssh - holdout: huggingface

## At a glance

|                                   | overall     | real voice  | synthetic voice |
|-----------------------------------|-------------|-------------|-----------------|
| intent accuracy (19)              | 69.8%       | 52.1%       | 85.8%           |
| command accuracy (93)             | 69.3%       | 51.0%       | 85.8%           |
| false accept (out of scope fired) | 6.2% (1/16) | 0.0% (0/10) | 16.7% (1/6)     |
| false reject (command ignored)    | 31.2%       | 52.3%       | 13.0%           |
| false wake (no wake word, fired)  | 0.0% (0/16) | 0.0% (0/7)  | 0.0% (0/9)      |
| slot exact                        | 98.7%       | 95.0%       | 100.0%          |
| latency p95                       | 4.93 s      | 4.88 s      | 4.98 s          |

**Pi:** real-time factor 0.013 (p95 0.014), inference 67 ms, CPU temp max 57.3 C, runtime CPU 13% mean, runtime RAM 111 MB peak, 416 MFLOP per inference

# Detailed metrics

## Classification

| metric                                            | 19 intents (+reject) | 93 commands (+reject) |
|---------------------------------------------------|----------------------|-----------------------|
| accuracy                                          | 69.8%                | 69.3%                 |
| balanced accuracy                                 | 66.4%                | 67.5%                 |
| precision (macro)                                 | 88.1%                | 88.9%                 |
| recall (macro)                                    | 66.4%                | 67.5%                 |
| F1 (macro)                                        | 72.6%                | 74.0%                 |
| F2 (macro)                                        | 67.6%                | 69.4%                 |
| false accept rate (OOS fired)                     | 6.2%                 | 6.2%                  |
| false reject rate (in-scope silent/rejected)      | 31.2%                | 31.2%                 |
| misfire rate (wrong command fired)                | 1.1%                 | 1.6%                  |
| accuracy 95% CI                                   | [63-76%]             | [63-75%]              |
| false accept 95% CI                               | [1-28%] (1/16)       | [1-28%]               |
| false wake rate (command without wake word fired) | 0.0% [0-19%] (0/16)  | 0.0% [0-19%] (0/16)   |

Responses: 98.0% of trials fired a command; no response: 4; extra fires: 0; wake detect rate: 98.5%

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice     | synthetic voice |
|--------------------------------|----------------|----------------|-----------------|
| clips (with wake word)         | 202            | 96             | 106             |
| **19 intents** accuracy        | 69.8% [63-76%] | 52.1% [42-62%] | 85.8% [78-91%]  |
| balanced accuracy              | 66.4%          | 51.5%          | 79.2%           |
| F1 (macro)                     | 72.6%          | 55.0%          | 81.1%           |
| F2 (macro)                     | 67.6%          | 51.5%          | 79.4%           |
| false accept rate              | 6.2% (1/16)    | 0.0% (0/10)    | 16.7% (1/6)     |
| false reject rate              | 31.2%          | 52.3%          | 13.0%           |
| misfire rate                   | 1.1%           | 1.2%           | 1.0%            |
| **93 commands** accuracy       | 69.3%          | 51.0%          | 85.8%           |
| balanced accuracy              | 67.5%          | 46.0%          | 85.5%           |
| F1 (macro)                     | 74.0%          | 44.8%          | 84.8%           |
| F2 (macro)                     | 69.4%          | 45.2%          | 85.1%           |
| misfire rate                   | 1.6%           | 2.3%           | 1.0%            |
| slot exact (intent right)      | 98.7% (n=78)   | 95.0% (n=20)   | 100.0% (n=58)   |
| latency p50 / p95              | 1.42 / 4.93 s  | 1.40 / 4.88 s  | 1.45 / 4.98 s   |
| false wake rate (no wake word) | 0.0% (0/16)    | 0.0% (0/7)     | 0.0% (0/9)      |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n  | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 15 | 100.0% | 0.0 min        | 0.000          | 0.000         | 0.000     |
| BRIGHTNESS      | 13 | 100.0% | 0.0 %          | 0.000          | 0.547         | 0.542     |
| COLOR           | 17 | 94.1%  | -              | -              | 0.039         | 0.035     |
| CREATE_REMINDER | 10 | 100.0% | -              | -              | 0.000         | 0.000     |
| TEMPERATURE     | 9  | 100.0% | 0.0 deg        | 0.000          | 0.472         | 0.463     |
| TIMER           | 14 | 100.0% | 0.0 s          | 0.000          | 0.357         | 0.408     |
| ALL             | 78 | 98.7%  | -              | 0.000          | 0.218         | 0.225     |

## Raspberry Pi

- **Raspberry Pi 5 Model B Rev 1.1**, 4 cores  up to 2400.0 MHz, RAM 4049.1 MB, Debian GNU/Linux 13 (trixie), kernel 6.18.50+rpt-rpi-2712, Python 3.13.5
- packages: numpy 2.2.4, sounddevice 0.5.6

| metric                                      | mean / p95 / max         |
|---------------------------------------------|--------------------------|
| response latency (command end -> Pi output) | 1.783 / 4.935 / 5.569 s  |
| latency p50 / p99                           | 1.423 / 5.458 s          |
| inference time (Pi-reported)                | 66.5 / 71.9 / 75.0 ms    |
| real-time factor (infer / audio window)     | 0.013 / 0.014 / 0.015    |
| CPU temperature                             | 54.5 / 55.6 / 57.3 C     |
| CPU use, whole Pi                           | 5.2 / 7.8 / 13.4 %       |
| CPU use, your runtime process               | 12.8 / 18.9 / 22.8 %     |
| RAM (RSS), your runtime process             | 108.7 / 109.8 / 110.7 MB |
| RAM used, whole Pi                          | 648.4 / 653.9 / 666.6 MB |
| CPU clock                                   | 1850 / 2400 / 2400 MHz   |
| load average (1 min)                        | 0.23 / 0.43 / 0.79       |
| runtime CPU-seconds per second of speech    | 1.280                    |
| runtime CPU share of wall time              | 12.7%                    |
| throttling flags seen                       | none                     |
| test wall time                              | 60.5 min                 |
| model parameters                            | 474,512                  |
| model size                                  | 1.91 MB                  |
| model FLOPs per inference                   | 416 MFLOP                |
| effective GFLOP/s (FLOPs / mean infer time) | 6.26                     |

## Most frequent confusions

**intent level:** TEMPERATURE -> REJECT (9); CREATE_REMINDER -> REJECT (8); PAUSE -> REJECT (6); BRIGHTNESS -> REJECT (5); TIMER -> REJECT (4); CALL -> REJECT (4); MESSAGE -> REJECT (3); ALARM -> REJECT (3); TIME -> REJECT (3); STOP -> REJECT (2)

**command level:** Temperature 22 degrees -> REJECT (2); Kill the lights -> REJECT (2); Pause -> REJECT (2); Pause audio -> REJECT (2); Pause song -> REJECT (2); Make a phone call -> REJECT (2); Remind me to Study -> REJECT (2); Stop playing -> REJECT (1); Set the temperature to 18 degrees -> REJECT (1); Timer 1 minute -> REJECT (1)

## Per-intent scores

| class           | n  | precision | recall | F1    | F2    |
|-----------------|----|-----------|--------|-------|-------|
| ALARM           | 18 | 100.0%    | 83.3%  | 90.9% | 86.2% |
| BRIGHTNESS      | 18 | 100.0%    | 72.2%  | 83.9% | 76.5% |
| CALL            | 6  | 100.0%    | 33.3%  | 50.0% | 38.5% |
| COLOR           | 18 | 100.0%    | 94.4%  | 97.1% | 95.5% |
| CREATE_REMINDER | 18 | 100.0%    | 55.6%  | 71.4% | 61.0% |
| LIGHT_OFF       | 6  | 75.0%     | 50.0%  | 60.0% | 53.6% |
| LIGHT_ON        | 6  | 100.0%    | 83.3%  | 90.9% | 86.2% |
| LIST_REMINDERS  | 6  | 100.0%    | 83.3%  | 90.9% | 86.2% |
| MESSAGE         | 6  | 100.0%    | 50.0%  | 66.7% | 55.6% |
| NEXT            | 6  | 100.0%    | 83.3%  | 90.9% | 86.2% |
| PAUSE           | 6  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| PLAY_MUSIC      | 6  | 83.3%     | 83.3%  | 83.3% | 83.3% |
| REJECT          | 16 | 20.5%     | 93.8%  | 33.7% | 54.7% |
| STOP            | 6  | 100.0%    | 50.0%  | 66.7% | 55.6% |
| TEMPERATURE     | 18 | 100.0%    | 50.0%  | 66.7% | 55.6% |
| TIME            | 6  | 100.0%    | 50.0%  | 66.7% | 55.6% |
| TIMER           | 18 | 100.0%    | 77.8%  | 87.5% | 81.4% |
| VOLUME_DOWN     | 6  | 83.3%     | 83.3%  | 83.3% | 83.3% |
| VOLUME_UP       | 6  | 100.0%    | 83.3%  | 90.9% | 86.2% |
| WEATHER         | 6  | 100.0%    | 66.7%  | 80.0% | 71.4% |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
