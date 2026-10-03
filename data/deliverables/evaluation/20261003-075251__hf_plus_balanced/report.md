# VCM benchmark - Arvir Jane R. Redondo (hf_plus, balanced) - 20261003-075251

Wake word: **Hey Delta** - trials: 202 with the wake word + 16 without - shuffle seed: 79276 - connection: ssh - holdout: huggingface

Mic check (Pi input default): signal-to-noise 22.4 dB, laptop speech -33.8 dBFS, room noise -56.2 dBFS - ok

## At a glance

|                                   | overall     | real voice  | synthetic voice |
|-----------------------------------|-------------|-------------|-----------------|
| intent accuracy (19)              | 84.7%       | 74.0%       | 94.3%           |
| command accuracy (93)             | 84.2%       | 72.9%       | 94.3%           |
| false accept (out of scope fired) | 6.2% (1/16) | 0.0% (0/10) | 16.7% (1/6)     |
| false reject (command ignored)    | 14.0%       | 25.6%       | 4.0%            |
| false wake (no wake word, fired)  | 0.0% (0/16) | 0.0% (0/7)  | 0.0% (0/9)      |
| slot exact                        | 98.9%       | 97.0%       | 100.0%          |
| latency p95                       | 3.25 s      | 2.91 s      | 3.60 s          |

**Pi:** real-time factor 0.012 (p95 0.014), inference 58 ms, CPU temp max 57.9 C, runtime CPU 12% mean, runtime RAM 146 MB peak, 416 MFLOP per inference

# Detailed metrics

## Classification

| metric                                            | 19 intents (+reject) | 93 commands (+reject) |
|---------------------------------------------------|----------------------|-----------------------|
| accuracy                                          | 84.7%                | 84.2%                 |
| balanced accuracy                                 | 83.0%                | 83.4%                 |
| precision (macro)                                 | 93.3%                | 94.0%                 |
| recall (macro)                                    | 83.0%                | 83.4%                 |
| F1 (macro)                                        | 85.9%                | 86.4%                 |
| F2 (macro)                                        | 83.6%                | 84.2%                 |
| false accept rate (OOS fired)                     | 6.2%                 | 6.2%                  |
| false reject rate (in-scope silent/rejected)      | 14.0%                | 14.0%                 |
| misfire rate (wrong command fired)                | 2.2%                 | 2.7%                  |
| accuracy 95% CI                                   | [79-89%]             | [78-89%]              |
| false accept 95% CI                               | [1-28%] (1/16)       | [1-28%]               |
| false wake rate (command without wake word fired) | 0.0% [0-19%] (0/16)  | 0.0% [0-19%] (0/16)   |

Responses: 100.0% of trials fired a command; no response: 0; extra fires: 0; wake detect rate: 100.0%

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice     | synthetic voice |
|--------------------------------|----------------|----------------|-----------------|
| clips (with wake word)         | 202            | 96             | 106             |
| **19 intents** accuracy        | 84.7% [79-89%] | 74.0% [64-82%] | 94.3% [88-97%]  |
| balanced accuracy              | 83.0%          | 72.6%          | 92.0%           |
| F1 (macro)                     | 85.9%          | 77.6%          | 92.1%           |
| F2 (macro)                     | 83.6%          | 73.7%          | 91.8%           |
| false accept rate              | 6.2% (1/16)    | 0.0% (0/10)    | 16.7% (1/6)     |
| false reject rate              | 14.0%          | 25.6%          | 4.0%            |
| misfire rate                   | 2.2%           | 3.5%           | 1.0%            |
| **93 commands** accuracy       | 84.2%          | 72.9%          | 94.3%           |
| balanced accuracy              | 83.4%          | 70.1%          | 95.0%           |
| F1 (macro)                     | 86.4%          | 69.1%          | 94.3%           |
| F2 (macro)                     | 84.2%          | 69.6%          | 94.7%           |
| misfire rate                   | 2.7%           | 4.7%           | 1.0%            |
| slot exact (intent right)      | 98.9% (n=93)   | 97.0% (n=33)   | 100.0% (n=60)   |
| latency p50 / p95              | 1.45 / 3.25 s  | 1.43 / 2.91 s  | 1.46 / 3.60 s   |
| false wake rate (no wake word) | 0.0% (0/16)    | 0.0% (0/7)     | 0.0% (0/9)      |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n  | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 15 | 100.0% | 0.0 min        | 0.000          | 0.000         | 0.000     |
| BRIGHTNESS      | 15 | 100.0% | 0.0 %          | 0.000          | 0.543         | 0.539     |
| COLOR           | 17 | 100.0% | -              | -              | 0.000         | 0.000     |
| CREATE_REMINDER | 16 | 100.0% | -              | -              | 0.000         | 0.000     |
| TEMPERATURE     | 15 | 93.3%  | 0.3 deg        | 0.033          | 0.509         | 0.485     |
| TIMER           | 15 | 100.0% | 0.0 s          | 0.000          | 0.348         | 0.408     |
| ALL             | 93 | 98.9%  | -              | 0.008          | 0.226         | 0.231     |

## Raspberry Pi

- **Raspberry Pi 5 Model B Rev 1.1**, 4 cores  up to 2400.0 MHz, RAM 4049.1 MB, Debian GNU/Linux 13 (trixie), kernel 6.18.50+rpt-rpi-2712, Python 3.13.5
- packages: numpy 2.2.4, sounddevice 0.5.6

| metric                                      | mean / p95 / max         |
|---------------------------------------------|--------------------------|
| response latency (command end -> Pi output) | 1.616 / 3.252 / 5.670 s  |
| latency p50 / p99                           | 1.450 / 5.368 s          |
| inference time (Pi-reported)                | 57.7 / 72.1 / 95.3 ms    |
| real-time factor (infer / audio window)     | 0.012 / 0.014 / 0.019    |
| CPU temperature                             | 54.6 / 56.2 / 57.9 C     |
| CPU use, whole Pi                           | 10.8 / 26.9 / 35.9 %     |
| CPU use, your runtime process               | 12.3 / 18.9 / 25.8 %     |
| RAM (RSS), your runtime process             | 144.8 / 145.0 / 145.8 MB |
| RAM used, whole Pi                          | 601.6 / 625.5 / 665.4 MB |
| CPU clock                                   | 2213 / 2400 / 2400 MHz   |
| load average (1 min)                        | 0.60 / 0.81 / 1.18       |
| runtime CPU-seconds per second of speech    | 1.223                    |
| runtime CPU share of wall time              | 12.3%                    |
| throttling flags seen                       | none                     |
| test wall time                              | 59.7 min                 |
| model parameters                            | 474,512                  |
| model size                                  | 1.91 MB                  |
| model FLOPs per inference                   | 416 MFLOP                |
| effective GFLOP/s (FLOPs / mean infer time) | 7.22                     |

## Most frequent confusions

**intent level:** BRIGHTNESS -> REJECT (3); TEMPERATURE -> REJECT (3); TIMER -> REJECT (3); ALARM -> REJECT (3); PAUSE -> REJECT (3); STOP -> PLAY_MUSIC (2); CREATE_REMINDER -> REJECT (2); VOLUME_DOWN -> VOLUME_UP (2); MESSAGE -> REJECT (2); TIME -> REJECT (2)

**command level:** Lower the volume -> Volume up (2); Pause song -> REJECT (2); Stop playing -> Play music (1); Adjust brightness to 20 percent -> REJECT (1); Reminder Study -> REJECT (1); Temperature 26 degrees -> REJECT (1); Timer 30 seconds -> REJECT (1); Temperature 18 degrees -> REJECT (1); Send my message -> REJECT (1); Adjust brightness to 100 percent -> REJECT (1)

## Per-intent scores

| class           | n  | precision | recall | F1     | F2     |
|-----------------|----|-----------|--------|--------|--------|
| ALARM           | 18 | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| BRIGHTNESS      | 18 | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| CALL            | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| COLOR           | 18 | 100.0%    | 94.4%  | 97.1%  | 95.5%  |
| CREATE_REMINDER | 18 | 100.0%    | 88.9%  | 94.1%  | 90.9%  |
| LIGHT_OFF       | 6  | 85.7%     | 100.0% | 92.3%  | 96.8%  |
| LIGHT_ON        | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| LIST_REMINDERS  | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| MESSAGE         | 6  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| NEXT            | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| PAUSE           | 6  | 100.0%    | 50.0%  | 66.7%  | 55.6%  |
| PLAY_MUSIC      | 6  | 71.4%     | 83.3%  | 76.9%  | 80.6%  |
| REJECT          | 16 | 36.6%     | 93.8%  | 52.6%  | 71.4%  |
| STOP            | 6  | 100.0%    | 50.0%  | 66.7%  | 55.6%  |
| TEMPERATURE     | 18 | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| TIME            | 6  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| TIMER           | 18 | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| VOLUME_DOWN     | 6  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| VOLUME_UP       | 6  | 71.4%     | 83.3%  | 76.9%  | 80.6%  |
| WEATHER         | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
