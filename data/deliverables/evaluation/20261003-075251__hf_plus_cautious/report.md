# VCM benchmark - Arvir Jane R. Redondo (hf_plus, cautious) - 20261003-075251

Wake word: **Hey Delta** - trials: 202 with the wake word + 16 without - shuffle seed: 79276 - connection: ssh - holdout: huggingface

Mic check (Pi input default): signal-to-noise 22.4 dB, laptop speech -33.8 dBFS, room noise -56.2 dBFS - ok

## At a glance

|                                   | overall     | real voice  | synthetic voice |
|-----------------------------------|-------------|-------------|-----------------|
| intent accuracy (19)              | 75.2%       | 60.4%       | 88.7%           |
| command accuracy (93)             | 75.2%       | 60.4%       | 88.7%           |
| false accept (out of scope fired) | 0.0% (0/16) | 0.0% (0/10) | 0.0% (0/6)      |
| false reject (command ignored)    | 25.3%       | 41.9%       | 11.0%           |
| false wake (no wake word, fired)  | 0.0% (0/16) | 0.0% (0/7)  | 0.0% (0/9)      |
| slot exact                        | 100.0%      | 100.0%      | 100.0%          |
| latency p95                       | 3.25 s      | 2.91 s      | 3.60 s          |

**Pi:** real-time factor 0.012 (p95 0.014), inference 58 ms, CPU temp max 57.9 C, runtime CPU 12% mean, runtime RAM 146 MB peak, 416 MFLOP per inference

# Detailed metrics

## Classification

| metric                                            | 19 intents (+reject) | 93 commands (+reject) |
|---------------------------------------------------|----------------------|-----------------------|
| accuracy                                          | 75.2%                | 75.2%                 |
| balanced accuracy                                 | 72.2%                | 73.4%                 |
| precision (macro)                                 | 94.0%                | 90.7%                 |
| recall (macro)                                    | 72.2%                | 73.4%                 |
| F1 (macro)                                        | 77.9%                | 78.6%                 |
| F2 (macro)                                        | 73.4%                | 74.9%                 |
| false accept rate (OOS fired)                     | 0.0%                 | 0.0%                  |
| false reject rate (in-scope silent/rejected)      | 25.3%                | 25.3%                 |
| misfire rate (wrong command fired)                | 1.6%                 | 1.6%                  |
| accuracy 95% CI                                   | [69-81%]             | [69-81%]              |
| false accept 95% CI                               | [0-19%] (0/16)       | [0-19%]               |
| false wake rate (command without wake word fired) | 0.0% [0-19%] (0/16)  | 0.0% [0-19%] (0/16)   |

Responses: 100.0% of trials fired a command; no response: 0; extra fires: 0; wake detect rate: 100.0%

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice     | synthetic voice |
|--------------------------------|----------------|----------------|-----------------|
| clips (with wake word)         | 202            | 96             | 106             |
| **19 intents** accuracy        | 75.2% [69-81%] | 60.4% [50-70%] | 88.7% [81-93%]  |
| balanced accuracy              | 72.2%          | 60.1%          | 82.3%           |
| F1 (macro)                     | 77.9%          | 65.4%          | 83.8%           |
| F2 (macro)                     | 73.4%          | 60.9%          | 82.3%           |
| false accept rate              | 0.0% (0/16)    | 0.0% (0/10)    | 0.0% (0/6)      |
| false reject rate              | 25.3%          | 41.9%          | 11.0%           |
| misfire rate                   | 1.6%           | 2.3%           | 1.0%            |
| **93 commands** accuracy       | 75.2%          | 60.4%          | 88.7%           |
| balanced accuracy              | 73.4%          | 56.3%          | 87.8%           |
| F1 (macro)                     | 78.6%          | 55.2%          | 87.1%           |
| F2 (macro)                     | 74.9%          | 55.6%          | 87.4%           |
| misfire rate                   | 1.6%           | 2.3%           | 1.0%            |
| slot exact (intent right)      | 100.0% (n=83)  | 100.0% (n=24)  | 100.0% (n=59)   |
| latency p50 / p95              | 1.45 / 3.25 s  | 1.43 / 2.91 s  | 1.46 / 3.60 s   |
| false wake rate (no wake word) | 0.0% (0/16)    | 0.0% (0/7)     | 0.0% (0/9)      |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n  | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 15 | 100.0% | 0.0 min        | 0.000          | 0.000         | 0.000     |
| BRIGHTNESS      | 12 | 100.0% | 0.0 %          | 0.000          | 0.543         | 0.536     |
| COLOR           | 17 | 100.0% | -              | -              | 0.000         | 0.000     |
| CREATE_REMINDER | 12 | 100.0% | -              | -              | 0.000         | 0.000     |
| TEMPERATURE     | 12 | 100.0% | 0.0 deg        | 0.000          | 0.473         | 0.463     |
| TIMER           | 15 | 100.0% | 0.0 s          | 0.000          | 0.348         | 0.408     |
| ALL             | 83 | 100.0% | -              | 0.000          | 0.210         | 0.218     |

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

**intent level:** BRIGHTNESS -> REJECT (6); CREATE_REMINDER -> REJECT (6); TEMPERATURE -> REJECT (6); PAUSE -> REJECT (5); STOP -> REJECT (3); TIMER -> REJECT (3); MESSAGE -> REJECT (3); ALARM -> REJECT (3); VOLUME_DOWN -> VOLUME_UP (2); TIME -> REJECT (2)

**command level:** Stop playing -> REJECT (2); Reminder Study -> REJECT (2); Lower the volume -> Volume up (2); Pause -> REJECT (2); Pause song -> REJECT (2); Make a phone call -> REJECT (2); Adjust brightness to 20 percent -> REJECT (1); Reminder Exercise -> REJECT (1); Temperature 26 degrees -> REJECT (1); Timer 30 seconds -> REJECT (1)

## Per-intent scores

| class           | n  | precision | recall | F1     | F2     |
|-----------------|----|-----------|--------|--------|--------|
| ALARM           | 18 | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| BRIGHTNESS      | 18 | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| CALL            | 6  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| COLOR           | 18 | 100.0%    | 94.4%  | 97.1%  | 95.5%  |
| CREATE_REMINDER | 18 | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| LIGHT_OFF       | 6  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| LIGHT_ON        | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| LIST_REMINDERS  | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| MESSAGE         | 6  | 100.0%    | 50.0%  | 66.7%  | 55.6%  |
| NEXT            | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| PAUSE           | 6  | 100.0%    | 16.7%  | 28.6%  | 20.0%  |
| PLAY_MUSIC      | 6  | 83.3%     | 83.3%  | 83.3%  | 83.3%  |
| REJECT          | 16 | 25.4%     | 100.0% | 40.5%  | 63.0%  |
| STOP            | 6  | 100.0%    | 33.3%  | 50.0%  | 38.5%  |
| TEMPERATURE     | 18 | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| TIME            | 6  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| TIMER           | 18 | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| VOLUME_DOWN     | 6  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| VOLUME_UP       | 6  | 71.4%     | 83.3%  | 76.9%  | 80.6%  |
| WEATHER         | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
