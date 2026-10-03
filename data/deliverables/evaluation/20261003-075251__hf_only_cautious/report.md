# VCM benchmark - Arvir Jane R. Redondo (hf_only, cautious) - 20261003-075251

Wake word: **Hey Delta** - trials: 202 with the wake word + 16 without - shuffle seed: 79276 - connection: ssh - holdout: huggingface

Mic check (Pi input default): signal-to-noise 22.4 dB, laptop speech -33.8 dBFS, room noise -56.2 dBFS - ok

## At a glance

|                                   | overall     | real voice  | synthetic voice |
|-----------------------------------|-------------|-------------|-----------------|
| intent accuracy (19)              | 35.6%       | 18.8%       | 50.9%           |
| command accuracy (93)             | 35.6%       | 18.8%       | 50.9%           |
| false accept (out of scope fired) | 0.0% (0/16) | 0.0% (0/10) | 0.0% (0/6)      |
| false reject (command ignored)    | 69.9%       | 90.7%       | 52.0%           |
| false wake (no wake word, fired)  | 0.0% (0/16) | 0.0% (0/7)  | 0.0% (0/9)      |
| slot exact                        | 100.0%      | 100.0%      | 100.0%          |
| latency p95                       | 3.25 s      | 2.91 s      | 3.60 s          |

**Pi:** real-time factor 0.011 (p95 0.014), inference 56 ms, CPU temp max 57.9 C, runtime CPU 12% mean, runtime RAM 146 MB peak, 416 MFLOP per inference

# Detailed metrics

## Classification

| metric                                            | 19 intents (+reject) | 93 commands (+reject) |
|---------------------------------------------------|----------------------|-----------------------|
| accuracy                                          | 35.6%                | 35.6%                 |
| balanced accuracy                                 | 32.2%                | 30.9%                 |
| precision (macro)                                 | 65.5%                | 50.1%                 |
| recall (macro)                                    | 32.2%                | 30.9%                 |
| F1 (macro)                                        | 38.1%                | 36.7%                 |
| F2 (macro)                                        | 32.2%                | 32.4%                 |
| false accept rate (OOS fired)                     | 0.0%                 | 0.0%                  |
| false reject rate (in-scope silent/rejected)      | 69.9%                | 69.9%                 |
| misfire rate (wrong command fired)                | 0.0%                 | 0.0%                  |
| accuracy 95% CI                                   | [29-42%]             | [29-42%]              |
| false accept 95% CI                               | [0-19%] (0/16)       | [0-19%]               |
| false wake rate (command without wake word fired) | 0.0% [0-19%] (0/16)  | 0.0% [0-19%] (0/16)   |

Responses: 100.0% of trials fired a command; no response: 0; extra fires: 0; wake detect rate: 100.0%

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice     | synthetic voice |
|--------------------------------|----------------|----------------|-----------------|
| clips (with wake word)         | 202            | 96             | 106             |
| **19 intents** accuracy        | 35.6% [29-42%] | 18.8% [12-28%] | 50.9% [42-60%]  |
| balanced accuracy              | 32.2%          | 14.1%          | 48.8%           |
| F1 (macro)                     | 38.1%          | 12.2%          | 52.2%           |
| F2 (macro)                     | 32.2%          | 11.7%          | 48.1%           |
| false accept rate              | 0.0% (0/16)    | 0.0% (0/10)    | 0.0% (0/6)      |
| false reject rate              | 69.9%          | 90.7%          | 52.0%           |
| misfire rate                   | 0.0%           | 0.0%           | 0.0%            |
| **93 commands** accuracy       | 35.6%          | 18.8%          | 50.9%           |
| balanced accuracy              | 30.9%          | 10.3%          | 49.5%           |
| F1 (macro)                     | 36.7%          | 9.4%           | 49.1%           |
| F2 (macro)                     | 32.4%          | 9.6%           | 49.0%           |
| misfire rate                   | 0.0%           | 0.0%           | 0.0%            |
| slot exact (intent right)      | 100.0% (n=35)  | 100.0% (n=4)   | 100.0% (n=31)   |
| latency p50 / p95              | 1.45 / 3.25 s  | 1.43 / 2.91 s  | 1.46 / 3.60 s   |
| false wake rate (no wake word) | 0.0% (0/16)    | 0.0% (0/7)     | 0.0% (0/9)      |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n  | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 8  | 100.0% | 0.0 min        | 0.000          | 0.000         | 0.000     |
| BRIGHTNESS      | 5  | 100.0% | 0.0 %          | 0.000          | 0.566         | 0.577     |
| COLOR           | 6  | 100.0% | -              | -              | 0.000         | 0.000     |
| CREATE_REMINDER | 5  | 100.0% | -              | -              | 0.000         | 0.000     |
| TEMPERATURE     | 6  | 100.0% | 0.0 deg        | 0.000          | 0.505         | 0.472     |
| TIMER           | 5  | 100.0% | 0.0 s          | 0.000          | 0.467         | 0.475     |
| ALL             | 35 | 100.0% | -              | 0.000          | 0.234         | 0.231     |

## Raspberry Pi

- **Raspberry Pi 5 Model B Rev 1.1**, 4 cores  up to 2400.0 MHz, RAM 4049.1 MB, Debian GNU/Linux 13 (trixie), kernel 6.18.50+rpt-rpi-2712, Python 3.13.5
- packages: numpy 2.2.4, sounddevice 0.5.6

| metric                                      | mean / p95 / max         |
|---------------------------------------------|--------------------------|
| response latency (command end -> Pi output) | 1.616 / 3.252 / 5.670 s  |
| latency p50 / p99                           | 1.450 / 5.368 s          |
| inference time (Pi-reported)                | 56.4 / 71.5 / 90.7 ms    |
| real-time factor (infer / audio window)     | 0.011 / 0.014 / 0.018    |
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
| effective GFLOP/s (FLOPs / mean infer time) | 7.38                     |

## Most frequent confusions

**intent level:** BRIGHTNESS -> REJECT (13); TIMER -> REJECT (13); CREATE_REMINDER -> REJECT (13); TEMPERATURE -> REJECT (12); COLOR -> REJECT (12); ALARM -> REJECT (10); CALL -> REJECT (6); WEATHER -> REJECT (6); MESSAGE -> REJECT (6); LIGHT_ON -> REJECT (6)

**command level:** Make a call -> REJECT (2); Adjust brightness to 100 percent -> REJECT (2); Set color to Blue -> REJECT (2); Timer 1 minute -> REJECT (2); Adjust brightness to 20 percent -> REJECT (2); Weather -> REJECT (2); Set an alarm for 8:00 AM -> REJECT (2); Reminder Study -> REJECT (2); Alarm 9:00 PM -> REJECT (2); Lower the volume -> REJECT (2)

## Per-intent scores

| class           | n  | precision | recall | F1     | F2     |
|-----------------|----|-----------|--------|--------|--------|
| ALARM           | 18 | 100.0%    | 44.4%  | 61.5%  | 50.0%  |
| BRIGHTNESS      | 18 | 100.0%    | 27.8%  | 43.5%  | 32.5%  |
| CALL            | 6  | 0.0%      | 0.0%   | 0.0%   | 0.0%   |
| COLOR           | 18 | 100.0%    | 33.3%  | 50.0%  | 38.5%  |
| CREATE_REMINDER | 18 | 100.0%    | 27.8%  | 43.5%  | 32.5%  |
| LIGHT_OFF       | 6  | 0.0%      | 0.0%   | 0.0%   | 0.0%   |
| LIGHT_ON        | 6  | 0.0%      | 0.0%   | 0.0%   | 0.0%   |
| LIST_REMINDERS  | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| MESSAGE         | 6  | 0.0%      | 0.0%   | 0.0%   | 0.0%   |
| NEXT            | 6  | 100.0%    | 50.0%  | 66.7%  | 55.6%  |
| PAUSE           | 6  | 100.0%    | 50.0%  | 66.7%  | 55.6%  |
| PLAY_MUSIC      | 6  | 100.0%    | 50.0%  | 66.7%  | 55.6%  |
| REJECT          | 16 | 11.0%     | 100.0% | 19.8%  | 38.1%  |
| STOP            | 6  | 100.0%    | 33.3%  | 50.0%  | 38.5%  |
| TEMPERATURE     | 18 | 100.0%    | 33.3%  | 50.0%  | 38.5%  |
| TIME            | 6  | 100.0%    | 33.3%  | 50.0%  | 38.5%  |
| TIMER           | 18 | 100.0%    | 27.8%  | 43.5%  | 32.5%  |
| VOLUME_DOWN     | 6  | 100.0%    | 33.3%  | 50.0%  | 38.5%  |
| VOLUME_UP       | 6  | 0.0%      | 0.0%   | 0.0%   | 0.0%   |
| WEATHER         | 6  | 0.0%      | 0.0%   | 0.0%   | 0.0%   |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
