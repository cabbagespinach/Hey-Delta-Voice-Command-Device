# Class benchmark (HF test): HF data only vs HF + our data

Both: BC-ResNet-6, same recipe, validation = our filtered validation clips (never HF), cutoffs chosen on validation. Test = HF test split (4,418 clips; no test speaker in either model's training).

| | HF data only | HF + our data |
|---|---|---|
| best epoch | 6 | 43 |
| cutoffs (cautious / balanced) | 0.890 / 0.670 | 0.610 / 0.340 |

| rule | who | HF data only: correct / wrong / rejected | HF + our data: correct / wrong / rejected |
|---|---|---|---|
| argmax | classmates | 90.3% / 7.7% / 2.0% (3557 clips) | 95.7% / 1.2% / 3.1% (3557 clips) |
| argmax | web | 52.9% / 16.7% / 30.3% (814 clips) | 63.9% / 9.4% / 26.7% (814 clips) |
| argmax | non-commands that trigger an action (mean over groups) | 27.7% | 19.1% |
| balanced | classmates | 80.1% / 1.8% / 18.1% (3557 clips) | 95.6% / 1.1% / 3.3% (3557 clips) |
| balanced | web | 37.7% / 2.0% / 60.3% (814 clips) | 63.6% / 8.4% / 28.1% (814 clips) |
| balanced | non-commands that trigger an action (mean over groups) | 2.1% | 17.0% |
| cautious | classmates | 63.7% / 0.2% / 36.0% (3557 clips) | 92.9% / 0.5% / 6.5% (3557 clips) |
| cautious | web | 25.4% / 0.3% / 74.3% (814 clips) | 57.8% / 3.7% / 38.5% (814 clips) |
| cautious | non-commands that trigger an action (mean over groups) | 0.0% | 8.5% |

## Per class, argmax accuracy on HF test

| class                       |   HF data only |   HF + our data |   difference |
|:----------------------------|---------------:|----------------:|-------------:|
| ALARM_6AM                   |           95   |            97.9 |          2.8 |
| ALARM_8AM                   |           97.9 |            96.5 |         -1.4 |
| ALARM_9PM                   |           86.5 |            90.8 |          4.3 |
| BRIGHTNESS_100              |           92.2 |            97.2 |          5   |
| BRIGHTNESS_20               |           84.4 |            96.5 |         12.1 |
| BRIGHTNESS_60               |           97.2 |            99.3 |          2.1 |
| CALL                        |           64.5 |            85.1 |         20.6 |
| COLOR_BLUE                  |           90.1 |            89.4 |         -0.7 |
| COLOR_GREEN                 |           88.7 |            91.5 |          2.8 |
| COLOR_RED                   |           73   |            96.5 |         23.4 |
| CREATE_REMINDER_DRINK_WATER |           96.5 |           100   |          3.5 |
| CREATE_REMINDER_EXERCISE    |           96.5 |            99.3 |          2.8 |
| CREATE_REMINDER_STUDY       |           95.7 |           100   |          4.3 |
| LIGHT_OFF                   |           70.9 |            78   |          7.1 |
| LIGHT_ON                    |           39   |            72.3 |         33.3 |
| LIST_REMINDERS              |           95   |            92.2 |         -2.8 |
| MESSAGE                     |           90.1 |            90.8 |          0.7 |
| NEXT                        |           92.2 |            85.1 |         -7.1 |
| PAUSE                       |           84.4 |            86.5 |          2.1 |
| PLAY_MUSIC                  |           58.2 |            58.9 |          0.7 |
| STOP                        |           83.7 |            74.5 |         -9.2 |
| TEMPERATURE_18              |           94.3 |            97.2 |          2.8 |
| TEMPERATURE_22              |           90.1 |            95   |          5   |
| TEMPERATURE_26              |           94.3 |            98.6 |          4.3 |
| TIME                        |           75.9 |            80.1 |          4.3 |
| TIMER_10S                   |           91.5 |            96.5 |          5   |
| TIMER_1MIN                  |           87.9 |            95.7 |          7.8 |
| TIMER_30S                   |           98.6 |            96.5 |         -2.1 |
| VOLUME_DOWN                 |           52.5 |            68.8 |         16.3 |
| VOLUME_UP                   |           60.3 |            76.6 |         16.3 |
| WEATHER                     |           53.2 |            70.9 |         17.7 |
| unknown                     |           72.3 |            80.9 |          8.5 |
