# Small-scale test: which commands need more real recordings?

Validation split only. `runs/probe_all` = all data; `runs/probe_no_owner` = no owner recordings or anything derived from them.

| | all data | without owner data |
|---|---|---|
| owner_macro_acc | 0.451 | 0.226 |
| web_macro_acc | 0.261 | 0.258 |
| synthetic_macro_acc | 0.582 | 0.639 |
| unknown_false_action | 0.296 | 0.289 |
| selection_score | 0.499 | 0.459 |

Owner `unknown` validation clips taken for a command: all data 38% (8 clips), without owner data 25%.

## Per command, owner validation session (sorted by accuracy without owner data)

| class              |   clips |   acc_all |   acc_no_owner |   drop |
|:-------------------|--------:|----------:|---------------:|-------:|
| lights_off         |       4 |      0    |           0    |   0    |
| pause              |       4 |      0    |           0    |   0    |
| set_timer_10       |       4 |      0    |           0    |   0    |
| time               |       4 |      0    |           0    |   0    |
| volume_down        |       4 |      0    |           0    |   0    |
| volume_up          |       5 |      0    |           0    |   0    |
| weather            |       4 |      0    |           0    |   0    |
| dim_lights_20      |       4 |      0.25 |           0    |   0.25 |
| set_temperature_22 |       4 |      0.5  |           0    |   0.5  |
| set_temperature_26 |       4 |      0.5  |           0    |   0.5  |
| set_alarm_6am      |       3 |      0.67 |           0    |   0.67 |
| play_music         |       4 |      0.75 |           0    |   0.75 |
| stop               |       4 |      0.75 |           0    |   0.75 |
| remind_study       |       3 |      1    |           0    |   1    |
| dim_lights_50      |       4 |      0    |           0.25 |  -0.25 |
| call_jane          |       4 |      0.25 |           0.25 |   0    |
| lights_on          |       4 |      0.5  |           0.25 |   0.25 |
| set_temperature_18 |       4 |      0.75 |           0.25 |   0.5  |
| list_reminders     |       4 |      1    |           0.25 |   0.75 |
| remind_trash       |       4 |      1    |           0.25 |   0.75 |
| next               |       7 |      0    |           0.43 |  -0.43 |
| set_timer_1        |       4 |      0.25 |           0.5  |  -0.25 |
| dim_lights_80      |       4 |      1    |           0.5  |   0.5  |
| text_jane          |       4 |      1    |           0.5  |   0.5  |
| set_alarm_7am      |       3 |      1    |           0.67 |   0.33 |
| unknown            |       8 |      0.62 |           0.75 |  -0.12 |
| set_alarm_9pm      |       3 |      0.33 |           1    |  -0.67 |
| set_timer_5        |       3 |      0.67 |           1    |  -0.33 |

## Most frequent confusions (all data)

|                                              |   clips |
|:---------------------------------------------|--------:|
| ('next', 'unknown')                          |       6 |
| ('time', 'unknown')                          |       4 |
| ('pause', 'unknown')                         |       4 |
| ('weather', 'unknown')                       |       4 |
| ('call_jane', 'unknown')                     |       3 |
| ('dim_lights_50', 'dim_lights_80')           |       3 |
| ('lights_off', 'play')                       |       2 |
| ('lights_off', 'unknown')                    |       2 |
| ('volume_up', 'call_jane')                   |       2 |
| ('set_timer_10', 'set_timer_1')              |       2 |
| ('lights_on', 'party')                       |       2 |
| ('set_temperature_22', 'set_temperature_18') |       2 |
| ('set_timer_1', 'set_alarm_9pm')             |       2 |
| ('dim_lights_20', 'dim_lights_50')           |       1 |
| ('dim_lights_20', 'dim_lights_80')           |       1 |

## Most frequent confusions (without owner data)

|                                   |   clips |
|:----------------------------------|--------:|
| ('next', 'unknown')               |       4 |
| ('pause', 'unknown')              |       4 |
| ('lights_off', 'unknown')         |       4 |
| ('stop', 'unknown')               |       4 |
| ('time', 'unknown')               |       4 |
| ('set_temperature_22', 'unknown') |       4 |
| ('play_music', 'unknown')         |       4 |
| ('remind_trash', 'unknown')       |       3 |
| ('set_temperature_18', 'unknown') |       3 |
| ('weather', 'unknown')            |       3 |
| ('set_temperature_26', 'unknown') |       3 |
| ('list_reminders', 'unknown')     |       3 |
| ('volume_down', 'unknown')        |       3 |
| ('dim_lights_20', 'unknown')      |       2 |
| ('call_jane', 'unknown')          |       2 |
