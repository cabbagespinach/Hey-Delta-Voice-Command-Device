# Pi A/B captures re-scored (balanced rule, each model's validation cutoffs)

correct / WRONG (wrong command) / asks again, over command captures; `non-cmd` = non-commands that made it act.

| condition | n | bcresnet6 | bcresnet6_ownernoise | non-cmd -> action |
|---|---|---|---|---|
| trained wording, quiet (you, talker #2, your voice message) | 82 | 82 / 0 / 0 | 81 / 0 / 1 | 0 0 |
| TV talk show | 33 | 23 / 0 / 10 | 25 / 0 / 8 | 0 0 |
| reworded commands, aircon | 93 | 75 / 3 / 15 | 69 / 1 / 23 | 1 1 |
|   ... wordings now in training | 47 | 33 / 2 / 12 | 28 / 1 / 18 | 0 0 |
|   ... HELD-OUT wordings (never trained) | 24 | 20 / 1 / 3 | 19 / 0 / 5 | 0 0 |
|   ... original trained wording | 22 | 22 / 0 / 0 | 22 / 0 / 0 | 1 1 |
| voice message, talker #3 | 7 | 1 / 0 / 6 | 0 / 0 / 7 | 0 0 |
| ALL | 215 | 181 / 3 / 31 | 175 / 1 / 39 | 1 1 |

## Every capture where the models differ (balanced)

| session | said | expected | bcresnet6 | bcresnet6_ownernoise |
|---|---|---|---|---|
| 125842 | Dim lights to 50% | dim_lights_50 | dim_lights_50 | unknown |
| 131935 | Party party | party | party | unknown |
| 131935 | Turn off lights | lights_off | unknown | lights_off |
| 131935 | Set temperature to 22 degrees | set_temperature_22 | set_temperature_22 | unknown |
| 131935 | What time is it? | time | unknown | time |
| 131935 | Text Jane | text_jane | unknown | text_jane |
| 131935 | Play | play | unknown | play |
| 132809 | Timer for one minute | set_timer_1 | set_timer_1 | unknown |
| 132809 | Turn the lights off | lights_off | lights_off | unknown |
| 132809 | Keep playing | play | unknown | play |
| 132809 | Pause it | pause | call_jane | pause |
| 132809 | Set a five minute timer | set_timer_5 | set_timer_5 | unknown |
| 132809 | What's the time now? | time | time | unknown |
| 132809 | Next song | next | unknown | next |
| 132809 | Wake me up at six A M | set_alarm_6am | set_alarm_6am | unknown |
| 132809 | Make it twenty two degrees | set_temperature_22 | set_temperature_22 | unknown |
| 132809 | Quieter | volume_down | volume_down | unknown |
| 132809 | (say something that is NOT a command) | unknown | party | volume_up |
| 132809 | What's the time? | time | time | unknown |
| 132809 | Lights on | lights_on | lights_on | unknown |
| 132809 | Send a message to Jane | text_jane | set_temperature_26 | unknown |
| 134101 | Skip | next | next | unknown |