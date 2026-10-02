# Phrasing test: other ways of saying each command

2340 clips: 121 phrasings x 20 TEST-split synthetic voices (6 Philippine-accent edge-tts, the rest Piper), assembled like captures. `trained` = the phrasing in the training data (same wording, voices never heard). 80 clips dropped because Whisper did not hear the phrasing.

## bcresnet6 (balanced cutoff 0.345; cautious 0.885)

|                  |   clips | correct   | wrong command   | rejected   | wrong as                                                                                                                |
|:-----------------|--------:|:----------|:----------------|:-----------|:------------------------------------------------------------------------------------------------------------------------|
| other phrasings  |    1761 | 73%       | 3%              | 24%        | call_jane, lights_off, next, pause, play, play_music, set_alarm_6am, set_temperature_26, stop, time, volume_up, weather |
| trained phrasing |     579 | 100%      | 0%              | 0%         |                                                                                                                         |

Per class, other phrasings only (worst first):

| class              |   clips | trained phrasing correct   | correct   | wrong command   | rejected   | wrong as                          |
|:-------------------|--------:|:---------------------------|:----------|:----------------|:-----------|:----------------------------------|
| text_jane          |      58 | 100%                       | 28%       | 9%              | 64%        | set_alarm_6am, set_temperature_26 |
| play               |      58 | 100%                       | 34%       | 2%              | 64%        | play_music                        |
| list_reminders     |      59 | 100%                       | 36%       | 0%              | 64%        |                                   |
| set_temperature_18 |      60 | 100%                       | 45%       | 0%              | 55%        |                                   |
| party              |      57 | 100%                       | 56%       | 21%             | 23%        | call_jane, time, volume_up        |
| time               |      60 | 100%                       | 57%       | 2%              | 42%        | weather                           |
| call_jane          |      52 | 100%                       | 60%       | 0%              | 40%        |                                   |
| next               |      78 | 100%                       | 63%       | 5%              | 32%        | lights_off                        |
| set_timer_1        |      60 | 100%                       | 65%       | 0%              | 35%        |                                   |
| stop               |      40 | 100%                       | 65%       | 10%             | 25%        | play                              |
| weather            |      60 | 100%                       | 65%       | 2%              | 33%        | call_jane                         |
| set_timer_5        |      60 | 100%                       | 67%       | 0%              | 33%        |                                   |
| set_timer_10       |      60 | 100%                       | 67%       | 0%              | 33%        |                                   |
| volume_down        |      73 | 100%                       | 67%       | 7%              | 26%        | call_jane, lights_off, volume_up  |
| set_alarm_6am      |      60 | 100%                       | 73%       | 0%              | 27%        |                                   |
| play_music         |      60 | 100%                       | 73%       | 20%             | 7%         | pause, stop                       |
| volume_up          |      78 | 100%                       | 74%       | 0%              | 26%        |                                   |
| set_alarm_7am      |      60 | 100%                       | 75%       | 0%              | 25%        |                                   |
| set_temperature_26 |      60 | 100%                       | 77%       | 0%              | 23%        |                                   |
| pause              |      33 | 100%                       | 79%       | 9%              | 12%        | call_jane, next                   |
| set_alarm_9pm      |      60 | 100%                       | 88%       | 0%              | 12%        |                                   |
| remind_study       |      37 | 100%                       | 89%       | 0%              | 11%        |                                   |
| set_temperature_22 |      60 | 100%                       | 95%       | 0%              | 5%         |                                   |
| lights_on          |     100 | 100%                       | 97%       | 3%              | 0%         | lights_off                        |
| dim_lights_50      |      59 | 100%                       | 98%       | 0%              | 2%         |                                   |
| dim_lights_20      |      60 | 100%                       | 100%      | 0%              | 0%         |                                   |
| dim_lights_80      |      59 | 100%                       | 100%      | 0%              | 0%         |                                   |
| lights_off         |     100 | 100%                       | 100%      | 0%              | 0%         |                                   |
| remind_trash       |      40 | 100%                       | 100%      | 0%              | 0%         |                                   |

Per phrasing:

| class              | phrasing                                  | trained   |   clips | correct   | wrong command   | rejected   | wrong as                          |
|:-------------------|:------------------------------------------|:----------|--------:|:----------|:----------------|:-----------|:----------------------------------|
| lights_on          | Turn on lights                            | True      |      20 | 100%      | 0%              | 0%         |                                   |
| lights_on          | Turn on the lights                        | False     |      20 | 100%      | 0%              | 0%         |                                   |
| lights_on          | Turn the lights on                        | False     |      20 | 95%       | 5%              | 0%         | lights_off                        |
| lights_on          | Switch on the lights                      | False     |      20 | 100%      | 0%              | 0%         |                                   |
| lights_on          | Switch the lights on                      | False     |      20 | 90%       | 10%             | 0%         | lights_off                        |
| lights_on          | Lights on                                 | False     |      20 | 100%      | 0%              | 0%         |                                   |
| lights_off         | Turn off lights                           | True      |      20 | 100%      | 0%              | 0%         |                                   |
| lights_off         | Turn off the lights                       | False     |      20 | 100%      | 0%              | 0%         |                                   |
| lights_off         | Turn the lights off                       | False     |      20 | 100%      | 0%              | 0%         |                                   |
| lights_off         | Switch off the lights                     | False     |      20 | 100%      | 0%              | 0%         |                                   |
| lights_off         | Switch the lights off                     | False     |      20 | 100%      | 0%              | 0%         |                                   |
| lights_off         | Lights off                                | False     |      20 | 100%      | 0%              | 0%         |                                   |
| play_music         | Play music                                | True      |      19 | 100%      | 0%              | 0%         |                                   |
| play_music         | Play some music                           | False     |      20 | 100%      | 0%              | 0%         |                                   |
| play_music         | Put on some music                         | False     |      20 | 30%       | 50%             | 20%        | pause                             |
| play_music         | Start the music                           | False     |      20 | 90%       | 10%             | 0%         | stop                              |
| weather            | What's the weather?                       | True      |      20 | 100%      | 0%              | 0%         |                                   |
| weather            | How's the weather?                        | False     |      20 | 100%      | 0%              | 0%         |                                   |
| weather            | What's the weather like today?            | False     |      20 | 95%       | 0%              | 5%         |                                   |
| weather            | Weather today                             | False     |      20 | 0%        | 5%              | 95%        | call_jane                         |
| time               | What time is it?                          | True      |       9 | 100%      | 0%              | 0%         |                                   |
| time               | What's the time?                          | False     |      20 | 100%      | 0%              | 0%         |                                   |
| time               | Tell me the time                          | False     |      20 | 0%        | 0%              | 100%       |                                   |
| time               | What's the time now?                      | False     |      20 | 70%       | 5%              | 25%        | weather                           |
| dim_lights_20      | Dim lights to twenty percent              | True      |      20 | 100%      | 0%              | 0%         |                                   |
| dim_lights_20      | Dim the lights to twenty percent          | False     |      20 | 100%      | 0%              | 0%         |                                   |
| dim_lights_20      | Set the lights to twenty percent          | False     |      20 | 100%      | 0%              | 0%         |                                   |
| dim_lights_20      | Lights at twenty percent                  | False     |      20 | 100%      | 0%              | 0%         |                                   |
| dim_lights_50      | Dim lights to fifty percent               | True      |      20 | 100%      | 0%              | 0%         |                                   |
| dim_lights_50      | Dim the lights to fifty percent           | False     |      20 | 100%      | 0%              | 0%         |                                   |
| dim_lights_50      | Set the lights to fifty percent           | False     |      20 | 100%      | 0%              | 0%         |                                   |
| dim_lights_50      | Lights at fifty percent                   | False     |      19 | 95%       | 0%              | 5%         |                                   |
| dim_lights_80      | Dim lights to eighty percent              | True      |      19 | 100%      | 0%              | 0%         |                                   |
| dim_lights_80      | Dim the lights to eighty percent          | False     |      20 | 100%      | 0%              | 0%         |                                   |
| dim_lights_80      | Set the lights to eighty percent          | False     |      20 | 100%      | 0%              | 0%         |                                   |
| dim_lights_80      | Lights at eighty percent                  | False     |      19 | 100%      | 0%              | 0%         |                                   |
| set_timer_1        | Set a timer for one minute                | True      |      20 | 100%      | 0%              | 0%         |                                   |
| set_timer_1        | Set a one minute timer                    | False     |      20 | 0%        | 0%              | 100%       |                                   |
| set_timer_1        | Start a timer for one minute              | False     |      20 | 100%      | 0%              | 0%         |                                   |
| set_timer_1        | Timer for one minute                      | False     |      20 | 95%       | 0%              | 5%         |                                   |
| set_timer_5        | Set a timer for five minutes              | True      |      20 | 100%      | 0%              | 0%         |                                   |
| set_timer_5        | Set a five minute timer                   | False     |      20 | 0%        | 0%              | 100%       |                                   |
| set_timer_5        | Start a timer for five minutes            | False     |      20 | 100%      | 0%              | 0%         |                                   |
| set_timer_5        | Timer for five minutes                    | False     |      20 | 100%      | 0%              | 0%         |                                   |
| set_timer_10       | Set a timer for ten minutes               | True      |      20 | 100%      | 0%              | 0%         |                                   |
| set_timer_10       | Set a ten minute timer                    | False     |      20 | 0%        | 0%              | 100%       |                                   |
| set_timer_10       | Start a timer for ten minutes             | False     |      20 | 100%      | 0%              | 0%         |                                   |
| set_timer_10       | Timer for ten minutes                     | False     |      20 | 100%      | 0%              | 0%         |                                   |
| set_alarm_6am      | Set an alarm at six A M                   | True      |      20 | 100%      | 0%              | 0%         |                                   |
| set_alarm_6am      | Set an alarm for six A M                  | False     |      20 | 100%      | 0%              | 0%         |                                   |
| set_alarm_6am      | Wake me up at six A M                     | False     |      20 | 25%       | 0%              | 75%        |                                   |
| set_alarm_6am      | Set an alarm for six in the morning       | False     |      20 | 95%       | 0%              | 5%         |                                   |
| set_alarm_7am      | Set an alarm at seven A M                 | True      |      19 | 100%      | 0%              | 0%         |                                   |
| set_alarm_7am      | Set an alarm for seven A M                | False     |      20 | 100%      | 0%              | 0%         |                                   |
| set_alarm_7am      | Wake me up at seven A M                   | False     |      20 | 40%       | 0%              | 60%        |                                   |
| set_alarm_7am      | Set an alarm for seven in the morning     | False     |      20 | 85%       | 0%              | 15%        |                                   |
| set_alarm_9pm      | Set an alarm at nine P M                  | True      |      20 | 100%      | 0%              | 0%         |                                   |
| set_alarm_9pm      | Set an alarm for nine P M                 | False     |      20 | 100%      | 0%              | 0%         |                                   |
| set_alarm_9pm      | Set an alarm for nine tonight             | False     |      20 | 80%       | 0%              | 20%        |                                   |
| set_alarm_9pm      | Set an alarm for nine in the evening      | False     |      20 | 85%       | 0%              | 15%        |                                   |
| set_temperature_18 | Set temperature to eighteen degrees       | True      |      20 | 100%      | 0%              | 0%         |                                   |
| set_temperature_18 | Set the temperature to eighteen degrees   | False     |      20 | 100%      | 0%              | 0%         |                                   |
| set_temperature_18 | Make it eighteen degrees                  | False     |      20 | 10%       | 0%              | 90%        |                                   |
| set_temperature_18 | Change the temperature to eighteen        | False     |      20 | 25%       | 0%              | 75%        |                                   |
| set_temperature_22 | Set temperature to twenty two degrees     | True      |      20 | 100%      | 0%              | 0%         |                                   |
| set_temperature_22 | Set the temperature to twenty two degrees | False     |      20 | 100%      | 0%              | 0%         |                                   |
| set_temperature_22 | Make it twenty two degrees                | False     |      20 | 95%       | 0%              | 5%         |                                   |
| set_temperature_22 | Change the temperature to twenty two      | False     |      20 | 90%       | 0%              | 10%        |                                   |
| set_temperature_26 | Set temperature to twenty six degrees     | True      |      20 | 100%      | 0%              | 0%         |                                   |
| set_temperature_26 | Set the temperature to twenty six degrees | False     |      20 | 100%      | 0%              | 0%         |                                   |
| set_temperature_26 | Make it twenty six degrees                | False     |      20 | 75%       | 0%              | 25%        |                                   |
| set_temperature_26 | Change the temperature to twenty six      | False     |      20 | 55%       | 0%              | 45%        |                                   |
| pause              | Pause the music                           | False     |      20 | 100%      | 0%              | 0%         |                                   |
| stop               | Stop                                      | True      |      18 | 100%      | 0%              | 0%         |                                   |
| stop               | Stop the music                            | False     |      20 | 100%      | 0%              | 0%         |                                   |
| stop               | Stop playing                              | False     |      20 | 30%       | 20%             | 50%        | play                              |
| next               | Next                                      | True      |      20 | 100%      | 0%              | 0%         |                                   |
| next               | Next song                                 | False     |      18 | 56%       | 6%              | 39%        | lights_off                        |
| next               | Play the next song                        | False     |      20 | 0%        | 15%             | 85%        | lights_off                        |
| next               | Skip                                      | True      |      20 | 100%      | 0%              | 0%         |                                   |
| next               | Skip this song                            | False     |      20 | 95%       | 0%              | 5%         |                                   |
| next               | Skip the song                             | False     |      20 | 100%      | 0%              | 0%         |                                   |
| volume_up          | Volume up                                 | True      |      20 | 100%      | 0%              | 0%         |                                   |
| volume_up          | Turn up the volume                        | False     |      20 | 95%       | 0%              | 5%         |                                   |
| volume_up          | Turn it up                                | False     |      20 | 5%        | 0%              | 95%        |                                   |
| volume_up          | Louder                                    | True      |      18 | 100%      | 0%              | 0%         |                                   |
| volume_up          | Make it louder                            | False     |      18 | 100%      | 0%              | 0%         |                                   |
| volume_up          | A bit louder                              | False     |      20 | 100%      | 0%              | 0%         |                                   |
| volume_down        | Volume down                               | True      |      19 | 100%      | 0%              | 0%         |                                   |
| volume_down        | Turn down the volume                      | False     |      20 | 85%       | 0%              | 15%        |                                   |
| volume_down        | Turn it down                              | False     |      20 | 25%       | 0%              | 75%        |                                   |
| volume_down        | Quieter                                   | False     |      13 | 85%       | 15%             | 0%         | call_jane, lights_off             |
| volume_down        | Lower the volume                          | False     |      20 | 80%       | 15%             | 5%         | volume_up                         |
| remind_trash       | Remind me to take out the trash           | True      |      20 | 100%      | 0%              | 0%         |                                   |
| remind_trash       | Remind me to take the trash out           | False     |      20 | 100%      | 0%              | 0%         |                                   |
| remind_trash       | Remind me about the trash                 | False     |      20 | 100%      | 0%              | 0%         |                                   |
| remind_study       | Remind me to study for my exam            | True      |      20 | 100%      | 0%              | 0%         |                                   |
| remind_study       | Remind me to study for my test            | False     |      19 | 84%       | 0%              | 16%        |                                   |
| remind_study       | Remind me to study                        | False     |      18 | 94%       | 0%              | 6%         |                                   |
| list_reminders     | What are my reminders?                    | True      |      20 | 100%      | 0%              | 0%         |                                   |
| list_reminders     | What reminders do I have?                 | False     |      20 | 0%        | 0%              | 100%       |                                   |
| list_reminders     | Do I have any reminders?                  | False     |      20 | 30%       | 0%              | 70%        |                                   |
| list_reminders     | Read my reminders                         | False     |      19 | 79%       | 0%              | 21%        |                                   |
| call_jane          | Call Jane                                 | True      |      13 | 100%      | 0%              | 0%         |                                   |
| call_jane          | Phone Jane                                | False     |      13 | 100%      | 0%              | 0%         |                                   |
| call_jane          | Give Jane a call                          | False     |      19 | 0%        | 0%              | 100%       |                                   |
| call_jane          | Call Jane please                          | False     |      20 | 90%       | 0%              | 10%        |                                   |
| text_jane          | Text Jane                                 | True      |      13 | 100%      | 0%              | 0%         |                                   |
| text_jane          | Send Jane a text                          | False     |      20 | 0%        | 0%              | 100%       |                                   |
| text_jane          | Message Jane                              | False     |      18 | 89%       | 0%              | 11%        |                                   |
| text_jane          | Send a message to Jane                    | False     |      20 | 0%        | 25%             | 75%        | set_alarm_6am, set_temperature_26 |
| play               | Play                                      | True      |      17 | 100%      | 0%              | 0%         |                                   |
| play               | Resume                                    | False     |      18 | 0%        | 6%              | 94%        | play_music                        |
| play               | Keep playing                              | False     |      20 | 90%       | 0%              | 10%        |                                   |
| play               | Continue playing                          | False     |      20 | 10%       | 0%              | 90%        |                                   |
| party              | Party party                               | True      |      17 | 100%      | 0%              | 0%         |                                   |
| party              | Party mode                                | False     |      20 | 35%       | 40%             | 25%        | volume_up                         |
| party              | Let's party                               | False     |      18 | 33%       | 22%             | 44%        | call_jane, time                   |
| party              | Party time                                | False     |      19 | 100%      | 0%              | 0%         |                                   |
| pause              | Pause it                                  | False     |      13 | 46%       | 23%             | 31%        | call_jane, next                   |
| pause              | Pause                                     | True      |      18 | 100%      | 0%              | 0%         |                                   |

## bcresnet6_ownernoise (balanced cutoff 0.575; cautious 0.945)

|                  |   clips | correct   | wrong command   | rejected   | wrong as                                                   |
|:-----------------|--------:|:----------|:----------------|:-----------|:-----------------------------------------------------------|
| other phrasings  |    1761 | 66%       | 2%              | 32%        | lights_on, pause, play, stop, time, volume_down, volume_up |
| trained phrasing |     579 | 100%      | 0%              | 0%         |                                                            |

Per class, other phrasings only (worst first):

| class              |   clips | trained phrasing correct   | correct   | wrong command   | rejected   | wrong as          |
|:-------------------|--------:|:---------------------------|:----------|:----------------|:-----------|:------------------|
| text_jane          |      58 | 100%                       | 7%        | 0%              | 93%        |                   |
| play               |      58 | 100%                       | 19%       | 0%              | 81%        |                   |
| set_temperature_18 |      60 | 100%                       | 38%       | 0%              | 62%        |                   |
| party              |      57 | 100%                       | 42%       | 11%             | 47%        | volume_up         |
| list_reminders     |      59 | 100%                       | 46%       | 2%              | 53%        | volume_down       |
| volume_down        |      73 | 100%                       | 47%       | 5%              | 48%        | play, volume_up   |
| time               |      60 | 100%                       | 50%       | 2%              | 48%        | volume_down       |
| call_jane          |      52 | 100%                       | 56%       | 0%              | 44%        |                   |
| set_temperature_26 |      60 | 100%                       | 57%       | 0%              | 43%        |                   |
| set_timer_10       |      60 | 100%                       | 58%       | 0%              | 42%        |                   |
| set_timer_1        |      60 | 100%                       | 60%       | 0%              | 40%        |                   |
| volume_up          |      78 | 100%                       | 63%       | 0%              | 37%        |                   |
| set_alarm_6am      |      60 | 100%                       | 65%       | 0%              | 35%        |                   |
| weather            |      60 | 100%                       | 65%       | 0%              | 35%        |                   |
| set_timer_5        |      60 | 100%                       | 65%       | 0%              | 35%        |                   |
| play_music         |      60 | 100%                       | 67%       | 22%             | 12%        | pause, stop, time |
| next               |      78 | 100%                       | 67%       | 0%              | 33%        |                   |
| stop               |      40 | 100%                       | 68%       | 0%              | 32%        |                   |
| pause              |      33 | 100%                       | 70%       | 0%              | 30%        |                   |
| remind_study       |      37 | 100%                       | 70%       | 0%              | 30%        |                   |
| set_alarm_7am      |      60 | 100%                       | 73%       | 0%              | 27%        |                   |
| set_temperature_22 |      60 | 100%                       | 80%       | 0%              | 20%        |                   |
| remind_trash       |      40 | 100%                       | 85%       | 0%              | 15%        |                   |
| dim_lights_50      |      59 | 100%                       | 92%       | 0%              | 8%         |                   |
| set_alarm_9pm      |      60 | 100%                       | 93%       | 0%              | 7%         |                   |
| lights_off         |     100 | 100%                       | 96%       | 2%              | 2%         | lights_on         |
| dim_lights_20      |      60 | 100%                       | 97%       | 0%              | 3%         |                   |
| dim_lights_80      |      59 | 100%                       | 98%       | 0%              | 2%         |                   |
| lights_on          |     100 | 100%                       | 99%       | 0%              | 1%         |                   |

Per phrasing:

| class              | phrasing                                  | trained   |   clips | correct   | wrong command   | rejected   | wrong as    |
|:-------------------|:------------------------------------------|:----------|--------:|:----------|:----------------|:-----------|:------------|
| lights_on          | Turn on lights                            | True      |      20 | 100%      | 0%              | 0%         |             |
| lights_on          | Turn on the lights                        | False     |      20 | 100%      | 0%              | 0%         |             |
| lights_on          | Turn the lights on                        | False     |      20 | 100%      | 0%              | 0%         |             |
| lights_on          | Switch on the lights                      | False     |      20 | 100%      | 0%              | 0%         |             |
| lights_on          | Switch the lights on                      | False     |      20 | 95%       | 0%              | 5%         |             |
| lights_on          | Lights on                                 | False     |      20 | 100%      | 0%              | 0%         |             |
| lights_off         | Turn off lights                           | True      |      20 | 100%      | 0%              | 0%         |             |
| lights_off         | Turn off the lights                       | False     |      20 | 100%      | 0%              | 0%         |             |
| lights_off         | Turn the lights off                       | False     |      20 | 85%       | 10%             | 5%         | lights_on   |
| lights_off         | Switch off the lights                     | False     |      20 | 100%      | 0%              | 0%         |             |
| lights_off         | Switch the lights off                     | False     |      20 | 95%       | 0%              | 5%         |             |
| lights_off         | Lights off                                | False     |      20 | 100%      | 0%              | 0%         |             |
| play_music         | Play music                                | True      |      19 | 100%      | 0%              | 0%         |             |
| play_music         | Play some music                           | False     |      20 | 90%       | 0%              | 10%        |             |
| play_music         | Put on some music                         | False     |      20 | 20%       | 55%             | 25%        | pause, time |
| play_music         | Start the music                           | False     |      20 | 90%       | 10%             | 0%         | stop        |
| weather            | What's the weather?                       | True      |      20 | 100%      | 0%              | 0%         |             |
| weather            | How's the weather?                        | False     |      20 | 100%      | 0%              | 0%         |             |
| weather            | What's the weather like today?            | False     |      20 | 95%       | 0%              | 5%         |             |
| weather            | Weather today                             | False     |      20 | 0%        | 0%              | 100%       |             |
| time               | What time is it?                          | True      |       9 | 100%      | 0%              | 0%         |             |
| time               | What's the time?                          | False     |      20 | 100%      | 0%              | 0%         |             |
| time               | Tell me the time                          | False     |      20 | 0%        | 0%              | 100%       |             |
| time               | What's the time now?                      | False     |      20 | 50%       | 5%              | 45%        | volume_down |
| dim_lights_20      | Dim lights to twenty percent              | True      |      20 | 100%      | 0%              | 0%         |             |
| dim_lights_20      | Dim the lights to twenty percent          | False     |      20 | 100%      | 0%              | 0%         |             |
| dim_lights_20      | Set the lights to twenty percent          | False     |      20 | 100%      | 0%              | 0%         |             |
| dim_lights_20      | Lights at twenty percent                  | False     |      20 | 90%       | 0%              | 10%        |             |
| dim_lights_50      | Dim lights to fifty percent               | True      |      20 | 100%      | 0%              | 0%         |             |
| dim_lights_50      | Dim the lights to fifty percent           | False     |      20 | 100%      | 0%              | 0%         |             |
| dim_lights_50      | Set the lights to fifty percent           | False     |      20 | 100%      | 0%              | 0%         |             |
| dim_lights_50      | Lights at fifty percent                   | False     |      19 | 74%       | 0%              | 26%        |             |
| dim_lights_80      | Dim lights to eighty percent              | True      |      19 | 100%      | 0%              | 0%         |             |
| dim_lights_80      | Dim the lights to eighty percent          | False     |      20 | 100%      | 0%              | 0%         |             |
| dim_lights_80      | Set the lights to eighty percent          | False     |      20 | 100%      | 0%              | 0%         |             |
| dim_lights_80      | Lights at eighty percent                  | False     |      19 | 95%       | 0%              | 5%         |             |
| set_timer_1        | Set a timer for one minute                | True      |      20 | 100%      | 0%              | 0%         |             |
| set_timer_1        | Set a one minute timer                    | False     |      20 | 0%        | 0%              | 100%       |             |
| set_timer_1        | Start a timer for one minute              | False     |      20 | 100%      | 0%              | 0%         |             |
| set_timer_1        | Timer for one minute                      | False     |      20 | 80%       | 0%              | 20%        |             |
| set_timer_5        | Set a timer for five minutes              | True      |      20 | 100%      | 0%              | 0%         |             |
| set_timer_5        | Set a five minute timer                   | False     |      20 | 0%        | 0%              | 100%       |             |
| set_timer_5        | Start a timer for five minutes            | False     |      20 | 100%      | 0%              | 0%         |             |
| set_timer_5        | Timer for five minutes                    | False     |      20 | 95%       | 0%              | 5%         |             |
| set_timer_10       | Set a timer for ten minutes               | True      |      20 | 100%      | 0%              | 0%         |             |
| set_timer_10       | Set a ten minute timer                    | False     |      20 | 0%        | 0%              | 100%       |             |
| set_timer_10       | Start a timer for ten minutes             | False     |      20 | 100%      | 0%              | 0%         |             |
| set_timer_10       | Timer for ten minutes                     | False     |      20 | 75%       | 0%              | 25%        |             |
| set_alarm_6am      | Set an alarm at six A M                   | True      |      20 | 100%      | 0%              | 0%         |             |
| set_alarm_6am      | Set an alarm for six A M                  | False     |      20 | 100%      | 0%              | 0%         |             |
| set_alarm_6am      | Wake me up at six A M                     | False     |      20 | 5%        | 0%              | 95%        |             |
| set_alarm_6am      | Set an alarm for six in the morning       | False     |      20 | 90%       | 0%              | 10%        |             |
| set_alarm_7am      | Set an alarm at seven A M                 | True      |      19 | 100%      | 0%              | 0%         |             |
| set_alarm_7am      | Set an alarm for seven A M                | False     |      20 | 100%      | 0%              | 0%         |             |
| set_alarm_7am      | Wake me up at seven A M                   | False     |      20 | 20%       | 0%              | 80%        |             |
| set_alarm_7am      | Set an alarm for seven in the morning     | False     |      20 | 100%      | 0%              | 0%         |             |
| set_alarm_9pm      | Set an alarm at nine P M                  | True      |      20 | 100%      | 0%              | 0%         |             |
| set_alarm_9pm      | Set an alarm for nine P M                 | False     |      20 | 100%      | 0%              | 0%         |             |
| set_alarm_9pm      | Set an alarm for nine tonight             | False     |      20 | 80%       | 0%              | 20%        |             |
| set_alarm_9pm      | Set an alarm for nine in the evening      | False     |      20 | 100%      | 0%              | 0%         |             |
| set_temperature_18 | Set temperature to eighteen degrees       | True      |      20 | 100%      | 0%              | 0%         |             |
| set_temperature_18 | Set the temperature to eighteen degrees   | False     |      20 | 100%      | 0%              | 0%         |             |
| set_temperature_18 | Make it eighteen degrees                  | False     |      20 | 10%       | 0%              | 90%        |             |
| set_temperature_18 | Change the temperature to eighteen        | False     |      20 | 5%        | 0%              | 95%        |             |
| set_temperature_22 | Set temperature to twenty two degrees     | True      |      20 | 100%      | 0%              | 0%         |             |
| set_temperature_22 | Set the temperature to twenty two degrees | False     |      20 | 100%      | 0%              | 0%         |             |
| set_temperature_22 | Make it twenty two degrees                | False     |      20 | 60%       | 0%              | 40%        |             |
| set_temperature_22 | Change the temperature to twenty two      | False     |      20 | 80%       | 0%              | 20%        |             |
| set_temperature_26 | Set temperature to twenty six degrees     | True      |      20 | 100%      | 0%              | 0%         |             |
| set_temperature_26 | Set the temperature to twenty six degrees | False     |      20 | 100%      | 0%              | 0%         |             |
| set_temperature_26 | Make it twenty six degrees                | False     |      20 | 30%       | 0%              | 70%        |             |
| set_temperature_26 | Change the temperature to twenty six      | False     |      20 | 40%       | 0%              | 60%        |             |
| pause              | Pause the music                           | False     |      20 | 100%      | 0%              | 0%         |             |
| stop               | Stop                                      | True      |      18 | 100%      | 0%              | 0%         |             |
| stop               | Stop the music                            | False     |      20 | 100%      | 0%              | 0%         |             |
| stop               | Stop playing                              | False     |      20 | 35%       | 0%              | 65%        |             |
| next               | Next                                      | True      |      20 | 100%      | 0%              | 0%         |             |
| next               | Next song                                 | False     |      18 | 83%       | 0%              | 17%        |             |
| next               | Play the next song                        | False     |      20 | 0%        | 0%              | 100%       |             |
| next               | Skip                                      | True      |      20 | 100%      | 0%              | 0%         |             |
| next               | Skip this song                            | False     |      20 | 85%       | 0%              | 15%        |             |
| next               | Skip the song                             | False     |      20 | 100%      | 0%              | 0%         |             |
| volume_up          | Volume up                                 | True      |      20 | 100%      | 0%              | 0%         |             |
| volume_up          | Turn up the volume                        | False     |      20 | 80%       | 0%              | 20%        |             |
| volume_up          | Turn it up                                | False     |      20 | 0%        | 0%              | 100%       |             |
| volume_up          | Louder                                    | True      |      18 | 100%      | 0%              | 0%         |             |
| volume_up          | Make it louder                            | False     |      18 | 94%       | 0%              | 6%         |             |
| volume_up          | A bit louder                              | False     |      20 | 80%       | 0%              | 20%        |             |
| volume_down        | Volume down                               | True      |      19 | 100%      | 0%              | 0%         |             |
| volume_down        | Turn down the volume                      | False     |      20 | 75%       | 0%              | 25%        |             |
| volume_down        | Turn it down                              | False     |      20 | 5%        | 0%              | 95%        |             |
| volume_down        | Quieter                                   | False     |      13 | 46%       | 8%              | 46%        | play        |
| volume_down        | Lower the volume                          | False     |      20 | 60%       | 15%             | 25%        | volume_up   |
| remind_trash       | Remind me to take out the trash           | True      |      20 | 100%      | 0%              | 0%         |             |
| remind_trash       | Remind me to take the trash out           | False     |      20 | 100%      | 0%              | 0%         |             |
| remind_trash       | Remind me about the trash                 | False     |      20 | 70%       | 0%              | 30%        |             |
| remind_study       | Remind me to study for my exam            | True      |      20 | 100%      | 0%              | 0%         |             |
| remind_study       | Remind me to study for my test            | False     |      19 | 100%      | 0%              | 0%         |             |
| remind_study       | Remind me to study                        | False     |      18 | 39%       | 0%              | 61%        |             |
| list_reminders     | What are my reminders?                    | True      |      20 | 100%      | 0%              | 0%         |             |
| list_reminders     | What reminders do I have?                 | False     |      20 | 0%        | 5%              | 95%        | volume_down |
| list_reminders     | Do I have any reminders?                  | False     |      20 | 60%       | 0%              | 40%        |             |
| list_reminders     | Read my reminders                         | False     |      19 | 79%       | 0%              | 21%        |             |
| call_jane          | Call Jane                                 | True      |      13 | 100%      | 0%              | 0%         |             |
| call_jane          | Phone Jane                                | False     |      13 | 100%      | 0%              | 0%         |             |
| call_jane          | Give Jane a call                          | False     |      19 | 0%        | 0%              | 100%       |             |
| call_jane          | Call Jane please                          | False     |      20 | 80%       | 0%              | 20%        |             |
| text_jane          | Text Jane                                 | True      |      13 | 100%      | 0%              | 0%         |             |
| text_jane          | Send Jane a text                          | False     |      20 | 0%        | 0%              | 100%       |             |
| text_jane          | Message Jane                              | False     |      18 | 22%       | 0%              | 78%        |             |
| text_jane          | Send a message to Jane                    | False     |      20 | 0%        | 0%              | 100%       |             |
| play               | Play                                      | True      |      17 | 100%      | 0%              | 0%         |             |
| play               | Resume                                    | False     |      18 | 0%        | 0%              | 100%       |             |
| play               | Keep playing                              | False     |      20 | 55%       | 0%              | 45%        |             |
| play               | Continue playing                          | False     |      20 | 0%        | 0%              | 100%       |             |
| party              | Party party                               | True      |      17 | 100%      | 0%              | 0%         |             |
| party              | Party mode                                | False     |      20 | 20%       | 30%             | 50%        | volume_up   |
| party              | Let's party                               | False     |      18 | 6%        | 0%              | 94%        |             |
| party              | Party time                                | False     |      19 | 100%      | 0%              | 0%         |             |
| pause              | Pause it                                  | False     |      13 | 23%       | 0%              | 77%        |             |
| pause              | Pause                                     | True      |      18 | 100%      | 0%              | 0%         |             |
