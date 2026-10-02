# Test results on unseen speakers

Only test clips whose speaker never appears in the training split. Command rates are macro averages over classes; `unknown_false_action` = share of unknown clips that trigger a command.

| model             | rule     | dataset              |   unseen_speakers |   command_clips |   command_classes |   correct |   wrong_command |   unknown_clips |   unknown_false_action |
|:------------------|:---------|:---------------------|------------------:|----------------:|------------------:|----------:|----------------:|----------------:|-----------------------:|
| bcresnet6_hf_only | argmax   | classmate_hf_optionb |                29 |            3368 |                31 |    0.9213 |          0.0684 |               0 |               nan      |
| bcresnet6_hf_only | argmax   | classmate_hf_real    |                 1 |             179 |                31 |    0.5914 |          0.2269 |               0 |               nan      |
| bcresnet6_hf_only | argmax   | classmate_hf_vcm     |                 2 |              10 |                 2 |    0.6667 |          0.0833 |               0 |               nan      |
| bcresnet6_hf_only | argmax   | web_hf_commonvoice   |                 8 |               0 |                 0 |  nan      |        nan      |              14 |                 0      |
| bcresnet6_hf_only | argmax   | web_hf_fsc           |                27 |             147 |                 6 |    0.4852 |          0.2368 |               6 |                 0.5    |
| bcresnet6_hf_only | argmax   | web_hf_multisensor   |                 6 |              11 |                 1 |    0.7273 |          0      |               0 |               nan      |
| bcresnet6_hf_only | argmax   | web_hf_slurp         |                36 |             553 |                20 |    0.4974 |          0.1565 |              21 |                 0.3333 |
| bcresnet6_hf_only | argmax   | web_hf_snips         |                11 |             101 |                 7 |    0.3624 |          0.3592 |               6 |                 0.5    |
| bcresnet6_hf_only | argmax   | web_hf_tas           |                 1 |               2 |                 2 |    1      |          0      |               0 |               nan      |
| bcresnet6_hf_only | cautious | classmate_hf_optionb |                29 |            3368 |                31 |    0.6645 |          0.0024 |               0 |               nan      |
| bcresnet6_hf_only | cautious | classmate_hf_real    |                 1 |             179 |                31 |    0.1462 |          0      |               0 |               nan      |
| bcresnet6_hf_only | cautious | classmate_hf_vcm     |                 2 |              10 |                 2 |    0.125  |          0      |               0 |               nan      |
| bcresnet6_hf_only | cautious | web_hf_commonvoice   |                 8 |               0 |                 0 |  nan      |        nan      |              14 |                 0      |
| bcresnet6_hf_only | cautious | web_hf_fsc           |                27 |             147 |                 6 |    0.1349 |          0      |               6 |                 0      |
| bcresnet6_hf_only | cautious | web_hf_multisensor   |                 6 |              11 |                 1 |    0.3636 |          0      |               0 |               nan      |
| bcresnet6_hf_only | cautious | web_hf_slurp         |                36 |             553 |                20 |    0.2183 |          0.001  |              21 |                 0      |
| bcresnet6_hf_only | cautious | web_hf_snips         |                11 |             101 |                 7 |    0.1402 |          0.0114 |               6 |                 0      |
| bcresnet6_hf_only | cautious | web_hf_tas           |                 1 |               2 |                 2 |    1      |          0      |               0 |               nan      |
| bcresnet6_hf_only | balanced | classmate_hf_optionb |                29 |            3368 |                31 |    0.8276 |          0.0172 |               0 |               nan      |
| bcresnet6_hf_only | balanced | classmate_hf_real    |                 1 |             179 |                31 |    0.3118 |          0.028  |               0 |               nan      |
| bcresnet6_hf_only | balanced | classmate_hf_vcm     |                 2 |              10 |                 2 |    0.4583 |          0      |               0 |               nan      |
| bcresnet6_hf_only | balanced | web_hf_commonvoice   |                 8 |               0 |                 0 |  nan      |        nan      |              14 |                 0      |
| bcresnet6_hf_only | balanced | web_hf_fsc           |                27 |             147 |                 6 |    0.2714 |          0.0124 |               6 |                 0.1667 |
| bcresnet6_hf_only | balanced | web_hf_multisensor   |                 6 |              11 |                 1 |    0.7273 |          0      |               0 |               nan      |
| bcresnet6_hf_only | balanced | web_hf_slurp         |                36 |             553 |                20 |    0.3474 |          0.016  |              21 |                 0      |
| bcresnet6_hf_only | balanced | web_hf_snips         |                11 |             101 |                 7 |    0.2593 |          0.0444 |               6 |                 0      |
| bcresnet6_hf_only | balanced | web_hf_tas           |                 1 |               2 |                 2 |    1      |          0      |               0 |               nan      |
