# Command classifier: test results

Cutoffs chosen on validation; test clips were never used for any choice. Command rates are macro averages over classes: **correct**, **wrong command** (the harmful error), **rejected** (answered 'unknown' = asks again).

## bcresnet6_hf_plus (191,672 parameters, best epoch 43)

Cutoffs: cautious 0.610, balanced 0.340

| rule | who | clips | correct | wrong command | rejected |
|---|---|---|---|---|---|
| argmax | web | 814 | 63.9% | 9.4% | 26.7% |
| argmax | classmates | 3557 | 95.7% | 1.2% | 3.1% |
| cautious | web | 814 | 57.8% | 3.7% | 38.5% |
| cautious | classmates | 3557 | 92.9% | 0.5% | 6.5% |
| balanced | web | 814 | 63.6% | 8.4% | 28.1% |
| balanced | classmates | 3557 | 95.6% | 1.1% | 3.3% |

Unknown clips that trigger a command (test):

| rule | web | mean |
|---|---|---|
| argmax | 19.1% | 19.1% |
| cautious | 8.5% | 8.5% |
| balanced | 17.0% | 17.0% |
