# Command classifier: test results

Cutoffs chosen on validation; test clips were never used for any choice. Command rates are macro averages over classes: **correct**, **wrong command** (the harmful error), **rejected** (answered 'unknown' = asks again).

## bcresnet6_hf_only (191,672 parameters, best epoch 6)

Cutoffs: cautious 0.890, balanced 0.670

| rule | who | clips | correct | wrong command | rejected |
|---|---|---|---|---|---|
| argmax | web | 814 | 52.9% | 16.7% | 30.3% |
| argmax | classmates | 3557 | 90.3% | 7.7% | 2.0% |
| cautious | web | 814 | 25.4% | 0.3% | 74.3% |
| cautious | classmates | 3557 | 63.7% | 0.2% | 36.0% |
| balanced | web | 814 | 37.7% | 2.0% | 60.3% |
| balanced | classmates | 3557 | 80.1% | 1.8% | 18.1% |

Unknown clips that trigger a command (test):

| rule | web | mean |
|---|---|---|
| argmax | 27.7% | 27.7% |
| cautious | 0.0% | 0.0% |
| balanced | 2.1% | 2.1% |
