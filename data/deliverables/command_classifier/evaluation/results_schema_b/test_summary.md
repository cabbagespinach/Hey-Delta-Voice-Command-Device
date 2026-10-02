# Command classifier: test results

Cutoffs chosen on validation; test clips were never used for any choice. Command rates are macro averages over classes: **correct**, **wrong command** (the harmful error), **rejected** (answered 'unknown' = asks again).

## bcresnet6_schema_b (191,672 parameters, best epoch 51)

Cutoffs: cautious 0.990, balanced 0.945

| rule | who | clips | correct | wrong command | rejected |
|---|---|---|---|---|---|
| argmax | owner | 59 | 98.4% | 0.0% | 1.6% |
| argmax | speaker2 | 48 | 100.0% | 0.0% | 0.0% |
| argmax | web | 1339 | 87.0% | 4.3% | 8.7% |
| argmax | synthetic | 834 | 98.5% | 0.0% | 1.6% |
| argmax | classmates | 1793 | 98.0% | 1.0% | 1.0% |
| cautious | owner | 59 | 39.5% | 0.0% | 60.5% |
| cautious | speaker2 | 48 | 54.4% | 0.0% | 45.6% |
| cautious | web | 1339 | 29.0% | 0.0% | 71.0% |
| cautious | synthetic | 834 | 51.8% | 0.0% | 48.2% |
| cautious | classmates | 1793 | 44.5% | 0.0% | 55.5% |
| balanced | owner | 59 | 90.5% | 0.0% | 9.5% |
| balanced | speaker2 | 48 | 86.5% | 0.0% | 13.5% |
| balanced | web | 1339 | 63.3% | 0.0% | 36.7% |
| balanced | synthetic | 834 | 89.7% | 0.0% | 10.3% |
| balanced | classmates | 1793 | 83.5% | 0.0% | 16.5% |

Unknown clips that trigger a command (test):

| rule | classmates | fragments | gsc | owner | reuse_fleurs | reuse_mswc | reuse_musan | reuse_owner | synthetic | web | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| argmax | 73.3% | 6.4% | 5.9% | 2.0% | 0.0% | 6.2% | 1.6% | 5.4% | 8.4% | 7.3% | 11.6% |
| cautious | 0.0% | 0.4% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.2% | 0.0% | 0.1% |
| balanced | 0.0% | 1.5% | 0.4% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.5% | 0.2% | 0.2% |

## dscnn_schema_b (194,888 parameters, best epoch 35)

Cutoffs: cautious 0.920, balanced 0.720

| rule | who | clips | correct | wrong command | rejected |
|---|---|---|---|---|---|
| argmax | owner | 59 | 72.1% | 8.7% | 19.1% |
| argmax | speaker2 | 48 | 88.1% | 2.4% | 9.5% |
| argmax | web | 1339 | 61.9% | 7.0% | 31.1% |
| argmax | synthetic | 834 | 93.8% | 1.9% | 4.3% |
| argmax | classmates | 1793 | 92.0% | 4.5% | 3.5% |
| cautious | owner | 59 | 18.2% | 0.0% | 81.8% |
| cautious | speaker2 | 48 | 54.4% | 0.0% | 45.6% |
| cautious | web | 1339 | 30.1% | 0.7% | 69.3% |
| cautious | synthetic | 834 | 66.8% | 0.0% | 33.2% |
| cautious | classmates | 1793 | 60.0% | 0.7% | 39.3% |
| balanced | owner | 59 | 41.7% | 1.6% | 56.8% |
| balanced | speaker2 | 48 | 75.0% | 0.0% | 25.0% |
| balanced | web | 1339 | 45.8% | 2.1% | 52.1% |
| balanced | synthetic | 834 | 84.5% | 0.4% | 15.1% |
| balanced | classmates | 1793 | 79.7% | 1.3% | 19.1% |

Unknown clips that trigger a command (test):

| rule | classmates | fragments | gsc | owner | reuse_fleurs | reuse_mswc | reuse_musan | reuse_owner | synthetic | web | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| argmax | 40.0% | 10.9% | 8.2% | 9.8% | 9.3% | 2.1% | 2.1% | 4.5% | 24.6% | 12.5% | 12.4% |
| cautious | 0.0% | 1.1% | 0.6% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.5% | 1.4% | 0.4% |
| balanced | 13.3% | 1.5% | 1.8% | 0.0% | 0.0% | 0.0% | 0.0% | 1.8% | 4.0% | 3.7% | 2.6% |
