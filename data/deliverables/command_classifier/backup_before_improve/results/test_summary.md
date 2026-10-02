# Command classifier: test results

Cutoffs chosen on validation; test clips were never used for any choice. Command rates are macro averages over classes: **correct**, **wrong command** (the harmful error), **rejected** (answered 'unknown' = asks again).

## bcresnet3 (55,914 parameters, best epoch 53)

Cutoffs: cautious 0.935, balanced 0.640

| rule | who | clips | correct | wrong command | rejected |
|---|---|---|---|---|---|
| argmax | owner | 73 | 98.7% | 1.3% | 0.0% |
| argmax | speaker2 | 73 | 97.6% | 1.7% | 0.7% |
| argmax | web | 845 | 81.5% | 3.1% | 15.4% |
| argmax | synthetic | 1111 | 98.8% | 0.4% | 0.8% |
| cautious | owner | 73 | 75.4% | 0.0% | 24.6% |
| cautious | speaker2 | 73 | 78.5% | 0.0% | 21.5% |
| cautious | web | 845 | 47.3% | 0.0% | 52.7% |
| cautious | synthetic | 1111 | 88.3% | 0.0% | 11.7% |
| balanced | owner | 73 | 93.0% | 1.3% | 5.8% |
| balanced | speaker2 | 73 | 94.1% | 0.0% | 5.9% |
| balanced | web | 845 | 73.6% | 0.9% | 25.5% |
| balanced | synthetic | 1111 | 97.9% | 0.2% | 1.9% |

Unknown clips that trigger a command (test):

| rule | fragments | owner | reuse_fleurs | reuse_mswc | reuse_musan | reuse_owner | synthetic | web | mean |
|---|---|---|---|---|---|---|---|---|---|
| argmax | 7.7% | 16.7% | 0.0% | 10.4% | 1.6% | 4.5% | 17.8% | 10.4% | 8.6% |
| cautious | 0.4% | 0.0% | 0.0% | 2.1% | 0.0% | 0.0% | 0.0% | 0.7% | 0.4% |
| balanced | 3.6% | 0.0% | 0.0% | 2.1% | 0.5% | 0.0% | 4.8% | 4.0% | 1.9% |

## bcresnet6 (191,286 parameters, best epoch 49)

Cutoffs: cautious 0.885, balanced 0.345

| rule | who | clips | correct | wrong command | rejected |
|---|---|---|---|---|---|
| argmax | owner | 73 | 100.0% | 0.0% | 0.0% |
| argmax | speaker2 | 73 | 100.0% | 0.0% | 0.0% |
| argmax | web | 845 | 88.0% | 1.5% | 10.5% |
| argmax | synthetic | 1111 | 99.3% | 0.2% | 0.5% |
| cautious | owner | 73 | 93.6% | 0.0% | 6.4% |
| cautious | speaker2 | 73 | 92.6% | 0.0% | 7.4% |
| cautious | web | 845 | 74.1% | 0.0% | 25.9% |
| cautious | synthetic | 1111 | 96.7% | 0.0% | 3.4% |
| balanced | owner | 73 | 100.0% | 0.0% | 0.0% |
| balanced | speaker2 | 73 | 100.0% | 0.0% | 0.0% |
| balanced | web | 845 | 88.0% | 1.3% | 10.7% |
| balanced | synthetic | 1111 | 99.2% | 0.2% | 0.6% |

Unknown clips that trigger a command (test):

| rule | fragments | owner | reuse_fleurs | reuse_mswc | reuse_musan | reuse_owner | synthetic | web | mean |
|---|---|---|---|---|---|---|---|---|---|
| argmax | 6.6% | 16.7% | 0.0% | 12.5% | 1.0% | 0.9% | 5.5% | 3.3% | 5.8% |
| cautious | 2.2% | 0.0% | 0.0% | 0.0% | 0.5% | 0.0% | 0.0% | 0.2% | 0.4% |
| balanced | 6.6% | 16.7% | 0.0% | 12.5% | 1.0% | 0.9% | 5.5% | 3.3% | 5.8% |

## bcresnet6_ownernoise (191,286 parameters, best epoch 53)

Cutoffs: cautious 0.945, balanced 0.575

| rule | who | clips | correct | wrong command | rejected |
|---|---|---|---|---|---|
| argmax | owner | 73 | 98.7% | 1.3% | 0.0% |
| argmax | speaker2 | 73 | 100.0% | 0.0% | 0.0% |
| argmax | web | 845 | 87.8% | 3.7% | 8.5% |
| argmax | synthetic | 1111 | 99.1% | 0.2% | 0.7% |
| cautious | owner | 73 | 78.6% | 0.0% | 21.4% |
| cautious | speaker2 | 73 | 66.7% | 0.0% | 33.3% |
| cautious | web | 845 | 53.5% | 0.0% | 46.5% |
| cautious | synthetic | 1111 | 90.6% | 0.0% | 9.4% |
| balanced | owner | 73 | 98.7% | 0.0% | 1.3% |
| balanced | speaker2 | 73 | 98.3% | 0.0% | 1.7% |
| balanced | web | 845 | 78.2% | 0.9% | 20.8% |
| balanced | synthetic | 1111 | 98.8% | 0.1% | 1.1% |

Unknown clips that trigger a command (test):

| rule | fragments | owner | reuse_fleurs | reuse_mswc | reuse_musan | reuse_owner | synthetic | web | mean |
|---|---|---|---|---|---|---|---|---|---|
| argmax | 8.8% | 25.0% | 0.0% | 12.5% | 1.0% | 3.6% | 5.5% | 5.3% | 7.7% |
| cautious | 0.7% | 0.0% | 0.0% | 4.2% | 0.0% | 0.0% | 0.0% | 0.2% | 0.6% |
| balanced | 4.4% | 0.0% | 0.0% | 8.3% | 0.5% | 1.8% | 1.4% | 2.0% | 2.3% |
