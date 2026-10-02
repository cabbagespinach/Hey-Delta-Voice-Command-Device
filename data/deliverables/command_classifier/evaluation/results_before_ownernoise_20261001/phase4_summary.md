# Command classifier: phase 4 summary

Recommended model: **bcresnet6** (validation selection score 0.950; bcresnet3 0.919, bcresnet6 0.950). Test results never chose anything.

## bcresnet3: 55,914 parameters, best epoch 53

Cutoffs (validation): cautious 0.935 (unknown -> action <= 2%), balanced 0.640 (<= 5%)

| rule | owner correct / wrong / asks again | speaker2 correct / wrong / asks again | public | synthetic | unknown -> action |
|---|---|---|---|---|---|
| argmax | 98.7% / 1.3% / 0.0% | 97.6% / 1.7% / 0.7% | 81.5% | 98.8% | 8.6% |
| cautious | 75.4% / 0.0% / 24.6% | 78.5% / 0.0% / 21.5% | 47.3% | 88.3% | 0.4% |
| balanced | 93.0% / 1.3% / 5.8% | 94.1% / 0.0% / 5.9% | 73.6% | 97.9% | 1.9% |

Streaming (40 streams, 40 min; whole Pi path):

| events | woke | rule | command correct | wrong | asks again | off-list -> action | end-to-end correct |
|---|---|---|---|---|---|---|---|
| clone (108) | 42% | argmax | 94.7% | 2.6% | 2.6% | 28.6% | 37.9% |
| clone (108) | 42% | cautious | 65.8% | 0.0% | 34.2% | 0.0% | 26.3% |
| clone (108) | 42% | balanced | 92.1% | 0.0% | 7.9% | 14.3% | 36.8% |
| distractor (32) | 0% | - | - | - | - | - | - |
| owner (74) | 82% | argmax | 54.1% | 6.6% | 39.3% | - | 44.6% |
| owner (74) | 82% | cautious | 26.2% | 0.0% | 73.8% | - | 21.6% |
| owner (74) | 82% | balanced | 44.3% | 1.6% | 54.1% | - | 36.5% |

False wake-ups: 34.5/hour; actions they caused per hour: {'argmax': 18.0, 'cautious': 6.0, 'balanced': 12.0}

## bcresnet6: 191,286 parameters, best epoch 49

Cutoffs (validation): cautious 0.885 (unknown -> action <= 2%), balanced 0.345 (<= 5%)

| rule | owner correct / wrong / asks again | speaker2 correct / wrong / asks again | public | synthetic | unknown -> action |
|---|---|---|---|---|---|
| argmax | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% | 88.0% | 99.3% | 5.8% |
| cautious | 93.6% / 0.0% / 6.4% | 92.6% / 0.0% / 7.4% | 74.1% | 96.7% | 0.4% |
| balanced | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% | 88.0% | 99.2% | 5.8% |

Streaming (40 streams, 40 min; whole Pi path):

| events | woke | rule | command correct | wrong | asks again | off-list -> action | end-to-end correct |
|---|---|---|---|---|---|---|---|
| clone (108) | 42% | argmax | 94.7% | 0.0% | 5.3% | 14.3% | 37.9% |
| clone (108) | 42% | cautious | 84.2% | 0.0% | 15.8% | 0.0% | 33.7% |
| clone (108) | 42% | balanced | 94.7% | 0.0% | 5.3% | 14.3% | 37.9% |
| distractor (32) | 0% | - | - | - | - | - | - |
| owner (74) | 82% | argmax | 62.3% | 1.6% | 36.1% | - | 51.4% |
| owner (74) | 82% | cautious | 47.5% | 0.0% | 52.5% | - | 39.2% |
| owner (74) | 82% | balanced | 62.3% | 1.6% | 36.1% | - | 51.4% |

False wake-ups: 34.5/hour; actions they caused per hour: {'argmax': 16.5, 'cautious': 10.5, 'balanced': 16.5}

---

Small-scale test: see `model/runs/probes/probe_report.md` and `model/runs/probes/shortcuts.json`.