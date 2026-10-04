# Latency on Raspberry Pi 5 Model B Rev 1.1

2026-10-04T21:01:32; ONNX Runtime 1.30.0, 1 thread, CPU; 600 wakeword / 100 command runs after 20 warm-up runs; script `data/deliverables/evaluation/pi_latency.py`.

| model | input | median ms | p95 ms | max ms | real-time factor | ONNX MB |
|---|---|---:|---:|---:|---:|---:|
| wakeword_bcresnet6 | 1.5 s | 12.06 | 14.49 | 22.61 | 0.008 | 1.98 |
| bcresnet6_hf_plus | 5 s | 50.53 | 59.71 | 69.96 | 0.0101 | 2.0 |
| bcresnet6_hf_only | 5 s | 50.65 | 62.91 | 80.46 | 0.0101 | 2.0 |
| bcresnet6_schema_b | 5 s | 50.63 | 66.6 | 83.01 | 0.0101 | 2.0 |
| dscnn_schema_b | 5 s | 88.97 | 102.3 | 112.27 | 0.0178 | 1.87 |

Wakeword while listening (one window every 100 ms): **12.4% of one core**. Peak memory with all models loaded: 99.0 MB. CPU temperature 36.4 -> 40.8 C; throttled: throttled=0x0.
