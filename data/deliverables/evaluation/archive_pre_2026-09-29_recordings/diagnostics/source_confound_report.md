# Source-confound diagnostic

Generated 2026-09-29T00:37:59+00:00 by `source_confound.py`. Model: pipeline-check baseline. Validation windows only; threshold 0.3343. Level targets: device peak -39.77 dB, device floor -54.2 dB, synthetic peak -10.85 dB (augmentation_config.json measurements).

## 1. Observational: real vs synthetic negatives

- 195 non-silent real negatives from 8 recordings vs 770 synthetic negatives.
- Median speech-band peak / floor: real -32.7 / -43.0 dB, synthetic -14.4 / -51.5 dB.
- Mean score: real 0.031, synthetic 0.015; above threshold: real 2.6%, synthetic 0.5%.
- Source AUC (score separating real from synthetic negatives): unmatched **0.716**; level-matched **0.539** (67 real windows, 255 synthetic matches within 6 dB).

| Real negative | Type | Peak dB | Floor dB | Score |
|---|---|---:|---:|---:|
| Manual-Fil-RPI13@-131ms_n | negative_confusable | -27.279 | -54.609 | 1.0 |
| Manual-BG-RPI8@-19ms_n | negative_real_background | -30.846 | -52.592 | 0.735 |
| Manual-Fil-RPI60@-598ms_n | negative_partial_wakeword | -45.428 | -53.983 | 0.713 |
| Manual-Fil-RPI29@-715ms_n | negative_partial_wakeword | -43.284 | -55.579 | 0.408 |
| Manual-BG-RPI16@93750ms_n | negative_real_background | -27.663 | -38.844 | 0.342 |
| Manual-BG-RPI16@114000ms_n | negative_real_background | -23.997 | -40.753 | 0.206 |
| Manual-BG-RPI16@90000ms_n | negative_real_background | -25.617 | -42.053 | 0.204 |
| Manual-BG-RPI16@94500ms_n | negative_real_background | -27.606 | -38.682 | 0.14 |
| Manual-BG-RPI16@90750ms_n | negative_real_background | -25.149 | -38.033 | 0.132 |
| Manual-Fil-RPI11@-707ms_n | negative_partial_wakeword | -46.841 | -53.652 | 0.124 |
| Manual-BG-RPI16@73500ms_n | negative_real_background | -30.553 | -42.433 | 0.105 |
| Manual-BG-RPI16@58500ms_n | negative_real_background | -29.553 | -41.87 | 0.102 |
| Manual-BG-RPI16@51000ms_n | negative_real_background | -25.387 | -41.795 | 0.086 |
| Manual-BG-RPI4@0ms_n | negative_real_background | -54.705 | -57.83 | 0.085 |
| Manual-BG-RPI9@0ms_n | negative_real_background | -41.365 | -52.701 | 0.08 |
| Manual-BG-RPI16@72750ms_n | negative_real_background | -30.542 | -43.658 | 0.072 |
| Manual-BG-RPI16@33000ms_n | negative_real_background | -22.24 | -41.246 | 0.056 |
| Manual-BG-RPI16@100500ms_n | negative_real_background | -25.47 | -41.034 | 0.056 |
| Manual-BG-RPI9@2250ms_n | negative_real_background | -40.74 | -52.115 | 0.056 |
| Manual-BG-RPI16@114750ms_n | negative_real_background | -23.493 | -40.733 | 0.053 |
| Manual-BG-RPI16@78750ms_n | negative_real_background | -7.703 | -42.38 | 0.041 |
| Manual-BG-RPI16@102750ms_n | negative_real_background | -25.687 | -35.857 | 0.039 |
| Manual-BG-RPI16@78000ms_n | negative_real_background | -7.743 | -42.733 | 0.038 |
| Manual-BG-RPI9@4500ms_n | negative_real_background | -49.548 | -52.944 | 0.036 |
| Manual-BG-RPI9@1500ms_n | negative_real_background | -39.375 | -52.375 | 0.036 |
| Manual-BG-RPI9@3000ms_n | negative_real_background | -49.245 | -52.145 | 0.035 |
| Manual-BG-RPI4@3750ms_n | negative_real_background | -34.603 | -56.862 | 0.032 |
| Manual-BG-RPI9@6000ms_n | negative_real_background | -48.946 | -51.873 | 0.029 |
| Manual-BG-RPI16@97500ms_n | negative_real_background | -28.367 | -39.977 | 0.029 |
| Manual-BG-RPI16@91500ms_n | negative_real_background | -26.252 | -40.085 | 0.027 |
| Manual-BG-RPI9@5250ms_n | negative_real_background | -49.646 | -52.693 | 0.027 |
| Manual-BG-RPI16@35250ms_n | negative_real_background | -28.201 | -40.034 | 0.026 |
| Manual-BG-RPI16@48000ms_n | negative_real_background | -22.311 | -42.496 | 0.025 |
| Manual-BG-RPI16@39000ms_n | negative_real_background | -25.487 | -34.139 | 0.024 |
| Manual-BG-RPI4@9000ms_n | negative_real_background | -55.034 | -57.277 | 0.021 |
| Manual-BG-RPI9@750ms_n | negative_real_background | -45.408 | -52.471 | 0.021 |
| Manual-BG-RPI4@9750ms_n | negative_real_background | -54.916 | -56.955 | 0.02 |
| Manual-BG-RPI16@113250ms_n | negative_real_background | -31.008 | -42.645 | 0.02 |
| Manual-BG-RPI9@3750ms_n | negative_real_background | -49.33 | -52.444 | 0.019 |
| Manual-BG-RPI16@106500ms_n | negative_real_background | -30.484 | -40.327 | 0.018 |
| Manual-BG-RPI16@45750ms_n | negative_real_background | -25.203 | -42.39 | 0.018 |
| Manual-BG-RPI16@12750ms_n | negative_real_background | -36.039 | -42.804 | 0.017 |
| Manual-BG-RPI4@8250ms_n | negative_real_background | -55.236 | -57.533 | 0.017 |
| Manual-BG-RPI16@59250ms_n | negative_real_background | -29.938 | -43.122 | 0.015 |
| Manual-BG-RPI16@96000ms_n | negative_real_background | -29.676 | -40.755 | 0.015 |
| Manual-BG-RPI16@112500ms_n | negative_real_background | -32.748 | -42.824 | 0.014 |
| Manual-BG-RPI16@42000ms_n | negative_real_background | -23.358 | -32.991 | 0.013 |
| Manual-BG-RPI16@123000ms_n | negative_real_background | -31.903 | -43.764 | 0.013 |
| Manual-BG-RPI16@12000ms_n | negative_real_background | -35.265 | -42.982 | 0.013 |
| Manual-BG-RPI16@26250ms_n | negative_real_background | -29.003 | -43.231 | 0.012 |
| Manual-BG-RPI16@103500ms_n | negative_real_background | -24.828 | -37.661 | 0.012 |
| Manual-BG-RPI16@38250ms_n | negative_real_background | -25.337 | -41.633 | 0.012 |
| Manual-BG-RPI4@3000ms_n | negative_real_background | -54.716 | -57.351 | 0.012 |
| Manual-BG-RPI16@51750ms_n | negative_real_background | -24.347 | -37.457 | 0.012 |
| Manual-BG-RPI16@14250ms_n | negative_real_background | -30.9 | -42.45 | 0.011 |
| Manual-BG-RPI16@111750ms_n | negative_real_background | -30.319 | -40.305 | 0.011 |
| Manual-BG-RPI16@5250ms_n | negative_real_background | -30.961 | -42.526 | 0.011 |
| Manual-BG-RPI16@36000ms_n | negative_real_background | -22.233 | -35.092 | 0.011 |
| Manual-BG-RPI16@27750ms_n | negative_real_background | -34.861 | -43.277 | 0.01 |
| Manual-BG-RPI16@102000ms_n | negative_real_background | -26.872 | -35.86 | 0.01 |
| Manual-BG-RPI4@1500ms_n | negative_real_background | -55.163 | -57.327 | 0.01 |
| Manual-BG-RPI16@85500ms_n | negative_real_background | -36.59 | -43.176 | 0.009 |
| Manual-BG-RPI16@39750ms_n | negative_real_background | -28.158 | -37.257 | 0.009 |
| Manual-BG-RPI4@2250ms_n | negative_real_background | -55.193 | -57.02 | 0.009 |
| Manual-BG-RPI16@36750ms_n | negative_real_background | -23.019 | -40.344 | 0.009 |
| Manual-BG-RPI16@24000ms_n | negative_real_background | -34.35 | -43.48 | 0.008 |
| Manual-BG-RPI16@101250ms_n | negative_real_background | -28.786 | -35.94 | 0.008 |
| Manual-BG-RPI16@22500ms_n | negative_real_background | -32.756 | -42.13 | 0.008 |
| Manual-BG-RPI16@30000ms_n | negative_real_background | -36.97 | -42.303 | 0.008 |
| Manual-BG-RPI16@68250ms_n | negative_real_background | -26.88 | -42.43 | 0.007 |
| Manual-BG-RPI16@24750ms_n | negative_real_background | -34.56 | -43.163 | 0.007 |
| Manual-BG-RPI4@6750ms_n | negative_real_background | -53.928 | -57.375 | 0.007 |
| Manual-BG-RPI16@21000ms_n | negative_real_background | -36.484 | -41.728 | 0.007 |
| Manual-BG-RPI16@88500ms_n | negative_real_background | -35.635 | -40.72 | 0.007 |
| Manual-BG-RPI16@32250ms_n | negative_real_background | -22.47 | -40.565 | 0.007 |
| Manual-BG-RPI16@34500ms_n | negative_real_background | -29.969 | -42.088 | 0.007 |
| Manual-BG-RPI16@50250ms_n | negative_real_background | -34.967 | -41.958 | 0.006 |
| Manual-BG-RPI4@5250ms_n | negative_real_background | -34.157 | -56.113 | 0.006 |
| Manual-BG-RPI16@87750ms_n | negative_real_background | -35.785 | -42.382 | 0.006 |
| Manual-BG-RPI4@10500ms_n | negative_real_background | -55.284 | -57.125 | 0.006 |
| Manual-BG-RPI16@20250ms_n | negative_real_background | -34.825 | -42.491 | 0.006 |
| Manual-BG-RPI16@122250ms_n | negative_real_background | -33.003 | -43.319 | 0.006 |
| Manual-BG-RPI16@86250ms_n | negative_real_background | -36.881 | -42.91 | 0.006 |
| Manual-BG-RPI16@105750ms_n | negative_real_background | -32.36 | -43.355 | 0.006 |
| Manual-BG-RPI16@49500ms_n | negative_real_background | -32.173 | -39.035 | 0.005 |
| Manual-BG-RPI16@69750ms_n | negative_real_background | -17.928 | -42.634 | 0.005 |
| Manual-BG-RPI4@750ms_n | negative_real_background | -54.574 | -57.053 | 0.005 |
| Manual-BG-RPI16@16500ms_n | negative_real_background | -38.316 | -43.667 | 0.005 |
| Manual-BG-RPI16@67500ms_n | negative_real_background | -32.158 | -44.077 | 0.005 |
| Manual-BG-RPI16@25500ms_n | negative_real_background | -28.995 | -42.747 | 0.005 |
| Manual-BG-RPI4@6000ms_n | negative_real_background | -34.528 | -56.675 | 0.005 |
| Manual-BG-RPI16@107250ms_n | negative_real_background | -29.245 | -37.306 | 0.005 |
| Manual-BG-RPI4@7500ms_n | negative_real_background | -55.419 | -57.63 | 0.005 |
| Manual-BG-RPI16@15750ms_n | negative_real_background | -38.213 | -43.127 | 0.005 |
| Manual-BG-RPI16@9750ms_n | negative_real_background | -35.681 | -41.973 | 0.005 |
| Manual-BG-RPI16@123750ms_n | negative_real_background | -30.111 | -43.485 | 0.005 |
| Manual-BG-RPI16@45000ms_n | negative_real_background | -28.797 | -43.39 | 0.005 |
| Manual-BG-RPI16@6750ms_n | negative_real_background | -35.491 | -42.66 | 0.005 |
| Manual-BG-RPI16@15000ms_n | negative_real_background | -31.554 | -42.273 | 0.004 |
| Manual-BG-RPI16@104250ms_n | negative_real_background | -24.224 | -41.983 | 0.004 |
| Manual-BG-RPI16@75000ms_n | negative_real_background | -18.424 | -43.004 | 0.004 |
| Manual-BG-RPI16@42750ms_n | negative_real_background | -23.634 | -42.909 | 0.004 |
| Manual-BG-RPI4@4500ms_n | negative_real_background | -33.678 | -55.558 | 0.004 |
| Manual-BG-RPI16@63000ms_n | negative_real_background | -21.796 | -41.975 | 0.004 |
| Manual-BG-RPI16@93000ms_n | negative_real_background | -32.667 | -40.175 | 0.004 |
| Manual-BG-RPI16@108750ms_n | negative_real_background | -39.303 | -43.872 | 0.004 |
| Manual-BG-RPI16@108000ms_n | negative_real_background | -29.278 | -42.66 | 0.004 |
| Manual-BG-RPI16@60750ms_n | negative_real_background | -32.934 | -42.54 | 0.004 |
| Manual-BG-RPI16@47250ms_n | negative_real_background | -22.473 | -43.606 | 0.004 |
| Manual-BG-RPI16@110250ms_n | negative_real_background | -32.137 | -41.536 | 0.004 |
| Manual-BG-RPI16@99000ms_n | negative_real_background | -30.006 | -39.649 | 0.004 |
| Manual-BG-RPI16@109500ms_n | negative_real_background | -33.997 | -44.037 | 0.004 |
| Manual-BG-RPI16@6000ms_n | negative_real_background | -33.067 | -42.349 | 0.004 |
| Manual-BG-RPI16@13500ms_n | negative_real_background | -35.847 | -42.751 | 0.003 |
| Manual-BG-RPI16@23250ms_n | negative_real_background | -33.113 | -43.098 | 0.003 |
| Manual-BG-RPI16@0ms_n | negative_real_background | -38.358 | -43.741 | 0.003 |
| Manual-BG-RPI16@57000ms_n | negative_real_background | -16.467 | -42.908 | 0.003 |
| Manual-BG-RPI16@95250ms_n | negative_real_background | -29.735 | -40.452 | 0.003 |
| Manual-BG-RPI16@120000ms_n | negative_real_background | -25.74 | -44.188 | 0.003 |
| Manual-BG-RPI16@9000ms_n | negative_real_background | -36.139 | -42.869 | 0.003 |
| Manual-BG-RPI16@105000ms_n | negative_real_background | -31.7 | -43.477 | 0.003 |
| Manual-BG-RPI16@40500ms_n | negative_real_background | -29.208 | -38.492 | 0.003 |
| Manual-BG-RPI16@117000ms_n | negative_real_background | -38.935 | -44.385 | 0.003 |
| Manual-BG-RPI16@96750ms_n | negative_real_background | -29.816 | -41.315 | 0.003 |
| Manual-BG-RPI16@29250ms_n | negative_real_background | -33.991 | -42.064 | 0.003 |
| Manual-BG-RPI16@30750ms_n | negative_real_background | -21.741 | -42.511 | 0.003 |
| Manual-BG-RPI16@87000ms_n | negative_real_background | -37.843 | -43.698 | 0.003 |
| Manual-BG-RPI16@84000ms_n | negative_real_background | -39.415 | -44.099 | 0.003 |
| Manual-BG-RPI16@98250ms_n | negative_real_background | -29.021 | -39.552 | 0.003 |
| Manual-BG-RPI16@84750ms_n | negative_real_background | -38.837 | -43.899 | 0.002 |
| Manual-BG-RPI16@11250ms_n | negative_real_background | -37.562 | -43.289 | 0.002 |
| Manual-BG-RPI16@74250ms_n | negative_real_background | -36.152 | -43.831 | 0.002 |
| Manual-BG-RPI16@99750ms_n | negative_real_background | -26.291 | -40.751 | 0.002 |
| Manual-BG-RPI16@46500ms_n | negative_real_background | -25.198 | -42.669 | 0.002 |
| Manual-BG-RPI16@31500ms_n | negative_real_background | -21.421 | -40.568 | 0.002 |
| Manual-BG-RPI16@117750ms_n | negative_real_background | -39.33 | -44.13 | 0.002 |
| Manual-BG-RPI16@18750ms_n | negative_real_background | -34.806 | -43.654 | 0.002 |
| Manual-BG-RPI16@3000ms_n | negative_real_background | -38.024 | -43.531 | 0.002 |
| Manual-BG-RPI16@19500ms_n | negative_real_background | -33.697 | -42.761 | 0.002 |
| Manual-BG-RPI16@56250ms_n | negative_real_background | -16.146 | -39.686 | 0.002 |
| Manual-BG-RPI16@66750ms_n | negative_real_background | -26.62 | -43.871 | 0.002 |
| Manual-BG-RPI16@62250ms_n | negative_real_background | -27.119 | -43.544 | 0.002 |
| Manual-BG-RPI16@80250ms_n | negative_real_background | -39.211 | -43.714 | 0.002 |
| Manual-BG-RPI16@2250ms_n | negative_real_background | -38.557 | -43.7 | 0.002 |
| Manual-BG-RPI16@3750ms_n | negative_real_background | -37.68 | -43.86 | 0.002 |
| Manual-BG-RPI16@60000ms_n | negative_real_background | -33.033 | -42.735 | 0.002 |
| Manual-BG-RPI16@10500ms_n | negative_real_background | -36.516 | -43.08 | 0.002 |
| Manual-BG-RPI16@8250ms_n | negative_real_background | -38.61 | -43.764 | 0.002 |
| Manual-BG-RPI16@115500ms_n | negative_real_background | -29.394 | -42.162 | 0.002 |
| Manual-BG-RPI16@43500ms_n | negative_real_background | -39.267 | -44.178 | 0.002 |
| Manual-BG-RPI16@111000ms_n | negative_real_background | -30.444 | -39.307 | 0.002 |
| Manual-BG-RPI16@1500ms_n | negative_real_background | -38.576 | -43.598 | 0.002 |
| Manual-BG-RPI16@4500ms_n | negative_real_background | -32.504 | -43.266 | 0.002 |
| Manual-BG-RPI16@48750ms_n | negative_real_background | -24.493 | -40.809 | 0.002 |
| Manual-BG-RPI16@21750ms_n | negative_real_background | -36.82 | -41.895 | 0.002 |
| Manual-BG-RPI16@92250ms_n | negative_real_background | -27.167 | -41.291 | 0.002 |
| Manual-BG-RPI16@75750ms_n | negative_real_background | -18.697 | -42.207 | 0.002 |
| Manual-BG-RPI16@118500ms_n | negative_real_background | -29.27 | -44.039 | 0.002 |
| Manual-BG-RPI16@17250ms_n | negative_real_background | -38.728 | -43.09 | 0.002 |
| Manual-BG-RPI16@120750ms_n | negative_real_background | -31.198 | -43.333 | 0.002 |
| Manual-BG-RPI16@750ms_n | negative_real_background | -38.501 | -43.325 | 0.002 |
| Manual-BG-RPI16@64500ms_n | negative_real_background | -39.295 | -44.101 | 0.002 |
| Manual-BG-RPI16@81750ms_n | negative_real_background | -39.548 | -44.16 | 0.002 |
| Manual-BG-RPI16@41250ms_n | negative_real_background | -26.609 | -37.427 | 0.001 |
| Manual-BG-RPI16@77250ms_n | negative_real_background | -39.395 | -43.704 | 0.001 |
| Manual-BG-RPI16@54750ms_n | negative_real_background | -34.059 | -42.023 | 0.001 |
| Manual-BG-RPI16@121500ms_n | negative_real_background | -38.053 | -43.413 | 0.001 |
| Manual-BG-RPI16@119250ms_n | negative_real_background | -25.017 | -43.838 | 0.001 |
| Manual-BG-RPI16@65250ms_n | negative_real_background | -39.446 | -44.08 | 0.001 |
| Manual-BG-RPI16@83250ms_n | negative_real_background | -39.113 | -44.209 | 0.001 |
| Manual-BG-RPI16@44250ms_n | negative_real_background | -39.564 | -44.074 | 0.001 |
| Manual-BG-RPI16@7500ms_n | negative_real_background | -37.932 | -43.053 | 0.001 |
| Manual-BG-RPI16@70500ms_n | negative_real_background | -17.809 | -43.102 | 0.001 |
| Manual-BG-RPI16@69000ms_n | negative_real_background | -27.114 | -42.414 | 0.001 |
| Manual-BG-RPI16@82500ms_n | negative_real_background | -39.122 | -43.725 | 0.001 |
| Manual-BG-RPI16@81000ms_n | negative_real_background | -38.902 | -44.091 | 0.001 |
| Manual-BG-RPI16@37500ms_n | negative_real_background | -26.577 | -42.063 | 0.001 |
| Manual-BG-RPI16@63750ms_n | negative_real_background | -22.617 | -42.984 | 0.001 |
| Manual-BG-RPI16@27000ms_n | negative_real_background | -38.637 | -43.553 | 0.001 |
| Manual-BG-RPI16@61500ms_n | negative_real_background | -38.541 | -43.487 | 0.001 |
| Manual-BG-RPI16@33750ms_n | negative_real_background | -26.466 | -42.199 | 0.001 |
| Manual-BG-RPI16@76500ms_n | negative_real_background | -34.874 | -43.266 | 0.001 |
| Manual-BG-RPI16@72000ms_n | negative_real_background | -39.709 | -44.167 | 0.001 |
| Manual-BG-RPI16@71250ms_n | negative_real_background | -39.659 | -44.256 | 0.001 |
| Manual-BG-RPI16@116250ms_n | negative_real_background | -33.184 | -43.699 | 0.001 |
| Manual-BG-RPI16@66000ms_n | negative_real_background | -26.772 | -43.6 | 0.001 |
| Manual-BG-RPI16@28500ms_n | negative_real_background | -33.951 | -41.908 | 0.001 |
| Manual-BG-RPI16@54000ms_n | negative_real_background | -27.063 | -38.681 | 0.0 |
| Manual-BG-RPI16@79500ms_n | negative_real_background | -25.52 | -43.158 | 0.0 |
| Manual-BG-RPI16@18000ms_n | negative_real_background | -38.854 | -43.91 | 0.0 |
| Manual-BG-RPI16@89250ms_n | negative_real_background | -29.695 | -42.402 | 0.0 |
| Manual-BG-RPI16@55500ms_n | negative_real_background | -34.671 | -40.836 | 0.0 |
| Manual-BG-RPI16@52500ms_n | negative_real_background | -26.517 | -36.244 | 0.0 |
| Manual-BG-RPI16@57750ms_n | negative_real_background | -29.238 | -43.708 | 0.0 |
| Manual-BG-RPI16@53250ms_n | negative_real_background | -26.898 | -29.809 | 0.0 |

## 2. Interventional: change only the level, re-score the same windows

| Change | n (groups) | Peak before → after (dB) | Mean score before → after | Mean change (95% cluster-bootstrap CI) | Above threshold before → after |
|---|---:|---|---|---|---|
| synthetic speech-like negatives -> device level | 670 (151) | -13.5 → -39.8 | 0.015 → 0.015 | -0.000 [-0.003, +0.002] | 4 → 5 |
| synthetic speech-like negatives -> device level + device floor | 670 (151) | -13.5 → -39.6 | 0.015 → 0.088 | +0.073 [+0.059, +0.096] | 4 → 35 |
| synthetic speech-like negatives -> device floor only (level unchanged) | 670 (151) | -13.5 → -13.5 | 0.015 → 0.039 | +0.024 [+0.019, +0.034] | 4 → 18 |
| synthetic speech-like negatives -> device level + pink floor at real-negative level (-43.0 dB) | 670 (151) | -13.5 → -37.7 | 0.015 → 0.091 | +0.076 [+0.060, +0.094] | 4 → 16 |
| synthetic speech-like negatives -> device level + real floor WITH hum (-54.2 dB) | 670 (151) | -13.5 → -39.5 | 0.015 → 0.032 | +0.017 [+0.012, +0.024] | 4 → 9 |
| synthetic speech-like negatives -> device level + real floor WITHOUT hum (-54.2 dB) | 670 (151) | -13.5 → -39.6 | 0.015 → 0.058 | +0.044 [+0.033, +0.059] | 4 → 17 |
| synthetic noise clips -> device level | 100 (100) | -57.6 → -39.8 | 0.018 → 0.039 | +0.021 [+0.017, +0.025] | 0 → 0 |
| synthetic positives -> device level + device floor | 27 (1) | -8.8 → -39.6 | 0.396 → 0.528 | +0.132 [+0.132, +0.132] | 13 → 20 |
| real RPI positives -> synthetic level | 15 (8) | -42.0 → -10.9 | 0.734 → 0.731 | -0.003 [-0.101, +0.061] | 11 → 13 |
| real RPI negatives -> synthetic level | 195 (8) | -32.7 → -10.9 | 0.031 → 0.090 | +0.059 [-0.037, +0.117] | 5 → 11 |
|   negative_confusable -> device level + device floor | 124 (120) | -10.9 → -39.6 | 0.068 → 0.178 | +0.109 [+0.083, +0.135] | 4 → 14 |
|   negative_media -> device level + device floor | 126 (14) | -13.6 → -39.6 | 0.007 → 0.059 | +0.052 [+0.036, +0.069] | 0 → 6 |
|   negative_partial_wakeword -> device level + device floor | 144 (1) | -12.2 → -39.6 | 0.000 → 0.047 | +0.047 [+0.047, +0.047] | 0 → 0 |
|   negative_general_speech -> device level + device floor | 276 (16) | -14.9 → -39.6 | 0.002 → 0.083 | +0.081 [+0.053, +0.118] | 0 → 15 |

Synthetic validation positives by source group: {'en_US-amy-medium': 27}.

