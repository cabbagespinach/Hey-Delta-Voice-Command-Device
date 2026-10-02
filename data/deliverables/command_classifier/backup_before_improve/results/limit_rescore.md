# Shorter listener time limit: Pi A/B captures re-scored (balanced rule)

Cut = speech start + limit (6.0 = today's setting). `cut` = mean capture length the model then gets.

## all sessions

| limit | cut (s) | bcresnet6 correct / wrong / asks | bcresnet6_ownernoise correct / wrong / asks | non-cmd -> action |
|---|---|---|---|---|
| 2.5 | 3.4 | 183/215  3  29 | 176/215  1  38 | 2 |
| 3.0 | 3.5 | 182/215  3  30 | 175/215  1  39 | 2 |
| 3.5 | 3.6 | 182/215  3  30 | 175/215  1  39 | 2 |
| 4.0 | 3.7 | 180/215  3  32 | 176/215  1  38 | 2 |
| 4.5 | 3.7 | 181/215  3  31 | 175/215  1  39 | 2 |
| 6.0 | 3.9 | 181/215  3  31 | 176/215  1  38 | 2 |

## TV talk show session

| limit | cut (s) | bcresnet6 correct / wrong / asks | bcresnet6_ownernoise correct / wrong / asks | non-cmd -> action |
|---|---|---|---|---|
| 2.5 | 3.5 | 25/33  0  8 | 26/33  0  7 | 0 |
| 3.0 | 4.0 | 24/33  0  9 | 25/33  0  8 | 0 |
| 3.5 | 4.3 | 24/33  0  9 | 24/33  0  9 | 0 |
| 4.0 | 4.7 | 22/33  0  11 | 26/33  0  7 | 0 |
| 4.5 | 5.0 | 23/33  0  10 | 25/33  0  8 | 0 |
| 6.0 | 6.0 | 23/33  0  10 | 25/33  0  8 | 0 |

## captures that hit the 6 s limit

| limit | cut (s) | bcresnet6 correct / wrong / asks | bcresnet6_ownernoise correct / wrong / asks | non-cmd -> action |
|---|---|---|---|---|
| 2.5 | 3.6 | 11/21  0  10 | 13/21  0  8 | 0 |
| 3.0 | 4.1 | 10/21  0  11 | 12/21  0  9 | 0 |
| 3.5 | 4.6 | 10/21  0  11 | 12/21  0  9 | 0 |
| 4.0 | 5.1 | 9/21  0  12 | 13/21  0  8 | 0 |
| 4.5 | 5.6 | 10/21  0  11 | 12/21  0  9 | 0 |
| 6.0 | 7.1 | 10/21  0  11 | 12/21  0  9 | 0 |

## all other captures

| limit | cut (s) | bcresnet6 correct / wrong / asks | bcresnet6_ownernoise correct / wrong / asks | non-cmd -> action |
|---|---|---|---|---|
| 2.5 | 3.4 | 172/194  3  19 | 163/194  1  30 | 2 |
| 3.0 | 3.5 | 172/194  3  19 | 163/194  1  30 | 2 |
| 3.5 | 3.5 | 172/194  3  19 | 163/194  1  30 | 2 |
| 4.0 | 3.5 | 171/194  3  20 | 163/194  1  30 | 2 |
| 4.5 | 3.5 | 171/194  3  20 | 163/194  1  30 | 2 |
| 6.0 | 3.5 | 171/194  3  20 | 164/194  1  29 | 2 |
