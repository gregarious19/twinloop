# v3 A2 — lumen dataset alignment

8131 healthy slices (ds < 0.25 or NaN, first 50 per file) from 200 files sampled with `default_rng(0)`; 4185 from clean files, 3946 from jit files. IoU(half-max `auto_mask`, T(stored label)).

**Verdict: shape mismatch (identity 0.398, best 0.403 < 0.6)**

| set | median IoU, identity / no shift | n |
|---|---|---|
| dataset, all | 0.398 | 8131 |
| dataset, clean | 0.397 | 4185 |
| dataset, jit | 0.399 | 3946 |
| fresh CPR of GT (20 branches, 5 dev cases) | 0.404 | 755 |

Best (transform, shift) combinations over 8 transforms × 7×7 shifts:

| transform | shift (dy, dx) | median IoU | clean | jit |
|---|---|---|---|---|
| identity | (0, -1) | 0.403 | 0.403 | 0.403 |
| identity | (1, -1) | 0.401 | 0.401 | 0.401 |
| identity | (0, -2) | 0.401 | 0.398 | 0.402 |
| identity | (1, -2) | 0.399 | 0.397 | 0.400 |
| identity | (0, 0) | 0.398 | 0.397 | 0.399 |
| identity | (-1, -1) | 0.398 | 0.396 | 0.401 |
| identity | (-1, -2) | 0.397 | 0.394 | 0.399 |
| identity | (1, 0) | 0.395 | 0.394 | 0.397 |

Nothing was retrained or changed.
