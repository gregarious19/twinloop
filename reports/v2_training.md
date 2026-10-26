# TwinLoop v2 training report
Date: 2026-10-26. Commit: 731abe9. Hardware: DGX Spark (all training ran on SPARK).
Validation = held-out `v2_train` cases only; no holdout/realfail/test case used.

| Model | Seed | Val metric | Steps | Hours | Deployed |
|---|---|---|---|---|---|
| gapfill | 0 | Dice 0.7856 | 40000 | 8.83 | no |
| gapfill | 1 | Dice 0.7853 | 40000 | 1.31 | no |
| gapfill | 2 | Dice 0.7893 | 40000 | 2.50 | yes |
| gapfill | mean ± SD | 0.7867 ± 0.0022 | | | |
| lumen | 0 | r_eq err mm 0.1251 | 30000 | 1.47 | no |
| lumen | 1 | r_eq err mm 0.1271 | 30000 | 1.26 | no |
| lumen | 2 | r_eq err mm 0.1249 | 30000 | 1.28 | yes |
| lumen | mean ± SD | 0.1257 ± 0.0012 | | | |
| decider | 0 | macro-F1 0.6321 | n/a | n/a | no |
| decider | 1 | macro-F1 0.6381 | n/a | n/a | yes |
| decider | 2 | macro-F1 0.6301 | n/a | n/a | no |
| decider | mean ± SD | 0.6335 ± 0.0042 | | | |

Hours are wall-clock as logged; the gap-filler seed 0 figure includes a pause (docker pause by the owner), so hours are not comparable between seeds.

## Decider per-class table (validation, deployed seed 1)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| keep | 0.979 | 0.842 | 0.905 | 4072 |
| extend | 0.367 | 0.837 | 0.510 | 306 |
| add | 0.143 | 0.333 | 0.200 | 24 |
| remove | 0.845 | 0.956 | 0.897 | 91 |
| restore | 0.726 | 0.954 | 0.825 | 175 |
| stop | 0.377 | 0.708 | 0.492 | 89 |

`add` has 24 validation rows and very low precision; treat its predictions as weak.

Caveats: one deployed seed per network; synthetic damage is not the same as real segmentation errors.

## Decider fine-tuning (C08)
Training rows: all 36 760 synthetic rows + 2 675 real rows (73 `v2_realfail` cases; first-pass
masks). Per-row weight = base (5.0 real, 1.0 synthetic) × class balance; `class_weight=None`,
other hyper-parameters unchanged, final fit `random_state=0`. 5-fold `GroupKFold` over real
cases; each fold trains on all synthetic rows + real rows of the other folds and is scored on the
held-out real rows only (macro-F1 over labels present).

| Fold | Held-out real rows | Macro-F1 |
|---|---|---|
| 0 | 525 | 0.530 |
| 1 | 542 | 0.419 |
| 2 | 543 | 0.397 |
| 3 | 541 | 0.429 |
| 4 | 524 | 0.398 |
| mean ± SD | | 0.435 ± 0.055 |

Pooled over folds (F1 / support): keep 0.766 / 2219, extend 0.194 / 131, add 0.000 / 8,
remove 0.611 / 247, restore 0.497 / 70. Only 8 real `add` rows exist, so `add` is effectively
untrained on real data.

Reference (not a CV, not used for selection): the synthetic-only decider (seed 1) scores macro-F1
0.310 (accuracy 0.693) on all 2 675 real rows. The synthetic validation macro-F1 (0.638) is not
comparable to the real-row scores: the decider is much weaker on real failures than on synthetic
damage. The fine-tuned model is deployed as `decider`; the previous one is `decider_synth_only`.
