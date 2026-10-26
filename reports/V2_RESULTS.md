# TwinLoop v2 results
Date: 2026-10-26. Commit: bcaed45 (+ results commit). Decider: fine-tuned (synthetic + real-failure rows). First-pass trainer: nnUNetTrainer (default; masks from `scripts/predict_firstpass.py`, D-V2-S13a).

FFR values are unvalidated research output, not clinical accuracy. Evaluation ran once; no parameter or model was changed after seeing these numbers.

## Regression set (n = 20, v2_holdout)
Means over 20 cases; `|r_min| err` is the mean over cases (all 20 have ≥ 1 lesion).

| Variant | Success | Dice | clDice | Branch recall | False-branch rate | Widened rate | \|r_min\| err mm | Review flags | Median total s |
|---|---|---|---|---|---|---|---|---|---|
| gt_v1 | 20/20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.003 | 0 | 10.6 |
| pred_v1 | 20/20 | 0.787 | 0.846 | 0.900 | 0.169 | 0.900 | 0.557 | 0 | 8.9 |
| pred_v2_rules | 20/20 | 0.779 | 0.822 | 0.910 | 0.171 | 0.895 | 0.547 | 13.2 | 82.7 |
| pred_v2_repair | 20/20 | 0.790 | 0.833 | 0.890 | 0.095 | 0.895 | 0.551 | 18.9 | 81.5 |
| pred_v2_repair_lumen | 20/20 | 0.790 | 0.833 | 0.890 | 0.095 | 0.854 | 0.856 | 18.9 | 81.5 |

Paired differences over the 20 cases (improved / worse / unchanged counted per case):

| Comparison | Metric | Mean diff | Median diff | Improved | Worse | Unchanged |
|---|---|---|---|---|---|---|
| repair − pred_v1 | branch recall | −0.010 | 0.000 | 1 | 5 | 14 |
| repair − pred_v1 | false-branch rate | −0.074 | −0.084 | 14 | 3 | 3 |
| repair − pred_v1 | widened rate | −0.005 | 0.000 | 1 | 0 | 19 |
| lumen − repair | \|r_min\| err mm | +0.305 | +0.248 | 1 | 19 | 0 |
| lumen − repair | widened rate | −0.042 | 0.000 | 3 | 2 | 15 |

(Improved = lower false-branch/widened/error, higher recall.)

**H1 (repair): FALSIFIED.** Branch recall changed by −0.010 (required ≥ +0.02): 1 case better, 5 worse. The other two conditions held (false-branch rate −0.074, a clear reduction; widened rate −0.005), but H1 fails if any one fails.
**H2 (lumen): FALSIFIED.** Mean |r_min| error rose from 0.551 to 0.856 mm (worse in 19 of 20 cases); widened rate fell by 0.042, but H2 requires lower r_min error.

## Observations (not tuned on)
- Repair mainly removes false branches (false-branch rate 0.169 → 0.095, better in 14/20 cases) but does not recover missing ones: recall fell slightly. Dice is +0.004 and clDice −0.013 versus `pred_v1`.
- The rules-only decider reaches recall 0.910 (+0.009, 3 cases better, none worse) but does not reduce false branches; the learned decider does the opposite. Neither meets H1.
- The learned decider is weak on real failures (5-fold CV macro-F1 0.435 on real rows vs 0.638 on synthetic validation; reports/v2_training.md), consistent with its limited effect.
- Lumen refinement systematically worsens the minimum-radius error. The model's own validation error is small (0.125 mm mean r_eq error), so the gap between that and +0.3 mm at the narrowest point suggests a failure on the first-pass trees or on stenoses specifically. The cause is not investigated here.
- `widened_rate` is ≈ 0.9 in every predicted variant: nearly every GT lesion looks less stenosed in the predicted tree.
- Repair adds about 70 s per case (median 82 s total, dominated by TotalSegmentator anatomy).

## Failures
None: 100/100 rows `ok`.

## Defaults
H1 and H2 are both falsified, so `V2Options.repair` and `lumen` stay False (D-V2-S16).

## Caveats
FFR unvalidated. Single-centre data. One deployed seed per network. Synthetic damage ≠ real errors. The evaluation was run as one 12-row run plus three parallel shards of the same script (D-V2-S16); results are identical to a single run in logic. n = 20 cases, so small differences (|Δ| < 0.02) are within noise.
