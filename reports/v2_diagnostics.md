# TwinLoop v2 diagnostics (S18): stenosis widening and lumen failure
Date: 2026-10-26. Commit: 2d15d72 (+ S18 commit). Analysis only: no model, parameter or default changed; `reports/V2_RESULTS.md` and `reports/v2_eval.csv` untouched.
Data: the 20 regression cases, GT masks and the same first-pass masks as S16. 23 281 GT centerline points (429 in GT lesions, 22 852 healthy), 86 GT lesions. 0 failed cases. Raw rows: `reports/v2_diag_points.csv`, `reports/v2_diag_lumen.csv`. Tables regenerate with `python scripts/diag_v2.py --summarize`.

Definitions. D1: `signed_err = r_pred − r_gt`, r_pred taken from the pred_v1 tree at the nearest predicted centerline point (valid if ≤ 2 mm). D2: radius read from the predicted mask (EDT in mm, the pipeline's method, trilinear read) at the same GT point: `signed_err_mask = r_predmask_at_gt − r_gt`. Control: the same EDT read on the GT mask (`r_gtmask_at_gt − r_gt`), which isolates the bias of reading an EDT at a point instead of along the smoothed centerline.

## 1. Radius error at GT centerline points (D1 and D2)
| Kind | Points | D1 mean signed (mm) | D1 median signed | D1 mean abs | Fraction d > 2 mm | Fraction same_branch False (d ≤ 2 mm) | D2 mean signed | D2 median signed | D2 mean abs | Fraction outside predicted mask | Control: GT-mask read − r_gt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lesion | 429 | +0.415 | +0.426 | 0.476 | 0.182 | 0.177 | −0.316 | −0.139 | 0.461 | 0.238 | −0.119 |
| healthy | 22 852 | +0.029 | +0.038 | 0.123 | 0.070 | 0.093 | −0.188 | −0.057 | 0.246 | 0.069 | −0.042 |
| all | 23 281 | +0.036 | +0.040 | 0.129 | 0.072 | 0.094 | −0.190 | −0.058 | 0.249 | 0.073 | −0.043 |

D1 mean/median/abs use only points with d ≤ 2 mm. Restricted to points on the same branch (d ≤ 2 mm): lesion n = 289, mean +0.422, median +0.445; healthy n = 19 287, mean +0.027, median +0.038.

## 2. Lumen model by centerline input (D3, 20 cases, 4.3 lesions per case)
| Input | Mean abs r_min err (mm) | Mean signed r_min err (mm) | Widened rate |
|---|---|---|---|
| pred_nolumen (reference: pred_v1 radii, no lumen) | 0.556 | +0.530 | 0.892 |
| (i) gt: GT centerlines | 0.797 | +0.797 | 0.677 |
| (ii) pred: predicted centerlines | 0.829 | +0.827 | 0.859 |
| (iii) gt_shift0.5 | 0.861 | +0.852 | 0.701 |
| (iii) gt_shift1.0 | 0.958 | +0.949 | 0.787 |

Per case (lower mean abs error is better): (i) gt vs (ii) pred: gt better in 8, worse in 12, equal 0. gt vs gt_shift0.5: better in 17, worse in 3. gt vs gt_shift1.0: better in 18, worse in 2. gt vs pred_nolumen: gt better in only 5 of 20. "No lumen found" warnings: 0 in every input. The `pred` input here is the first-pass tree without repair (S16's lumen variant ran after repair).

## 3. Verdicts
- **Is the predicted mask systematically wider?** No, not everywhere. On healthy points the pred_v1 centerline radius is +0.029 mm (median +0.038) above GT, about 3 % of a typical radius. At GT lesion points it is +0.415 mm (median +0.426). So the excess is concentrated at lesions: the predicted mask does not reproduce the GT narrowings.
- **Is the centerline/matching a significant contributor?** Partly, at lesions. 18.2 % of lesion points have no predicted centerline point within 2 mm (healthy 7.0 %), and 17.7 % of matched lesion points land on a different branch (healthy 9.3 %); 23.8 % of lesion points lie outside the predicted mask (healthy 6.9 %). These misses count as widened in the S16 metric. They do not explain the bias itself: with same-branch matches only, the lesion bias is unchanged (+0.422). D2 is not a clean width measurement: reading an EDT at an off-centre point is biased low even on the GT mask (control −0.042 healthy, −0.119 lesion), and D2 − control is still negative (healthy −0.146, lesion −0.197), meaning the GT centerline sits closer to the predicted mask's edge than to its centre. D1 and D2 therefore have opposite signs at lesions, which fits "wider where it is centred, but misplaced relative to GT" rather than a pure mask-width or pure centerline cause.
- **Does the lumen model work on GT centerlines?** No. With GT centerlines the r_min error is 0.797 mm, not the 0.1–0.2 mm that would support the centerline-mismatch hypothesis; it is worse than no lumen at all (0.556 mm; better in 5 of 20 cases), and GT centerlines are no better than predicted ones (better in 8 of 20). The error is almost entirely positive (signed +0.797): the lumen radius at the narrowest point is larger than the mask-EDT radius. Degradation with a perpendicular offset is gentle: 0.797 → 0.861 (0.5 mm) → 0.958 mm (1.0 mm). The centerline-mismatch hypothesis is therefore not the main cause of the S16 lumen result; the centerline contributes a smaller additional error.

## 4. Suggested fixes (not done)
- Check the radius definition: the lumen network is trained and validated on the area-equivalent radius of CPR masks (0.125 mm validation error), while the S16 metric and pipeline radii are EDT (inscribed) radii. Compare the two definitions on GT masks along GT centerlines at the same points; if they differ by ~0.8 mm at stenoses, convert or re-evaluate on one definition. This is a hypothesis; I did not test it.
- Treat `widened_rate` as partly measurement-limited: GT lesions are local dips in the GT mask radius, so a smoother prediction is expected to look wider even when correct (see caveats). Calibrate with an upper-bound control before using it to judge changes.
- Address narrowings at the source: stenosis-aware segmentation (loss weighting or targeted refinement at candidate lesions) rather than post-hoc radius refinement.

## 5. Caveats
n = 20 cases (86 lesions, 429 lesion points); these cases were already used in S16, so this is diagnostics, not a new test. Lesion points are selected as local radius minima of the GT mask, so some positive D1 bias is expected from regression to the mean (GT dips partly reflect mask noise) even for a perfect-but-smoother prediction; this analysis cannot separate that from real under-segmentation of stenoses. The D2 read is point-based and off-centre biased (see control). Matching is nearest-point within 2 mm, as in `stenosis_metrics`. Shift directions are random per branch with seed 0 (one draw, not averaged). FFR unvalidated.
