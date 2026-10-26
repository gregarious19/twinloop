# TwinLoop v2 diagnostics 2 (S19): radius definition and widened-rate calibration
Date: 2026-10-26. Commit: c41abde (+ S19 commit). Analysis only: no model, library code, parameter or default changed; `reports/V2_RESULTS.md`, `reports/v2_eval.csv` and the S18 files untouched.
Data: the 20 regression cases, GT masks and the S16/S18 first-pass masks. 0 failed rows. Raw rows: `reports/v2_diag2_radius.csv` (23 281 GT centerline points), `reports/v2_diag2_calib.csv` (100 rows). Tables regenerate with `python scripts/diag_v2b.py --summarize`. The `pred_v1` row here reproduces S16 exactly (max per-case difference 0 in Dice and widened rate).

## 1. Which radius is the lumen net trained on, and what does `refine_radii` write?
- **Training target:** the per-slice area-equivalent radius `r_eq = sqrt(A/π)` of the CPR cross-section (0.2 mm pixels), where A is the area of the centre component of the GT mask (`center_component`, `src/twinloop/cpr.py:92`; `r_eq_mm`, `cpr.py:106`). Training data are built with `center_component` on the straightened GT mask (`train/make_lumen_data.py:40`) and the validation metric is `|r_eq(pred component) − r_eq(GT)|` (`train/train_lumen.py:109-110`).
- **What `refine_radii` writes:** `r_eq_mm` of the predicted centre component per slice (`src/twinloop/lumen.py:59`), smoothed with a Gaussian σ = 1 sample (`lumen.py:64`) and interpolated onto `branch.radius_mm` (`lumen.py:65`).
- **What it is compared to:** the pipeline's v1 radius is the inscribed radius, the Euclidean distance transform of the mask along the skeleton (`src/twinloop/centerline.py:215-219`; no half-voxel term in the default parameters). `stenosis_metrics` compares against that radius. So the code does write an area-equivalent radius into a field that is otherwise inscribed: H-def is possible in principle.

## 2. C1: inscribed vs area-equivalent radius on GT
`r_ins` = the GT branch radius (pipeline/EDT) at the point; `r_eq` = area-equivalent radius of the GT-mask CPR slice, same geometry as the lumen model (48 px, 0.2 mm, 0.5 mm steps), using the component containing the slice centre only (0 if the centre pixel is outside the mask); `ecc = r_eq / r_ins`. Points with `r_ins = 0` are excluded.

| Kind | n | Mean r_eq − r_ins (mm) | Median r_eq − r_ins | Mean ecc | Median ecc | Fraction r_eq = 0 |
|---|---|---|---|---|---|---|
| lesion (all points in a lesion run) | 429 | +0.632 | +0.456 | 1.657 | 1.393 | 0.030 |
| healthy | 22 852 | +0.314 | +0.232 | 1.214 | 1.152 | 0.000 |
| lesion minimum (narrowest point) | 76 | +0.250 | +0.227 | 1.204 | 1.188 | 0.079 |

Lumen radius at the 76 GT lesion minima on branches passing the S18 filter (S18 used all 86 lesions, so its numbers differ slightly). `r_eq` uses the slice-centre component; `r_eq_nn` uses the nearest component when the centre is outside, which is the definition the net was trained and validated on.

| Lumen input | n valid | vs r_ins: mean abs / signed (mm) | vs r_eq (strict): mean abs / signed | vs r_eq_nn: mean abs / signed |
|---|---|---|---|---|
| (i) GT centerlines | 76 | 0.914 / +0.914 | 0.763 / +0.664 | 0.507 / +0.387 |
| (ii) predicted centerlines | 56 | 0.810 / +0.804 | 0.736 / +0.593 | 0.509 / +0.332 |

(ii) has fewer valid points because 20 GT minima have no predicted centerline point within 2 mm.

## 3. C2: widened rate against smoothed GT
GT mask blurred as float (σ_voxels = σ_mm / spacing per axis), thresholded at 0.5, run through the v1 pipeline path (no repair, no lumen), scored against `g_gt` with the S16 metrics. Means over 20 cases; 4.3 GT lesions per case. Biases are the mean signed `r_pred − r_gt` over matched GT points (≤ 2 mm).

| Source | Dice | clDice | Branch recall | False-branch rate | \|r_min\| err (mm) | Widened rate | Healthy bias (mm) | Lesion bias (mm) |
|---|---|---|---|---|---|---|---|---|
| pred_v1 | 0.787 | 0.845 | 0.900 | 0.169 | 0.556 | 0.900 | +0.033 | +0.465 |
| GT σ 0.25 mm | 0.998 | 0.997 | 1.000 | 0.000 | 0.220 | 0.283 | +0.008 | +0.180 |
| GT σ 0.5 mm | 0.956 | 0.993 | 1.000 | 0.000 | 0.337 | 0.518 | −0.006 | +0.252 |
| GT σ 0.75 mm | 0.896 | 0.991 | 1.000 | 0.000 | 0.335 | 0.491 | −0.084 | +0.217 |
| GT σ 1.0 mm | 0.789 | 0.985 | 0.995 | 0.000 | 0.339 | 0.649 | −0.241 | +0.121 |

Widened lesions / GT lesions by GT severity (pooled over the 20 cases; ds from the GT detector, threshold 0.25):

| Source | ds 0.25–0.5 | ds 0.5–0.7 | ds 0.7–1.0 |
|---|---|---|---|
| lesions per bin | 48 | 34 | 4 |
| pred_v1 | 40/48 (0.83) | 33/34 (0.97) | 4/4 (1.00) |
| GT σ 0.25 mm | 4/48 (0.08) | 19/34 (0.56) | 3/4 (0.75) |
| GT σ 0.5 mm | 9/48 (0.19) | 32/34 (0.94) | 4/4 (1.00) |
| GT σ 0.75 mm | 8/48 (0.17) | 31/34 (0.91) | 4/4 (1.00) |
| GT σ 1.0 mm | 16/48 (0.33) | 34/34 (1.00) | 4/4 (1.00) |

## 4. Verdicts
- **Is H-def true? Partly, and not enough.** The two definitions differ: area-equivalent exceeds inscribed by +0.25 mm at lesion minima (ecc 1.20) and +0.31 mm on healthy points (ecc 1.21). Comparing the lumen output like-for-like (GT centerlines, `r_eq_nn`, the trained definition) drops the signed error from +0.914 to +0.387 mm, so about 0.53 mm of 0.91 (58 %) is the definition mismatch. The like-for-like error is still 0.51 mm (0.76 mm with the strict centre-only `r_eq`), well above the ≲ 0.2 mm that would confirm H-def. The rest is a genuine overestimate of the lumen at stenoses, in contrast to its 0.125 mm all-slice validation error.
- **Which smoothing level matches pred_v1, and how much of its 0.90 does smoothing explain?** No single level matches both. Dice 0.787 is matched by σ 1.0 mm (0.789), which gives widened 0.649 (72 % of 0.90) but a healthy bias of −0.241 mm, unlike pred_v1's +0.033. The healthy bias is matched by σ 0.25–0.5 mm (+0.008 / −0.006), giving widened 0.283–0.518 (31–58 % of 0.90). Smoothing alone therefore explains roughly a third to three quarters of the widened rate. Notably, even σ 0.25 mm (Dice 0.998) already widens 28 % of lesions: the metric is very sensitive to tiny mask changes.
- **Are severe GT lesions widened by pred_v1 more than by matched-smoothed GT?** No. For ds ≥ 0.5, pred_v1 widens 37/38 lesions and GT smoothed at σ ≥ 0.5 mm widens 35–38/38; for ds ≥ 0.7 it is 4/4 for both. Only 4 lesions are in the severe bin, so this is weak evidence. The excess of pred_v1 over smoothing is in the mild bin (ds 0.25–0.5): 0.83 versus 0.08–0.33.

## 5. Suggested fixes (not done)
- Use one radius definition across the pipeline: train or convert the lumen output to the inscribed radius (or compute pipeline radii as area-equivalent) so the solver and metric compare like with like. This removes about 0.5 mm of the 0.9 mm lumen error.
- Add a lesion-stratified validation metric for the lumen net (error on slices with ds ≥ 0.25) before any further training, since the 0.125 mm validation error is dominated by healthy slices and hides the remaining ~0.4–0.5 mm overestimate at stenoses.
- Replace or supplement `widened_rate` with a smoothing-aware version (for example scoring against GT blurred at σ ≈ 0.5 mm, or a ds tolerance) and always report it by severity bin, since σ 0.25 mm blur alone widens 28 % of lesions.

## 6. Caveats
n = 20 cases (86 GT lesions, 76 minima on branches passing the S18 filter, only 4 lesions with ds ≥ 0.7); these cases were already used in S16 and S18, so this is diagnostics, not a new test. The strict `r_eq` is 0 at 7.9 % of lesion minima (the GT centre pixel is outside the mask, i.e. centerline/mask misalignment), which is why the nearest-component variant is also shown. Gaussian blur shrinks thin vessels (healthy bias −0.241 mm at σ 1.0 mm), so the smoothed-GT rows are not a perfect stand-in for network error. The 0.53 mm "definition share" is derived from the difference of the two means over the same 76 points. FFR unvalidated.
