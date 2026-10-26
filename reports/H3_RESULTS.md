# H3 results — lumen v3 on the 30-case stenosis set (S24)

Pre-registered in SPEC_V2 §13c (S21); run once; nothing changed after seeing results. Cases: `reports/v3_stenosis_set.json`. All rows: first-pass or GT mask as stated, inscribed radii, no repair, no anatomy, v1 diameter-law flow. `pred_lumen3` settings (frozen in S23 on validation cases): radius_out=ins, fallback=interp, sigma=0.0.

Rows ok / total: gt_v1 30/30, pred_v1 30/30, pred_lumen_old 30/30, pred_lumen3 30/30, gt_lumen3 30/30.
Verdict computed on the 30 cases where pred_v1 and pred_lumen3 both ran.

## 1. Stenosis error per source and severity bin

`ds_err_pp` = 100·(ds_pred − ds_gt), negative = stenosis under-called; `rmin_err` = r_pred − r_gt (mm). Bins by ds_gt (§13b). All ok cases of each source.

| source | bin | n | n_missed | mean ds_err_pp | median ds_err_pp | mean abs ds_err_pp | mean rmin_err | mean abs rmin_err |
|---|---|---|---|---|---|---|---|---|
| gt_v1 | mild | 52 | 0 | +0.0 | +0.0 | 0.1 | +0.003 | 0.004 |
| gt_v1 | moderate | 69 | 0 | +0.0 | +0.0 | 0.0 | -0.000 | 0.000 |
| gt_v1 | severe | 22 | 0 | -0.0 | -0.0 | 0.1 | -0.001 | 0.002 |
| gt_v1 | moderate+severe | 91 | 0 | -0.0 | +0.0 | 0.0 | -0.000 | 0.001 |
| pred_v1 | mild | 52 | 13 | -21.2 | -21.4 | 22.5 | +0.283 | 0.323 |
| pred_v1 | moderate | 69 | 21 | -47.9 | -50.6 | 47.9 | +0.774 | 0.789 |
| pred_v1 | severe | 22 | 8 | -63.7 | -64.9 | 63.7 | +1.024 | 1.024 |
| pred_v1 | moderate+severe | 91 | 29 | -51.5 | -53.8 | 51.5 | +0.830 | 0.842 |
| pred_lumen_old | mild | 52 | 13 | -13.9 | -14.2 | 18.3 | +0.472 | 0.499 |
| pred_lumen_old | moderate | 69 | 21 | -45.8 | -50.5 | 45.8 | +1.027 | 1.027 |
| pred_lumen_old | severe | 22 | 8 | -66.8 | -69.1 | 66.8 | +1.264 | 1.264 |
| pred_lumen_old | moderate+severe | 91 | 29 | -50.5 | -54.5 | 50.5 | +1.081 | 1.081 |
| pred_lumen3 | mild | 52 | 13 | -15.9 | -18.6 | 20.1 | +0.348 | 0.394 |
| pred_lumen3 | moderate | 69 | 21 | -45.4 | -48.3 | 45.4 | +0.893 | 0.893 |
| pred_lumen3 | severe | 22 | 8 | -65.5 | -67.7 | 65.5 | +1.154 | 1.154 |
| pred_lumen3 | moderate+severe | 91 | 29 | -49.9 | -52.4 | 49.9 | +0.952 | 0.952 |
| gt_lumen3 | mild | 52 | 0 | -8.9 | -7.5 | 14.2 | +0.294 | 0.337 |
| gt_lumen3 | moderate | 69 | 0 | -26.2 | -32.5 | 27.1 | +0.549 | 0.549 |
| gt_lumen3 | severe | 22 | 0 | -36.7 | -46.2 | 36.7 | +0.723 | 0.726 |
| gt_lumen3 | moderate+severe | 91 | 0 | -28.7 | -34.2 | 29.4 | +0.591 | 0.591 |

## 2. Down-calls and false significant lesions

| source | down50 | down70 | false_sig lesions (total, ok cases) |
|---|---|---|---|
| gt_v1 | 0/91 | 0/22 | 6 |
| pred_v1 | 61/91 | 14/22 | 14 |
| pred_lumen_old | 61/91 | 14/22 | 6 |
| pred_lumen3 | 62/91 | 14/22 | 9 |
| gt_lumen3 | 63/91 | 16/22 | 8 |

## 3. H3 verdict

**H3 is FALSIFIED.**

| condition | pred_lumen3 | pred_v1 | difference | holds |
|---|---|---|---|---|
| PRIMARY: mean ds_err_pp (moderate+severe) pred_lumen3 − pred_v1 >= +10 pp | -49.90 | -51.47 | +1.57 | NO |
| GUARD (a): down50 pred_lumen3 <= pred_v1 | 62.00 | 61.00 | +1.00 | NO |
| GUARD (b): false_sig_lesions increase <= 5 | 9.00 | 14.00 | -5.00 | yes |
| GUARD (c): mild-bin mean |ds_err_pp| increase <= 5 pp | 20.15 | 22.53 | -2.38 | yes |

## 4. Secondary (no verdict): minimum FFR per case versus gt_v1

Cases with gt_v1 min FFR < 0.80: 16 of 30.

| source | n cases | mean abs diff vs gt_v1 | gt_v1 <0.80 also <0.80 | false calls (<0.80 where gt_v1 >= 0.80) |
|---|---|---|---|---|
| pred_v1 | 30 | 0.184 | 0/16 | 1 |
| pred_lumen_old | 30 | 0.202 | 0/16 | 0 |
| pred_lumen3 | 30 | 0.198 | 0/16 | 0 |
| gt_lumen3 | 30 | 0.140 | 5/16 | 1 |

Lumen v3 fallback fraction (slices whose centre pixel is not in a predicted component), pred_lumen3: mean 0.000, max 0.000.

## 5. Caveats

- Lesions are defined on GT-mask radius profiles, not invasive angiography; FFR values are unvalidated (no invasive-FFR data) and not clinically accurate.
- n = 30 cases, single centre (ImageCAS); case set enriched for GT stenoses by design.
- `gt_lumen3` (GT mask + lumen v3) is an oracle for perfect centerlines, reported without a verdict; `pred_lumen_old` is the deployed lumen model, for reference.
- Missed lesions (no predicted point within 2 mm) are excluded from error means and counted in `n_missed`; `down50` counts only matched lesions.
