# Lumen v3 training (S23)

## Training-set counts (train part of v2_train, before training)

Files: 17090 train, 2544 validation. Bins by per-slice `ds` (inscribed): healthy < 0.25, mild [0.25, 0.5), moderate [0.5, 0.7), severe >= 0.7. A *run* is a maximal contiguous stretch of slices with ds >= 0.25 inside one file, binned by its maximum ds.

| version | measure | healthy | mild | moderate | severe |
|---|---|---|---|---|---|
| clean | slices | 642253 | 11059 | 2806 | 246 |
| clean | lesion runs | n/a | 2323 | 940 | 122 |
| jit | slices | 629379 | 10152 | 2430 | 191 |
| jit | lesion runs | n/a | 2237 | 832 | 93 |
| all | slices | 1271632 | 21211 | 5236 | 437 |
| all | lesion runs | n/a | 4560 | 1772 | 215 |

## Lumen v3 (S23)

Validation = `jit` branch files of the v2_train validation cases (case_id % 10 == 3); mean |r(pred) − r(GT)| in mm over slices with a GT lumen, per severity bin of the slice (healthy ds < 0.25, mild [0.25, 0.5), moderate+ ≥ 0.5). Model selection = moderate+ r_ins. Kill criterion (moderate+ r_ins > 0.5 mm at 25 % of steps) never triggered.

| model | seed | best step | r_ins healthy | r_ins mild | r_ins mod+ | r_eq healthy | r_eq mild | r_eq mod+ | hours | deployed as lumen_v3 |
|---|---|---|---|---|---|---|---|---|---|---|
| old `lumen` (reference) | 2 | 30000 | 0.177 | 0.344 | 0.453 | 0.120 | 0.286 | 0.478 | 1.28 | n/a |
| lumen v3 | 0 | 26000 | 0.184 | 0.330 | 0.415 | 0.129 | 0.290 | 0.449 | 0.57 | no |
| lumen v3 | 1 | 28000 | 0.183 | 0.334 | 0.425 | 0.127 | 0.286 | 0.446 | 0.58 | no |
| lumen v3 | 2 | 28000 | 0.181 | 0.328 | 0.408 | 0.122 | 0.286 | 0.445 | 0.57 | yes |

Validation slices (healthy / mild / mod+): 96262 / 1420 / 368; the moderate+ figure rests on few slices from few lesions.

Frozen inference settings (validation cases only, see `reports/v3_lumen_freeze_grid.csv`): radius_out=ins, fallback=interp, sigma=0.0. Moderate+ |error| at GT lesion minima on GT centerlines shifted 0.5 mm: 1.086 mm (n = 77 lesions, 85 cases), always an over-estimate of the GT-graph radius. This is far above the slice-level error (~0.4 mm); the ruler differs (GT-graph radius is the 3-D EDT value).

