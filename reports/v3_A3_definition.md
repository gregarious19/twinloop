# v3 A3 — definition selection and Gate G1

Qualifying: retention (modplus survival under σ_b 0.5 mm) >= 0.8 and count >= 0.4 * count(D0). Selection uses sel cases only; dev is shown, never used.

## sel (selection)

| definition | σ_s | modplus count | retention | image-supported (non-calc.) | n non-calc. | qualifies |
|---|---|---|---|---|---|---|
| D0 | 0 | 777 | 0.048 | 0.051 | 673 | no |
| D1 | 0.5 | 634 | 0.038 | 0.050 | 543 | no |
| D2 | 1 | 284 | 0.025 | 0.025 | 237 | no |
| D3 | 1.5 | 85 | 0.024 | 0.014 | 72 | no |
| D4 | 2 | 11 | 0.000 | 0.000 | 9 | no |

## dev (check only)

| definition | σ_s | modplus count | retention | image-supported (non-calc.) | n non-calc. | qualifies |
|---|---|---|---|---|---|---|
| D0 | 0 | 105 | 0.019 | 0.024 | 85 | no |
| D1 | 0.5 | 88 | 0.011 | 0.014 | 70 | no |
| D2 | 1 | 44 | 0.000 | 0.000 | 36 | no |
| D3 | 1.5 | 12 | 0.000 | 0.000 | 11 | no |
| D4 | 2 | 2 | 0.000 | 0.000 | 2 | no |

halfmax_usable: False; dataset_aligned: False

**Gate G1: FAIL.** no definition has retention >= 0.8 with count >= 0.4 * count(D0) = 311; best retention 0.048 at sigma_s 0 (count 777)
