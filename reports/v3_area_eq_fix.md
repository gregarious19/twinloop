# area_eq junction fix (S22)

First 50 `v2_train` case ids (2..77), GT masks only; 50 scored, 0 failed. Lesions: v1 `detect_lesions` on the GT graph (first two trees), binned by `Lesion.ds`: mild [0.25, 0.5), moderate [0.5, 0.7), severe >= 0.7. `area_eq_fixed` skips slices within 1.5 x local inscribed radius + 1.0 mm of a junction and interpolates across them.

| radius | mild | moderate | severe | total | total / inscribed | severe / inscribed |
|---|---|---|---|---|---|---|
| inscribed | 85 | 72 | 13 | 170 | 1.00 | 1.00 |
| area_eq | 370 | 19 | 0 | 389 | 2.29 | 0.00 |
| area_eq_fixed | 148 | 8 | 6 | 162 | 0.95 | 0.46 |

## Adoption criterion (S22; not applied, defaults unchanged)

- area_eq: total 2.29 x inscribed (need <= 1.25), severe 0.00 x inscribed (need >= 0.50) -> DOES NOT MEET the criterion.
- area_eq_fixed: total 0.95 x inscribed (need <= 1.25), severe 0.46 x inscribed (need >= 0.50) -> DOES NOT MEET the criterion.

Cases where the fix changed the lesion count vs unfixed area_eq: 49 of 50.

Caveats: GT-mask lesions on 50 training cases; descriptive only; no clinical claim.
