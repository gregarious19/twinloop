# Re-annotated lumen dataset (S25)

19634 files, 1496689 slices. Source `/data/v2/datasets/lumen`, output `/data/v2/datasets/lumen_reannot`.
New mask = pixelwise minimum (AND) of the auto half-maximum mask and the stored label mask, reduced to its centre component; if that is empty the stored label mask is kept. Areas are in pixels (0.2 mm). Not validated against any reference; `ds` is copied from the original files.

| bin | slices | mean label | mean auto | mean new | new/label | fell back to label |
|---|---|---|---|---|---|---|
| healthy (ds<0.25) | 1466017 | 301.1 | 303.9 | 147.7 | 0.49 | 7 |
| mild | 24184 | 321.3 | 493.5 | 177.7 | 0.55 | 52 |
| moderate | 5977 | 315.7 | 725.4 | 207.0 | 0.66 | 325 |
| severe | 511 | 276.9 | 771.5 | 190.2 | 0.69 | 78 |
| all | 1496689 | 301.5 | 308.8 | 148.5 | 0.49 | 462 |
