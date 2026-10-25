# v2 FFR sensitivity to resting-flow mode

GT masks of the 5 smallest-id v2_train cases, highres anatomy, v1 centerlines and lesions. FFR values are unvalidated.

| case | mode | myo_ml | min_ffr tree_0 | min_ffr tree_1 |
|---|---|---|---|---|
| 2 | diameter_law |  | 0.954 | 0.983 |
| 2 | myocardial_mass | 89.4 | 0.901 | 0.980 |
| 2 | mass x0.8 | 89.4 | 0.928 | 0.984 |
| 2 | mass x1.2 | 89.4 | 0.873 | 0.977 |
| 3 | diameter_law |  | 0.978 | 0.913 |
| 3 | myocardial_mass | 101.8 | 0.983 | 0.933 |
| 3 | mass x0.8 | 101.8 | 0.986 | 0.951 |
| 3 | mass x1.2 | 101.8 | 0.979 | 0.914 |
| 7 | diameter_law |  | 0.915 | 0.980 |
| 7 | myocardial_mass | 84.0 | 0.938 | 0.990 |
| 7 | mass x0.8 | 84.0 | 0.953 | 0.992 |
| 7 | mass x1.2 | 84.0 | 0.921 | 0.988 |
| 9 | diameter_law |  | 0.910 | 0.973 |
| 9 | myocardial_mass | 113.0 | 0.864 | 0.976 |
| 9 | mass x0.8 | 113.0 | 0.898 | 0.982 |
| 9 | mass x1.2 | 113.0 | 0.830 | 0.970 |
| 11 | diameter_law |  | 0.939 | 0.975 |
| 11 | myocardial_mass | 73.9 | 0.828 | 0.981 |
| 11 | mass x0.8 | 73.9 | 0.872 | 0.984 |
| 11 | mass x1.2 | 73.9 | 0.785 | 0.977 |

Across modes the min FFR of a tree changes by up to 0.15 (median 0.03) within one case.

The aorta position does not enter the 1D model, so an aortic-root shift has no FFR effect.
