# Progress — v2 (SPARK)

Update at the end of every task and before ending a session. Paste verification output verbatim
(trim long logs to the relevant lines).

## Done
<!-- - [S00] YYYY-MM-DD — <one line> — verification:
```
<output>
```
-->
- [S00] 2026-10-25 — image twinloop:v2 built; TotalSegmentator 2.18.0 installs cleanly; paths/config_v2/v2split + tests — verification:
```
v1 state: branch v2, v1 modules present, v1 tests 58 passed (twinloop:dev); containers: twinloop-train Up, twinloop-server Up
before/after: nnunetv2==2.8.1, torch 2.14.0a0+b2c75dd062.nv26.09 unchanged (diff -> SAME); Python 3.12.3
splits_final.json: <class 'list'> 5 ['train', 'val'] ['ImageCAS_0001', 'ImageCAS_0004', 'ImageCAS_0006']  (matches SPEC)
import totalsegmentator, sklearn, pyarrow, joblib -> ok
tests/test_v2_basics.py: 4 passed; ruff check/format clean
role counts: Counter({'v2_train': 640, 'v2_holdout': 87, 'v2_realfail': 73})
```
- [S01] 2026-10-25 — anatomy.py part 1 (Anatomy, run_totalseg, segment_anatomy, qc) — verification:
```
CPU: tests/test_anatomy.py 5 passed; full suite 67 passed; ruff clean
GPU+data (case 2): 1 passed; myocardium 89.4 mL; sources all heartchambers_highres;
qc {aorta_present: True, aorta_connected: False, myo_present: True, ostia_ok: False}
TotalSegmentator wall: total (roi_subset aorta heart) 291 s incl. weight download; highres 1647 s incl. weight download (~2 min on rerun inside pytest)
ls ts_highres: aorta heart_atrium_left heart_atrium_right heart_myocardium heart_ventricle_left heart_ventricle_right pulmonary_artery
ls ts_total: aorta heart
```
Surprises: class names differ from spec (D-V2-S01a); aorta_connected False on a real case (SPEC ISSUE D-V2-S01c, test only reports it); concurrent TS runs corrupt weight download (D-V2-S01b).
- [S02] 2026-10-25 — graph.py (build_graph, main_path, graph_summary, endings) + phantom kinds taper, two_trees_gap — verification:
```
tests/test_graph.py: 5 passed; full suite: 72 passed; ruff clean
```
Note: two_trees_gap removes only the tube voxels (x > 20 mm), so the second tree's cylinder at x=15 stays intact, per spec ("the tube's voxels").
- [S03] 2026-10-25 — find_ostia, lv_axis, label_branches, coverage_check (anatomy.py part 2); graph.py gained public end_direction/endings_for_tree — verification:
```
tests/test_anatomy_geometry.py + test_graph.py: 14 passed; full suite 81 passed; ruff clean
```
- [S07] 2026-10-25 — cpr.py (phys_to_index_zyx, rmf, straighten_points/straighten, center_component, r_eq_mm) — done BEFORE S06 (order swap, see DECISIONS D-V2-S07a) — verification:
```
tests/test_cpr.py: 5 passed; ruff clean
```

## In progress
WAITING ON OWNER: (1) [licence now works] (2) create the private git remote and push (SPEC §15)
Note: docker needs `sg docker -c "./scripts/dk ..."` in this shell (docker group membership not active in the session).

## BLOCKED
<!-- - [Sxx] error: … | tried: 1) … 2) … 3) … | hypothesis: … | next independent task taken: Syy -->

## Notes for next session
<!-- Observations, sanity-check results, things the owner should look at, bug reports for CLUSTER. -->
