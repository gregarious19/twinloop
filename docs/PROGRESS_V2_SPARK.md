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
- [S05] 2026-10-25 — metrics.py, scripts/make_regression_set.py, reports/v2_regression_set.json — verification:
```
tests/test_metrics.py: 4 passed
unique regression case ids: 20; rerun of make_regression_set.py exits 1 ("regression set already exists")
case_ids: [236,124,44,914,8,14,752,742,836,144,688,4,366,744,352,6,34,38,68,82]
```
Notes: branch_recall / false_branch_rate / false_branches take `ref: sitk.Image` (or a sitk mask) because physical points need the grid; with no GT branches recall is 1.0 and false-branch rate 0.0.
- [S06] 2026-10-25 — damage.py, scripts/v2_damage_preview.py — verification:
```
tests/test_damage.py: 8 passed
v2_damage_preview --n 10: counts per type {'truncate': 11, 'gap': 16, 'shrink': 14, 'delete_branch': 11, 'false_branch': 5, 'occlusion': 3}; skipped damages: 0; ~4-5 s per case
```
Spec gaps resolved (D-V2-S06a): gap/shrink draw `len` before `s` (s range depends on it); shrink uses s ~ U(5, L-5-len) like gap; a try that changes 0 voxels is rejected.
- [S04] 2026-10-25 — scripts/v2_inspect.py; run on cases 2, 3, 7 (GT masks, highres anatomy) — verification:
```
OK case=2 trees=2 fragments=0 labels=['LAD', 'LCX', 'LM', 'RCA', 'other'] flags=10 t=167
OK case=3 trees=2 fragments=0 labels=['LAD', 'LCX', 'LM', 'RCA', 'other'] flags=11 t=172
OK case=7 trees=2 fragments=0 labels=['LAD', 'LCX', 'LM', 'RCA', 'other'] flags=14 t=138
ls $TWINLOOP_DATA/v2/inspect/*.json | wc -l -> 3
```
Sanity (no parameters tuned, no bugs found): see Notes for next session.
- [S08] 2026-10-25 — regions.py (propose_regions, region_features 31 features, label_synthetic/match_records, label_real), scripts/v2_region_stats.py — verification:
```
tests/test_regions.py: 6 passed
v2_region_stats --n 10 (seed 0): matched/total per type: truncate 9/11, gap 12/16, shrink 13/14, delete_branch 0/11, false_branch 3/5, occlusion 3/3
label counts: {'keep': 279, 'restore': 13, 'extend': 21, 'remove': 3, 'stop': 3}   (no 'add')
mean regions per case 31.9; seconds per case 8.3
```
Surprise: SPEC ISSUE on mask_ahead_frac test (D-V2-S08a).
- [S10] 2026-10-25 — gapfill.py (extract_patch, apply_gapfill), modelio.py (load_model), repair.py (RepairAction, RulesDecider, run_repair) — verification:
```
tests/test_repair.py: 6 passed (OracleNet closes the two_trees_gap gap: > 20 new voxels, one component fewer; all -10 net -> (mask, 0); run_repair applies an extend; gapfill=None -> flagged; load_model errors)
```
Decisions: D-V2-S10a.
- [S11] 2026-10-25 — lumen.py (refine_radii) — verification:
```
tests/test_lumen.py: 4 passed (stenosis min r 0.75±0.15, median 1.5±0.15 from wrong 3.0 mm v1 radii; all -10 model keeps radii and warns; branch < 10 mm unchanged; window starts)
```
- [S17] 2026-10-25 — [x] skipped: ${TWINLOOP_DATA}/raw/mmwhs does not exist (only raw/imagecas).
- [S12] 2026-10-25 — V2Options + v2 argument in run_pipeline, CLI flags, V2Out / CaseResult.v2 — verification:
```
./scripts/dk pytest -q -> 124 passed (tests/test_pipeline_v2.py: 5 new; test_cli_ok_and_failed now monkeypatches segment_anatomy)
real case (id 2, GT mask, highres anatomy):
OK case_id=s12_case2 trees=2 lesions=1 min_ffr=0.901 total_s=163.2
v2: flow_mode=myocardial_mass, myocardium_ml=89.4, qc {aorta_present True, aorta_connected False, myo_present True, ostia_ok False},
    branch_labels RCA/LM/LAD/LCX/other, repair_actions [], review_flags = abrupt_end per ending (no ffr_uncertain), timings segmentation 143.1 s (anatomy) centerline 2.8 s
```
Decisions: D-V2-S12a/b/c. S09 is still waiting for the sensitivity report (see In progress).
- [S09] 2026-10-25 — flow.py (tree_rest_flows, ffr_uncertainty_flags, min_ffr), solver `q_rest_override_m3s`, scripts/v2_sensitivity.py, reports/v2_sensitivity.md — verification:
```
tests/test_flow.py + tests/test_solver.py: 13 passed (worked example 98.0 / 49.0 mL/min; auto without myocardium -> diameter_law; flag at baseline FFR 0.79, none for lesion-free or r_min 0.8)
reports/v2_sensitivity.md: 20 rows (5 cases x 4 modes), cases 2, 3, 7, 9, 11
```
Observation: tree_0 min FFR moves by up to 0.15 between modes within one case (median 0.03; case 11: 0.939 diameter law, 0.828 myocardial mass, 0.785 / 0.872 for mass x1.2 / x0.8). Resting MBF 1.0 mL/min/g kept (D-V2-S09a). FFR unvalidated.
- [C00] 2026-10-26 — environment on SPARK (no venv; twinloop:v2 image) — verification:
```
./scripts/dk python -c "import torch,sklearn;print(torch.__version__,torch.cuda.is_available(),sklearn.__version__)"
2.14.0a0+b2c75dd062.nv26.09 True 1.9.1
df -h ~/twinloop-data: 3.7T total, 2.7T free (>= 100 GB needed: OK); nproc 20; free -g available 104 GB
```
Decision D-V2-S (all training on SPARK).
- [C01] 2026-10-26 — src/twinloop/nets.py (GapfillNet, LumenNet), train/common.py, tests/test_nets.py — verification:
```
tests/test_nets.py: 5 passed; full suite 129 passed; ruff check/format clean
```
Side fix: modelio.load_model now raises RuntimeError("v2 model not available: .../model.pt") if model.pt is missing; test_load_model_errors also drops the cached twinloop.nets attribute now that nets.py exists.
- [C02] 2026-10-26 — data verified (local, no rsync) — verification:
```
imagesTr 800 == labelsTr 800; splits_final.json exists; role counts Counter({'v2_train': 640, 'v2_holdout': 87, 'v2_realfail': 73}) == S00
```
imagesTs/labelsTs exist on SPARK (v1 test split, normal here); the C-tasks never read them (`paths.case_image_path` rejects test ids).

- [C03] 2026-10-26 — datasets (gapfill, lumen done; decider running) — verification so far:
```
gapfill: 20 _done_shard, 640/640 cases, 0 errors, 8474 samples {truncate 1527, taper_neg 1924, gap 1558, shrink 1516, delete_branch 1145, occlusion 804}; 5 % of positives have an empty input mask (D-V2-C03b)
lumen:   20 _done_shard, 19634 branch files, 0 errors, 4.8 GB
```
- [C04/C05 sanity] gap-filler PASS (init 1.6664, Dice 0.9973); lumen PASS (init 1.3663, Dice 1.0000); timing 0.302 s/step (gapfill) and 2.03 s/step (lumen), D-V2-C04b.

- [C06] 2026-10-26 — decider trained, 3 seeds (container twinloop-v2-train-decider, CPU) — verification:
```
dataset: 20 shards, 36760 rows, 640 cases, 0 errors; labels keep 31732, extend 2257, restore 1302, remove 730, stop 568, add 171
seed 0: train 0.7387, val 0.6321 PASS
seed 1: train 0.7406, val 0.6381 PASS
seed 2: train 0.7215, val 0.6301 PASS
seed 0 per-class (val): see metrics.json; weakest classes add (F1 ~0.17-0.22) and stop (~0.49), extend precision ~0.37 (recall 0.84)
```
Note: sklearn's early_stopping='auto' is active for > 10 000 rows, so the 300-iteration fit stops after a few seconds; the spec fixes only max_iter, learning_rate, class_weight, random_state, so this was left as is.

## In progress
- PAUSED by owner request (docker pause twinloop-v2-train-gapfill, 2026-10-26 ~03:30 IST). Resume with: `docker unpause twinloop-v2-train-gapfill` (the chain script keeps waiting on it; do NOT docker stop it, the chain would then start the lumen job).
- Session end 2026-10-26: done S00-S12, S17 (skipped), C00-C03, C06. Waiting on background jobs: GPU chain (gapfill 3 seeds, then lumen 3 seeds, see below; expect gapfill done ~10 h after start, lumen ~2 days more).
- Next when the chain finishes: C07 (export + reports/v2_training.md). Next when `git log main` shows `T09:`: S12M, S13, S14, C08, S15, S16.
- [C04/C05] GPU chain started 2026-10-26: `~/twinloop-data/v2/tmp/scr/chain_train.sh` (log `chain_train.log` next to it). Containers `twinloop-v2-train-gapfill` (seeds 0,1,2; ~3.4 h each) then `twinloop-v2-train-lumen` (seeds 0,1,2; ~17 h each). Runs in `/data/v2/runs/<model>/seed<k>/` (train_log.csv, ckpt.pt, best.pt, metrics.json). Seeds resume from ckpt.pt: if a container died, rerun the same loop for the model (`./scripts/dkd twinloop-v2-train-<m> bash -c 'for s in 0 1 2; do python -m train.train_<m> --data /data/v2/datasets/<m> --out /data/v2/runs/<m>/seed$s --seed $s; done'`) but only for seeds without metrics.json. Check: `docker logs --tail 20 <name>`, `docker inspect -f '{{.State.ExitCode}}' <name>`.
WAITING ON OWNER: (1) [licence now works] (2) create the private git remote and push (SPEC §15)
WAITING: v1 T09 not on main yet (main HEAD = 4e990d1 T05; training still running in twinloop-train). S12M, S13-S16 are blocked on it. S00-S12 are done; only S17 (optional, skipped unless raw/mmwhs exists) is open.
Note: docker needs `sg docker -c "./scripts/dk ..."` in this shell (docker group membership not active in the session).

## BLOCKED
<!-- - [Sxx] error: … | tried: 1) … 2) … 3) … | hypothesis: … | next independent task taken: Syy -->

## Notes for next session
<!-- Observations, sanity-check results, things the owner should look at, bug reports for CLUSTER. -->
### Handoff 2026-10-25
- Next: S12M once `git log main` shows `T09:`. S17: `${TWINLOOP_DATA}/raw/mmwhs` does not exist -> mark skipped.
- WAITING ON OWNER: (2) create the private git remote and push branch v2 (SPEC §15). CLUSTER needs S07/S08-era code via that push.
- Bug reports for CLUSTER: none. Test-suite: 124 passed. Open SPEC ISSUES: D-V2-S01c (aorta_connected), D-V2-S08a (mask_ahead_frac).
- Docker in this shell needs `sg docker -c "./scripts/dk ..."`.

### S08 design-check observations (not tuned)
- `delete_branch` (action `add`) matched 0/11: the decider's `add` class would get no synthetic positives from the current data. Examples (cases 2, 3, 7): the nearest probe region to the deleted branch's origin is 24.9, 22.3 and 48.5 mm away; only 10 probe regions exist in those 3 cases (2–5 per case). Probes are proposed every 5 mm on branches with r >= 1 mm, require a ray with mean HU >= 250 entirely outside the mask and are suppressed near junctions and each other, so the point at a deleted branch's origin is rarely probed. Owner/planner should decide whether to relax probe rules (SPEC change); I did not.
- `gap` 12/16 and `false_branch` 3/5 unmatched cases not examined further (>= 50 %: no action required).

### S04 sanity observations (cases 2, 3, 7; GT masks)
- All 3: 2 trees, 0 fragments, labels LM/LAD/LCX/RCA all present, no graph warnings. Myocardium 84–102 mL, aorta 86–114 mL.
- `aorta_connected` False in 3/3 (known SPEC ISSUE D-V2-S01c: ascending/descending pieces).
- `ostia_ok` True only in case 3. Case 2: left ostium 2.8 mm from the TotalSegmentator aorta, RCA root 19.9 mm away; no RCA point is closer (checked all branch starts/ends), so the GT RCA annotation does not reach the aortic root / TS aorta ends short; not an axis or unit bug. Cases 2 and 7 therefore get `ostia_ok` False.
- Every GT ending is flagged `abrupt_end` (10/11/14 flags, no other kinds): ImageCAS GT vessels stop at r_end 1.0–1.5 mm, which the spec's abrupt rule (r_end >= 0.8 and ratio >= 0.75) calls abrupt. The flag is therefore uninformative on GT masks; expect many on predictions too. Not tuned.
- Time per case ~130–170 s, almost all heartchambers_highres.


- 2026-10-26 10:03 IST: gap-filler training (twinloop-v2-train-gapfill) unpaused by user request; resumed from step ~17100.

### S12M — Merge v1 main into v2 (2026-10-26)
Merged `main` (9c52737, "T09, T10": commit subject is not `T09:` but contains T09) with no conflicts. `app/jobs.py` now passes `V2Options()`; `tests/test_api.py` has an autouse fixture disabling `segment_anatomy`. `./scripts/dk pytest -q` → 130 passed, 2 deselected; ruff check + format clean. Server check skipped: v1 `twinloop-server` is running. `scripts/dk` default trainer is still `nnUNetTrainer` (v1 default, = the full-training trainer whose checkpoint_final exists).

### In progress (2026-10-26 12:42 IST)
- S13: `twinloop-v2-predict-firstpass` (detached, started 12:42 IST) writes /data/v2/firstpass/ (160 cases, regression set first). Then run: `./scripts/dk python scripts/eval_v2.py --cases regression --variants gt_v1,pred_v1 --out reports/v2_eval_firstpass.csv`. `scripts/eval_v2.py` smoke-tested on gt_v1 (case 236 ok).
- C04 gap-filler seed 1 running (`twinloop-v2-train-gapfill`, chain continues to seed 2 then lumen).

### S13 — First-pass evaluation (2026-10-26)
`scripts/eval_v2.py --cases regression --variants gt_v1,pred_v1 --out reports/v2_eval_firstpass.csv` → 40 rows, 40 ok, 0 failed. First-pass masks from `scripts/predict_firstpass.py` (D-V2-S13a).
```
variant  dice   cl_dice  branch_recall  false_branch_rate  widened_rate  mean_abs_rmin_err_mm
gt_v1    1.000  1.000    1.000          0.000              0.000         0.003
pred_v1  0.787  0.845    0.900          0.169              0.900         0.556
```
Lowest branch_recall (pred_v1): case 366 (0.762), case 688 (0.765), case 236 (0.773).
Note: pred_v1 widened_rate 0.90 is very high (nearly every GT lesion looks less stenosed in the first-pass tree). Reported as measured; not tuned on.

### S14 — Real-failure decider dataset (2026-10-26)
`scripts/make_decider_real.py` (v2_realfail only, first-pass masks from `/data/v2/firstpass/`, D-V2-S13a) → `/data/v2/datasets/decider_real/decider_real.parquet`: cases used 73, skipped 0, rows 2675; labels {keep 2219, remove 247, extend 131, restore 70, add 8}. No rsync needed (D-V2-S: training on this machine; C08 reads it in place). Only 8 `add` rows, so `add` stays weak for the fine-tune.

### C04 — Gap-filler trained (2026-10-26)
Seeds 0/1/2 best val new-voxel Dice 0.7856 / 0.7853 / 0.7893 (all 40000 steps, none killed; kill threshold 0.30). C07 will pick seed 2.

### In progress (2026-10-26 15:09 IST)
- C05: `twinloop-v2-train-lumen` (chain_train.sh) started 15:01 IST, seed 0 at ~0.63 s/step (est. ~5 h/seed, ~16 h for 3 seeds; first estimate). Kill check at step 7500. Unblocked after it: C07 → S15 → C08 → S16. Do not `docker stop` or unpause-pause the chain container carelessly: the script waits on it and moves to the next seed on exit.
- Stale: S13/S14 done; predictions in /data/v2/firstpass complete (160).
- WAITING ON OWNER: create a private git remote and push branch v2 (I never push).

### C05 lumen crash fixed (2026-10-26 ~15:45 IST)
First lumen run died at the step-2000 validation: `TypeError: Got unsupported ScalarType BFloat16` in `lumen._predict_logits` (autocast output .numpy()). Fix: `.float()` before `.cpu().numpy()` (src/twinloop/lumen.py), regression test `test_predict_logits_accepts_bfloat16_output`; also removed an unused import in scripts/predict_firstpass.py. 131 passed, ruff clean. Relaunched `twinloop-v2-train-lumen` (seeds 0-2, skips seeds with metrics.json; the old chain_train.sh is finished, nothing else waits on it). Everything runs on SPARK for now (owner decision), so C07 etc. run here too. Check validation passed: `docker logs --tail 5 twinloop-v2-train-lumen` after step 2000.

### C05 / C07 / C08 done on SPARK (2026-10-26)
- C05: lumen seeds 0/1/2 best val r_eq err 0.1251 / 0.1271 / 0.1249 mm (30000 steps each, none killed; best step 28000/30000/30000).
- C07: `train/export_models.py` -> `~/twinloop-data/models/v2/{gapfill,lumen,decider}`; deployed gapfill seed 2 (0.7893), lumen seed 2 (0.1249 mm), decider seed 1 (macro-F1 0.6381). Loading checks pass. `reports/v2_training.md` written.
- C08: `train/finetune_decider.py`; 5-fold grouped CV on real rows macro-F1 0.435 ± 0.055 (synthetic-only on all real rows 0.310). `decider_synth_only/` kept. See report section.
- Everything ran on SPARK by owner decision; no rsync needed.

### S15 — Models integrated (2026-10-26)
`load_model` loads gapfill (GapfillNet), lumen (LumenNet), decider (FullProba). sklearn 1.9.1 in container = `sklearn_version` in decider config. CLI `--repair --lumen`:
```
OK case_id=s15_2 trees=2 lesions=4 min_ffr=0.978 total_s=92.4   (v2_train 2, GT mask)  n_actions=6  review_flags=15
OK case_id=s15_3 trees=2 lesions=7 min_ffr=0.989 total_s=72.0   (v2_train 3, GT mask)  n_actions=6  review_flags=16
OK case_id=s15_1 trees=2 lesions=5 min_ffr=0.980 total_s=80.5   (v2_realfail 1, first-pass mask)  n_actions=37  review_flags=16
```
Timings (s): anatomy 71/51/52, repair 10/10/18, lumen 0.7/0.6/0.6. All < 300 s. Observations (not tuned): lumen step takes <1 s, so I have not verified it changes radii; repair on GT masks still proposes extend/stop actions (GT vessels end at r~1 mm, see S04 notes). FFR values unvalidated.
