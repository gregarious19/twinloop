# TwinLoop v3 — tasks (SPARK)

Work top to bottom. Spec references are to docs/SPEC_V3.md. Commit after each task:
`git commit -m "V<nn>: <title>"`. Every GPU task: check `nvidia-smi` shows no other twinloop
process and `free -g` shows ≥ 20 GB available; otherwise wait (`sleep 1800`) and recheck.
Log every GPU job's wall hours in PROGRESS_V3 ("GPU hours used: x of 96").

---

## [ ] V00 — Session setup and state check
**Depends on:** nothing. **Files:** docs/PROGRESS_V3.md.
**Steps:**
1. `uname -m` → `aarch64`; `git branch --show-current` → `v3`; `git log --oneline -3` shows
   `V00: v3 handoff docs` on top (the owner's commit).
2. `cat scripts/dk scripts/dkd`; confirm `./scripts/dk head -1 CLAUDE.md` prints `# TwinLoop — CLAUDE.md (v3)`.
3. `docker ps -a --filter name=twinloop`: copy the table into PROGRESS_V3 "Notes". If
   `twinloop-v2-xfit` is running, note it: GPU tasks must wait for it (never stop it).
4. `./scripts/dk python -c "from twinloop.v2split import load_roles; from collections import Counter; r=load_roles(); print(Counter(r.values())); print(sum(1 for c,v in r.items() if v=='v2_train' and c%10==3))"`.
5. `./scripts/dk pytest -q` and both ruff commands; record pass counts.
6. `./scripts/dk env | grep -i nnunet` (expect nnUNet_raw, nnUNet_preprocessed, nnUNet_results
   under /data); `ls /data/v2/firstpass | wc -l`, `ls /data/v2/datasets/lumen | head -3`, `ls /data/v2/datasets/lumen_reannot | wc -l`.
**Acceptance:** roles = `{'v2_train': 640, 'v2_holdout': 87, 'v2_realfail': 73}`; dev count printed;
tests and ruff pass; firstpass count 160.
**Verify:** paste outputs of steps 1–6. Expected step 4: `Counter({'v2_train': 640, 'v2_holdout': 87,
'v2_realfail': 73})` then a number near 85.

## [ ] V01 — Radius smoothing, blur, v3 paths, pipeline option
**Depends on:** V00. **Files:** `src/twinloop/radius_smooth.py`, `src/twinloop/paths.py`,
`src/twinloop/pipeline.py`, `tests/test_radius_smooth.py`, `tests/test_pipeline_v3.py`.
**Steps:**
1. Add `v3_dir()` to `paths.py` (§2.1).
2. Write `radius_smooth.py`: `smooth_radius`, `smooth_trees`, `smooth_graph` (§4.2) and
   `blur_mask` (§4.1). Read `scripts/diag_v2b.py` first for the S19 blur; keep §4.1 behaviour.
3. Golden file (before touching `pipeline.py`): run the `stenosis` phantom (build it the way
   `tests/test_pipeline_phantom.py` does) through `run_pipeline` with
   `V2Options(anatomy=False, flow_mode="diameter_law")` and store `{"trees": ..., "lesions": ...}` of
   its `result.json` as `tests/data/v3_golden_stenosis.json` (create `tests/data/` if missing; this
   small JSON may be committed).
4. Add `radius_smooth_mm: float = 0.0` to `V2Options`; in `run_pipeline`, right after the lumen
   block and before `lesion_maps = ...`, call `smooth_trees(trees, v2.radius_smooth_mm)` when
   `v2 is not None and v2.radius_smooth_mm > 0`. (`halfmax` comes in V08.)
5. Tests: the three worked examples of §4.2 (tolerance 1e-3); σ 0 returns a copy (not the same
   object); `blur_mask` on a 20×20×20 cube keeps origin/spacing/direction and values ⊂ {0,1};
   `tests/test_pipeline_v3.py`: (a) the phantom run after the edit, with and without
   `radius_smooth_mm=0.0`, equals the golden `trees` and `lesions` exactly; (b) with
   `radius_smooth_mm=1.0` the minimum radius over all branches is ≥ the golden minimum radius.
**Acceptance:** all tests pass; ruff clean.
**Verify:** `./scripts/dk pytest -q tests/test_radius_smooth.py tests/test_pipeline_v3.py` → all
passed; `./scripts/dk pytest -q` → all passed (count = V00 count + new tests).

## [ ] V02 — Vertical slice: lesion audit on 3 cases
**Depends on:** V01. **Files:** `src/twinloop/halfmax.py` (only `auto_mask` now),
`scripts/v3_lesion_audit.py`, `scripts/v3_audit_report.py`, `tests/test_halfmax.py`.
**Steps:**
1. Read `src/twinloop/cpr.py` (`straighten`, `center_component`, `r_ins_2d_mm`) and resolve the
   first row of §16. Log the answer in DECISIONS_V3.
2. Move `auto_mask` into `halfmax.py` with the §4.5 fix and the HU clip; test: blank slice →
   (48,48) all False; a disc of 500 HU radius 8 px on 0 HU → mask area within 15 % of the disc.
3. Write `v3_lesion_audit.py` exactly as §6.2 (`--cases first3` = the 3 lowest `v2_train` ids).
4. Write `v3_audit_report.py` producing the §6.2 table (selection logic comes in V05).
5. Run: `./scripts/dk python scripts/v3_lesion_audit.py --cases first3 --sigmas 0,1.0 --out /data/v3/audit/slice.csv`
   then `./scripts/dk python scripts/v3_audit_report.py --csv /data/v3/audit/slice.csv --out /data/v3/audit/slice.md`.
**Acceptance:** CSV has the §6.2 header; 3 cases, every row `status=ok`; for σ_s 0 the lesion count
equals `sum(len(detect_lesions(b)))` over GT branches of those cases (print both); `survive_*`,
`image_ds`, `iou_ref` non-empty for most rows; runtime per case printed (< 120 s).
**Verify:** `head -3 /data/v3/audit/slice.csv`; `./scripts/dk cat /data/v3/audit/slice.md`; paste the lesion-count check.

## [ ] V03 — A2: lumen-dataset alignment diagnosis (CPU)
**Depends on:** V02 (uses `auto_mask`). **Files:** `scripts/v3_a2_alignment.py`,
`reports/v3_A2_alignment.md`, `reports/v3_A2_alignment.json`.
**Steps:** exactly §6.1 steps 1–5 and the outcome rule.
**Acceptance:** both report files exist; JSON has all six keys; the verdict line is one of the
three in §6.1; the clean/jit split is shown.
**Verify:** `cat reports/v3_A2_alignment.json` and the verdict line.

## [ ] V04 — A1: full lesion audit on all 640 v2_train cases (CPU)
**Depends on:** V02. **Files:** `/data/v3/audit/lesions.csv`, `reports/v3_A1_lesion_audit.md`.
**Steps:**
1. Run detached: `./scripts/dkd twinloop-v3-audit python -u scripts/v3_lesion_audit.py --cases all
   --sigmas 0,0.5,1.0,1.5,2.0 --out /data/v3/audit/lesions.csv --workers 8`. Expected ≈ 30–60 min.
   Limit 4 h.
2. When done: `./scripts/dk python scripts/v3_audit_report.py --csv /data/v3/audit/lesions.csv
   --out reports/v3_A1_lesion_audit.md`.
**Acceptance:** 640 distinct case_ids in the CSV; failed cases listed in the report (≤ 10 allowed;
more → BLOCKED protocol); report shows sel / dev / all tables and the `halfmax_usable` line.
**Verify:** `./scripts/dk python -c "import pandas as p; d=p.read_csv('/data/v3/audit/lesions.csv'); print(d.case_id.nunique(), (d.status!='ok').sum())"` → `640 <small number>`.

## [ ] V05 — A3: choose the definition, Gate G1
**Depends on:** V03, V04. **Files:** `scripts/v3_audit_report.py` (`--select`),
`reports/v3_definition.json`, `reports/v3_A3_definition.md`, DECISIONS_V3.
**Steps:**
1. Implement §6.3 exactly; run `./scripts/dk python scripts/v3_audit_report.py --csv /data/v3/audit/lesions.csv --select`.
2. Log σ* (or G1 failure) as a decision `D-V3-V05` with the numbers.
3. If G1 fails: mark V06–V13 `[x] skipped (G1 failed)`, commit, go to V14.
**Acceptance:** JSON matches §6.3 keys; selection used sel cases only (the script asserts every
case it selects on has `id % 10 != 3`); report table covers D0–D4 on sel and dev.
**Verify:** `cat reports/v3_definition.json`.

## [ ] V06 — eval_v3.py; A4 re-score of the 50 seen cases, Gate G2
**Depends on:** V05 (G1 pass). **Files:** `scripts/eval_v3.py`, `scripts/v3_h4_report.py`
(`--mode a4`), `reports/v3_A4_rescore.md`, `reports/v3_a4_cases.csv`, `reports/v3_a4_lesions.csv`.
**Steps:**
1. Write `eval_v3.py` (§10) and `v3_h4_report.py` (§8 metrics; `--mode a4` table, `--mode h4`
   verdicts; h4 mode used only in V13).
2. Smoke: `--cases seen --variants gt_ref,pred_v1 --sigma 0 --limit 2` (add a `--limit` option)
   into `/data/v3/smoke/` files (`<task>_cases.csv`, `<task>_lesions.csv`); check the 2 `gt_ref` rows have `dice` 1.0.
3. Run: `--cases seen --variants gt_ref,gt_blur025,gt_blur05,pred_v1` for `--sigma 0` and
   `--sigma <σ*>` into `reports/v3_a4_cases.csv` / `reports/v3_a4_lesions.csv` (both sigmas in the
   same files).
4. Report + G2 (§6.4). Sanity: at σ 0, restricted to the 30 stenosis-set cases, gt_ref must give 16
   positives and pred_v1 must call 0 of them (reports/H3_RESULTS.md §4 used the same options:
   inscribed, no anatomy, diameter-law flow). If not, find why before continuing (BLOCKED protocol).
5. If G2 fails: mark V07–V13 `[x] skipped (G2 failed)`, commit, go to V14.
**Acceptance:** 50 × 4 × 2 = 400 case rows, failures listed; the σ 0 stenosis-set check matches;
G2 verdict written in the report and in DECISIONS_V3 (`D-V3-V06`).
**Verify:** `./scripts/dk python scripts/v3_h4_report.py --mode a4 --cases reports/v3_a4_cases.csv --lesions reports/v3_a4_lesions.csv --out reports/v3_A4_rescore.md` and paste the G2 line.

## [ ] V07 — H4 case set and pre-registration (commit BEFORE any Phase B training)
**Depends on:** V06 (G2 pass). **Files:** `scripts/v3_make_h4_set.py`, `reports/v3_h4_set.json`,
`reports/v3_h4_prereg.json`.
**Steps:**
1. H4 set per §2.3; assert no id % 5 == 0, none of the 50 seen ids, every case has a first-pass mask.
2. Count positives from GT only: run `eval_v3.py --cases h4 --variants gt_ref --sigma <σ*>
   --out /data/v3/eval/h4_gt_ref_cases.csv --lesions-out /data/v3/eval/h4_gt_ref_lesions.csv`
   (reference only, no prediction is run) and count P (rows with `min_ffr` < 0.80).
   These rows are reused in V13 (same command, same output files: copy them into the V13 files).
3. Write `reports/v3_h4_set.json`: `{"case_ids": [...], "rule": "<§2.3 text>", "n": n}`.
4. Write `reports/v3_h4_prereg.json`: `{"sigma_s_mm": σ*, "n_cases": n, "n_positive": P,
   "underpowered": P < 10, "arms": {"arm1": true, "arm2": <halfmax_usable>, "arm3": <dataset_aligned>},
   "criteria": "SPEC_V3 §7 at commit <git rev-parse HEAD>"}`.
5. Commit with message `V07: H4 pre-registration`.
**Acceptance:** both files committed; `git log -1 --format=%H` recorded in PROGRESS_V3; no GPU
training has started yet in v3.
**Verify:** `cat reports/v3_h4_prereg.json`; `git show --stat HEAD`.

## [ ] V08 — Arm 2: half-max narrowing (skip if `arms.arm2` is false)
**Depends on:** V07. **Files:** `src/twinloop/halfmax.py`, `src/twinloop/pipeline.py`,
`tests/test_halfmax.py`.
**Steps:**
1. If `arms.arm2` is false: mark `[x] skipped (halfmax not usable, V04)` and stop.
2. Implement `halfmax_narrow` (§9.3); add `halfmax: bool = False` to `V2Options`; call it after the
   lumen block and before smoothing (§5).
3. Tests: on the `stenosis` phantom, with the mask dilated by 1 voxel, no branch radius increases and
   at least one decreases; default options unchanged output (extend V01's identity test).
4. Smoke: `eval_v3.py --cases dev10 --variants arm2_halfmax --sigma <σ*>` into `/data/v3/smoke/` files (`<task>_cases.csv`, `<task>_lesions.csv`).
**Acceptance:** tests pass; smoke §11 rule (≥ 9/10 ok).
**Verify:** `./scripts/dk pytest -q tests/test_halfmax.py` and the smoke status counts.

## [ ] V09 — Arm 1: Dataset512 (CPU)
**Depends on:** V07. **Files:** `scripts/v3_dataset512.py`.
**Steps:**
1. Read `scripts/xfit_setup.py` (how Dataset511 copied plans and symlinked images).
2. Build `/data/nnUNet_raw/Dataset512_ImageCASlesion/` per §9.1 (640 cases, labels 0/1/2 under σ*),
   `dataset.json`, plans copy with `dataset_name` changed, `splits_final.json` (one fold).
3. `./scripts/dk nnUNetv2_preprocess -d 512 -plans_name nnUNetPlans -c 3d_fullres -np 8` (§16 if
   it asks for a fingerprint). Limit 3 h.
4. Print: label voxel counts summed over all cases per class; number of cases with ≥ 1 lesion voxel.
**Acceptance:** 640 images and 640 labels; every label has the same size/spacing/origin as its
image (script asserts); preprocessing exits 0; the share of cases with ≥ 1 lesion voxel is
reported, and if it is below 30 % write BLOCKED (lesion class too rare to matter) and go to V14.
**Verify:** `ls /data/nnUNet_raw/Dataset512_ImageCASlesion/labelsTr | wc -l` → 640; paste step-4 counts.

## [ ] V10 — Arm 1: sanity run, then 3 replicates (GPU)
**Depends on:** V09. **Files:** `scripts/v3_train_arm1.sh`, `reports/v3_arm1_training.md`.
**Steps:**
1. Verify §16 rows on `-pretrained_weights` and `nnUNetTrainer_250epochs`; log answers.
2. Sanity (§11): `nnUNetTrainer_5epochs` into `/data/v3/nnunet_results/sanity`; check the three
   sanity conditions; predict 1 dev case.
3. `scripts/v3_train_arm1.sh`: loop k = 0,1,2; skip k if
   `/data/v3/nnunet_results/rep{k}/Dataset512_ImageCASlesion/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth`
   exists; run the §9.1 train command; then the next k. Launch: `./scripts/dkd twinloop-v3-arm1 bash scripts/v3_train_arm1.sh`.
4. At epoch 50 of each replicate, apply the kill rule (§11); enforce the 12 h limit.
5. When all finish: `reports/v3_arm1_training.md` table: replicate, epochs done, final vessel /
   lesion pseudo-Dice, wall hours, killed (y/n), validation crash (y/n).
**Acceptance:** 3 replicates finished or killed, each documented; GPU hours logged.
**Verify:** `ls /data/v3/nnunet_results/rep*/Dataset512_ImageCASlesion/*/fold_0/checkpoint_final.pth`; `cat reports/v3_arm1_training.md`.

## [ ] V11 — Arm 3: LumenNet on re-annotated data (GPU; skip if `arms.arm3` is false)
**Depends on:** V07; GPU free (after V10's chain). **Files:** `reports/v3_arm3_training.md`.
**Steps:**
1. If `arms.arm3` is false: mark `[x] skipped (dataset not aligned, V03)` and stop.
2. Train seeds 0,1,2 (§9.2) in one detached container `twinloop-v3-arm3`; 3 h limit per seed.
3. Export each seed as `lumen_v3_reannot_s{k}`; load each with `twinloop.modelio.load_model`.
4. Smoke: `eval_v3.py --cases dev10 --variants arm3_s0,arm3_s1,arm3_s2 --sigma <σ*>` into `/data/v3/smoke/` files (`<task>_cases.csv`, `<task>_lesions.csv`).
5. Report table: seed, best step, validation errors as printed by the script, killed, hours.
**Acceptance:** 3 seeds finished or killed; exported models load; smoke rule met.
**Verify:** `cat reports/v3_arm3_training.md`.

## [ ] V12 — Arm 1 predictions (GPU)
**Depends on:** V10. **Files:** `scripts/v3_predict_arm1.py`.
**Steps:**
1. For each non-killed replicate k: predict `dev10` cases and the H4 cases (§9.1 inference) into
   `/data/v3/pred/arm1_rep{k}/`; resumable; binary masks.
2. Smoke: `eval_v3.py --cases dev10 --variants arm1_rep0,arm1_rep1,arm1_rep2 --sigma <σ*>` into
   `/data/v3/smoke/` files (`<task>_cases.csv`, `<task>_lesions.csv`); apply the §11 smoke rule (Dice ≥ 0.60).
**Acceptance:** H4-case mask count = n_cases per non-killed replicate; smoke rule met.
**Verify:** `ls /data/v3/pred/arm1_rep*/ | wc -l` per folder; smoke summary.

## [ ] V13 — H4 evaluation (ONCE)
**Depends on:** V08, V11, V12 (each done or skipped). **Files:** `reports/v3_h4_cases.csv`,
`reports/v3_h4_lesions.csv`, `reports/V3_RESULTS.md`.
**Steps:**
1. `cp /data/v3/eval/h4_gt_ref_cases.csv reports/v3_h4_cases.csv` and
   `cp /data/v3/eval/h4_gt_ref_lesions.csv reports/v3_h4_lesions.csv`.
2. Run once, detached (`twinloop-v3-h4`), limit 8 h:
   `eval_v3.py --cases h4 --variants gt_ref,gt_blur05,pred_v1,<included arm variants> --sigma <σ*>
   --out reports/v3_h4_cases.csv --lesions-out reports/v3_h4_lesions.csv` (gt_ref rows are skipped as done).
3. `./scripts/dk python scripts/v3_h4_report.py --mode h4 --cases reports/v3_h4_cases.csv
   --lesions reports/v3_h4_lesions.csv --prereg reports/v3_h4_prereg.json --out reports/V3_RESULTS.md`
   following the §13 template.
4. Log the verdict as `D-V3-V13`. No default changes in any case (D-V3-12).
**Acceptance:** one row per (case, variant); failures listed; verdict per arm and for H4 per §7.
**Verify:** `grep -n "H4" reports/V3_RESULTS.md | head`.

## [ ] V14 — Phase C (conditional) and wrap-up
**Depends on:** V05 or V06 (gate failure), or V13.
**Files:** `reports/v3_external_data.md`, docs/PROGRESS_V3.md.
**Steps:**
1. If G1 or G2 failed or H4 is not SUPPORTED: write `reports/v3_external_data.md` (§14). Otherwise
   mark this step skipped.
2. Final checks: `./scripts/dk pytest -q`, both ruff commands.
3. PROGRESS_V3 "Notes for next session": a 10-line summary (gates, σ*, verdicts, GPU hours used,
   open questions) and `WAITING ON OWNER: git push origin v3`.
**Acceptance:** tests and ruff pass; the report the gates lead to exists; summary written.
**Verify:** paste test/ruff output and the summary.
