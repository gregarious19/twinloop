# TwinLoop v2 — SPARK tasks (DGX Spark)

Work strictly in order unless a task is BLOCKED or WAITING ON OWNER (then do the next task whose
dependencies are met). Paths are relative to the repo root. Commands run on the host from the repo
root; `./scripts/dk <cmd>` runs `<cmd>` in the container. `SPEC §x` = docs/SPEC_V2.md section x.
`v1 Tn` = task Tn in docs/TASKS.md. Log verification output in docs/PROGRESS_V2_SPARK.md.
Commit after each task: `git add -A && git commit -m "S<nn>: <title>"` (the commit message prefix is
how the CLUSTER session checks dependencies — keep it exact).

**Order of work on SPARK:** S00 → S12 (v2 Track A, needs no trained model) → S12M (merge v1 `main`
into `v2`, once the v1 session has finished v1 T09) → S13 → S14 → S15 → S16. S17 is optional.
You work in the v2 worktree on branch `v2` (CLAUDE.md "Where v2 lives"). v1 tasks are never yours.

**Never** stop, restart or remove the `twinloop-train` container. Before any GPU run (TotalSegmentator,
nnU-Net, model inference) check `free -g`; if the "available" column is < 20, wait or do a CPU task.

---

## [x] S00 — v1 state check, dependencies, v2 scaffolding
**Depends on:** none
**Files:** `requirements.txt`, `requirements.lock`, `Dockerfile` (only if the fallback is needed), `scripts/dk`, `scripts/dkd`, `src/twinloop/paths.py`, `src/twinloop/config_v2.py`, `src/twinloop/v2split.py`, `tests/test_v2_basics.py`, `docs/PROGRESS_V2_SPARK.md`
**Steps:**
1. Record v1 state in PROGRESS_V2_SPARK.md: `git log --oneline | head -30`, `ls src/twinloop`, `docker ps -a --filter name=twinloop --format '{{.Names}} {{.Status}}'`, and `./scripts/dk pytest -q` (summary line only).
2. Track A needs these v1 modules working: `config.py io.py phantom.py centerline.py lesions.py solver.py schema.py casefiles.py pipeline.py segmentation.py` (segmentation may be the T04 stub). Also run `git branch --show-current` (must print `v2`) and `git log --oneline main | head -5`. If any listed module is missing or its v1 tests fail, write BLOCKED: "v1 T02–T04 incomplete on branch v2 — owner: tell the v1 session to finish them, then run S12M early" and stop (do not do v1 tasks yourself).
3. Record the versions that must not change: `./scripts/dk pip freeze | grep -iE "^(torch|nnunetv2)==" | tee /tmp/before.txt` (paste into PROGRESS). Also `./scripts/dk python --version` → write into docs/DECISIONS_V2.md under "Filled in during execution" as `D-V2-S00a — Container Python <x.y.z>`.
4. In `scripts/dk` and `scripts/dkd`, change the image name `twinloop:dev` to `twinloop:v2` (the v1 session keeps using `twinloop:dev`; never rebuild or retag `twinloop:dev`). Append to `requirements.txt`: `scikit-learn`, `pyarrow`, `joblib`, `TotalSegmentator` (unpinned for now). Build: `docker build -t twinloop:v2 .`. In every later step of all tasks, "rebuild" means `docker build -t twinloop:v2 .`. Never run `scripts/dkd` with the names `twinloop-train` or `twinloop-prep` (v1's containers); if you need a detached job, name it `twinloop-v2-<purpose>` and update the allowed names in `scripts/dkd`'s comment.
   - If pip fails to resolve TotalSegmentator against the pinned `nnunetv2`: remove `TotalSegmentator` from requirements.txt; create `requirements-nodeps.txt` with `TotalSegmentator`; add to the Dockerfile, after the existing RUN line: `COPY requirements-nodeps.txt /tmp/requirements-nodeps.txt` and `RUN pip install --no-deps -r /tmp/requirements-nodeps.txt`; rebuild; run `./scripts/dk pip check`; for every reported *missing* requirement of TotalSegmentator (not version complaints about torch/nnunetv2), add the package to requirements.txt; rebuild. Log in DECISIONS_V2.md.
5. Verify unchanged: `./scripts/dk pip freeze | grep -iE "^(torch|nnunetv2)==" | diff /tmp/before.txt -` must print nothing. If it differs, revert requirements.txt/Dockerfile to the last commit, rebuild, and use the step-4 fallback.
6. Pin: `./scripts/dk pip freeze > requirements.lock`; rewrite the four new lines as `name==<version from requirements.lock>` (TotalSegmentator in whichever file it lives). Rebuild. Log versions in DECISIONS_V2.md.
7. In `scripts/dk` and `scripts/dkd`, add `-e TOTALSEG_HOME_DIR=/data/totalseg \` after the `-e TWINLOOP_DATA=/data` line. `mkdir -p "${TWINLOOP_DATA:-$HOME/twinloop-data}/totalseg" "${TWINLOOP_DATA:-$HOME/twinloop-data}/v2" "${TWINLOOP_DATA:-$HOME/twinloop-data}/models/v2"`.
8. Write `paths.py` (SPEC §2.1), `config_v2.py` (SPEC §3, verbatim), `v2split.py` (SPEC §2.2). First inspect the split file and paste the output: `./scripts/dk python -c "import json;js=json.load(open('/data/nnUNet_preprocessed/Dataset501_ImageCAS/splits_final.json'));print(type(js),len(js),list(js[0].keys()),js[0]['val'][:3])"`. If the format differs from SPEC §2.2, adapt the parser only and log it.
9. `tests/test_v2_basics.py`: `resolve("/data/x")` with `TWINLOOP_DATA=/tmp/d` → `/tmp/d/x`; `case_image_path(10)` raises the exact test-split ValueError; `load_roles` on a fake splits file written to `tmp_path` (monkeypatch `TWINLOOP_DATA`) with train `["ImageCAS_0001", "ImageCAS_0002"]`, val `["ImageCAS_0003", "ImageCAS_0004"]` → `{1: "v2_train", 2: "v2_train", 3: "v2_realfail", 4: "v2_holdout"}` (test cases never appear; ids 5, 10 excluded); config dataclasses are frozen and have the exact SPEC §3 defaults (check 5 values).
10. Print role counts on real data and paste: `./scripts/dk python -c "from twinloop.v2split import load_roles;import collections;print(collections.Counter(load_roles().values()))"`. Expected roughly v2_train ~640, v2_realfail ~80, v2_holdout ~80.
11. Create `docs/PROGRESS_V2_SPARK.md` from the template if it does not exist. Under "In progress" write: `WAITING ON OWNER: (1) ./scripts/dk totalseg_set_license -l <key>  (2) create the private git remote and push (SPEC §15)`.
12. Commit `S00: v2 setup`.

**Acceptance:** image builds; torch/nnunetv2 versions unchanged; `./scripts/dk python -c "import totalsegmentator, sklearn, pyarrow, joblib; print('ok')"` prints ok; new tests pass; ruff clean; role counts pasted.
**Verify:**
```
./scripts/dk python -c "import totalsegmentator, sklearn, pyarrow, joblib; print('ok')"   # ok
./scripts/dk pytest -q tests/test_v2_basics.py                                             # all passed
./scripts/dk ruff check . && ./scripts/dk ruff format --check .
```

---

## [x] S01 — TotalSegmentator anatomy + quality checks
**Depends on:** S00; owner has set the licence (for the highres part)
**Files:** `src/twinloop/anatomy.py` (part 1: `Anatomy`, `run_totalseg`, `segment_anatomy`, qc), `tests/test_anatomy.py`
**Steps:**
1. Licence check: `./scripts/dk bash -c 'test -f /data/totalseg/config.json && echo present || echo missing'`. If missing, keep the WAITING ON OWNER line, implement steps 3–4 and test the `total` fallback only; come back for step 2 when it is present.
2. Pick the smallest-id `v2_train` case `N`; its container path is `/data/nnUNet_raw/Dataset501_ImageCAS/imagesTr/ImageCAS_{N:04d}_0000.nii.gz` (call it `IMG`). Check `free -g` (≥ 20 available). Run
   `./scripts/dk TotalSegmentator -i IMG -o /data/v2/tmp/ts_highres -ta heartchambers_highres` and
   `./scripts/dk TotalSegmentator -i IMG -o /data/v2/tmp/ts_total -ta total --roi_subset aorta heart`.
   Paste `ls` of both folders and the wall times. Record the highres file names in DECISIONS_V2.md (resolves SPEC §5.1 UNCERTAIN).
3. Implement `run_totalseg`, `segment_anatomy` and the qc keys `aorta_present`, `aorta_connected`, `myo_present` (SPEC §5.1–5.2). `ostia_ok` is set to False here and filled by `find_ostia` in S03. `lv_cavity` = `ventricle_left` from highres (None otherwise).
4. Tests (CPU, monkeypatch `anatomy.run_totalseg` to return synthetic masks built with numpy on a 100³ grid, spacing 1 mm): highres success → source `heartchambers_highres` for all; highres raising RuntimeError("licence") → aorta from `total`, myocardium None, `myo_present` False, the message in `qc_messages`; a two-piece aorta (70 %/30 %) → `aorta_connected` False; aorta of 10 mL → `aorta_present` False.
5. GPU+data test (`@pytest.mark.gpu`, `@pytest.mark.data`): `segment_anatomy` on the S01 case → `aorta_present`, `aorta_connected`, `myo_present` all True; myocardium volume printed.
6. Commit `S01: TotalSegmentator anatomy`.

**Acceptance:** CPU tests pass; GPU test passes when the licence is present (otherwise noted as pending in PROGRESS).
**Verify:** `./scripts/dk pytest -q tests/test_anatomy.py` and `./scripts/dk pytest -q -m "gpu and data" tests/test_anatomy.py`

---

## [x] S02 — Vessel graph
**Depends on:** S00
**Files:** `src/twinloop/graph.py`, `src/twinloop/phantom.py` (add kinds only), `tests/test_graph.py`
**Steps:**
1. Add phantom kinds to `make_phantom` without changing existing kinds: `"taper"` = cylinder along z at x = y = 28 mm from z = 10 to 70 mm whose radius goes linearly from 1.5 mm (z = 10) to 0.3 mm (z = 70); `"two_trees_gap"` = `"two_trees"` with the tube's voxels for 38 ≤ z ≤ 42 mm removed (mask and image).
2. Implement `graph.py` per SPEC §4 (including `fragments`, `main_path`, `graph_summary`, endings).
3. Tests: tube → 1 tree, 0 fragments, the distal ending (the one that is not the root) has kind `abrupt` (flat cut, r ≈ 1.5); taper → its distal ending has kind `taper`; two_trees_gap → `len(trees) == 2`, `len(fragments) == 1`; y_tree → `main_path(root)` has 2 branch ids starting with the root; `json.dumps(graph_summary(g))` works and floats have ≤ 2 decimals; empty mask → "No coronary tree found in mask".
4. Commit `S02: vessel graph`.

**Acceptance / Verify:** `./scripts/dk pytest -q tests/test_graph.py` → all passed; existing v1 phantom tests still pass (`./scripts/dk pytest -q`).

---

## [x] S03 — Ostia, LV axis, branch labels, coverage check
**Depends on:** S01, S02
**Files:** `src/twinloop/anatomy.py` (part 2), `tests/test_anatomy_geometry.py`
**Steps:**
1. Implement `find_ostia`, `lv_axis`, `label_branches`, `coverage_check` per SPEC §5.3 (messages exactly as in the coverage table).
2. Tests with hand-built graphs (write a helper `make_branch(branch_id, parent, start, direction, length_mm, radius_mm)` at 0.5 mm spacing like v1 `test_solver.py`) and numpy masks on a 160³ grid, spacing 0.5 mm, origin 0, identity direction:
   - aorta = cylinder radius 12 mm along z at x = 40, y = 40; a "left" tree whose root starts at the aorta surface on the +x side and a "right" tree starting on the −x side → `tree_side` left/right correct, both ostia distances ≤ 1 mm, `ostia_ok` True. Move one root 6 mm away → `ostia_ok` False.
   - myocardium = ellipsoidal shell (semi-axes 20, 20, 40 mm, thickness 8 mm) centred at (40, 40, 40) with long axis along z; aorta centroid at z = 75 → `lv_axis` apex has z < 40.
   - left tree: LM root 10 mm with 2 children: child A heading toward the apex (−z) 60 mm, child B heading +y 40 mm → labels LM / LAD (A path) / LCX (B path). Without myocardium → warning "LAD/LCX not labelled: no myocardium". Left root 40 mm long → warning "LM not identified" and a `missing_main_branch` flag from `coverage_check`.
   - coverage: RCA main path 31.2 mm with abrupt end → exactly the two flags of the SPEC §5.3 worked example (`"RCA main path 31.2 mm < 50 mm"` and an `abrupt_end`); a single tree → `missing_tree` flag with message "Only 1 coronary tree(s) found".
3. Commit `S03: anatomy geometry`.

**Acceptance / Verify:** `./scripts/dk pytest -q tests/test_anatomy_geometry.py` → all passed.

---

## [x] S04 — Vertical slice: `v2_inspect` on real cases
**Depends on:** S03
**Files:** `scripts/v2_inspect.py`
**Steps:**
1. `scripts/v2_inspect.py --case-id N [--mask-path PATH] [--no-anatomy] --out DIR`: load image (`case_image_path`) and mask (GT `case_label_path` unless `--mask-path`) with `twinloop.io`; `segment_anatomy` (workdir `DIR/<id>_ts`) unless `--no-anatomy`; `build_graph`; `find_ostia`; `label_branches`; `coverage_check`; write `DIR/<id>.json` = `{"case_id", "graph": graph_summary(g), "qc", "qc_messages", "myo_volume_ml", "aorta_volume_ml", "coverage": [...], "timings_s": {"anatomy", "graph"}}` (numpy → lists). Print one line `OK case=<id> trees=<n> fragments=<n> labels=<sorted set> flags=<n> t=<s>`.
2. Run on the 3 smallest-id `v2_train` cases: `./scripts/dk python scripts/v2_inspect.py --case-id <id> --out /data/v2/inspect` (check `free -g` first). Paste the OK lines.
3. Sanity (do NOT tune parameters; fix only genuine bugs such as axis order or unit errors, and log what you fixed): on GT masks expect 2 trees, `ostia_ok` True, labels containing LAD, LCX and RCA when `myo_present`. Write what you see per case in PROGRESS Notes.
4. Commit `S04: v2 inspect slice`.

**Acceptance:** 3 JSON files exist and the script exits 0 for each; sanity observations recorded.
**Verify:** `ls ${TWINLOOP_DATA:-$HOME/twinloop-data}/v2/inspect/*.json | wc -l` ≥ 3.

---

## [x] S05 — Metrics and the regression set
**Depends on:** S02
**Files:** `src/twinloop/metrics.py`, `scripts/make_regression_set.py`, `reports/v2_regression_set.json`, `tests/test_metrics.py`
**Steps:**
1. Implement SPEC §13 metric functions.
2. Tests: identical masks → dice 1, cl_dice 1; disjoint → 0; both empty → dice 1; y_tree phantom GT vs y_tree with one child removed → branch_recall 2/3 ± 0.01; GT = tube phantom, pred = tube + a side tube (radius 1.0 mm, 15 mm long, starting on the tube axis at z = 40 mm, pointing +x) → `false_branch_rate > 0`, and the pred branch whose points have mean x > 30 mm is the one counted as false; pred = GT → 0; stenosis phantom GT vs tube prediction → widened_rate 1.0.
3. `scripts/make_regression_set.py` implements the SPEC §13 rule (images and GT of `v2_holdout` only; no predictions). Write `reports/v2_regression_set.json`. If it already exists, refuse to overwrite (exit 1 with "regression set already exists").
4. Commit `S05: metrics and regression set`.

**Acceptance:** tests pass; JSON lists 20 distinct `v2_holdout` case ids.
**Verify:** `./scripts/dk pytest -q tests/test_metrics.py`; `./scripts/dk python -c "import json;d=json.load(open('reports/v2_regression_set.json'));print(len(set(d['case_ids'])))"` → 20.

---

## [x] S06 — Damage generator
**Depends on:** S02
**Files:** `src/twinloop/damage.py`, `scripts/v2_damage_preview.py`, `tests/test_damage.py`
**Steps:**
1. Implement SPEC §6 exactly (interval voxel rule, 6 types, non-overlap, rng).
2. Tests on the y_tree phantom (`DamageParams` with `n_damages_per_case=1`, and force a type by passing `type_weights` with a single 1.0): gap → voxel count decreases and the number of 26-connected components increases; truncate → all removed voxels have z beyond the seed point along the branch; delete_branch → one child branch's voxels gone (graph rebuilt has 1 branch fewer... assert `len(build_graph(mask_out).trees[0].branches) < 3`); shrink → median radius in the interval decreases by ≥ 25 %; false_branch → voxels added and all added voxels have image HU ≤ 150; occlusion → image_out differs from image only on removed voxels and their mean HU is within 50 ± 10; the same rng seed gives identical outputs twice.
3. `scripts/v2_damage_preview.py --n 10`: on the 10 smallest-id `v2_train` cases (seed 0) run `damage_case`; print counts per damage type and the number of skipped damages; for the first 2 cases save `mask_out` and records JSON to `/data/v2/tmp/damage_preview/`.
4. Commit `S06: damage generator`.

**Acceptance:** tests pass; over 10 cases every one of the 6 types occurs ≥ 3 times.
**Verify:** `./scripts/dk pytest -q tests/test_damage.py`; `./scripts/dk python scripts/v2_damage_preview.py --n 10` (paste counts).

---

## [x] S07 — CPR straightening
**Depends on:** S02
**Files:** `src/twinloop/cpr.py`, `tests/test_cpr.py`
**Steps:**
1. Implement `phys_to_index_zyx`, `rmf`, `straighten`, `center_component`, `r_eq_mm` (SPEC §9.1–9.3).
2. Tests: `phys_to_index_zyx` round-trips with `TransformContinuousIndexToPhysicalPoint` on an image with non-identity direction (diag(−1, −1, 1)) to 1e-6; `rmf` on a helix: U ⟂ T (|dot| < 1e-6), all unit length, consecutive U change smoothly (angle between U_i and U_{i+1} < 10°); tube phantom (GT mask) straightened along its own graph branch → middle slice `r_eq_mm` = 1.5 ± 0.15 mm and the image's centre pixel HU ≈ 400; stenosis phantom → min `r_eq` over slices = 0.75 ± 0.15.
3. Commit `S07: CPR`.

**Acceptance / Verify:** `./scripts/dk pytest -q tests/test_cpr.py` → all passed.

---

## [x] S08 — Region proposals, features, labels
**Depends on:** S06, S07 (uses `rmf` and `phys_to_index_zyx`)
**Files:** `src/twinloop/regions.py`, `scripts/v2_region_stats.py`, `tests/test_regions.py`
**Steps:**
1. Implement SPEC §7 (proposals, `region_features` returning an (n, 31) array in `FEATURE_NAMES` order, `label_synthetic`, `label_real`).
2. Tests: tube phantom → ≥ 1 `ending` region, `region_features(...)` has shape (n, 31) with no exceptions; two_trees_gap phantom → an ending region near z = 38 mm with `mask_ahead_frac` < 0.5 (the gap) and `in_fragment` = 0; y_tree with a forced `false_branch` damage → a `branch` region labelled `remove` by `label_synthetic`; forced `gap` → an `ending` labelled `extend`; `label_real` with pred = GT → all `keep`.
3. `scripts/v2_region_stats.py --n 10 --out /data/v2/tmp/region_stats.csv`: on the 10 smallest-id `v2_train` cases, seed 0: damage → build_graph(mask_out) → regions → labels; print per damage type `matched/total`, label counts, mean regions per case, seconds per case.
4. These numbers are design checks; do not change parameters to improve them. If a type matches < 50 %, describe the most common reason (look at 2 examples) in PROGRESS Notes.
5. Commit `S08: regions and features`.

**Acceptance:** tests pass; stats printed and pasted; seconds per case < 120.
**Verify:** `./scripts/dk pytest -q tests/test_regions.py`

---

## [ ] S09 — Myocardial-mass flow, FFR uncertainty, sensitivity check
**Depends on:** S03
**Files:** `src/twinloop/flow.py`, `src/twinloop/solver.py` (add `q_rest_override_m3s` only), `scripts/v2_sensitivity.py`, `reports/v2_sensitivity.md`, `tests/test_flow.py`
**Steps:**
1. Verify the resting myocardial blood flow value (SPEC §11 UNCERTAIN) with WebSearch; record the source URL and value in DECISIONS_V2.md; change `rest_flow_ml_min_per_g` only per SPEC §11's rule.
2. Add `q_rest_override_m3s` to `solve_tree`; all v1 solver tests must still pass.
3. Implement `tree_rest_flows` and `ffr_uncertainty_flags` (SPEC §11, exact signatures).
4. Tests: SPEC §11 worked example (98.0 / 49.0 mL/min ± 0.1); `auto` without myocardium → diameter_law + exact warning text; uncertainty flag appears on a hand-built tube with a lesion whose baseline outlet FFR is 0.80 ± 0.02 (adjust the lesion r_min in the test until baseline FFR is in that band, then assert the flag) and does not appear for a lesion-free tube.
5. `scripts/v2_sensitivity.py --n 5`: on the 5 smallest-id `v2_train` cases with GT masks and anatomy: run v1 extract → lesions → solve with (a) diameter_law, (b) myocardial mass, (c) mass × 0.8, (d) mass × 1.2. Write `reports/v2_sensitivity.md` with a table `case | mode | myo_ml | min_ffr tree_0 | min_ffr tree_1` and one sentence on the size of the changes. (The aorta position does not enter the 1D model, so an aortic-root shift has no FFR effect; state that in the report.)
6. Commit `S09: flow modes`.

**Acceptance:** tests pass; report exists with 20 rows (5 cases × 4 modes) or a reason per missing row.
**Verify:** `./scripts/dk pytest -q tests/test_flow.py tests/test_solver.py`

---

## [ ] S10 — Gap-filler inference and the repair loop
**Depends on:** S08
**Files:** `src/twinloop/gapfill.py`, `src/twinloop/repair.py`, `src/twinloop/modelio.py`, `tests/test_repair.py`
**Steps:**
1. `gapfill.py`: `extract_patch`, `apply_gapfill` (SPEC §8.1). `apply_gapfill` accepts any `torch.nn.Module` mapping (B,3,64,64,64) → (B,1,64,64,64) logits.
2. `modelio.py`: `load_model(name, device)` per SPEC §8.3; imports `twinloop.nets` lazily inside the function (the file is written by CLUSTER; until it is pulled, a missing import raises `RuntimeError("v2 model not available: twinloop.nets not found")`).
3. `repair.py`: `RepairAction`, `RulesDecider`, `run_repair` (SPEC §10).
4. Tests (CPU):
   - `extract_patch` on the tube phantom: `x.shape == (3,64,64,64)`, seed channel max at the patch voxel nearest the seed (offset 0), image channel values within [−1, 1].
   - Test helper `OracleNet(nn.Module)`: constructed with the full GT mask (`sitk.Image`); its `forward(x)` ignores `x` and returns logits `+10` where the GT is 1 and `−10` elsewhere **on the patch grid of the most recent call**. To know that grid, `apply_gapfill` must call `model.set_reference(ref)` before `forward` **if the model has that method** (`hasattr(model, "set_reference")`); real nets do not have it. `OracleNet.set_reference(ref)` resamples the GT onto `ref` (nearest) and stores it.
   - Use phantom `two_trees_gap` (damaged) with GT = `two_trees`: `apply_gapfill(OracleNet(gt), image, damaged_mask, seed, direction, 8.0, …)` at the ending at the gap → > 20 new voxels and the 26-connected component count of the result is one lower.
   - A fake model returning all `−10` → `(mask, 0)` (nothing found).
   - `run_repair` on the same pair with `RulesDecider` and `OracleNet(gt)` → final component count lower; actions contain an `extend` with result `applied`; with `models["gapfill"] = None` → that action's result is `flagged`.
5. Commit `S10: repair loop`.

**Acceptance / Verify:** `./scripts/dk pytest -q tests/test_repair.py` → all passed.

---

## [ ] S11 — Lumen refinement
**Depends on:** S07
**Files:** `src/twinloop/lumen.py`, `tests/test_lumen.py`
**Steps:**
1. Implement `refine_radii` (SPEC §9.4) taking any module mapping (B,T,1,48,48) → logits.
2. Tests with a fake module that returns `+10` where the input slice HU (de-normalised) > 200 and `−10` elsewhere: on the stenosis phantom the refined branch's min radius = 0.75 ± 0.15 and median radius = 1.5 ± 0.15; branches shorter than 10 mm unchanged; a fake module returning all −10 → radii unchanged (empty components keep v1 radius).
3. Commit `S11: lumen refinement`.

**Acceptance / Verify:** `./scripts/dk pytest -q tests/test_lumen.py` → all passed.

---

## [ ] S12 — Pipeline integration
**Depends on:** S09, S10, S11
**Files:** `src/twinloop/pipeline.py`, `src/twinloop/schema.py`, `tests/test_pipeline_v2.py`
**Steps:**
1. Add `V2Options` and the `v2` argument to `run_pipeline`, the CLI flags, `V2Out` and `CaseResult.v2` (SPEC §12). Order inside the pipeline when `v2` is set: segmentation (or given mask) → anatomy (if on) → repair (if on; models loaded once via `load_model`; missing models → raise the `RuntimeError` from `load_model`) → v1 `extract_trees` on the final mask → lumen refinement (if on) → lesions → `tree_rest_flows` → solve with overrides → uncertainty flags → result. `v2=None` must behave exactly like v1.
2. `app/jobs.py` does not exist on `v2` yet (v1 T06 happens on `main`); it is wired in S12M. Implement the SPEC §12 robustness rules (anatomy errors caught, file names).
3. Log the new optional `result.json` field `v2` in DECISIONS_V2.md (CLAUDE.md rule).
4. Tests (monkeypatch `twinloop.anatomy.segment_anatomy` to return a synthetic `Anatomy` with a 140 mL myocardium and qc all True): stenosis phantom with `V2Options(anatomy=True, repair=False, lumen=False, flow_mode="auto")` → `result.v2.flow_mode == "myocardial_mass"`, `myocardium_ml == 140`; `V2Options(anatomy=False)` → flow_mode `diameter_law`; `segment_anatomy` raising RuntimeError → case still succeeds, warning starts with "Anatomy unavailable:", flow_mode `diameter_law`; `v2=None` → `result.v2 is None` and every v1 test still passes; `repair=True` with `TWINLOOP_DATA` pointing to an empty `tmp_path` → `run_pipeline` raises RuntimeError containing "v2 model not available".
5. Real case (data, GPU for TotalSegmentator; check `free -g`): `./scripts/dk python -m twinloop.pipeline --image <smallest v2_train image> --mask <its GT>` → paste the OK line and the `v2` block of result.json.
6. Commit `S12: v2 pipeline integration`.

**Acceptance:** all tests (`./scripts/dk pytest -q`) pass; real case prints OK with `v2.qc` filled.
**Verify:** `./scripts/dk pytest -q` → 0 failed.

---

## [ ] S12M — Merge v1 `main` into `v2`
**Depends on:** S12; v1 T09 finished on `main`
**Steps:**
1. Check v1 progress without leaving the worktree: `git show main:docs/PROGRESS.md | grep -nE "T0[6-9]|T1[0-2]"` and `git log --oneline main | head -20`. Proceed only if a commit `T09:` exists on `main` (training finished, trainer default set). Otherwise write `WAITING: v1 T09 not on main yet` under In progress and end the session (nothing else on SPARK is unblocked except S17).
2. Make sure the working tree is clean (`git status --porcelain` empty), then `git merge --no-ff main -m "S12M: merge v1 main into v2"`.
3. Conflict rules (resolve, never discard either side's feature):
   - `scripts/dk`, `scripts/dkd`: keep image `twinloop:v2`, keep `TOTALSEG_HOME_DIR`, take v1's `TWINLOOP_NNUNET_TRAINER` default and any other v1 change.
   - `requirements.txt`/`requirements.lock`: union; v1 pins win for packages both touched, except keep v2's four added packages; rebuild `twinloop:v2` and rerun the S00 step-5 version check.
   - `src/twinloop/pipeline.py`, `schema.py`: keep v1's changes and v2's `V2Options`/`v2` field.
   - `CLAUDE.md`: keep the v2 version. `docs/*`: keep both (different files).
4. Wire `app/jobs.py` (from v1 T06): pass `V2Options()` to `run_pipeline`. In `tests/test_api.py`, add an autouse fixture that monkeypatches `twinloop.anatomy.segment_anatomy` to raise `RuntimeError("disabled in tests")`.
5. `./scripts/dk pytest -q` and ruff must pass. Commit (the merge commit plus a follow-up `S12M: wire v2 into app`).

**Acceptance:** branch `v2` contains `main`'s `T09:` commit; all tests pass; `curl` health check works after `bash scripts/run_server.sh` (it publishes 127.0.0.1:8000 — if the v1 session's `twinloop-server` is running, skip the server check and note it).
**Verify:** `git branch --contains $(git log main --format=%h --grep '^T09:' | head -1)` lists `v2`; `./scripts/dk pytest -q` → 0 failed.

---

## [ ] S13 — First-pass evaluation (Track B)
**Depends on:** S05, S12M (training finished, `TWINLOOP_NNUNET_TRAINER` set)
**Files:** `scripts/eval_v2.py`, `reports/v2_eval_firstpass.csv`
**Steps:**
1. Check the validation predictions exist: `ls ${TWINLOOP_DATA:-$HOME/twinloop-data}/nnUNet_results/Dataset501_ImageCAS/*/fold_0/validation/ | head` (SPEC §13 fallback if not).
2. Implement `scripts/eval_v2.py` (SPEC §13). Case dirs under `/data/v2/eval/<variant>/<case_id>/`. Resumable: skip a case whose row already exists for that variant.
3. Run `--cases regression --variants gt_v1,pred_v1 --out reports/v2_eval_firstpass.csv`.
4. In PROGRESS Notes: mean dice, cl_dice, branch_recall, false_branch_rate, widened_rate for pred_v1, and the 3 cases with the lowest branch_recall.
5. Commit `S13: first-pass eval`.

**Acceptance:** 40 rows (20 cases × 2 variants), failures included with their error.
**Verify:** `./scripts/dk python -c "import pandas as pd;d=pd.read_csv('reports/v2_eval_firstpass.csv');print(d.groupby('variant')[['dice','branch_recall']].mean(), len(d))"`

---

## [ ] S14 — Real-failure decider dataset
**Depends on:** S08, S12M
**Files:** `scripts/make_decider_real.py`
**Steps:**
1. For every `v2_realfail` case: image, GT, first-pass mask (validation prediction); `build_graph` on pred (skip case on error, count it) and on GT; `propose_regions`; `region_features`; `label_real`. Write `/data/v2/datasets/decider_real/decider_real.parquet` (columns as `decider_synth`, `seed = -1`).
2. Print: cases used/skipped, rows, label counts.
3. Under "In progress": `WAITING ON OWNER: rsync -av $TWINLOOP_DATA/v2/datasets/decider_real/ <cluster>:<CLUSTER_DATA>/v2/datasets/decider_real/`.
4. Commit `S14: real-failure dataset` (the script only; never data).

**Acceptance:** parquet exists, ≥ 1 row per used case, label counts pasted.
**Verify:** `./scripts/dk python -c "import pandas as pd;d=pd.read_parquet('/data/v2/datasets/decider_real/decider_real.parquet');print(len(d), d.label.value_counts().to_dict())"`

---

## [ ] S15 — Integrate trained models
**Depends on:** S12; CLUSTER C07 finished and the owner has synced `models/v2/` and pulled `nets.py`
**Steps:**
1. `ls ${TWINLOOP_DATA:-$HOME/twinloop-data}/models/v2/*/config.json` must list `gapfill`, `lumen`, `decider`. Otherwise WAITING ON OWNER (rsync from SPEC §1).
2. `./scripts/dk python -c "from twinloop.modelio import load_model; [print(n, type(load_model(n,'cuda')).__name__) for n in ['gapfill','lumen','decider']]"`.
3. Check sklearn version equality: `./scripts/dk python -c "import sklearn;print(sklearn.__version__)"` must equal `sklearn_version` in `models/v2/decider/config.json`; if not, BLOCKED (owner must align versions).
4. Run the CLI with `--repair --lumen` on the 2 smallest-id `v2_train` cases using their GT masks, and on 1 `v2_realfail` case using its validation prediction as `--mask`. Paste the OK lines, `n_actions`, review flags, and timings.
5. Commit `S15: models integrated`.

**Acceptance:** 3 OK lines; total time per case < 300 s.

---

## [ ] S16 — Final v2 evaluation and defaults
**Depends on:** S13, S15 (and S14 + CLUSTER C08 if done; otherwise evaluate with the synthetic-only decider and say so)
**Files:** `reports/v2_eval.csv`, `reports/V2_RESULTS.md`, `src/twinloop/pipeline.py` (defaults only)
**Steps:**
1. `./scripts/dk python scripts/eval_v2.py --cases regression --variants gt_v1,pred_v1,pred_v2_rules,pred_v2_repair,pred_v2_repair_lumen --out reports/v2_eval.csv`. Run once. Do not change any parameter or model after seeing these numbers.
2. Write `reports/V2_RESULTS.md` from the template below; H1/H2 verdicts per SPEC §14.
3. Defaults: if H1 SUPPORTED set `V2Options.repair` default True; if H2 SUPPORTED set `lumen` True. Otherwise keep False. Log in DECISIONS_V2.md with the numbers.
4. Commit `S16: v2 evaluation`.

```markdown
# TwinLoop v2 results
Date: YYYY-MM-DD. Commit: <sha>. Decider: <fine-tuned | synthetic-only>. First-pass trainer: <class>.
## Regression set (n = 20, v2_holdout)
| Variant | Success | Dice | clDice | Branch recall | False-branch rate | Widened rate | |r_min| err mm | Review flags | Median total s |
|---|---|---|---|---|---|---|---|---|---|
| gt_v1 | | | | | | | | | |
| pred_v1 | | | | | | | | | |
| pred_v2_rules | | | | | | | | | |
| pred_v2_repair | | | | | | | | | |
| pred_v2_repair_lumen | | | | | | | | | |
Paired (pred_v2_repair − pred_v1): branch recall median __, improved __/20, worse __/20.
H1 (repair): SUPPORTED / FALSIFIED — <numbers>.
H2 (lumen): SUPPORTED / FALSIFIED — <numbers>.
## Failures
- case <id> <variant>: <error>
## Caveats
FFR unvalidated. Single-centre data. One deployed seed per network. Synthetic damage ≠ real errors.
```

**Acceptance:** 100 rows in `reports/v2_eval.csv`; RESULTS has no empty cells ("n/a — reason" where needed).
**Verify:** `./scripts/dk python -c "import pandas as pd;d=pd.read_csv('reports/v2_eval.csv');print(len(d), d.groupby('variant').status.value_counts().to_dict())"`

---

## [ ] S17 — (Optional) TotalSegmentator error on MM-WHS
**Depends on:** S01. **Skip** (mark `[x] skipped` with the reason in PROGRESS) unless `${TWINLOOP_DATA}/raw/mmwhs/` exists (the owner has not downloaded MM-WHS).
**Steps (only if present):** for each MM-WHS CT training case with labels: run `segment_anatomy`; compare myocardium (MM-WHS label value 205) and ascending aorta (label 820) with Dice after cropping both masks to the bounding box of the MM-WHS aorta label in z; write `reports/v2_mmwhs.csv` (`case,dice_myo,dice_aorta,vol_myo_ts_ml,vol_myo_gt_ml`). Label values are UNCERTAIN: print `np.unique` of one label file first and adapt.
