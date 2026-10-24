# TwinLoop v1 — Tasks

Work strictly in order unless a task is BLOCKED (then move to the next task whose dependencies are all done). Every path below is relative to the repo root. Every command runs on the **host** from the repo root unless it starts with `./scripts/dk`, which runs it inside the container. `SPEC §x` = section x of docs/SPEC.md.

Status markers: change `[ ]` to `[x]` only after verification passes and the output is pasted in docs/PROGRESS.md.

---

## [x] T00 — Machine check, container, repo scaffold
**Depends on:** none
**Files:** `Dockerfile`, `.dockerignore`, `.gitignore`, `pyproject.toml`, `requirements.txt`, `requirements.lock`, `scripts/dk`, `scripts/check_env.sh`, `src/twinloop/__init__.py`, `app/__init__.py`, `tests/test_smoke.py`, `reports/.gitkeep`
**Steps:**
1. Run and paste into PROGRESS.md: `uname -m; nvidia-smi; free -h; df -h "$HOME"; docker --version; docker info | grep -i runtime`. Expected: `aarch64`, a GB10 GPU, ~128 GB total memory. If `df` shows < 250 GB free for `$HOME`, log BLOCKED ("ImageCAS + converted copies + preprocessing need ~250 GB") but continue with T02–T04 and T06–T07, which need no data.
2. If `git rev-parse --is-inside-work-tree` fails, run `git init -b main`. Configure nothing global.
3. Create `.gitignore`, `pyproject.toml` exactly as in SPEC §3, and `requirements.txt` exactly as in SPEC §2.3. `.dockerignore`: `.git`, `**/__pycache__`, `*.nii.gz`, `*.zip`, `reports/*.csv`.
4. Pick the base image tag with the procedure in SPEC §2.3 (try `docker pull nvcr.io/nvidia/pytorch:26.09-py3` first, then older tags). Write the Dockerfile from SPEC §2.3 with `ARG BASE_IMAGE=` set to the tag that worked.
5. `docker build -t twinloop:dev .` (if pip resolution fails, apply the fallback in SPEC §2.3).
6. Create `scripts/dk` and `scripts/dkd` from SPEC §2.3, `chmod +x scripts/dk scripts/dkd`, `mkdir -p "$HOME/.kaggle" "${TWINLOOP_DATA:-$HOME/twinloop-data}"`.
7. Pin versions: `./scripts/dk pip freeze > requirements.lock`. Then rewrite each line of `requirements.txt` as `name==<version from requirements.lock>` (keep `uvicorn[standard]==<v>`; do not pin torch — it comes from the base image). Rebuild the image. Record the final versions table in DECISIONS.md.
8. `scripts/check_env.sh`:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   python - <<'EOF'
   import torch, SimpleITK, skimage, scipy, networkx, fastapi, nnunetv2, pydantic
   assert torch.cuda.is_available(), "CUDA not available"
   x = torch.randn(1, 1, 64, 64, 64, device="cuda")
   conv = torch.nn.Conv3d(1, 8, 3, padding=1).cuda()
   y = conv(x); torch.cuda.synchronize()
   print("torch", torch.__version__, "device", torch.cuda.get_device_name(0), "out", tuple(y.shape))
   EOF
   which nnUNetv2_train nnUNetv2_predict nnUNetv2_plan_and_preprocess > /dev/null
   echo "ENV OK"
   ```
9. `tests/test_smoke.py`: `def test_import(): import twinloop` (passes because PYTHONPATH includes src).
10. Commit: `git add -A && git commit -m "T00: container and scaffold"`.

**Acceptance:** image `twinloop:dev` exists; check_env prints the device line and `ENV OK`; pytest passes; ruff clean; `requirements.txt` fully pinned.
**Verify:**
```
./scripts/dk bash scripts/check_env.sh   # last line: ENV OK
./scripts/dk pytest -q                   # 1 passed
./scripts/dk ruff check .                # All checks passed!
```

---

## [x] T01 — Download ImageCAS (background) and build the manifest
**Depends on:** T00 (steps 1–4); steps 5–7 also need `twinloop.io` from T03 — do them after T03 if the download finishes earlier.
**Files:** `scripts/download_imagecas.sh`, `scripts/make_manifest.py`
**Owner action (done):** the owner saved a new-style Kaggle API token (a single string) at `~/.kaggle/access_token` (chmod 600). There is no `kaggle.json`. Never print, echo, cat or log the token.
**Steps:**
0. Auth check. Inside the container the folder is mounted at `/kaggle` and `KAGGLE_CONFIG_DIR=/kaggle`. Run `./scripts/dk kaggle --version` and `./scripts/dk kaggle datasets list -s imagecas`.
   - If the list prints → auth works, go to step 1.
   - If it fails with an auth/credentials error: the installed `kaggle` may be too old for access tokens. Run `./scripts/dk pip index versions kaggle`, set `kaggle==<newest>` in requirements.txt, rebuild the image, and retry. If it still fails, change `scripts/dk` and `scripts/dkd` to also pass `-e KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/access_token" 2>/dev/null)"` (the token then stays only in the local container env; never write it to a file in the repo) and retry. Log which method worked in DECISIONS.md.
   - If all three fail, log BLOCKED: "Owner: check the Kaggle token at ~/.kaggle/access_token is current (regenerate on kaggle.com → Settings → API if needed), then tell Claude to resume T01", and continue with T02.
1. Find the dataset: `./scripts/dk kaggle datasets list --user xiaoweixumedicalai` and also `./scripts/dk kaggle datasets list -s imagecas`. Paste output into PROGRESS.md. Choose the dataset(s) owned by `xiaoweixumedicalai` whose title mentions ImageCAS. If it is split into several datasets (e.g. parts 1–4), download all of them. Record the exact slug(s) in DECISIONS.md. If none exist, follow the BLOCKED rule in SPEC §4.1. If the download returns HTTP 403, log BLOCKED with "Owner: open the ImageCAS dataset page on kaggle.com while logged in and accept its terms/request access, then tell Claude to resume T01" and continue with T02.
2. Before downloading, check size: `./scripts/dk kaggle datasets files <slug>` and compare with `df -h`. Need free space ≥ 2.5 × dataset size.
3. `scripts/download_imagecas.sh <slug> [<slug> ...]`: for each slug runs `kaggle datasets download -d "$slug" -p /data/raw/imagecas --unzip`, then finds any remaining `*.zip` under `/data/raw/imagecas` and unzips each in place with `python -m zipfile -e <zip> <its dir>` followed by deleting that zip **only after** a successful extract (these zips are our own downloads, not source data). Ends with `echo DOWNLOAD DONE`.
4. Start it detached (survives the session): `./scripts/dkd twinloop-download bash scripts/download_imagecas.sh <slugs>`.
   Monitor with `docker logs --tail 5 twinloop-download`. While it runs, proceed to T02 and come back to steps 5–7 when `docker logs` shows `DOWNLOAD DONE` (then `docker rm twinloop-download`). If it exits without `DOWNLOAD DONE`, paste the last 30 log lines and rerun step 4 (kaggle skips nothing, so a rerun re-downloads; that is acceptable).
5. `scripts/make_manifest.py --root DIR --out CSV` implements SPEC §4.2 exactly: recursively find `*.nii.gz` / `*.nii`, skip paths containing `(1)`, pair image/label by `case_id`, then for each pair (multiprocessing `Pool(8)`) load the image with `twinloop.io.load_image` (reorients to LPS) and read size and spacing. Print: `cases=<n> train=<n> test=<n> unpaired=<n>`; list unpaired files in PROGRESS.md.
6. Run: `./scripts/dk python scripts/make_manifest.py --root /data/raw/imagecas --out /data/raw/imagecas/manifest.csv`.
7. Commit scripts (never data).

**Acceptance:** manifest exists; `cases` between 990 and 1000; every row has both paths existing; split counts ≈ 800/200; all `size_x == size_y == 512`; spacings within 0.2–0.6 mm.
**Verify:**
```
./scripts/dk python -c "import pandas as pd; m=pd.read_csv('/data/raw/imagecas/manifest.csv'); print(len(m), m.split.value_counts().to_dict(), m.size_x.unique(), m[['spacing_x','spacing_z']].describe().loc[['min','max']].to_dict())"
```
Expected like `1000 {'train': 800, 'test': 200} [512] {...min ~0.29, max ~0.45...}`.

---

## [x] T02 — Config, schema stubs and the 1D solver (pure CPU)
**Depends on:** T00
**Files:** `src/twinloop/config.py`, `src/twinloop/centerline.py` (only the `Branch` and `VesselTree` dataclasses for now), `src/twinloop/lesions.py` (only the `Lesion` dataclass), `src/twinloop/solver.py`, `tests/test_solver.py`
**Steps:**
1. `config.py`: SPEC §5.6 verbatim. Add `MMHG_TO_PA = 133.322`.
2. Dataclasses `Branch`, `VesselTree` (SPEC §5.3) and `Lesion` (SPEC §5.4).
3. `solver.py`: `branch_pressure_drop`, `outlet_weights(tree, p) -> dict[str, float]`, `resting_flow_m3s(tree, p) -> float`, `solve_tree` — SPEC §5.5. Helper `downstream_outlets(tree) -> dict[branch_id, list[outlet_id]]`. Arrays: `numpy.float64`. Convert mm → m (× 1e-3) once at the start.
4. Verify the flow-law constants (SPEC §5.5 "Uncertain"). Log the outcome in DECISIONS.md.
5. Tests in `tests/test_solver.py` (build trees by hand from numpy arrays; no images). Helper `straight_branch(branch_id, parent, start, direction, length_mm, radius_mm)` with 0.5 mm spacing:
   - `test_poiseuille_tube`: SPEC §5.5 example 1 → 88.03 Pa ± 0.5 %.
   - `test_young_tsai`: build r with a lesion object (r_ref 1.5, r_min 0.75) on a branch so short that viscous ≈ 0; isolate by calling the turbulent-term helper `turbulent_drop_pa(q, r_ref_mm, r_min_mm, p)` → 145.11 Pa ± 0.5 %.
   - `test_monotone`: drop array non-decreasing, `out[0] == 0`.
   - `test_murray`: two outlets r 1.5 and 1.0 → weights 0.7714 / 0.2286 (± 1e-3).
   - `test_flow_law`: root radius 1.5 mm → 23.55 mL/min ± 0.1 %.
   - `test_single_tube_coupled`: one branch r 1.5, 50 mm; Pa 90 mmHg → hyperemic flow 96.96 mL/min ± 0.5 % and outlet FFR 0.9881 ± 0.001; `converged` True.
   - `test_symmetric_y`: root (r 2, 30 mm) + two equal children (r 1.5, 35 mm): child flows equal within 1e-6 relative; root flow = sum of children within 1e-9 relative; FFR non-increasing along every branch and continuous at the junction (child[0] == parent[-1] within 1e-9).
   - `test_stenosis_lowers_ffr`: same tube with a lesion → outlet FFR strictly lower than without.
6. Commit.

**Acceptance:** all tests pass; ruff clean.
**Verify:** `./scripts/dk pytest -q tests/test_solver.py` → `8 passed`.

---

## [x] T03 — I/O, phantoms, centerlines, lesions
**Depends on:** T02
**Files:** `src/twinloop/io.py`, `src/twinloop/phantom.py`, `src/twinloop/centerline.py`, `src/twinloop/lesions.py`, `tests/test_io.py`, `tests/test_centerline.py`, `tests/test_lesions.py`
**Steps:**
1. `io.py` per SPEC §5.1 (error strings exact).
2. `phantom.py` per SPEC §6.9. Build masks analytically on the voxel-centre grid with numpy (`np.indices`, physical coordinate = index × spacing because origin 0 and identity direction). Image = `400 * mask` as Int16. For `y_tree`, union of three capsule-free cylinders (point-to-segment distance ≤ radius, clipped to the segment's length).
   `write_dicom_series(img, out_dir)`:
   ```python
   img16 = sitk.Cast(img, sitk.sitkInt16)
   w = sitk.ImageFileWriter(); w.KeepOriginalImageUIDOn()
   sx, sy, sz = img16.GetSpacing()
   uid = "1.2.826.0.1.3680043.2.1125." + str(int(time.time()))
   for k in range(img16.GetDepth()):
       sl = img16[:, :, k]
       z = img16.TransformIndexToPhysicalPoint((0, 0, k))
       for tag, val in [("0020|000e", uid + ".1"), ("0020|000d", uid), ("0008|0060", "CT"),
                        ("0020|0032", "\\".join(map(str, z))), ("0020|0013", str(k + 1)),
                        ("0028|0030", f"{sy}\\{sx}"), ("0018|0050", str(sz)),
                        ("0020|0037", "1\\0\\0\\0\\1\\0")]:
           sl.SetMetaData(tag, val)
       w.SetFileName(str(out_dir / f"slice_{k:04d}.dcm")); w.Execute(sl)
   ```
3. `centerline.py` `extract_trees` per SPEC §5.3, steps 1–10 exactly.
4. `lesions.py` `reference_radius`, `detect_lesions` per SPEC §5.4.
5. Tests:
   - `test_io.py`: NIfTI round-trip keeps array and spacing; a RAS-oriented image (direction diag(-1,-1,1)) comes back with direction ≈ identity after `load_image`; DICOM zip of `make_phantom("tube")[0]` loads with the same size and spacing (± 1e-3); `.txt` path → the exact unsupported-file ValueError; mask with wrong size → "Mask shape does not match image".
   - `test_centerline.py`: tube → 1 tree, 1 branch, length 54–61 mm, median radius of middle 80 % = 1.5 ± 0.15 mm, root point is an end of the tube (z < 15 or z > 65). Record whether `radius_half_voxel_correction` True or False is closer to 1.5 mm in DECISIONS.md and keep the closer one as the default. y_tree → 1 tree, 3 branches, root has 2 children, root median radius 2.0 ± 0.2, root starts near z = 5 (z < 12) because it is the widest end. two_trees → 2 trees, `tree_0` has the larger volume. Empty mask → "No coronary tree found in mask".
   - `test_lesions.py`: the worked example in SPEC §5.4; a constant-radius branch → 0 lesions; a branch whose last 2 mm taper to r 0.5 → 0 lesions (end exclusion); stenosis phantom through `extract_trees` + `detect_lesions` → exactly 1 lesion, ds 0.35–0.60, position z within 40 ± 2 mm.
6. Commit.

**Acceptance:** all listed tests pass; `extract_trees` on the y_tree phantom runs in < 10 s.
**Verify:** `./scripts/dk pytest -q tests/test_io.py tests/test_centerline.py tests/test_lesions.py` → all passed, 0 failed.

---

## [x] T04 — Pipeline + result.json + CLI (first vertical slice)
**Depends on:** T03 (and T01 for step 5 only)
**Files:** `src/twinloop/schema.py`, `src/twinloop/pipeline.py`, `src/twinloop/segmentation.py` (stub: `model_available()` and `segment_nnunet()` that raises the "Segmentation model not available…" RuntimeError), `tests/test_pipeline_phantom.py`
**Steps:**
1. `schema.py` per SPEC §6.3 (pydantic v2). Rounding happens when building the models.
2. `pipeline.py` `run_pipeline` + CLI per SPEC §5.7. Lesion IDs `L1..Ln` in order (tree_0 before tree_1; branches in BFS order; by `min_idx`). `params` = `dataclasses.asdict(params)`. Write `status.json` with the SPEC §6.4 fields. Use `jobs`-independent helpers for case IDs and atomic JSON writing in `src/twinloop/casefiles.py` (`new_case_id()`, `write_json_atomic(path, obj)`), so `app/jobs.py` reuses them later.
3. Tests (`tmp_path` as case dir): stenosis phantom image + mask → `result.json` validates via `CaseResult.model_validate_json`; 1 tree; 1 lesion; that lesion's `ffr_distal` < `ffr_proximal`; `summary.min_ffr` between 0.80 and 0.99; disclaimer string exact; timings keys exactly `load, segmentation, centerline, solver, total` (segmentation = 0.0 when mask given). Tube phantom → 0 lesions and min_ffr > 0.97. two_trees → 2 trees.
4. Run the CLI on a phantom written to `/data/runs/phantom/stenosis_img.nii.gz` + mask (add `python -m twinloop.phantom --kind stenosis --out /data/runs/phantom` to write both files).
5. **Real-data slice** (only if T01 is done; otherwise note "pending T01" in PROGRESS.md and do it when T01 finishes): take the first 3 `train` cases of the manifest, run the CLI with `--mask <label_path>`. Paste the `OK ...` lines. Sanity: ≥ 1 tree each, root flow 50–400 mL/min hyperemic, min_ffr between 0.5 and 1.0, total < 60 s. If a case violates these, write what you see in PROGRESS.md Notes (do not tune parameters on the data; only fix genuine bugs, e.g. orientation or unit errors).
6. Commit.

**Acceptance:** phantom tests pass; CLI prints `OK ...` on the phantom; (when data exists) 3 real cases print `OK`.
**Verify:**
```
./scripts/dk pytest -q tests/test_pipeline_phantom.py
./scripts/dk python -m twinloop.phantom --kind stenosis --out /data/runs/phantom
./scripts/dk python -m twinloop.pipeline --image /data/runs/phantom/stenosis_img.nii.gz --mask /data/runs/phantom/stenosis_mask.nii.gz
# -> OK case_id=c..._...... trees=1 lesions=1 min_ffr=0.9xx total_s=...
```

---

## [x] T05 — nnU-Net dataset, preprocessing, epoch benchmark, launch training (background)
**Depends on:** T01, T04
**Files:** `scripts/convert_to_nnunet.py`, `scripts/train.sh`
**Steps:**
1. `convert_to_nnunet.py --manifest /data/raw/imagecas/manifest.csv --out /data/nnUNet_raw/Dataset501_ImageCAS --workers 8`: for every manifest row, load image + label with `twinloop.io` (LPS), write image to `imagesTr/ImageCAS_{case_id:04d}_0000.nii.gz` (train) or `imagesTs/...` (test), label (UInt8 0/1) to `labelsTr/ImageCAS_{case_id:04d}.nii.gz` or `labelsTs/...`. Skip files that already exist (resumable). Write `dataset.json`:
   ```json
   {"channel_names": {"0": "CT"}, "labels": {"background": 0, "coronary": 1},
    "numTraining": <number of train cases>, "file_ending": ".nii.gz", "name": "ImageCAS"}
   ```
2. Run conversion detached: `./scripts/dkd twinloop-prep python scripts/convert_to_nnunet.py --manifest /data/raw/imagecas/manifest.csv --out /data/nnUNet_raw/Dataset501_ImageCAS --workers 8`; wait for exit code 0. Then `./scripts/dkd twinloop-prep nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres --verify_dataset_integrity -np 4`; wait for exit code 0 (expect 1–4 h). Must end without errors; `/data/nnUNet_preprocessed/Dataset501_ImageCAS/nnUNetPlans.json` must exist. Paste the 3d_fullres `patch_size`, `batch_size` and `spacing` from that file into DECISIONS.md.
3. Find trainer classes: `./scripts/dk bash -c 'grep -rhoE "class nnUNetTrainer_[0-9]+epochs?\b" $(python -c "import nnunetv2,os;print(os.path.dirname(nnunetv2.__file__))")'`. Paste the list.
4. Benchmark: `./scripts/dkd twinloop-train nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_5epochs` (if that class is missing, use the smallest listed; expected 30–90 min). If it errors mentioning `torch.compile`/inductor/triton, rerun with `nnUNet_compile=false ./scripts/dkd twinloop-train ...`, use `nnUNet_compile=false` for every later nnU-Net command (put `export nnUNet_compile=false` at the top of `scripts/train.sh` and change the default in `scripts/dk`/`scripts/dkd` to `false`), and log in DECISIONS.md. Read epoch times from `/data/nnUNet_results/Dataset501_ImageCAS/nnUNetTrainer_5epochs__nnUNetPlans__3d_fullres/fold_0/training_log_*.txt` (lines `Epoch time: X s`). t = median of epochs 2–5.
5. Choose E = the largest of 1000 / 500 / 250 / 100 with an existing class (`nnUNetTrainer` is 1000) such that `E × t × 1.1 ≤ 259200` (72 h). If even 100 does not fit, log BLOCKED with t and stop T05. Record t, E and the class in DECISIONS.md.
6. `scripts/train.sh <TrainerClass> [--c]`: refuses to start (prints "training already running", exit 1) if `docker ps -q --filter name=twinloop-train` is non-empty; otherwise `./scripts/dkd twinloop-train nnUNetv2_train 501 3d_fullres 0 -tr "$1" ${2:-}`. `--c` resumes from the latest checkpoint (use after a reboot/crash).
7. Launch `bash scripts/train.sh <TrainerClass>`. Confirm after ~10 min that `docker logs --tail 20 twinloop-train` shows epoch 0 finished. Write in PROGRESS.md: start time (UTC), trainer, expected end = start + E × t.
8. Commit.

**Acceptance:** `dataset.json` correct (`numTraining` equals train count); preprocessing done; training container running with epoch logs appearing.
**Verify:**
```
ls ${TWINLOOP_DATA:-$HOME/twinloop-data}/nnUNet_raw/Dataset501_ImageCAS/imagesTr | wc -l   # = train count
docker ps --filter name=twinloop-train --format '{{.Status}}'   # Up ...
docker logs --tail 5 twinloop-train                               # shows Epoch / train_loss lines
```

---

## [x] T06 — Job runner and HTTP API
**Depends on:** T04
**Files:** `app/jobs.py`, `app/server.py`, `tests/test_api.py`, `scripts/run_server.sh`
**Steps:**
1. `jobs.py`: `JobManager(cases_root: Path)` with `submit(case_dir) -> None`, `queue_length() -> int`, `recover()` (SPEC §6.4 restart rule), worker thread that loads the case (`status loading`), calls `run_pipeline` with `on_step` updating `status.json`, catches all exceptions (→ `failed`). Cases root from env `TWINLOOP_DATA` → `<root>/runs/cases`.
2. `server.py`: SPEC §6.5 exactly. Mount `app/static` at `/static`; `GET /` returns `app/static/index.html`. Start the worker and call `recover()` in a FastAPI lifespan handler. Create a placeholder `app/static/index.html` containing `<h1>TwinLoop</h1>` (replaced in T07).
3. `scripts/run_server.sh` per SPEC §7.
4. `tests/test_api.py` (FastAPI `TestClient` used as a context manager so the lifespan starts the worker; `monkeypatch.setenv("TWINLOOP_DATA", str(tmp_path))`, `monkeypatch.setenv("nnUNet_results", str(tmp_path / "no_results"))` and `monkeypatch.setenv("TWINLOOP_NNUNET_TRAINER", "NoSuchTrainer")` so no real model is ever used in tests; create the app after setting env — expose a `create_app()` factory in server.py and `app = create_app()` at module level; phantom files are written into `tmp_path` with `make_phantom` + `save_nifti`):
   - health → 200 with keys `status, cuda, model_available, queue_length`.
   - upload stenosis phantom image + mask → 202; poll status (sleep 0.2 s, max 60 s) until `done`; result → 200 and validates as `CaseResult`.
   - `test_dicom_zip_upload`: tube phantom as DICOM zip + its mask → `done`.
   - `.txt` upload → 400 with the exact detail.
   - `aortic_pressure_mmhg=200` → 400.
   - image without mask and no model → final status `failed`, error contains "Segmentation model not available".
   - unknown / malformed case id → 404; result before done → 409 (submit, then immediately request result — if it raced to done, skip that assertion).
5. Run the server with `bash scripts/run_server.sh`; `curl -s http://127.0.0.1:8010/api/health`.
6. Commit.

**Acceptance:** all API tests pass; server responds on 127.0.0.1:8010 only (`ss -ltnp | grep 8010` shows `127.0.0.1:8010`).
**Verify:** `./scripts/dk pytest -q tests/test_api.py` → all passed; `curl -s http://127.0.0.1:8010/api/health` → `{"status":"ok","cuda":true,...}`.

---

## [x] T07 — Web viewer
**Depends on:** T06
**Files:** `app/static/index.html`, `app/static/app.js`, `app/static/style.css`, `app/static/vendor/three.module.js`, `app/static/vendor/OrbitControls.js`, `scripts/ui_smoke.py` (optional, see step 5)
**Steps:**
1. Download the two vendor files with `curl -fsSL -o` from the URLs in SPEC §6.6. In `OrbitControls.js`, the import is `from 'three'`; the importmap in index.html resolves it — do not edit the vendor file.
2. Build the UI per SPEC §6.6: every state in the state table, the colour map, legend, tooltips, lesion table, case list. Plain ES modules. All fetch errors go through one `api()` helper that sets the "Server not reachable" banner.
3. Colour function in JS must match: `ffrColor(0.70) = #d73027`, `ffrColor(0.80) = #fee08b`, `ffrColor(0.90) = #1a9850`, `ffrColor(1.0) = #4575b4`, linear in RGB between stops, clamped outside.
4. Restart the server (`bash scripts/run_server.sh`).
5. Automated checks: `curl -s http://127.0.0.1:8010/ | grep -c "TwinLoop CT-FFR"` ≥ 1; `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8010/static/vendor/three.module.js` = 200; if `node` exists on the host, `node --check app/static/app.js` passes. Optional: if `python -c "import playwright"` works inside or outside the container and a Chromium is available, write `scripts/ui_smoke.py` that uploads the stenosis phantom + mask, waits for "done", and saves `reports/ui_smoke.png`; otherwise skip and say so.
6. Add to PROGRESS.md under "Notes for next session" an **owner manual checklist**: open http://127.0.0.1:8010 → upload `stenosis_img.nii.gz` + mask → see a tube coloured from blue to yellow/green, lesion L1 in the table, click L1 moves the camera, click tube shows the tooltip, banner visible; stop server mid-poll → grey banner appears.
7. Commit.

**Acceptance:** automated checks in step 5 pass.
**Verify:** the three commands in step 5 with the stated outputs.

---

## [ ] T08 — Real segmentation wrapper, baseline, evaluation scripts
**Depends on:** T05 (benchmark checkpoint exists), T06
**Files:** `src/twinloop/segmentation.py` (full), `scripts/eval_seg.py`, `scripts/eval_pipeline.py`, `tests/test_segmentation_gpu.py`
**Steps:**
1. Implement `segment_nnunet` and `segment_threshold_baseline` per SPEC §5.2. Output mask: read `<out_dir>/case.nii.gz`, `CopyInformation` from the input image, cast UInt8.
2. `tests/test_segmentation_gpu.py` (`@pytest.mark.gpu` and `@pytest.mark.data`): with `TWINLOOP_NNUNET_TRAINER=nnUNetTrainer_5epochs` (the benchmark model — it is weak, the test only checks mechanics), segment the first **train** case of the manifest; assert same size as input and values ⊆ {0, 1}.
3. `scripts/eval_seg.py --method nnunet|threshold --split test --out reports/seg_eval.csv [--limit N]`: for `nnunet`, one batch `nnUNetv2_predict -i /data/nnUNet_raw/Dataset501_ImageCAS/imagesTs -o /data/runs/eval/nnunet_pred -d 501 -c 3d_fullres -f 0 -tr $TWINLOOP_NNUNET_TRAINER --disable_tta` (timed), then Dice/clSens per case vs `labelsTs` (SPEC §8 formulas). For `threshold`, run the baseline on each test image. Append rows (do not overwrite other methods' rows: read existing CSV, drop rows of the same method, write back). Print `method=<m> n=<n> dice_mean=<x.xxxx> dice_sd=<x.xxxx> clsens_mean=<x.xxxx>`.
4. `scripts/eval_pipeline.py --n 20 --out reports/pipeline_eval.csv`: SPEC §8 pipeline consistency; case dirs under `/data/runs/eval/pipeline/<case_id>_<source>/`; prints the summary numbers used in RESULTS.md.
5. **Mechanics check only (no test data):** `eval_seg.py` needs `--split train` to read images/labels from `imagesTr/labelsTr` (and `--limit N` = first N cases by `case_id`; for `nnunet` with `--limit`, copy those N images into a temp input folder before the batch predict). Run:
   `./scripts/dk python scripts/eval_seg.py --method threshold --split train --limit 3 --out /data/runs/eval/mechanics_check.csv`
   `TWINLOOP_NNUNET_TRAINER=nnUNetTrainer_5epochs ./scripts/dk python scripts/eval_pipeline.py --split train --n 2 --out /data/runs/eval/mechanics_pipeline.csv`
   These outputs go to `/data/runs/eval/`, NOT `reports/`. Do not run anything on test cases here. Do not run these while `twinloop-train` is running if `free -h` shows < 20 GB available; wait instead.
6. Commit.

**Acceptance:** GPU test passes; both scripts run end to end on train cases and write CSVs with the SPEC §8 columns.
**Verify:**
```
TWINLOOP_NNUNET_TRAINER=nnUNetTrainer_5epochs ./scripts/dk pytest -q -m "gpu and data" tests/test_segmentation_gpu.py   # 1 passed
./scripts/dk head -3 /data/runs/eval/mechanics_check.csv   # header + rows
```

---

## [ ] T09 — Wait for training to finish
**Depends on:** T05
**Steps:**
1. `docker ps -a --filter name=twinloop-train --format '{{.Status}}'` and `docker logs --tail 30 twinloop-train`.
2. Apply the kill criterion in SPEC §8 (EMA pseudo-Dice < 0.30 after epoch 50 → stop, BLOCKED).
3. If the container exited with an error: paste the last 50 log lines; if the cause is transient (OOM-kill, reboot) resume with `bash scripts/train.sh <TrainerClass> --c` (max 3 resumes, then BLOCKED).
4. If still running: write the current epoch, latest pseudo-Dice and ETA in PROGRESS.md "In progress", and do any remaining independent task (T12 steps 1–3 can be drafted). If nothing else is left, end the session with that note — the next session resumes here.
5. Done when `/data/nnUNet_results/Dataset501_ImageCAS/<TrainerClass>__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth` exists and the container exited with code 0.
6. Set `TWINLOOP_NNUNET_TRAINER=<TrainerClass>` as the default in `scripts/dk` and `scripts/run_server.sh` (replace `nnUNetTrainer` in the `${TWINLOOP_NNUNET_TRAINER:-...}` fallback). Commit.

**Acceptance:** `checkpoint_final.pth` exists; final validation pseudo-Dice copied into PROGRESS.md.
**Verify:** `ls -la ${TWINLOOP_DATA:-$HOME/twinloop-data}/nnUNet_results/Dataset501_ImageCAS/*/fold_0/checkpoint_final.pth`

---

## [ ] T10 — Segmentation evaluation on the test split
**Depends on:** T08, T09
**Steps:**
1. Ensure training is not running (`docker ps` shows no `twinloop-train`).
2. `./scripts/dk python scripts/eval_seg.py --method nnunet --split test --out reports/seg_eval.csv`
3. `./scripts/dk python scripts/eval_seg.py --method threshold --split test --out reports/seg_eval.csv`
4. Create `reports/RESULTS.md` from the SPEC §8 template; fill the Segmentation table and the hypothesis verdict. Do not re-run with other settings.
5. Commit `reports/`.

**Acceptance:** `seg_eval.csv` has one row per test case per method; RESULTS.md segmentation section filled with numbers matching the printed summaries.
**Verify:** `./scripts/dk python -c "import pandas as pd; d=pd.read_csv('reports/seg_eval.csv'); print(d.groupby('method').dice.agg(['count','mean','std']))"`

---

## [ ] T11 — Pipeline consistency evaluation
**Depends on:** T10
**Steps:**
1. `./scripts/dk python scripts/eval_pipeline.py --n 20 --out reports/pipeline_eval.csv`
2. Fill the "Pipeline consistency", "Failures and warnings" and "Caveats" sections of RESULTS.md, including every failed case with its error text. Fill the single-scan inference time (median/p90 of `t_segmentation`, pred source) into the segmentation table.
3. Commit.

**Acceptance:** 40 rows in `pipeline_eval.csv` (20 cases × 2 sources); RESULTS.md has no empty cells (use "n/a" with a reason where a value cannot exist).
**Verify:** `./scripts/dk python -c "import pandas as pd; d=pd.read_csv('reports/pipeline_eval.csv'); print(len(d), d.groupby('mask_source').status.value_counts().to_dict())"`

---

## [ ] T12 — README, end-to-end demo check, final pass
**Depends on:** T07, T11
**Files:** `README.md`
**Steps:**
1. README: what TwinLoop is (2 sentences + disclaimer), requirements (DGX Spark/ARM64, Docker, NGC access, Kaggle token), setup commands, how to run the server and CLI, how to resume training, results summary (link `reports/RESULTS.md`), known limitations (SPEC §5.4 and §5.5 lists + single-centre data + no validation), non-goals (SPEC §6.10).
2. Restart the server with the final trainer. Through the API (curl), upload the first **test** image *without* a mask; poll until `done`; record total time (must be < 300 s) and min FFR in PROGRESS.md.
   ```
   curl -s -F image=@<host path of imagesTs/ImageCAS_0005_0000.nii.gz> http://127.0.0.1:8010/api/cases
   ```
   (Use the smallest test `case_id` that exists.)
3. Full checks: `./scripts/dk pytest -q`, `./scripts/dk pytest -q -m "gpu and data"`, `./scripts/dk ruff check .`, `./scripts/dk ruff format --check .`.
4. Ensure the owner manual UI checklist (T07) is at the top of PROGRESS.md "Notes for next session".
5. Commit `"T12: v1 complete"`.

**Acceptance:** all checks pass; API demo finished `done` in < 300 s; README covers every item in step 1.
**Verify:** paste outputs of step 3 and the step 2 status JSON into PROGRESS.md.
