# Progress

Update this file after every task. Newest entries at the top of each section.

## Done
### T05 — nnU-Net dataset, preprocessing, benchmark, training launched (2026-10-24)
Files: `scripts/convert_to_nnunet.py`, `scripts/train.sh`. Decisions: D40, D41.
```
ls .../imagesTr | wc -l   # 800  (labelsTr 800, imagesTs 200, labelsTs 200; dataset.json numTraining 800)
nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres --verify_dataset_integrity -np 4   # exit 0, 800 cases in 58 min; nnUNetPlans.json exists
benchmark nnUNetTrainer_5epochs: epoch times 143.5 (compile), 104.6, 104.4, 104.6, 104.6 s  -> t = 104.5 s
```
Chosen: `nnUNetTrainer` (1000 epochs), est. 31.9 h with the 1.1 margin. **Training start 2026-10-24 19:38 UTC; expected end about 2026-10-26 ~01:00 UTC** (1000 × ~105-108 s, plus final validation). Epoch 3 reached at 19:45 with train_loss -0.54 and pseudo-Dice 0.58 (just a progress sign, not a result). Container `twinloop-train`; resume after a crash/reboot with `bash scripts/train.sh nnUNetTrainer --c`. Kill rule: after epoch 50 if EMA pseudo-Dice < 0.30, stop.

### T01 — ImageCAS download and manifest (2026-10-24)
Files: `scripts/make_manifest.py` (SPEC §4.2). Download finished with `DOWNLOAD DONE`; the data came as 5 split zips (`1-200` … `801-1000`, last part `*.change2zip`), joined and extracted with `bash scripts/join_split_zips.sh` (`JOIN DONE`). Originals kept under `/data/raw/imagecas/`. Files are `<id>.img.nii.gz` / `<id>.label.nii.gz`.
Verification:
```
./scripts/dk python scripts/make_manifest.py --root /data/raw/imagecas --out /data/raw/imagecas/manifest.csv
cases=1000 train=800 test=200 unpaired=0
1000 {'train': 800, 'test': 200} [512] size_y [512]; spacing_x min 0.289 max 0.465; spacing_z 0.5 (min=max); all image/label paths exist
ruff check: All checks passed!; ruff format --check: 23 files already formatted
```
Manifest took 75 s. `/data/raw/` is now frozen (manifest exists).

### T07 — Web viewer (2026-10-24)
Files: `app/static/index.html`, `app.js`, `style.css`, `color.js` (D37), `vendor/three.module.js` + `vendor/OrbitControls.js` (three.js r160 from the SPEC URLs, unmodified).
Verification commands and output (server restarted with `bash scripts/run_server.sh`):
```
curl -s http://127.0.0.1:8000/ | grep -c "TwinLoop CT-FFR"                                    # 2   (>= 1)
curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/static/vendor/three.module.js     # 200
node --check app/static/app.js                                                                 # (no output, passes)
```
Extra checks beyond the task: `ffrColor` returns exactly #d73027 / #fee08b / #1a9850 / #4575b4 at 0.70 / 0.80 / 0.90 / 1.00 and clamps outside (plain node test). A headless Chromium 145 (existing Playwright download, driven over CDP; D38) uploaded the stenosis phantom + mask and showed: title, always-visible disclaimer banner, summary (min FFR 0.94, 1 lesion), lesion row `L1 | tree_0_b0 | 42.4 | 6.5 | 0.99 | 0.95 | 0.05`, legend, a 762x631 WebGL canvas, clicking L1 places the white marker, clicking the tube shows `FFR 0.98 · ⌀ 1.6 mm · tree_0_b0`, and zero console errors. Also checked in the browser: unsupported file -> red text "Only .nii.gz, .nii or .zip files are supported"; image without mask (no model) -> red box "Segmentation model not available; provide a mask or finish training" + "Try another file" button; fetch failure -> grey banner "Server not reachable — retrying…", which clears on recovery.
Not verified by me: the spinner / upload progress bar visuals (the phantom finishes in 0.1 s), the < 900 px narrow layout, Firefox/Safari. See the owner checklist.

### T06 — Job runner and HTTP API (2026-10-24)
Files: `app/jobs.py`, `app/server.py`, `app/static/index.html` (placeholder), `scripts/run_server.sh`, `tests/test_api.py`.
Verification commands and output:
```
./scripts/dk pytest -q tests/test_api.py      # 17 passed, 1 warning (starlette: httpx vs httpx2 deprecation, harmless)
bash scripts/run_server.sh
{"status":"ok","cuda":true,"model_available":false,"queue_length":0}
ss -ltnp | grep 8000
LISTEN 0 4096 127.0.0.1:8000 0.0.0.0:*
```
Also: curl upload of the stenosis phantom image + mask -> `{"case_id":"c20261024144253_3287fe","status":"queued"}` -> `done` in 0.1 s; the T04 CLI case appears in `GET /api/cases`; `GET /` returns 200. The server container `twinloop-server` is left running.
Decisions: D35, D36.

### T04 — Pipeline + result.json + CLI (2026-10-24)
Files: `src/twinloop/schema.py`, `casefiles.py`, `pipeline.py`, `segmentation.py` (stub: `model_available()` real, `segment_nnunet()` raises the not-available error), `tests/test_pipeline_phantom.py`.
Verification commands and output:
```
./scripts/dk pytest -q tests/test_pipeline_phantom.py     # 6 passed in 1.14s
./scripts/dk python -m twinloop.phantom --kind stenosis --out /data/runs/phantom
./scripts/dk python -m twinloop.pipeline --image /data/runs/phantom/stenosis_img.nii.gz --mask /data/runs/phantom/stenosis_mask.nii.gz
OK case_id=c20261024144113_6ae41c trees=1 lesions=1 min_ffr=0.945 total_s=0.1
```
Full suite: 41 passed; ruff check and format --check clean.
Step 5 (real-data slice, train cases 1, 2, 3 with their label masks), run after T01:
```
OK case_id=c20261024172233_7c0945 trees=2 lesions=5 min_ffr=0.484 total_s=5.1
OK case_id=c20261024172239_6903cf trees=2 lesions=1 min_ffr=0.954 total_s=5.1
OK case_id=c20261024172246_d2a991 trees=2 lesions=4 min_ffr=0.913 total_s=4.9
```
Sanity: all have 2 trees, converged, hyperemic root flow 93–320 mL/min (limit 50–400), total about 5 s (< 60 s). **One value outside the 0.5–1.0 sanity range:** case 1 min_ffr 0.484, from lesion L4 (DS 64.7 %, r_min 0.465 mm, r_ref 1.318 mm, in a small distal branch tree_0_b6). It follows from the model on that mask, not from an orientation or unit error that I could find; no parameters were changed. Unvalidated value, not a clinical statement.

### T03 — I/O, phantoms, centerlines, lesions (2026-10-24)
Files: `src/twinloop/io.py`, `phantom.py` (incl. `write_dicom_series`, `python -m twinloop.phantom`), `centerline.py` (`extract_trees`), `lesions.py`, `tests/test_io.py`, `test_centerline.py`, `test_lesions.py`.
Verification command:
```
./scripts/dk pytest -q tests/test_io.py tests/test_centerline.py tests/test_lesions.py
```
Output:
```
..........................                                               [100%]
26 passed in 2.32s
```
Full suite 35 passed; `ruff check .` All checks passed!; `ruff format --check .` 14 files already formatted. `extract_trees` on y_tree: 0.1 s (limit 10 s).
Measured on phantoms: tube length 58.4 mm, 1 branch; y_tree 3 branches, root median radius 2.04 mm, root starts at z=6.4; stenosis lesion DS 0.42 (true 0.50), length 6.5 mm, min at z=39.8 mm; tube has 0 lesions. Sanity check: with `ds_threshold` temporarily 0.95, 5 of 10 lesion tests failed (so they do detect changes); restored to 0.25.
Decisions: D33 (half-voxel correction default False), D34 (interpretation notes).

### T02 — Config, schema stubs and the 1D solver (2026-10-24)
Files: `src/twinloop/config.py` (SPEC §5.6 + `MMHG_TO_PA`), `centerline.py` (`Branch`, `VesselTree` only), `lesions.py` (`Lesion` only), `solver.py`, `tests/test_solver.py`.
Verification command:
```
./scripts/dk pytest -q tests/test_solver.py
```
Output:
```
........                                                                 [100%]
8 passed in 0.05s
```
Notes: flow-law constants (1.43·d^2.55) could not be verified by WebSearch, kept per SPEC §5.5 (see D29). Verified by hand that `test_single_tube_coupled` expectations (96.96 mL/min, FFR 0.9881) follow from the spec constants and they pass.

### T00 — Machine check, container, repo scaffold (2026-10-24)
Machine check: `aarch64`; NVIDIA GB10, driver 580.159.03; 121 GiB RAM (113 GiB available at end); 3.1 TB free on `$HOME`; Docker 29.2.1 with runtime `runc` only (GPU passthrough via `--gpus all` works anyway); git 2.43.0 (`git init -b main` done).
Base image: `nvcr.io/nvidia/pytorch:26.09-py3` (D25). Two SPEC issues found and fixed minimally: D26 (torch constraint string), D27 (ruff formatting Markdown).
Verification commands and output:
```
./scripts/dk bash scripts/check_env.sh
torch 2.14.0a0+b2c75dd062.nv26.09 device NVIDIA GB10 out (1, 8, 64, 64, 64)
ENV OK

./scripts/dk pytest -q          # 9 passed (1 smoke + 8 solver from T02; smoke alone: 1 passed)
./scripts/dk ruff check .       # All checks passed!
./scripts/dk ruff format --check .   # 9 files already formatted
```
`requirements.txt` fully pinned to `==` versions (D28); `requirements.lock` has 269 `name==version` lines (torch excluded, it comes from the base image).
Surprises: another process on the host (not ours) holds ~38 GB of GPU memory (`nvidia-smi`), relevant to the ~70 GB GPU budget for training. The image pull took > 10 min.

## In progress
**T01 (ImageCAS download) is running** in detached container `twinloop-download` (dataset `xiaoweixumedicalai/imagecas`, 83 GB; ~7 MB/s; started ~20:03 IST; at 20:19 IST it was 7 % done; ETA about 23:30-00:00 IST on 2026-10-24). It survives the session.
Resume order:
1. `docker logs --tail 1 twinloop-download | tail -c 150` -> done when `docker ps -a --filter name=twinloop-download` shows `Exited (0)` and the log ends with `SPLIT ARCHIVES PRESENT ...` then `DOWNLOAD DONE`. If it exited non-zero, paste the last 30 log lines here and rerun `./scripts/dkd twinloop-download bash scripts/download_imagecas.sh xiaoweixumedicalai/imagecas` (re-downloads everything; that is accepted by TASKS T01 step 4). Then `docker rm twinloop-download`.
2. `bash scripts/join_split_zips.sh` on the HOST (D32: split zips; needs host `zip`/`unzip`, both present). Expect 5 directories `~/twinloop-data/raw/imagecas/{1-200,201-400,401-600,601-800,801-1000}/`, each with a `.extracted` marker. Look at the extracted layout (file names) before writing the manifest: SPEC §4.2 says case_id = first integer in the basename and a basename containing `label` is a label.
3. T01 steps 5-7: write `scripts/make_manifest.py` (uses `twinloop.io.load_image`, `Pool(8)`), run it, check the acceptance numbers (990-1000 cases, ~800/200 split, size_x == size_y == 512), commit.
4. T04 step 5: first 3 train cases with their label masks through the CLI (`OK ...` lines; sanity ranges in TASKS T04 step 5). Do not tune parameters on data.
5. Then T05 (nnU-Net conversion, preprocessing, benchmark, launch training), T08, T09 (wait), T10, T11, T12.
Only one detached training container may exist (`twinloop-train`); never `docker rm -f` it while running.
Server container `twinloop-server` is currently running (127.0.0.1:8010); restart with `bash scripts/run_server.sh` after code changes.

## BLOCKED
(none)

Resolved: T00 docker-permission block (2026-10-24). The user `pranay` is now in the `docker` group; `docker ps` works.

<!-- Format:
### T<id> — <short title>
Error: <exact message>
Tried: 1) ... 2) ... 3) ...
Best hypothesis: ...
Needs from owner (if any): ...
-->

## Notes for next session
### Owner manual UI checklist (T07) — please do this once
1. Open http://127.0.0.1:8010 (start with `bash scripts/run_server.sh` if needed). The yellow disclaimer banner must be visible at all times.
2. Upload `~/twinloop-data/runs/phantom/stenosis_img.nii.gz` with `stenosis_mask.nii.gz` as the optional mask -> a tube coloured blue (proximal) to green/yellow, lesion L1 in the table.
3. Click L1 -> the camera moves to it and a white sphere appears. Click the tube -> tooltip like `FFR 0.98 · ⌀ 1.6 mm · tree_0_b0`.
4. Upload a larger file (a real ImageCAS image once available) and watch the progress bar and the spinner labels (Waiting in queue / Reading scan / Finding coronary arteries / Measuring vessels / Computing FFR).
5. Stop the server mid-poll (`docker stop twinloop-server`) -> grey banner "Server not reachable — retrying…"; start it again -> banner disappears.
6. Shrink the window below 900 px: panels stack vertically.
- Never run `ruff format .` on files other than code without `extend-exclude = ["docs"]` in place (it is now in pyproject.toml). Before that fix, ruff rewrote code blocks in docs/SPEC.md and docs/TASKS.md; they were restored to the original text (752 / 302 lines). There is no git baseline for the docs because the first commit came after that, so if the docs look odd, compare line counts.
- `docker build` takes ~3 min after the first time (layers cached). The base image is 25 GB.
- Kaggle auth is untested (T01 step 0).
