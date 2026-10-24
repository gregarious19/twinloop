# Decisions

Format: `D<n> — <decision>. Reason: <one line>.` Pre-filled by the planner; append new ones at the end (with the task ID that made them).

## Product and scope
- D1 — v1 is a local research prototype: pipeline + FastAPI + browser viewer, no auth, server on 127.0.0.1 only. Reason: the owner wants something runnable to show people; deployment and accounts come later.
- D2 — Docker image is the deployment unit; AWS deployment is out of scope for v1. Reason: one task later, and the image already isolates the ARM environment.
- D3 — No regulatory work (CDSCO, ISO 13485) in v1. Reason: the owner explicitly defers legal work until the product is built.
- D4 — Every output carries a "not validated, not for clinical use" disclaimer. Reason: no invasive-FFR data exists in the project.

## Platform
- D5 — All Python inside the NGC PyTorch container; torch never pip-installed. Reason: DGX Spark is ARM64 + Blackwell; NGC ships a matching torch build, PyPI wheels may not.
- D6 — Base image tag: chosen in T00 by trying 26.09 down to 25.09 (fill in below). Reason: the newest tag that pulls and passes check_env gives the best GB10 support.
- D7 — Long jobs run in detached named containers (`scripts/dkd`). Reason: they must survive the Claude Code session ending.
- D8 — Training cap 72 h of GPU time; epoch count chosen from a measured benchmark. Reason: GB10 memory bandwidth is lower than datacenter GPUs; the owner accepted a longer budget.

## Data
- D9 — ImageCAS from the official Kaggle account `xiaoweixumedicalai`; no unofficial mirrors. Reason: licence and data integrity.
- D10 — Fixed split: test if `case_id % 5 == 0` (~200), else train (~800). Reason: deterministic, documented, never changes; avoids depending on the official split file whose availability is uncertain.
- D11 — Everything reoriented to LPS with `sitk.DICOMOrient` on load, including nnU-Net training copies. Reason: training and inference must see the same orientation; DICOM-native convention.

## Segmentation
- D12 — nnU-Net v2 `3d_fullres`, fold 0 only, TTA disabled at inference. Reason: 5 folds would take ~5× the budget; TTA ~8× slower inference; the prototype needs a single fast model.
- D13 — Inference via the `nnUNetv2_predict` CLI in a subprocess. Reason: the CLI is the most stable interface across nnU-Net versions and isolates GPU memory.
- D14 — Baseline: HU threshold 250–700, 2 largest components. Reason: the simplest possible baseline, proves the model learns something non-trivial.
- D15 — One training seed instead of ≥ 3. Reason: one run takes up to 72 h; uncertainty reported across test cases instead.

## Geometry and physics
- D16 — Centerlines with scikit-image skeletonize + networkx, not VMTK. Reason: VMTK has unreliable ARM64 builds; pure Python installs everywhere.
- D17 — Own steady-state 1D solver (Poiseuille + Young–Tsai turbulent loss, Murray outlet split, flow scaling law, hyperemia factor 0.24), not SimVascular. Reason: transparent, testable against analytic cases, no hard install on ARM.
- D18 — Aortic pressure default 90 mmHg, user-adjustable 60–140. Reason: typical mean aortic pressure; patient-specific value improves estimates.
- D19 — Resting flow law q = 1.43·d^2.55 (mL/min, d in mm), van der Giessen et al. 2011 — constants to be verified in T02. Reason: no myocardial mass segmentation in v1.
- D20 — No junction losses and no pulsatility in v1. Reason: keeps the solver simple; listed as a known limitation.
- D21 — Lesion detection by local 80th-percentile reference radius, DS ≥ 25 %, with 2 mm end exclusion zones. Reason: simple, deterministic; ostial and long diffuse lesions are known misses.

## App
- D22 — Single in-process worker thread, one case at a time. Reason: one GPU; no need for Celery/Redis.
- D23 — three.js 0.160.0 vendored into app/static/vendor. Reason: works offline in hospitals, no build step.
- D24 — Results stored as files under `/data/runs/cases/<case_id>/` (no database). Reason: simplest; easy to inspect and back up.

## Filled in during execution
<!-- Sonnet appends here: D25 onward -->
- D25 — (T00) Base image tag: `nvcr.io/nvidia/pytorch:26.09-py3` (the first tag in the list; pulled anonymously, no NGC login needed; passes check_env.sh on the GB10: torch 2.14.0a0+b2c75dd062.nv26.09, CUDA forward-compat mode with driver 580.159.03). Reason: SPEC §2.3 procedure, first tag that pulls and passes.
- D26 — (T00) SPEC ISSUE: the Dockerfile's constraint line `torch==<torch.__version__>` can never match. pip's metadata version of the base image's torch is `2.14.0a0+b2c75dd062.nv26.9.68203377` while `torch.__version__` prints `...nv26.09`, so `pip install -c` fails with ResolutionImpossible for any package that depends on torch (nnunetv2, timm). Fix applied (minimal): the constraint and the final assertion read the version from `importlib.metadata.version('torch')` instead of `torch.__version__`. The SPEC §2.3 no-deps fallback was tried first, then reverted: it was unnecessary once the constraint was correct, so `requirements-nodeps.txt` does not exist. Reason: smallest change that keeps torch pinned to the base image.
- D27 — (T00) SPEC ISSUE: `ruff format` formats Python code blocks inside Markdown, so `ruff format .` rewrote the examples in docs/SPEC.md and docs/TASKS.md, and `ruff format --check .` would fail on the unmodified docs. Fix: `extend-exclude = ["docs"]` under `[tool.ruff]` in pyproject.toml (a one-line addition to the SPEC §3 content). The docs files were restored to their original text (752 / 302 lines) after I formatted them by accident; only the cosmetic code-block layout had changed.
- D28 — (T00) Pinned versions (requirements.txt == requirements.lock; torch comes from the base image, not pinned): nnunetv2 2.8.1, SimpleITK 2.5.6, nibabel 5.4.2, scikit-image 0.26.0, scipy 1.18.1, networkx 3.6.1, pandas 3.0.5, pydantic 2.13.5, fastapi 0.162.0, uvicorn[standard] 0.54.0, python-multipart 0.0.32, httpx 0.28.1, kaggle 2.2.5, pytest 9.1.1, ruff 0.17.2. `pip check` reports only base-image issues (cupti-python, quack-kernels, flash-attn-4 vs nvidia-cutlass-dsl), none from our packages.
- D29 — (T02) Flow-law constants (q_rest = 1.43 · d^2.55 mL/min, van der Giessen et al. 2011) could NOT be verified: one WebSearch found the paper (J Biomech 44(6):1089–1095) but not the equation. The exponent 2.55 lies within the 2.06–3.20 range reported for coronary trees in the literature. Kept SPEC values; flagged in PROGRESS.md. Reason: SPEC §5.5 says keep and note when unverifiable.
- D30 — (T01) Kaggle auth: `kaggle` 2.2.5 is the newest release but does not read `access_token` from `KAGGLE_CONFIG_DIR` (inside the container `HOME=/tmp`). Method that worked: the second fallback in TASKS T01 step 0, `-e KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/access_token")"` added to `scripts/dk` and `scripts/dkd`. The token exists only in the container environment, never written to a file.
- D31 — (T01) Dataset slug: `xiaoweixumedicalai/imagecas` (official, 89,108,870,662 bytes, last updated 2023-11-30). One dataset, no parts. Other Kaggle copies (`rj1007/imagecas`, `shakthipraveen/*`, ...) are unofficial and not used (D9). Kaggle lists the licence as **CC BY-NC 4.0 (non-commercial)**: the owner should check this against any commercial plan for models trained on it.
- D32 — (T01) The upload is 5 split zip sets (`1-200`, `201-400`, `401-600`, `601-800`, `801-1000`; each `<n>.change2zip` + `.z01`–`.z04`), plus the authors' code folder and `imageCAS_data_split.xlsx`. Python `zipfile` cannot read split archives and the image has no `zip`/`7z`. Solution without a Dockerfile change or new dependency: `scripts/join_split_zips.sh` runs on the host with the host's Info-ZIP `zip -s 0` + `unzip`, extracting each set into its own `raw/imagecas/<n>/` directory; originals are kept. `download_imagecas.sh` prints a reminder when split archives are present. The official split spreadsheet is NOT used; D10's split stays.
- D33 — (T03) `CenterlineParams.radius_half_voxel_correction` default changed True -> False, as SPEC §5.3 step 3 / TASKS T03 instruct when the uncorrected value is closer on the phantoms. Measured (median radius over the middle 80 % of the root branch, phantom truth in brackets): tube corrected 1.382 / raw 1.565 (1.5); y_tree root corrected 1.857 / raw 2.041 (2.0); stenosis minimum radius corrected 0.599 / raw 0.783 (0.75). Phantoms only; no ImageCAS case was used.
- D34 — (T03) Interpretation notes for SPEC §5.3: (a) junction clusters (step 7) are formed from degree>=3 voxels connected by edges of the spanning tree (not merely 26-adjacent in the skeleton graph), so contraction can never create a loop; (b) when scoring root candidates (step 6) the path walk stops before a degree!=2 voxel, so the junction bulge radius is not counted.
- D35 — (T06) The uploaded mask is stored as `input_mask.nii.gz` or `input_mask.nii`, matching the upload's extension (SPEC §6.4 names only `input_mask.nii.gz`). Reason: SimpleITK picks the NIfTI reader from the file name, so a plain `.nii` saved under a `.nii.gz` name would fail to read. Mask uploads other than `.nii`/`.nii.gz` get the same 400 "Only .nii.gz, .nii or .zip files are supported" message.
- D36 — (T06) `queue_length` in /api/health counts queued plus running jobs. Reason: SPEC only says "queue_length"; this is what a client needs to know whether the worker is busy.
