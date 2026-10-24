# TwinLoop — CLAUDE.md

## Mission
Build TwinLoop v1: a research prototype that takes a coronary CT angiography scan (NIfTI or zipped DICOM), segments the coronary arteries with nnU-Net v2, extracts vessel centerlines and radii, solves a steady-state 1D blood-flow model, and shows a 3D coronary tree coloured by CT-FFR in a local web app. Goal: a cheap, fast alternative to HeartFlow for Indian hospitals. There is no invasive-FFR ground truth: FFR values are unvalidated and must never be presented as accurate.

**Done** = every task in docs/TASKS.md is `[x]` with verification output in docs/PROGRESS.md: container builds, all tests and lint pass, nnU-Net trained and evaluated on the ImageCAS test split (reports/RESULTS.md), and an uploaded scan renders as a coloured FFR tree at http://127.0.0.1:8000 in under 5 minutes.

Full design: docs/SPEC.md. Task list: docs/TASKS.md. Decisions: docs/DECISIONS.md. Status: docs/PROGRESS.md.

## Machine
NVIDIA DGX Spark (GB10), **ARM64/aarch64**, ~128 GB unified CPU+GPU memory. All Python runs inside the NGC PyTorch container. Never `pip install torch` and never install Python packages on the host.

## Stack (versions pinned in requirements.txt during T00; torch comes from the base image)
- Base image `nvcr.io/nvidia/pytorch:<tag>-py3` (tag chosen in T00, recorded in DECISIONS.md)
- nnU-Net v2 (`3d_fullres`, fold 0), SimpleITK, scikit-image, scipy, networkx, pandas, pydantic v2
- FastAPI + uvicorn, plain HTML/JS + three.js 0.160.0 (vendored, no build step)
- pytest, ruff (line length 100)

## Commands (run from the repo root on the host)
```
docker build -t twinloop:dev .                          # build image
./scripts/dk bash scripts/check_env.sh                  # env check -> "ENV OK"
./scripts/dk pytest -q                                  # tests (excludes gpu/data markers)
./scripts/dk pytest -q -m "gpu and data"                # GPU + data tests
./scripts/dk ruff check . && ./scripts/dk ruff format --check .   # lint
./scripts/dk ruff format .                              # format
./scripts/dk python -m twinloop.pipeline --image /data/... [--mask /data/...]  # CLI
bash scripts/run_server.sh                              # web app -> http://127.0.0.1:8000
./scripts/dkd <twinloop-name> <cmd...>                  # detached long job
docker logs --tail 20 twinloop-train                    # training progress
```
`scripts/dk` mounts the repo at `/workspace/twinloop` and `$TWINLOOP_DATA` (default `$HOME/twinloop-data`) at `/data`. Paths in container commands use `/data/...`.

## Repo map
- `src/twinloop/` — core library: `config.py` (all parameters), `io.py`, `phantom.py`, `segmentation.py`, `centerline.py`, `lesions.py`, `solver.py`, `schema.py` (result.json models), `casefiles.py`, `pipeline.py` (CLI)
- `app/` — FastAPI server (`server.py`), job runner (`jobs.py`), static web UI (`static/`)
- `scripts/` — docker helpers, data download/manifest/conversion, training, evaluation, server launcher
- `tests/` — pytest suites (phantom-based; no real data needed except `gpu`/`data` markers)
- `reports/` — RESULTS.md and evaluation CSVs (committed)
- `docs/` — SPEC, TASKS, DECISIONS, PROGRESS

## Hard rules
- Never delete or modify anything under `/data/raw/` after the manifest exists. Never commit data, model weights, or any `.nii*`/`.zip`.
- Never use ImageCAS test-split cases (`case_id % 5 == 0`) except in T10–T11 evaluation and the T12 demo run. Never tune any parameter, threshold or epoch count on them. Report results honestly, including failures.
- Never change `result.json` fields, API routes, status names, or error strings defined in SPEC §5–6. You may add optional fields only if logged in DECISIONS.md.
- Never add a dependency without adding it pinned to requirements.txt and logging it in DECISIONS.md.
- Never bind the server to anything but 127.0.0.1 on the host. No auth exists, so it must stay local.
- Never print, copy or commit secrets (`~/.kaggle/access_token`, NGC key). Refer to them by env var/file name only.
- Never start `twinloop-train` while it is already running (check `docker ps`); never `docker rm -f` it while running.
- Never write any claim of clinical accuracy. The disclaimer string in SPEC §6.3 must appear in every result and UI screen.
- Never `git push`. Commit locally after each finished task: `git commit -m "T<id>: <title>"`.
- Change parameter defaults only where TASKS.md says to; log every change in DECISIONS.md.

## Working protocol
- Work one task at a time from docs/TASKS.md, in order.
- Before you start a task, re-read its acceptance criteria. After it's done, run its verification command and paste the result into docs/PROGRESS.md.
- If something is ambiguous, choose the option marked DEFAULT in the spec. If there isn't one, choose the simplest reversible option, log it in docs/DECISIONS.md, and keep going. Don't stop to ask unless the action is destructive or irreversible.
- If the same error survives 3 different fix attempts, stop. Write the error, what you tried, and your best hypothesis into docs/PROGRESS.md under BLOCKED, then move to the next independent task.
- Never mark a task done without passing verification.

## Session start checklist
1. Read docs/PROGRESS.md ("In progress", "BLOCKED", "Notes for next session").
2. `docker ps -a --filter name=twinloop` to see running/finished background jobs.
3. Continue with the first unfinished task whose dependencies are done.
