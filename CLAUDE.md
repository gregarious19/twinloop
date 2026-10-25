# TwinLoop — CLAUDE.md

## Mission
TwinLoop is a research prototype that estimates CT-FFR from a coronary CT angiography scan: nnU-Net
segments the coronary arteries, centerlines and radii feed a steady 1D blood-flow model, and a local
web app shows the coronary tree coloured by FFR. It aims to be a cheap, fast alternative to HeartFlow
for Indian hospitals. FFR values are unvalidated (no invasive-FFR data) and are never presented as
accurate. **v1** (docs/SPEC.md, docs/TASKS.md) builds the base pipeline. **v2** (docs/SPEC_V2.md)
adds TotalSegmentator anatomy (aorta, myocardium), a segmentation repair loop (decider + gap-filler),
CPR-based lumen refinement and myocardial-mass flow.

**Done (v2)** = every task in your machine's task file is `[x]` (or `[x] skipped` where allowed) with
verification output in your progress file; all tests and ruff pass; reports/V2_RESULTS.md is filled.

## Where v2 lives
v2 is developed on git branch `v2` in a **separate worktree folder** (e.g. `~/twinloop-v2`) of the
same repo. The original folder (e.g. `~/twinloop`, branch `main`) belongs to a separate v1 Claude
session that finishes v1 (training, API, UI, evaluation). Never `cd` into, edit, commit in, or check
out branches in the v1 folder; never run `git checkout`/`git switch` here. You may read `main`
(`git log main`, `git show main:docs/PROGRESS.md`). v1 code reaches `v2` only through the merge
task in docs/TASKS_V2_SPARK.md. The v2 Docker image tag is `twinloop:v2` (v1 keeps `twinloop:dev`).

## Which machine am I on? (decide first, every session)
| `uname -m` | Machine | Work from | Log progress in | Log decisions in |
|---|---|---|---|---|
| `aarch64` | SPARK (DGX Spark, Docker) | docs/TASKS_V2_SPARK.md | docs/PROGRESS_V2_SPARK.md | docs/DECISIONS_V2.md |
| `x86_64` with `sbatch` | CLUSTER (Slurm, 80 GB GPU) | docs/TASKS_V2_CLUSTER.md | docs/PROGRESS_CLUSTER.md | docs/DECISIONS_CLUSTER.md |
Edit only files your machine owns (SPEC_V2 §1 table). CLUSTER owns `src/twinloop/nets.py`,
`train/`, `cluster/`, `tests/test_nets.py`, `requirements-cluster.txt`, its progress/decisions files and
`reports/v2_training.md`; SPARK owns everything else.

## Stack
- SPARK: base image `nvcr.io/nvidia/pytorch:<tag>-py3` (tag in docs/DECISIONS.md D25); nnU-Net v2,
  TotalSegmentator, SimpleITK, scikit-image, scipy, networkx, pandas, pyarrow, scikit-learn, joblib,
  pydantic v2, FastAPI, three.js 0.160.0 (vendored); pins in requirements.txt / requirements.lock.
- CLUSTER: venv (Python minor = SPARK's), torch 2.5.1 (CUDA wheel), pins in requirements-cluster.txt
  generated from requirements.lock; scikit-learn version must equal SPARK's.
- pytest, ruff (line length 100) on both.

## Commands
SPARK (host, repo root; `./scripts/dk` runs inside the container, data at `/data`):
```
docker build -t twinloop:v2 .
./scripts/dk pytest -q                       # CPU tests (excludes gpu/data markers)
./scripts/dk pytest -q -m "gpu and data"     # GPU + data tests
./scripts/dk ruff check . && ./scripts/dk ruff format --check .
./scripts/dk python -m twinloop.pipeline --image /data/... [--mask /data/...] [--repair] [--lumen]
bash scripts/run_server.sh                   # web app on http://127.0.0.1:8000
docker ps -a --filter name=twinloop          # background jobs (twinloop-train = first-pass training)
free -g                                      # check before any GPU run: need >= 20 GB available
```
CLUSTER (login node, repo root):
```
source cluster/cluster.env && source $VENV/bin/activate
python -m pytest -q tests/test_nets.py       # only short CPU tests on the login node
ruff check train src/twinloop/nets.py tests/test_nets.py && ruff format --check train src/twinloop/nets.py tests/test_nets.py
cluster/submit.sh gpu|cpu [sbatch flags] <job script> [args]   # every real job goes through Slurm
squeue -u $USER ; sacct -j <id> --format=JobID,JobName,State,Elapsed,ExitCode -X
```

## Repo map
- `src/twinloop/` — v1: config, io, phantom, segmentation, centerline, lesions, solver, schema,
  casefiles, pipeline. v2: paths, config_v2, v2split, graph, anatomy, damage, regions, cpr, gapfill,
  lumen, repair, flow, modelio, metrics; nets (CLUSTER-owned).
- `app/` — FastAPI server, job runner, static web UI. `scripts/` — docker helpers, data, eval, v2
  scripts. `train/` — CLUSTER dataset generators and training scripts. `cluster/` — Slurm setup.
- `tests/` — pytest (phantoms; `gpu`/`data` markers for real data). `reports/` — results (committed).
- `docs/` — SPEC, SPEC_V2, TASKS, TASKS_V2_SPARK, TASKS_V2_CLUSTER, DECISIONS*, PROGRESS*.

## Hard rules
- Never use ImageCAS test-split cases (`case_id % 5 == 0`) in v2. v1 uses them only in v1 T10–T12.
  Never tune any parameter, threshold, model or epoch count on test, `v2_holdout` or `v2_realfail`
  cases (SPEC_V2 §2.2). Report all results honestly, including failures and killed runs.
- Never stop, restart or `docker rm` the `twinloop-train` container while it runs.
- Never delete or modify `/data/raw/`. Never commit data, weights, `.nii*`, `.npz`, `.parquet`, `.zip`.
- Never print, read, copy or commit secrets: `~/.kaggle/access_token`, the NGC key, the TotalSegmentator
  licence key or `/data/totalseg/config.json` (existence checks only).
- Never change v1 `result.json` fields, API routes, status names or error strings. Optional new fields
  only if logged in your decisions file (v2 adds exactly one: `v2`).
- Never add a dependency without pinning it (requirements.txt on SPARK, requirements-cluster.txt on
  CLUSTER) and logging it.
- Never bind the server to anything but 127.0.0.1. Never write any claim of clinical accuracy; the
  v1 disclaimer string stays on every result and screen.
- Never `git push`, never copy data between machines: write `WAITING ON OWNER: <exact command>` in
  your progress file and continue with the next independent task.
- Commit locally after each finished task with the exact prefix: `git commit -m "S<nn>: <title>"`
  (SPARK v2) or `"C<nn>: <title>"` (CLUSTER), always on branch `v2`.
- Change parameter defaults only where a task says so; log every change.

## Working protocol
- Work one task at a time from docs/TASKS.md, in order.
- Before you start a task, re-read its acceptance criteria. After it's done, run its verification command and paste the result into docs/PROGRESS.md.
- If something is ambiguous, choose the option marked DEFAULT in the spec. If there isn't one, choose the simplest reversible option, log it in docs/DECISIONS.md, and keep going. Don't stop to ask unless the action is destructive or irreversible.
- If the same error survives 3 different fix attempts, stop. Write the error, what you tried, and your best hypothesis into docs/PROGRESS.md under BLOCKED, then move to the next independent task.
- Never mark a task done without passing verification.

For v2, read the protocol with your machine's files substituted: "docs/TASKS.md" → your task file
from the table above, "docs/PROGRESS.md" → your progress file, "docs/DECISIONS.md" → your decisions
file. v1 tasks (docs/TASKS.md) are done by the v1 session in the v1 folder, never by you.

## Session start checklist
1. `uname -m` → pick your row in the machine table.
2. Read your progress file ("In progress", "BLOCKED", "WAITING ON OWNER", "Notes for next session").
   SPARK also reads docs/PROGRESS_CLUSTER.md "Notes for next session" for bug reports to fix first.
3. SPARK: `docker ps -a --filter name=twinloop`. CLUSTER: `squeue -u $USER` and `sacct` for job IDs
   listed in your progress file.
4. Continue with the first unfinished task whose dependencies are met.
