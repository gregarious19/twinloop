# TwinLoop — CLAUDE.md (v3)

## Mission
TwinLoop estimates CT-FFR from a coronary CT angiography (CCTA) scan: nnU-Net segments the coronary
arteries, centerlines and radii feed a steady 1D blood-flow model. FFR values are unvalidated (no
invasive-FFR data) and are never presented as accurate. v2 is finished; its three hypotheses
(H1, H2, H3) were all falsified. The first-pass pipeline flags 0 of 16 cases whose ground-truth (GT)
mask gives min FFR < 0.80, but a σ 0.5 mm blur of the GT mask (Dice 0.956) loses its stenoses almost
as badly, so the GT stenosis "ruler" itself is in doubt.
**v3** first audits that ruler (Phase A), then, only if two gates pass, tests three fixes against
one pre-registered hypothesis H4 (Phase B). If a gate fails, it researches external data (Phase C).
Full design: docs/SPEC_V3.md. Tasks: docs/TASKS_V3.md.

**Done (v3)** = every task in docs/TASKS_V3.md is `[x]` (or `[x] skipped (<reason>)` where a gate
or condition says so) with verification output in docs/PROGRESS_V3.md; `./scripts/dk pytest -q` and
both ruff commands pass; the final report the gates lead to is written (reports/V3_RESULTS.md, or
reports/v3_external_data.md when a gate failed).

## Where v3 lives
Branch `v3` in the worktree folder `twinloop-v3` (sibling of `twinloop-v2`). SPARK only (DGX
Spark, `uname -m` = `aarch64`); there is no CLUSTER work in v3. Never `cd` into, edit or commit in
the `twinloop-v2` or `twinloop` (v1, branch `main`) folders. Never run `git checkout`/`git switch`.
You may read other branches (`git show v2:<path>`, `git log v2`).

## Stack (unchanged from v2; no new dependencies in v3)
Docker image `twinloop:v2` (base `nvcr.io/nvidia/pytorch:26.09-py3`, Python 3.12, torch
2.14.0a0 NGC build, nnU-Net 2.8.1, TotalSegmentator 2.18.0, SimpleITK 2.5.6, scikit-image 0.26.0,
scipy 1.18.1, networkx 3.6.1, pandas 3.0.5, scikit-learn 1.9.1, joblib, pyarrow, pydantic 2.13.5).
Pins: requirements.txt / requirements.lock. Do not rebuild the image. pytest, ruff (line length 100).

## Commands (host, repo root of `twinloop-v3`; `./scripts/dk` runs inside the container, data at `/data`)
If `docker ps` says "permission denied", prefix every docker-using command with `sg docker -c "..."`.
```
./scripts/dk pytest -q                               # CPU tests (excludes gpu/data markers)
./scripts/dk pytest -q -m "gpu and data"             # GPU + data tests
./scripts/dk ruff check . && ./scripts/dk ruff format --check .
./scripts/dk python scripts/<script>.py [args]       # every v3 script runs this way
./scripts/dkd <container-name> <command>             # detached long job (survives the session)
docker ps -a --filter name=twinloop                  # background jobs
docker logs --tail 20 <container-name>               # job progress
nvidia-smi ; free -g                                 # before any GPU job: >= 20 GB RAM available
```

## Repo map
- `src/twinloop/` — pipeline library (v1 + v2 modules). v3 adds `radius_smooth.py`, `halfmax.py`
  and `paths.v3_dir()`; v3 adds two `V2Options` fields (`radius_smooth_mm`, `halfmax`).
- `scripts/` — v3 scripts are named `v3_*.py` / `eval_v3.py` (list in SPEC_V3 §12).
- `train/` — v2 training scripts (read and run; do not edit). `tests/` — pytest.
- `reports/` — committed results (Markdown, JSON, CSV < 2 MB). Large CSVs go to `/data/v3/`.
- `docs/` — SPEC_V3, TASKS_V3, DECISIONS_V3, PROGRESS_V3 (yours); SPEC_V2 and older files are
  background only, never edited.
- Data: `/data/nnUNet_raw/Dataset501_ImageCAS/` (images, GT labels), `/data/v2/` (v2 outputs,
  first-pass masks in `/data/v2/firstpass/`), `/data/models/v2/` (deployed models), `/data/v3/`
  (everything v3 writes).

## Hard rules
- Never read or use ImageCAS test-split cases (`case_id % 5 == 0`); `paths.case_image_path` raises
  on them. Never tune any parameter, threshold, model, epoch count or arm choice on `v2_holdout`
  or `v2_realfail` cases. Phase A uses `v2_train` cases only (plus the 50 already-seen cases in
  V06, as SPEC_V3 §6.4 states).
- H4 is evaluated exactly ONCE (task V13). Never rerun it, never change code, settings or the
  case set after seeing any H4 number. A failed row is recorded, never dropped or retried silently.
- Report every result honestly, including falsified hypotheses, failed gates, killed runs and
  crashes. Never describe FFR as accurate or validated; never write "diagnosis".
- Never delete, overwrite or modify anything under `/data/raw/`, `/data/nnUNet_raw/Dataset501_*`,
  `/data/nnUNet_preprocessed/Dataset501_*`, `/data/nnUNet_results/`, `/data/v2/`, `/data/models/v2/*`
  existing folders. Adding new folders (Dataset512, new model folders) is allowed where a task says so.
- Never stop, restart or `docker rm` a container you did not start in v3 (for example
  `twinloop-v2-xfit`, `twinloop-train`, `twinloop-server`). Never run two TotalSegmentator jobs at
  once (v3 does not need TotalSegmentator: always `anatomy=False`).
- One GPU job at a time. Do not start a GPU job while another container is using the GPU
  (`nvidia-smi` shows a twinloop process): wait and poll every 30 min (`sleep 1800`).
- Total GPU budget 96 h for v3. Log each GPU job's wall hours in PROGRESS_V3. Stop launching jobs if
  the next one would exceed the budget, and write BLOCKED.
- Never change v1 `result.json` fields, API routes, status names or error strings. New `V2Options`
  fields default to "off", so every existing output stays identical.
- No new dependencies. Never commit data, weights, `.nii*`, `.npz`, `.npy`, `.parquet`, `.pth`, `.zip`.
- Never print, read, copy or commit secrets (`~/.kaggle/access_token`, NGC key, TotalSegmentator
  licence, `/data/totalseg/config.json`).
- Never `git push`. Write `WAITING ON OWNER: git push origin v3` in PROGRESS_V3 and continue.
- Commit locally after each finished task: `git commit -m "V<nn>: <title>"` on branch `v3`.
- Phase C: research only. No downloads, no account creation, no emails.

## Working protocol
- Work one task at a time from docs/TASKS.md, in order.
- Before you start a task, re-read its acceptance criteria. After it's done, run its verification command and paste the result into docs/PROGRESS.md.
- If something is ambiguous, choose the option marked DEFAULT in the spec. If there isn't one, choose the simplest reversible option, log it in docs/DECISIONS.md, and keep going. Don't stop to ask unless the action is destructive or irreversible.
- If the same error survives 3 different fix attempts, stop. Write the error, what you tried, and your best hypothesis into docs/PROGRESS.md under BLOCKED, then move to the next independent task.
- Never mark a task done without passing verification.

For v3, read that protocol with these substitutions: "docs/TASKS.md" → docs/TASKS_V3.md,
"docs/PROGRESS.md" → docs/PROGRESS_V3.md, "docs/DECISIONS.md" → docs/DECISIONS_V3.md
(the unsuffixed files belong to v1 and are never edited).
A task whose detached job is still running stays under "In progress"; meanwhile continue with the
next task whose dependencies are met. Gate tasks (V05, V06) may mark later tasks
`[x] skipped (G<n> failed)`: that is the only allowed way to skip a task besides the conditions
written inside a task.

## Session start checklist
1. `uname -m` must print `aarch64`; `git branch --show-current` must print `v3`.
2. Read docs/PROGRESS_V3.md ("In progress", "BLOCKED", "WAITING ON OWNER", "Notes for next session").
3. `docker ps -a --filter name=twinloop` and check every container listed under "In progress".
4. Continue with the first unfinished task in docs/TASKS_V3.md whose dependencies are met.
