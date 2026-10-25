# Decisions — CLUSTER

Format: `D-C-<n> — <decision>. Reason: <one line>.` The planner's decisions for the whole project
are in docs/DECISIONS_V2.md (read-only for CLUSTER). Append CLUSTER decisions below, prefixed with
the task ID.

## Pre-filled
- D-C-1 — Python environment = venv at `$TWINLOOP_DATA/venv` created with `uv` at SPARK's Python minor version; no Docker/Apptainer. Reason: simplest on a Slurm cluster; no SPARK-owned file (.gitignore) needs changing.
- D-C-2 — torch 2.5.1 from the PyTorch CUDA wheel index matching the driver (cu124/cu121/cu118). Reason: a known-good, widely available version; the nets use only basic layers.
- D-C-3 — Library versions copied from SPARK's requirements.lock; scikit-learn must match exactly. Reason: the decider is a joblib pickle loaded on SPARK.
- D-C-4 — All sbatch partition/account/QOS/GPU flags live in `cluster/slurm_flags_{gpu,cpu}.txt`, read by `cluster/submit.sh`. Reason: one place to change if the cluster setup changes.
- D-C-5 — Dataset generation as 20-shard CPU array jobs, resumable, with per-shard error logs. Reason: ~640 cases; failures of single cases must not stop the run.
- D-C-6 — Training scripts checkpoint every 2000 steps and resume automatically. Reason: wall-time limits and preemption.

## Filled in during execution
<!-- CLUSTER appends here: D-C-7 onward -->
