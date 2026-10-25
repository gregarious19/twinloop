# TwinLoop v2 — CLUSTER tasks (Slurm, 80 GB GPUs)

You are on the CLUSTER (`uname -m` = `x86_64`, `command -v sbatch` succeeds). This machine only
builds a Python environment, writes the network code (`src/twinloop/nets.py`), generates training
data with the library code written on SPARK, trains the three v2 models, and exports them.
Read docs/SPEC_V2.md §1, §2, §8, §14 before starting. `SPEC §x` = docs/SPEC_V2.md.

Rules specific to this machine (in addition to CLAUDE.md):
- Edit only CLUSTER-owned files (SPEC §1 table): `src/twinloop/nets.py`, `train/**`, `cluster/**`,
  `tests/test_nets.py`, `requirements-cluster.txt`, `docs/PROGRESS_CLUSTER.md`,
  `docs/DECISIONS_CLUSTER.md`, `reports/v2_training.md`.
- Never run GPU work or anything heavier than a 2-minute test on the login node. Use `sbatch`.
- Never use more than the resources in SPEC §14 per job. Never submit more than 3 GPU jobs at once.
- Never read, copy or create test-split cases (`case_id % 5 == 0`); `v2_holdout` cases are never
  used here at all; `v2_realfail` only as `decider_real` rows in C08.
- Log every verification output in `docs/PROGRESS_CLUSTER.md`. Before each commit run
  `ruff check train src/twinloop/nets.py tests/test_nets.py && ruff format train src/twinloop/nets.py tests/test_nets.py`
  (SPARK lints the whole repo). Commit after each task: `git add -A && git commit -m "C<nn>: <title>"`.
  Never `git push` (the owner pushes). Never commit data, logs or weights.
- Long jobs outlive your session. After submitting, write the job IDs and how to check them under
  "In progress", then continue with other tasks or end the session.

Slurm job checks: `squeue -u $USER`; `sacct -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode -X`;
logs in `$TWINLOOP_DATA/v2/slurm/`.

---

## [x] C00 — Read the cluster notes, build the environment
**Depends on:** SPARK S00 pulled (needs the pinned `requirements.txt`/`requirements.lock` and `config_v2.py`)
**Files:** `cluster/SLURM_NOTES.md` (owner-filled; read only, except the "Extracted" section), `cluster/cluster.env`, `cluster/slurm_flags_gpu.txt`, `cluster/slurm_flags_cpu.txt`, `cluster/submit.sh`, `cluster/jobs/env_check.sh`, `requirements-cluster.txt`, `cluster/make_requirements.py`, `docs/PROGRESS_CLUSTER.md`, `docs/DECISIONS_CLUSTER.md`
**Steps:**
1. Read `cluster/SLURM_NOTES.md`. It must contain the owner's cluster instructions and the filled "Required facts" table (partition, account, QOS, GPU request syntax, max wall time, module commands, scratch/data path, internet on login node yes/no). If any required fact is empty and cannot be found in the pasted instructions, write `WAITING ON OWNER: fill <fact> in cluster/SLURM_NOTES.md` and stop this session.
2. Write the "Extracted" section at the end of SLURM_NOTES.md: one line per fact with the value you will use.
3. `cluster/slurm_flags_gpu.txt` — one line of sbatch flags, e.g. `--partition=<P> --account=<A> --qos=<Q> --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=<min(24:00:00, max wall)>` (omit account/qos if the notes say they are not used; use the notes' GPU syntax if it differs from `--gres=gpu:1`). `cluster/slurm_flags_cpu.txt` — same without the GPU flag, `--cpus-per-task=8 --mem=32G --time=04:00:00`.
4. `cluster/submit.sh`:
   ```bash
   #!/usr/bin/env bash
   # usage: cluster/submit.sh gpu|cpu [extra sbatch flags...] <job script> [args...]
   set -euo pipefail
   KIND="$1"; shift
   REPO="$(cd "$(dirname "$0")/.." && pwd)"
   source "$REPO/cluster/cluster.env"
   mkdir -p "$TWINLOOP_DATA/v2/slurm"
   # shellcheck disable=SC2046
   sbatch $(cat "$REPO/cluster/slurm_flags_${KIND}.txt") \
     --output="$TWINLOOP_DATA/v2/slurm/%x_%A_%a.out" "$@"
   ```
5. `cluster/cluster.env` (no secrets; the venv lives outside the repo so no SPARK-owned `.gitignore` change is needed):
   ```bash
   export REPO=<absolute path of this repo>
   export TWINLOOP_DATA=<data path from the notes>/twinloop-data
   export VENV=$TWINLOOP_DATA/venv
   export PYTHONPATH=$REPO/src:$REPO
   <module load lines from the notes, if any>
   ```
6. Python: the venv must use the same Python minor version as the SPARK container (see `D-V2-S00a` in docs/DECISIONS_V2.md). DEFAULT tool `uv`: `python3 -m pip install --user uv` then `~/.local/bin/uv venv --python <x.y> $VENV`. If uv or internet is unavailable, use the conda/module Python from the notes with the same minor version; if none exists, use the closest available minor and log it.
7. `cluster/make_requirements.py`: read `requirements.lock` (from SPARK) and write `requirements-cluster.txt` with exact pins (`name==version` from the lock) for: `numpy scipy SimpleITK scikit-image networkx pandas pyarrow scikit-learn joblib pydantic pytest ruff`. Do not include torch, nnunetv2 or TotalSegmentator. If the lock spells a name differently (e.g. `simpleitk`), match case-insensitively.
8. Torch: submit a GPU check of the driver first: `cluster/submit.sh gpu --job-name=drv --wrap="nvidia-smi"`; read the CUDA version in the header. Choose the wheel index: CUDA ≥ 12.4 → `cu124`; ≥ 12.1 → `cu121`; ≥ 11.8 → `cu118`; else BLOCKED. Install: `$VENV/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/<cuXXX>` then `$VENV/bin/python -m pip install -r requirements-cluster.txt`. If a pinned version has no wheel for this Python, use the nearest available version **except scikit-learn, which must match exactly** (the decider is pickled on CLUSTER and loaded on SPARK) — if scikit-learn cannot match, BLOCKED. Log every deviation in DECISIONS_CLUSTER.md.
9. `cluster/jobs/env_check.sh`:
   ```bash
   #!/usr/bin/env bash
   #SBATCH --job-name=env_check
   set -euo pipefail
   source "$REPO/cluster/cluster.env"; source "$VENV/bin/activate"; cd "$REPO"
   python - <<'EOF'
   import torch, SimpleITK, skimage, scipy, networkx, sklearn, pyarrow, pandas
   assert torch.cuda.is_available()
   x = torch.randn(2, 3, 64, 64, 64, device="cuda"); c = torch.nn.Conv3d(3, 16, 3, padding=1).cuda()
   print("torch", torch.__version__, torch.cuda.get_device_name(0), tuple(c(x).shape), "sklearn", sklearn.__version__)
   import twinloop.config_v2, twinloop.centerline
   print("ENV OK")
   EOF
   ```
   Note: `$REPO` is not set before sourcing; `submit.sh` sources `cluster.env` and Slurm exports the environment by default (`--export=ALL`). If the notes say the cluster does not export the environment, put the absolute repo path in the script instead.
10. Submit `cluster/submit.sh gpu cluster/jobs/env_check.sh`; wait for it (`sacct`); paste its log.
11. Create `docs/PROGRESS_CLUSTER.md` and `docs/DECISIONS_CLUSTER.md` from the templates (if absent). Commit `C00: cluster environment`.

**Acceptance:** env_check log ends with `ENV OK`; sklearn version equals SPARK's pin.
**Verify:** `tail -3 $TWINLOOP_DATA/v2/slurm/env_check_*.out`

---

## [x] C01 — Network code and training utilities
**Depends on:** C00
**Files:** `src/twinloop/nets.py`, `train/__init__.py`, `train/common.py`, `tests/test_nets.py`
**Steps:**
1. `nets.py`: `GapfillNet` and `LumenNet` exactly as SPEC §8.1–8.2 (constructor `__init__(self, channels=(16, 32, 64, 128))`). `LumenNet.forward(x)` takes (B, T, 1, 48, 48) and an optional `pad_mask` (B, T) bool (True = padding), passed as `src_key_padding_mask` to the transformer.
2. `train/common.py`:
   - `set_seed(seed)` (python, numpy, torch, cuda).
   - `cosine_lr(step, total, warmup, base_lr)`: linear warmup to base_lr over `warmup` steps, then cosine to 0 at `total`.
   - `soft_dice_loss(logits, target, eps=1.0)`: `1 − (2Σ(pt)+eps)/(Σp+Σt+eps)` with p = sigmoid, summed over all non-batch dims, averaged over the batch.
   - `CsvLogger(path, columns)`: appends rows, writes the header once, flushes each row.
   - `save_ckpt(path, model, opt, step, best_metric)` / `load_ckpt(path, model, opt) -> (step, best_metric)`; write to `path.tmp` then `os.replace`.
   - `split_case_ids(ids) -> (train_ids, val_ids)`: val = `case_id % 10 == 3`.
   - `git_commit() -> str` (`git rev-parse --short HEAD`, "unknown" on error).
3. `tests/test_nets.py` (CPU, small): GapfillNet on (1,3,64,64,64) → (1,1,64,64,64); LumenNet on (2,16,1,48,48) → (2,16,1,48,48), with and without pad_mask; `soft_dice_loss` of a perfect prediction (logits ±20) < 1e-3; `cosine_lr` at step 0 = 0, at warmup = base_lr, at total = 0.
4. Commit `C01: nets and training utilities`.

**Acceptance / Verify:** `source cluster/cluster.env && $VENV/bin/python -m pytest -q tests/test_nets.py` → all passed (CPU test is short; allowed on the login node).

---

## [x] C02 — Data arrives on the cluster
**Depends on:** C00
**Steps:**
1. If `$TWINLOOP_DATA/nnUNet_raw/Dataset501_ImageCAS/imagesTr` is missing: `WAITING ON OWNER: rsync from SPEC §1 (SPARK → CLUSTER)`, and do C01 meanwhile.
2. Verify: count `imagesTr/*.nii.gz` equals count `labelsTr/*.nii.gz`; `imagesTs` does not exist on the cluster (if it does, BLOCKED: test data must not be here); `splits_final.json` exists.
3. Print role counts (as in SPARK S00 step 10, with `PYTHONPATH` set) and compare with SPARK's numbers in docs/PROGRESS_V2_SPARK.md; they must be identical. Note: `paths.case_image_path` raises for test ids — nothing here should ever ask for them.
4. Commit `C02: data verified` (progress file only).

**Acceptance:** counts equal; role counts identical to SPARK's.

---

## [ ] C03 — Training datasets (Slurm CPU arrays)
**Depends on:** C01, C02, SPARK S06, S07, S08 pulled (check: `git log --oneline | grep -cE " S0(6|7|8):"` = 3)
**Files:** `train/make_gapfill_data.py`, `train/make_lumen_data.py`, `train/make_decider_data.py`, `train/shards.py`, `cluster/jobs/make_data.sh`
**Steps:**
1. `train/shards.py`: `shard_cases(role="v2_train", shard=k, nshards=20) -> list[int]` = sorted `v2_train` ids with `index % nshards == k`.
2. The three generators follow SPEC §14 "Datasets" exactly. Common CLI: `--shard K --nshards 20 --out DIR`. Each case in a try/except: on error, append `case_id,error` to `DIR/errors_shard_{K:02d}.csv` and continue. When a shard finishes, write `DIR/_done_shard_{K:02d}`. Skip cases whose outputs already exist (resumable).
   - gapfill: `DIR/{case:04d}_s{seed}_{damage_id}.npz` (positives and occlusion negatives) and `DIR/{case:04d}_s{seed}_neg{j}.npz` (taper negatives); `np.savez_compressed` with keys `x` (float16), `y` (uint8), `case_id`, `type`.
   - lumen: `DIR/{case:04d}_{branch_id}_{clean|jit}.npz` with keys `img` (float16, normalised as SPEC §8.1), `mask`, `ds`, `case_id`, `jitter`.
   - decider: `DIR/shard_{K:02d}.parquet`.
   - RNG for damage: `np.random.default_rng(1000003 * seed + case_id)`; jitter rng: `np.random.default_rng(7 * case_id + 1)`.
3. `cluster/jobs/make_data.sh <which>` (array job): sources env, activates venv, runs `python -m train.make_<which>_data --shard $SLURM_ARRAY_TASK_ID --nshards 20 --out $TWINLOOP_DATA/v2/datasets/<which>`.
4. Smoke test first on the login node limited to 1 case: add `--limit 1` to each generator; run each for shard 0 with `--limit 1` and an output dir under `$TWINLOOP_DATA/v2/tmp/`; check files open and shapes are right. Then submit the three arrays: `cluster/submit.sh cpu --array=0-19 --job-name=data_<which> cluster/jobs/make_data.sh <which>` for `gapfill`, `lumen`, `decider`.
5. After all finish: count `_done_shard_*` = 20 per dataset; total errors; sample counts. Paste. If > 5 % of cases errored in a dataset, look at the 3 most frequent error messages, write them under Notes as a SPARK bug report (SPEC §1), and continue only if ≥ 90 % of cases produced data.
6. Commit `C03: training datasets`.

**Acceptance:** 20 done markers per dataset; ≥ 90 % of `v2_train` cases contributed to each dataset.
**Verify:** `ls $TWINLOOP_DATA/v2/datasets/{gapfill,lumen,decider}/_done_shard_* | wc -l` → 60.

---

## [ ] C04 — Train the gap-filler
**Depends on:** C03
**Files:** `train/train_gapfill.py`, `cluster/jobs/train_gapfill.sh`
**Steps:**
1. `train/train_gapfill.py --data DIR --out RUNDIR --seed K [--steps 40000] [--sanity]` per SPEC §14 recipe: dataset of npz files, train/val split by `split_case_ids`; `DataLoader(num_workers=12, pin_memory=True, persistent_workers=True)`; bf16 autocast; log every 100 steps and validate every 2000 steps to `RUNDIR/train_log.csv` (`step,loss,lr,val_metric,elapsed_s`; also `val_neg_newvox`); checkpoint `RUNDIR/ckpt.pt` every 2000 steps (resume automatically if it exists); save `RUNDIR/best.pt` (state_dict) when the val metric improves; at the end write `RUNDIR/metrics.json` `{"best_val_new_dice", "best_step", "final_step", "hours"}`. Apply the SPEC §14 kill criterion inside the script at step 10 000 (write `metrics.json` with `"killed": true` and exit 3).
   `--sanity`: 4 fixed positive training samples, no augmentation, lr 1e-3, 1000 steps; print init loss and final train new-voxel Dice; exit 0 if Dice > 0.8 else 2.
2. `cluster/jobs/train_gapfill.sh` (array index = seed): `python -m train.train_gapfill --data $TWINLOOP_DATA/v2/datasets/gapfill --out $TWINLOOP_DATA/v2/runs/gapfill/seed$SLURM_ARRAY_TASK_ID --seed $SLURM_ARRAY_TASK_ID "$@"`.
3. Sanity: `cluster/submit.sh gpu --array=0 --job-name=gf_sanity cluster/jobs/train_gapfill.sh --sanity`. Paste init loss and final Dice. Fail → BLOCKED for this model (do C05/C06).
4. Full: `cluster/submit.sh gpu --array=0-2 --job-name=gapfill cluster/jobs/train_gapfill.sh`. Record job ID under In progress. If a job hits its wall time, resubmit the same command (it resumes from `ckpt.pt`).
5. When done, paste each seed's `metrics.json`. Commit `C04: gap-filler trained`.

**Acceptance:** ≥ 1 seed finished without kill; best val new-voxel Dice recorded for all seeds.

---

## [ ] C05 — Train the lumen model
**Depends on:** C03
**Files:** `train/train_lumen.py`, `cluster/jobs/train_lumen.sh`
**Steps:** same structure as C04 with SPEC §14 LumenNet recipe:
1. A sample = one random window of up to 128 consecutive slices from one branch file (uniform window start); batch of 4 windows padded to the longest with zeros and a `pad_mask`; loss ignores padded slices; slices with `ds ≥ 0.25` weighted ×3 in BCE. Val: every val branch, full length, windows 128 / stride 96 with averaged logits (as SPEC §9.4); metric = mean over val slices with a non-empty GT component of |r_eq(center_component(pred > 0.5)) − r_eq(GT)| in mm (empty prediction → r_eq 0). `metrics.json`: `{"best_val_req_err_mm", "best_step", "final_step", "hours"}`. Kill at step 10 000 if error > 0.5 mm.
2. `--sanity`: 4 fixed windows, 1000 steps, lr 1e-3, train Dice > 0.8.
3. Sanity job, then `--array=0-2` full job. Paste metrics. Commit `C05: lumen model trained`.

**Acceptance:** ≥ 1 seed finished without kill; metrics for all seeds recorded.

---

## [ ] C06 — Train the decider (synthetic)
**Depends on:** C03
**Files:** `train/train_decider.py`, `cluster/jobs/train_decider.sh`
**Steps:**
1. `train/train_decider.py --data DIR --out RUNDIR --seed K`: concatenate `shard_*.parquet`; X = the 31 columns in `twinloop.regions.FEATURE_NAMES` order; y = `label` mapped by `ACTIONS`; split by `split_case_ids`; fit SPEC §14 model; report train and val macro-F1, per-class precision/recall/support on val (`sklearn.metrics.classification_report(output_dict=True)`), confusion matrix; save `RUNDIR/decider.joblib` (`joblib.dump`), `RUNDIR/metrics.json` `{"train_macro_f1", "val_macro_f1", "per_class": {...}}`. Sanity rule (SPEC §14) checked in the script: exit 2 if violated.
2. Run as a CPU job: `cluster/submit.sh cpu --array=0-2 --job-name=decider cluster/jobs/train_decider.sh`.
3. Paste val macro-F1 per seed and the per-class table for seed 0. Commit `C06: decider trained`.

**Acceptance:** 3 seeds finished; sanity passed.

---

## [ ] C07 — Export models and the training report
**Depends on:** C04, C05, C06 (export whichever finished; list missing ones as BLOCKED)
**Files:** `train/export_models.py`, `reports/v2_training.md`
**Steps:**
1. `train/export_models.py`: for each model pick the deployed seed — gapfill: max `best_val_new_dice`; lumen: min `best_val_req_err_mm`; decider: max `val_macro_f1` (ties → lowest seed). Killed runs are never deployed. Write `$TWINLOOP_DATA/models/v2/<name>/` per SPEC §8.3: `model.pt` (state_dict from `best.pt`) or `decider.joblib`; `config.json` with the fields of SPEC §8.3 plus `"sklearn_version"` (decider), `"torch_version"`, `"val_metric_name"`, `"val_metric"`, `"seed"`, `"git_commit"`; `metrics.json` copied from the run.
2. Verify loading: `python -c` that builds `GapfillNet()`/`LumenNet()`, `load_state_dict(strict=True)`, runs one forward pass on zeros on CPU; `joblib.load` the decider and `predict_proba` on one zero row of 31 features → shape (1, 6).
3. `reports/v2_training.md` from the SPEC §14 template: every seed of every model (including killed/failed, with reason), mean ± SD of the val metric over finished seeds, deployed seed, decider per-class table.
4. Under In progress: `WAITING ON OWNER: rsync -av <cluster>:<CLUSTER_DATA>/models/v2/ $TWINLOOP_DATA/models/v2/ (SPARK) and git push/pull so SPARK has nets.py`.
5. Commit `C07: models exported`.

**Acceptance:** three model folders with `config.json`; loading checks pass; report complete.

---

## [ ] C08 — Fine-tune the decider on real failures
**Depends on:** C06; SPARK S14 dataset synced to `$TWINLOOP_DATA/v2/datasets/decider_real/decider_real.parquet`
**Files:** `train/finetune_decider.py`
**Steps:**
1. Implement SPEC §14 "Decider fine-tuning" exactly (weights, grouped 5-fold CV over real cases, final fit seed 0).
2. Print per-fold macro-F1 (real rows of the held-out fold only), mean ± SD; per-class report pooled over folds.
3. Export: move the current `models/v2/decider/` to `models/v2/decider_synth_only/`; write the fine-tuned model to `models/v2/decider/` with `config.json` (`"trained_on": "v2_train synthetic + v2_realfail real"`, CV metrics).
4. Add a "Decider fine-tuning" section to `reports/v2_training.md` with the CV table. WAITING ON OWNER: rsync models again. Commit `C08: decider fine-tuned`.

**Acceptance:** CV metrics reported; new decider folder loads (`predict_proba` shape (1, 6)).
