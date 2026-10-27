# Progress — v3 (SPARK)

Update at the end of every task and before ending a session. Paste verification output verbatim
(trim long logs to the relevant lines). GPU hours used: 0 of 96.

## Done
- [V00] 2026-10-27 — setup/state check passed — verification:
```
uname -m: aarch64 ; branch: v3 ; top commit: 347027b V00: v3 handoff docs
./scripts/dk head -1 CLAUDE.md -> # TwinLoop — CLAUDE.md (v3)
roles: Counter({'v2_train': 640, 'v2_holdout': 87, 'v2_realfail': 73}); dev (id%10==3) = 85
pytest: 146 passed, 2 deselected, 1 warning ; ruff check: All checks passed ; ruff format: 102 files already formatted
nnUNet_raw/preprocessed/results = /data/nnUNet_*  (nnUNet_compile=true, n_proc_DA=8)
/data/v2/firstpass: 160 ImageCAS_*.nii.gz masks (+3 nnU-Net metadata files, 163 entries total)
/data/v2/datasets/lumen: files like 0002_tree_0_b0_clean.npz / _jit.npz ; lumen_reannot: 19634 entries
nvidia-smi: GB10, memory.used N/A (unified memory); free -g: 92 GB available
```
- [V01] 2026-10-27 — radius_smooth.py (smooth_radius/trees/graph, blur_mask), paths.v3_dir, V2Options.radius_smooth_mm + call in run_pipeline, golden phantom JSON (tests/data/v3_golden_stenosis.json) — verification:
```
pytest tests/test_radius_smooth.py tests/test_pipeline_v3.py: 8 passed
pytest -q: 154 passed, 2 deselected (146 + 8 new)
ruff check: All checks passed ; ruff format --check: 105 files already formatted
```
Note: blur_mask follows SPEC §4.1 (same procedure as diag_v2b.blurred, which uses the identical sigma/threshold).
<!-- - [V00] YYYY-MM-DD — <one line> — verification:
```
<output>
```
-->

## In progress
<!-- task, container name, start time (UTC), how to check, expected end -->

## BLOCKED
<!-- - [Vxx] error: <exact message> | tried: 1) … 2) … 3) … | hypothesis: … | next independent task taken: Vyy -->

## WAITING ON OWNER
<!-- exact command or decision needed -->

## Notes for next session
Docker (`docker ps -a --filter name=twinloop`, 2026-10-27, needs `sg docker -c`):
- RUNNING: twinloop-v2-xfit, twinloop-v2-train-lumen-reannot (both foreign to v3: GPU tasks
  V10-V12 must wait for them; never stop them), twinloop-serve-v2-{api,worker,db}-1.
- Exited: twinloop-v2-evalv3-s0/s1/s2, twinloop-v2-server.
Host `ls /data` fails (data is container-path /data = $HOME/twinloop-data); use ./scripts/dk for paths.
<!-- observations, gate results, things the owner should look at -->
