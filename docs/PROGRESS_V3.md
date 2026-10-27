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
- [V02] 2026-10-27 — halfmax.auto_mask (+tests), v3_lesion_audit.py, v3_audit_report.py; slice audit of the 3 lowest v2_train ids (2, 3, 7), σ_s 0 and 1.0 — verification:
```
case 2: 2 rows, 12s ; case 3: 6 rows, 12s ; case 7: 6 rows, 12s   (all status=ok, header = §6.2)
lesion count at σ_s 0: sum(detect_lesions over GT branches) = 8 ; CSV σ_s=0 ok rows = 8
re-run after ruff reformat: output CSV identical
slice.md: median iou_ref over all D0 lesions 0.358; halfmax_usable: False (3 cases only, not a verdict)
pytest -q: 156 passed, 2 deselected (154 + 2 halfmax) ; ruff check / format: clean
```
Surprising: image_ds is 0 for most lesions (half-max area at the dip is not smaller than the flank), iou_ref ≈ 0.3–0.45. Both are consistent with the v2 findings; the full audit (V04) decides. Decisions D-V3-V02a–d logged (d = eval_v3.py name clash).
- [V03] 2026-10-27 — A2 alignment diagnosis (scripts/v3_a2_alignment.py) — verification:
```
8131 healthy slices from 200 files
shape mismatch (identity 0.398, best 0.403 < 0.6)
{"identity_median_iou": 0.3977, "best_transform": "identity", "best_shift": [0, -1],
 "best_median_iou": 0.4030, "fresh_identity_median_iou": 0.4036, "dataset_aligned": false}
clean 0.397 / jit 0.399 ; fresh CPR of GT (20 branches, 5 dev cases, 755 slices) 0.404
```
Verdict: NOT a misalignment (no transform/shift helps; the fresh straightening of the GT gives the same IoU): half-max and label disagree in shape. dataset_aligned=false, so arm 3 (V11) will be skipped per §9.2.
<!-- - [V00] YYYY-MM-DD — <one line> — verification:
```
<output>
```
-->

## In progress
V04: container twinloop-v3-audit (CPU, 8 workers), started 2026-10-27 09:04 UTC, check `sg docker -c "docker logs --tail 5 twinloop-v3-audit"`; limit 4 h; output /data/v3/audit/lesions.csv (resumable).

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
