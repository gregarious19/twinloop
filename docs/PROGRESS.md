# Progress

Update this file after every task. Newest entries at the top of each section.

## Done
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
T01: download running in container `twinloop-download` (started 2026-10-24 ~20:03 IST, dataset `xiaoweixumedicalai/imagecas`, 89.1 GB). Check: `docker logs --tail 5 twinloop-download` (done when it prints `DOWNLOAD DONE`, then `docker rm twinloop-download`). Next for T01: run `bash scripts/join_split_zips.sh` on the host (split archives, D32), then steps 5–7 (manifest) after T03 provides `twinloop.io`.
Download progress (20:08 IST): ~6 MB/s of 83 GB -> ETA about 4 h (about 00:00 IST). Tip: the log is one huge tqdm line, use `docker logs --tail 1 twinloop-download | tail -c 150`.\nNext task that does not need data: T04 (pipeline + CLI).

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
- Never run `ruff format .` on files other than code without `extend-exclude = ["docs"]` in place (it is now in pyproject.toml). Before that fix, ruff rewrote code blocks in docs/SPEC.md and docs/TASKS.md; they were restored to the original text (752 / 302 lines). There is no git baseline for the docs because the first commit came after that, so if the docs look odd, compare line counts.
- `docker build` takes ~3 min after the first time (layers cached). The base image is 25 GB.
- Kaggle auth is untested (T01 step 0).
