# Progress

Update this file after every task. Newest entries at the top of each section.

## Done
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
Next: T01 (ImageCAS download via Kaggle, detached) then T03. T01 steps 5–7 need `twinloop.io` from T03.

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
