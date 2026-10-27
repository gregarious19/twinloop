# Decisions — v3 (SPARK)

Format: `D-V3-<n> — <decision>. Reason: <one line>.` Pre-filled by the planner (owner-approved
defaults, 2026-10-27). Append new ones at the end as `D-V3-<task id><letter>` (e.g. `D-V3-V02a`).

## Scope and process
- D-V3-01 — v3 = audit the stenosis reference first (Phase A), then a pre-registered H4 (Phase B) only if Gates G1 and G2 pass; otherwise Phase C. Reason: GT σ 0.5 mm blur under-calls modplus lesions about as badly as the segmenter (−47.7 vs −51.2 pp), so the reference may not measure what H4 needs.
- D-V3-02 — SPARK only, 96 GPU-hours total, one GPU job at a time, wait for foreign containers (e.g. twinloop-v2-xfit) instead of sharing the GPU. Reason: owner default; known TotalSegmentator/GPU contention issues.
- D-V3-03 — No new dependencies, image `twinloop:v2` reused, no rebuild. Reason: owner default; everything needed is installed.
- D-V3-04 — Area-equivalent radius is not a candidate definition. Reason: S20/S22 showed it inflates mild lesion counts (2.3–3×) and removes most severe lesions; the junction fix failed its criterion.
- D-V3-05 — Candidate robust definition = Gaussian smoothing of the inscribed radius profile along arc length, σ_s ∈ {0, 0.5, 1.0, 1.5, 2.0} mm; pick the smallest that keeps ≥ 80 % of modplus lesions under a σ 0.5 mm GT blur and ≥ 40 % of D0's modplus count. Reason: one-voxel dips vanish while ≥ 3 mm narrowings survive (SPEC §4.2 worked examples); smallest σ = smallest change to the reference.
- D-V3-06 — The same σ_s smooths radii before the solver for every row, so FFR reference and predictions use one ruler. Reason: case-level FFR < 0.80 from GT is itself driven by the dips.
- D-V3-07 — Selection on v2_train "sel" cases (id % 10 != 3); "dev" cases (id % 10 == 3) check only. Reason: no tuning on holdout; a held-back check inside v2_train.
- D-V3-08 — Image support threshold image_ds ≥ 0.25 on non-calcified lesions; half-max method usable if median healthy-flank IoU(half-max, label) ≥ 0.60; calcified = HU ≥ 800 within 2 mm. Reason: 0.25 = v1 lesion threshold; 800 HU = v2 regression-set calcium rule; 0.60 IoU = clearly above the 0.33 implied by the S25 overlap.
- D-V3-09 — Gate G2: σ 0.5 mm blurred GT must reach sensitivity ≥ 0.50 on the 50 seen cases. Reason: if a near-perfect smooth mask cannot, no segmenter can pass H4; uses spent cases only.
- D-V3-10 — H4 set = v2_holdout ∪ v2_realfail minus the 50 seen cases (expected 110), no enrichment. Reason: largest unspent pool; the decider (realfail's only prior use) is not part of v3.
- D-V3-11 — H4 bar: sens gain ≥ +0.25 and sens ≥ 0.40; guards FPR +0.10, modplus ds_err_pp +10 pp, branch recall −0.02, down50 not higher; ≥ 2 of 3 replicates; < 10 positives → INCONCLUSIVE. Reason: owner defaults, plus SPEC_V2 §13b standing guard.
- D-V3-12 — No pipeline default changes in v3, whatever H4 says. Reason: GT-mask reference is not clinical truth; a default change needs owner review of V3_RESULTS.md.
- D-V3-13 — Re-annotated LumenNet (old S26, now S28) cancelled in v2; revived as arm 3 only if the A2 dataset IoU check passes. Reason: healthy-slice AND area is half the label area, which suggests misalignment.

## Arms
- D-V3-14 — Arm 1 = nnU-Net fine-tune from the deployed checkpoint with a second "lesion" label class (zone = r_ref + 1 mm around lesion runs ± 2 mm), nnUNetTrainer_250epochs, 3 replicates, checkpoint_final. Reason: lesion-centred oversampling and a lesion Dice term without patching nnU-Net; 250 epochs ≈ 7.3 h fits the budget.
- D-V3-15 — Arm 1 split: train = v2_train id % 10 != 3, val = dev; holdout/realfail/test never in Dataset512. Reason: nnU-Net logs validation Dice; it must never see H4 cases.
- D-V3-16 — Arm 2 = half-max narrowing min(r_v1, r_halfmax) on non-calcified slices, only if the half-max method is usable. Reason: simplest non-learned fix; radii only decrease.
- D-V3-17 — Arm 3 uses the S23 recipe and frozen inference settings (ins, interp, σ 0). Reason: no new tuning; isolates the effect of the labels.
- D-V3-18 — xfit predictions are not used in v3. Reason: weaker 100-epoch masks; dev smoke uses blurred-GT masks instead, which always exist.

## Repo
- D-V3-19 — Branch v3 in worktree twinloop-v3; docs suffixed _V3; task IDs V00–V14; commit prefix "V<nn>:". Reason: avoids the S25/S26 ID clash of v2.
- D-V3-20 — New outputs under /data/v3/ and Dataset512 folders only; large CSVs in /data/v3/, small ones in reports/. Reason: never modify v2 data; keep the repo light.

## Filled in during execution
<!-- append D-V3-V<nn><letter> entries below -->
- D-V3-V02a — §16 row 1 resolved: `cpr.straighten(image, mask, branch, cp)` accepts a mask (`straighten_points` resamples it with order 0) and returns it under key `"mask"` (uint8, >0), next to `"img"`, `"s_mm"`, `"points_mm"`. Reason: read src/twinloop/cpr.py.
- D-V3-V02b — `halfmax.auto_mask` thresholds with strict `sm > thr` instead of the old `>=`. Reason: SPEC §11 requires a blank slice to give an all-False (48,48) array; with `>=` a constant slice has thr == value and selects the whole slice. Differs from the old rule only on exact ties.
- D-V3-V02c — Audit CSV: a case/σ_s with no GT lesion writes one row with `status=nolesions` (other columns empty); `report` counts cases from non-failed rows. Reason: lesions/case denominators, resumability and the "640 distinct case_ids" check need zero-lesion cases to appear. If a blurred-GT graph cannot be built (no tree), its lesions count as missed / not surviving.
- D-V3-V02d — SPEC ISSUE: `scripts/eval_v3.py` already exists on this branch (v2 S24, H3 lumen evaluation, commit 196258f; also on branch v2). SPEC_V3 §10 wants a new script of the same name. Proposed fix (applied in V06): replace it with the v3 script; the v2 version stays in git history (196258f) and on branch v2. Not touched in V02.
- D-V3-V05 — Gate G1 FAILED. On the 555 sel cases the modplus lesion count is 777 / 634 / 284 / 85 / 11 for σ_s 0 / 0.5 / 1 / 1.5 / 2 mm, and retention under a σ_b 0.5 mm blur of the GT is 0.048 / 0.038 / 0.025 / 0.024 / 0.000 (needs >= 0.80 with count >= 311). No definition qualifies, so σ* is null (reports/v3_definition.json, v3_A3_definition.md). dev cases agree (retention 0.019 at D0, 0 from D2). Image support is also near zero (5 % of non-calcified D0 modplus lesions have half-max image_ds >= 0.25), but half-max was not usable (median iou_ref 0.357) and dataset_aligned is false. V06–V13 skipped per the spec; H4 was never run. Note: "survive" is strict (the blurred graph must still show ds >= 0.5 at the lesion), and smoothing the profile removes real lesions faster than it makes them blur-robust, so this family of definitions cannot fix the ruler.
