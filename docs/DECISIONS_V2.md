# Decisions — v2 (SPARK)

Format: `D-V2-<n> — <decision>. Reason: <one line>.` Pre-filled by the planner. SPARK appends new
ones at the end (prefix with the task ID). CLUSTER writes to docs/DECISIONS_CLUSTER.md instead.

## Scope and machines
- D-V2-1 — v2 = TotalSegmentator anatomy, coverage check, damage generator, decider + gap-filler repair loop, CPR lumen refinement, myocardial-mass flow, evaluation harness. No plaque classes. Reason: no plaque labels exist (ImageCAS is lumen-only).
- D-V2-2 — Two machines: SPARK writes all library code and runs inference/pipeline/app/evaluation; CLUSTER (Slurm, 80 GB GPUs) generates training data and trains. Reason: the owner wants training in one place and the pipeline on the Spark; the Spark GPU is busy with first-pass training.
- D-V2-3 — Strict file ownership per machine (SPEC_V2 §1), separate progress and decision files. Reason: two Claude sessions commit to one repo; disjoint files make every owner `git pull` conflict-free.
- D-V2-4 — The owner does all git pushes and data rsyncs; sessions write `WAITING ON OWNER` lines. Reason: no cross-machine credentials for Claude; keeps data movement visible.
- D-V2-5 — v2 lives on branch `v2` in a separate git worktree folder with its own CLAUDE.md and Docker tag `twinloop:v2`; the existing v1 session keeps the original folder (`main`, `twinloop:dev`) and finishes v1. v1 is merged into `v2` once (S12M) after v1 T09. Reason: owner's instruction — the running v1 agent must not see v2 instructions; Track A needs only v1 T02–T04 code.

## Data
- D-V2-6 — v2 roles from nnU-Net fold 0: its validation cases split by parity into `v2_holdout` (evaluation) and `v2_realfail` (real first-pass errors for decider fine-tuning); all other train cases are `v2_train`. Reason: these are the only cases with honest first-pass predictions that the segmentation model never trained on; the test split stays untouched.
- D-V2-7 — First-pass masks for evaluation come from nnU-Net's own fold-0 validation predictions. Reason: same model, no extra inference cost. Caveat: nnU-Net's validation may use mirroring TTA while the app disables TTA; logged as a known difference.
- D-V2-8 — MM-WHS and ASOCA are not used (not downloaded); MM-WHS check is an optional task. Reason: owner has only ImageCAS.
- D-V2-9 — Regression set (20 `v2_holdout` cases: 10 highest calcium, 5 most lesions, 5 lowest ids) fixed once in S05 from GT and images only. Reason: hard cases over-represented; chosen before any v2 model exists.

## Anatomy
- D-V2-10 — TotalSegmentator used directly (no own aorta/myocardium model yet). Myocardium + aorta from `heartchambers_highres` (academic licence, non-commercial); fallback: aorta from the free `total` task, flow falls back to the v1 diameter law. Reason: owner choice; train an own model later before commercial use.
- D-V2-11 — `TOTALSEG_HOME_DIR=/data/totalseg`. Reason: containers run with `HOME=/tmp`; weights and licence must persist.
- D-V2-12 — TotalSegmentator is installed without changing the pinned nnunetv2 (no-deps fallback if needed). Reason: the first-pass model is being trained with that nnunetv2 version.
- D-V2-13 — Coarse branch labels only (LM, LAD, LCX, RCA, other) from tree side (x relative to the aorta) and the LV long axis (PCA of the myocardium). No dominance, no diagonal/marginal names. Reason: ImageCAS labels are binary; robust geometric rules exist only for the main vessels.

## Repair loop
- D-V2-14 — `build_graph` keeps up to 8 components (2 main trees + fragments). Reason: a gap splits a vessel; v1 would drop the distal piece silently.
- D-V2-15 — Actions: keep, extend, add, remove, restore, stop. `stop` (likely occlusion) is never filled, only flagged. Reason: filling a true occlusion would hide the most important finding.
- D-V2-16 — Decider = scikit-learn HistGradientBoostingClassifier on 31 hand-made geometry + intensity features, no anatomy features. Reason: fast on CPU, little data, inspectable; anatomy-free means datasets can be made on CLUSTER without TotalSegmentator and the decider works when anatomy fails. An LLM can be added later.
- D-V2-17 — Gap-filler = small 3D U-Net conditioned by cropping (64³ patch, 0.4 mm) + a seed-heatmap channel; may return "nothing found" (< 20 new voxels). Reason: simpler than a general promptable model; location is the only prompt needed.
- D-V2-18 — Training data from deliberately damaged GT masks (gap, truncate, delete branch, shrink, false branch, occlusion with intensity edit), then real failures from `v2_realfail`. Reason: no human correction labels exist; the occlusion and shrink types teach the model to respect true narrowing.
- D-V2-19 — Max 3 repair iterations; decider confidence < 0.6 → flagged for review, no change. Reason: bounded runtime; uncertain cases go to a human.
- D-V2-20 — Repair and lumen refinement are off by default until S16 shows they beat v1 on the regression set. Reason: never ship a change that is not measured to help.

## Lumen and flow
- D-V2-21 — Lumen model = per-slice 2D U-Net encoder/decoder + transformer across ≤ 128 straightened cross-sections (48×48 px, 0.2 mm, every 0.5 mm along the centerline). Reason: compares a narrowing with the segments before and after it, as a reader does on CPR.
- D-V2-22 — Rotation-minimising frame (double reflection) for CPR. Reason: no twisting of cross-sections along curved vessels.
- D-V2-23 — Myocardial-mass flow: q_rest = 1.0 mL/min/g × (myocardium mL × 1.05 g/mL), split between trees by Σ outlet r³; `auto` mode falls back to the v1 diameter law without a valid myocardium. The 1.0 value is verified in S09. Reason: patient-specific flow is the main point of segmenting the myocardium.
- D-V2-24 — FFR uncertainty flag: re-solve with lesion radii ×0.9 and ×1.1; flag if the tree's min FFR crosses 0.80. Reason: segmentation error near the cut-off flips decisions; the report must say so.
- D-V2-25 — No 3D CFD, no VMTK in v2. Reason: unreliable ARM builds; the 1D solver stays.

## Training (details in SPEC_V2 §14)
- D-V2-26 — 3 seeds per network on CLUSTER, deploy the best by validation metric (validation = `v2_train` cases with id % 10 == 3); only the deployed seed is evaluated end to end. Reason: protocol needs ≥ 3 seeds; end-to-end evaluation of all seeds costs too much.
- D-V2-27 — Gap-filler validation metric = Dice of the new voxels only. Reason: copying the input mask already gives a high full-patch Dice.
- D-V2-28 — Models exchanged as files under `models/v2/<name>/` with `config.json` (incl. sklearn/torch versions). Reason: SPARK loads exactly what CLUSTER trained; version mismatches are detectable.

## Filled in during execution
<!-- SPARK appends here: D-V2-S00a onward -->
- D-V2-S00a — Container Python 3.12.3 (base nvcr.io/nvidia/pytorch:26.09-py3, torch 2.14.0a0+b2c75dd062.nv26.09, nnunetv2 2.8.1 unchanged).
- D-V2-S00b — TotalSegmentator installed directly via requirements.txt (pip resolved without touching torch/nnunetv2; no-deps fallback not needed). Pins: scikit-learn==1.9.1, pyarrow==25.0.1, joblib==1.6.0, TotalSegmentator==2.18.0. Reason: S00 step 4–6. Side effect: certifi bumped 2026.7.22 → 2026.10.23 (harmless). `docker` needs `sg docker -c "..."` in this shell (user is in the docker group, session predates it).
- D-V2-S00c — `torch==` is not printed by `pip freeze` (installed from a local wheel, appears as `torch @ file:///...`); the "unchanged" check compares `^(torch|nnunetv2)(==| @)` lines instead. Reason: spec grep pattern would match only nnunetv2.
- D-V2-S01a — `heartchambers_highres` output files (resolves SPEC §5.1 UNCERTAIN): aorta, heart_atrium_left, heart_atrium_right, heart_myocardium, heart_ventricle_left, heart_ventricle_right, pulmonary_artery. So myocardium = `heart_myocardium`, LV cavity = `heart_ventricle_left` (spec said `myocardium`/`ventricle_left`). `total --roi_subset aorta heart` gives aorta, heart. Reason: observed on ImageCAS_0002.
- D-V2-S01b — Never run two TotalSegmentator invocations concurrently while weights may be downloading: they share `tmp_download_file.zip` and corrupt each other. Reason: happened once in S01.
- D-V2-S01c — SPEC ISSUE: the `aorta_connected` rule (largest 26-component ≥ 95 %) fails on real case ImageCAS_0002: TotalSegmentator aorta (highres and total) has 2 big components (57 %/43 % highres, 61 %/39 % total) — likely ascending and descending aorta separated because the cardiac FOV cuts the arch. Rule kept as specified (no tuning); S01 GPU test asserts only aorta_present and myo_present and prints qc. Proposed fix for the owner: count a split into ≤2 large pieces as connected, or drop the key. Check on more cases in S04.
