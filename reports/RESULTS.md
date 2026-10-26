# TwinLoop v1 results
Date: 2026-10-26. Hardware: DGX Spark (GB10). Container: nvcr.io/nvidia/pytorch:26.09-py3. Code commit under test: 115ace2.

## Segmentation (ImageCAS test split, n = 200)
Trainer: nnUNetTrainer (1000 epochs), fold 0, 1 seed (deviation: compute budget). TTA disabled.
| Method | Dice mean ± SD | Dice median | clSens mean ± SD | Infer s median (p90) |
|---|---|---|---|---|
| nnU-Net 3d_fullres | 0.8078 ± 0.0555 | 0.8104 | 0.9114 ± 0.0536 | 26.9 (n/a, amortised batch) |
| Threshold baseline | 0.0077 ± 0.0041 | 0.0072 | 0.4428 ± 0.1159 | 1.0 (1.5) |
Hypothesis (Dice ≥ 0.75): SUPPORTED (mean test Dice 0.8078; 31 of 200 cases are below 0.75, worst 0.622 for case 200).

Notes:
- nnU-Net `infer_s` is one batch `nnUNetv2_predict` over all 200 images divided by 200, so it includes start-up and shares the GPU with another job on this machine. A per-scan time comes from the pipeline evaluation (T11), not yet run.
- Training-time figure for reference only (not a test result): best EMA pseudo-Dice 0.8422 on fold-0 held-out training cases. The post-training validation pass was skipped (DECISIONS D42).
- The threshold baseline (HU 250–700, 2 largest components) captures the aorta and heart chambers, so its Dice is near zero as expected. Its clSens only shows it covers part of the vessels.

## Pipeline consistency (first 20 test cases)
Not yet run (T11).

## Failures and warnings
- None in the segmentation evaluation: 200 of 200 test cases scored for both methods.

## Caveats
FFR values are unvalidated (no invasive FFR ground truth). Single training run. Single-centre, single-scanner training data.
