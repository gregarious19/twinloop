#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import torch, SimpleITK, skimage, scipy, networkx, fastapi, nnunetv2, pydantic
assert torch.cuda.is_available(), "CUDA not available"
x = torch.randn(1, 1, 64, 64, 64, device="cuda")
conv = torch.nn.Conv3d(1, 8, 3, padding=1).cuda()
y = conv(x); torch.cuda.synchronize()
print("torch", torch.__version__, "device", torch.cuda.get_device_name(0), "out", tuple(y.shape))
PY
which nnUNetv2_train nnUNetv2_predict nnUNetv2_plan_and_preprocess > /dev/null
echo "ENV OK"
