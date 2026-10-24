#!/usr/bin/env bash
# usage: scripts/train.sh <TrainerClass> [--c]
# Starts nnU-Net v2 fold-0 training detached as `twinloop-train`. --c resumes from the latest checkpoint.
set -euo pipefail
if [ -n "$(docker ps -q --filter name=twinloop-train)" ]; then
  echo "training already running"; exit 1
fi
cd "$(dirname "$0")/.."
./scripts/dkd twinloop-train nnUNetv2_train 501 3d_fullres 0 -tr "$1" ${2:-}
