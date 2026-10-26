"""First-pass masks for v2 evaluation when fold-0 validation predictions are missing (SPEC_V2 §13).

Runs nnUNetv2_predict (v1 settings: fold 0, --disable_tta) on v2_holdout / v2_realfail cases into
<data>/v2/firstpass/ImageCAS_XXXX.nii.gz. Resumable. Order: regression set first, then the rest.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from twinloop.paths import case_image_path, v2_dir
from twinloop.v2split import load_roles

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    roles = load_roles()
    reg = json.loads((REPO / "reports/v2_regression_set.json").read_text())["case_ids"]
    rest = sorted(c for c, r in roles.items() if r in ("v2_holdout", "v2_realfail"))
    order = [*[int(c) for c in reg], *[c for c in rest if c not in set(reg)]]
    out = v2_dir() / "firstpass"
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in order if not (out / f"ImageCAS_{c:04d}.nii.gz").exists()]
    print(f"{len(todo)} of {len(order)} cases to predict", flush=True)
    trainer = os.environ.get("TWINLOOP_NNUNET_TRAINER", "nnUNetTrainer")
    for i in range(0, len(todo), 10):  # batches of 10: resumable, one model load each
        batch = todo[i : i + 10]
        inp = v2_dir() / "tmp" / "firstpass_in"
        inp.mkdir(parents=True, exist_ok=True)
        for old in inp.iterdir():
            old.unlink()
        for c in batch:
            (inp / f"ImageCAS_{c:04d}_0000.nii.gz").symlink_to(case_image_path(c))
        cmd = ["nnUNetv2_predict", "-i", str(inp), "-o", str(out), "-d", "501", "-c", "3d_fullres",
               "-f", "0", "-tr", trainer, "--disable_tta"]  # fmt: skip
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"nnUNetv2_predict failed on batch {batch}", file=sys.stderr)
            return r.returncode
        print(f"done {i + len(batch)}/{len(todo)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
