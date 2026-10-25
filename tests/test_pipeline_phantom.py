import json

import pytest

from twinloop.config import PipelineParams
from twinloop.phantom import make_phantom
from twinloop.pipeline import main, run_pipeline
from twinloop.schema import DISCLAIMER, CaseResult


def run(kind, tmp_path):
    img, mask = make_phantom(kind)
    return run_pipeline(img, mask, PipelineParams(), tmp_path, "c_test", f"{kind}.nii.gz")


def test_stenosis(tmp_path):
    res = run("stenosis", tmp_path)
    again = CaseResult.model_validate_json((tmp_path / "result.json").read_text())
    assert again == res
    assert len(res.trees) == 1
    assert len(res.lesions) == 1
    les = res.lesions[0]
    assert les.lesion_id == "L1"
    assert les.ffr_distal < les.ffr_proximal
    assert 0.80 < res.summary.min_ffr < 0.99
    assert res.disclaimer == DISCLAIMER
    assert list(res.timings_s) == ["load", "segmentation", "centerline", "solver", "total"]
    assert res.timings_s["segmentation"] == 0.0
    assert res.input.has_mask is True
    for name in ("image.nii.gz", "mask.nii.gz", "result.json"):
        assert (tmp_path / name).is_file()


def test_tube(tmp_path):
    res = run("tube", tmp_path)
    assert len(res.lesions) == 0
    assert res.summary.min_ffr > 0.97
    assert res.summary.n_significant_branches == 0
    assert res.trees[0].converged


def test_two_trees(tmp_path):
    res = run("two_trees", tmp_path)
    assert [t.tree_id for t in res.trees] == ["tree_0", "tree_1"]


def test_y_tree_flows_and_steps(tmp_path):
    img, mask = make_phantom("y_tree")
    steps = []
    res = run_pipeline(img, mask, PipelineParams(), tmp_path, "c", "y.nii.gz", on_step=steps.append)
    assert steps == ["extracting", "solving"]
    br = {b.branch_id: b for b in res.trees[0].branches}
    root = br[res.trees[0].root_branch_id]
    assert root.flow_ml_min == pytest.approx(
        sum(br[c].flow_ml_min for c in root.children), rel=0.01
    )
    assert res.trees[0].q_hyper_ml_min == root.flow_ml_min


def test_no_mask_and_no_model_fails_with_message(tmp_path, monkeypatch):
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "none"))
    img, _ = make_phantom("tube")
    with pytest.raises(RuntimeError, match="Segmentation model not available"):
        run_pipeline(img, None, PipelineParams(), tmp_path, "c", "t.nii.gz")


def test_cli_ok_and_failed(tmp_path, capsys, monkeypatch):
    from twinloop.io import save_nifti

    def no_anatomy(*args, **kwargs):
        raise RuntimeError("disabled in tests")

    monkeypatch.setattr("twinloop.anatomy.segment_anatomy", no_anatomy)  # v2 CLI default is on

    img, mask = make_phantom("stenosis")
    save_nifti(img, tmp_path / "img.nii.gz")
    save_nifti(mask, tmp_path / "mask.nii.gz")
    case = tmp_path / "case"
    rc = main(
        [
            "--image",
            str(tmp_path / "img.nii.gz"),
            "--mask",
            str(tmp_path / "mask.nii.gz"),
            "--case-dir",
            str(case),
            "--case-id",
            "c_cli",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("OK case_id=c_cli trees=1 lesions=1 min_ffr=0.")
    assert json.loads((case / "status.json").read_text())["status"] == "done"

    bad = tmp_path / "bad.txt"
    bad.write_text("x")
    rc = main(["--image", str(bad), "--case-dir", str(tmp_path / "bad_case")])
    assert rc == 1
    assert (
        capsys.readouterr().out.strip() == "FAILED Only .nii.gz, .nii or .zip files are supported"
    )
    status = json.loads((tmp_path / "bad_case" / "status.json").read_text())
    assert status["status"] == "failed"
