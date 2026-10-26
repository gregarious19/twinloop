# TwinLoop v2 — Specification (repair loop, anatomy, lumen refinement)

This document is self-contained and extends TwinLoop v1 (docs/SPEC.md). Read §0 and §1 first.
If anything here conflicts with your own judgment, follow this document. Where two reasonable
options exist, the chosen one is marked **DEFAULT**. v1 rules in CLAUDE.md still apply.

---

## 0. Background (what v1 already does, what v2 adds)

v1 pipeline (already built in `src/twinloop/`): load CCTA → nnU-Net v2 coronary mask (binary) →
`centerline.extract_trees()` (skeleton → up to 2 `VesselTree`s of `Branch`es with `points_mm`,
`radius_mm`, `s_mm`) → `lesions.detect_lesions()` → `solver.solve_tree()` (1D steady flow) →
`result.json` + web viewer. The nnU-Net "first-pass" model trains on the DGX Spark (v1 T05/T09).

v2 adds, around the first pass:
1. **Vessel graph** with ending types and coarse anatomical branch names.
2. **Anatomy** from TotalSegmentator: aorta, myocardium; quality checks; coronary ostia.
3. **Coverage check**: rule-based flags for missing/short main vessels and abrupt endings.
4. **Damage generator**: corrupts ground-truth (GT) masks to create training data.
5. **Region proposals + decider**: each candidate region gets one action:
   `keep | extend | add | remove | restore | stop`.
6. **Gap-filler**: a small 3D network that segments only inside a patch around a seed point.
7. **Repair loop**: steps 1–6 repeated (max 3 iterations).
8. **CPR + lumen model**: straightened vessel cross-sections → transformer → lumen radius per
   0.5 mm, replacing the skeleton-based radius before lesion detection and the solver.
9. **Myocardial-mass flow mode** for the solver, plus an FFR uncertainty flag.
10. **Evaluation harness** for all of the above.

Non-goals for v2: plaque classes (no labels exist), 3D CFD/VMTK, dominance detection, naming of
diagonal/marginal branches, any change to v1 API routes/status names/error strings, any FFR
accuracy claim, use of the ImageCAS test split.

---

## 1. Two machines

| Name in docs | Hardware | Role | Task file | Progress file |
|---|---|---|---|---|
| **SPARK** | NVIDIA DGX Spark, GB10, aarch64, ~70 GB usable GPU memory, Docker | All library code, unit tests, inference, pipeline, web app, evaluation | `docs/TASKS_V2_SPARK.md` | `docs/PROGRESS_V2_SPARK.md` |
| **CLUSTER** | Ubuntu, x86_64, Slurm, 80 GB GPU nodes, faculty-level partition | All v2 **training** and training-dataset generation | `docs/TASKS_V2_CLUSTER.md` | `docs/PROGRESS_CLUSTER.md` |

How to tell where you are: run `uname -m`. `aarch64` → SPARK. `x86_64` and `command -v sbatch`
succeeds → CLUSTER. Work only the task file for your machine.

**Code ownership (avoids git conflicts between the two sessions):**
| Owner | Files |
|---|---|
| CLUSTER | `src/twinloop/nets.py`, `train/**`, `cluster/**`, `tests/test_nets.py`, `requirements-cluster.txt`, `docs/PROGRESS_CLUSTER.md`, `docs/DECISIONS_CLUSTER.md`, `reports/v2_training.md` |
| SPARK | everything else (all other `src/`, `scripts/`, `app/`, `tests/`, `docs/`, `reports/`) |

A session never edits a file owned by the other machine. If a CLUSTER task finds a bug in a
SPARK-owned file, it writes the bug, a minimal reproduction and the proposed patch under
"Notes for next session" in PROGRESS_CLUSTER.md, works around it locally only if the workaround
lives in `train/`, and continues. The SPARK session reads PROGRESS_CLUSTER.md at session start
(after a `git pull` done by the owner) and fixes such bugs first.

A CLUSTER task that needs SPARK code names the SPARK task (e.g. "needs S07 pulled"). Check with
`git log --oneline | grep -E "^[0-9a-f]+ S07:"`; if absent, write `WAITING ON OWNER: git pull (needs S07)`
and do the next task whose dependencies are met.

**Sync (owner does these; Sonnet never runs `git push` and never copies data between machines):**
- Code: a private git remote. All v2 work is on branch `v2`. The owner pushes `v2` from SPARK
  (`git push -u origin v2` from the v2 worktree) and clones/pulls it on CLUSTER
  (`git clone -b v2 <remote> twinloop`), and pushes CLUSTER commits back the same way.
- Data SPARK → CLUSTER (once, after CLUSTER task C00). `<cluster>` = the owner's ssh login for the
  cluster (e.g. `user@hpc.example.edu`); `<CLUSTER_DATA>` = the value of `TWINLOOP_DATA` written in
  `cluster/cluster.env` by C00.
  ```
  # on SPARK host
  D=${TWINLOOP_DATA:-$HOME/twinloop-data}
  ssh <cluster> "mkdir -p <CLUSTER_DATA>/nnUNet_raw/Dataset501_ImageCAS <CLUSTER_DATA>/nnUNet_preprocessed/Dataset501_ImageCAS"
  rsync -av --progress $D/nnUNet_raw/Dataset501_ImageCAS/imagesTr $D/nnUNet_raw/Dataset501_ImageCAS/labelsTr \
        $D/nnUNet_raw/Dataset501_ImageCAS/dataset.json <cluster>:<CLUSTER_DATA>/nnUNet_raw/Dataset501_ImageCAS/
  rsync -av $D/nnUNet_preprocessed/Dataset501_ImageCAS/splits_final.json \
        <cluster>:<CLUSTER_DATA>/nnUNet_preprocessed/Dataset501_ImageCAS/
  ```
  Never copy `imagesTs`/`labelsTs` (test split) to CLUSTER.
- Models CLUSTER → SPARK (after C07): `rsync -av <cluster>:<CLUSTER_DATA>/models/v2/ $D/models/v2/`
- Real-failure dataset SPARK → CLUSTER (after S14): `rsync -av $D/v2/datasets/decider_real/ <cluster>:<CLUSTER_DATA>/v2/datasets/decider_real/`

A Sonnet session that needs a sync writes `WAITING ON OWNER: <exact rsync/git command>` under
"In progress" in its progress file, then continues with the next independent task.

---

## 2. Data, paths and splits

### 2.1 Data root
- Env var `TWINLOOP_DATA`. SPARK containers: `/data`. CLUSTER: the path in `cluster/cluster.env`.
- `src/twinloop/paths.py`:
```python
def data_root() -> Path            # Path(os.environ.get("TWINLOOP_DATA", "/data"))
def resolve(p: str | Path) -> Path # if str(p) starts with "/data/": data_root() / str(p)[6:]; else Path(p)
def case_image_path(case_id: int) -> Path
    # data_root()/nnUNet_raw/Dataset501_ImageCAS/imagesTr/ImageCAS_{case_id:04d}_0000.nii.gz
def case_label_path(case_id: int) -> Path
    # data_root()/nnUNet_raw/Dataset501_ImageCAS/labelsTr/ImageCAS_{case_id:04d}.nii.gz
def models_dir() -> Path           # data_root()/models/v2
def v2_dir() -> Path               # data_root()/v2
```
`case_image_path`/`case_label_path` raise `ValueError("Test-split case not allowed in v2: {id}")`
when `case_id % 5 == 0`. These nnUNet_raw files are already LPS-oriented (v1 T05 converted them).
Load them with `twinloop.io.load_image` / `load_mask` anyway (idempotent reorientation).

### 2.2 v2 case roles (`src/twinloop/v2split.py`)
nnU-Net fold 0 was trained on part of the train split; its validation cases (~20 % of train) were
never trained on. File: `data_root()/nnUNet_preprocessed/Dataset501_ImageCAS/splits_final.json`.
Format (verify in S00 by printing `type(js)`, `len(js)`, `js[0].keys()`, `js[0]["val"][:3]`):
a JSON list of 5 dicts `{"train": [...], "val": [...]}`, entries like `"ImageCAS_0007"`.
Case id = `int(entry.split("_")[1])`.

```python
def load_roles() -> dict[int, str]
```
Roles, applied in this order:
| Role | Rule | Used for |
|---|---|---|
| `test` | `case_id % 5 == 0` | NEVER used in v2 (not even read) |
| `v2_holdout` | in fold-0 `val` and `case_id % 2 == 0` | evaluation only (regression set, end-to-end) |
| `v2_realfail` | in fold-0 `val` and `case_id % 2 == 1` | real first-pass failures for decider fine-tuning |
| `v2_train` | every other non-test case | synthetic training data, sanity checks on SPARK |

Validation inside v2 model training (early stopping/model selection): `v2_train` cases with
`case_id % 10 == 3` (role stays `v2_train`; training scripts split by this rule). **No parameter in
this spec may be tuned on `v2_holdout`, `v2_realfail` or `test`.** All thresholds are fixed here.

---

## 3. Parameters (`src/twinloop/config_v2.py`, frozen dataclasses, exact defaults)

```python
@dataclass(frozen=True)
class GraphParams:
    end_window_mm: float = 2.0          # radius at end = median radius over last 2 mm
    taper_ref_from_end_mm: float = 5.0  # reference window: 5..10 mm from the end
    taper_ref_to_end_mm: float = 10.0
    abrupt_min_radius_mm: float = 0.8
    abrupt_taper_ratio: float = 0.75    # abrupt if r_end >= 0.8 and r_end/r_ref >= 0.75
    direction_window_mm: float = 3.0    # end direction = unit(p_last - p_at(L-3mm))
    max_components: int = 8             # components kept by build_graph (2 main + fragments)

@dataclass(frozen=True)
class AnatomyParams:
    totalseg_timeout_s: int = 900
    aorta_min_ml: float = 15.0; aorta_max_ml: float = 400.0
    aorta_main_component_frac: float = 0.95
    myo_min_ml: float = 50.0; myo_max_ml: float = 350.0
    ostium_max_dist_mm: float = 3.0
    lm_max_length_mm: float = 25.0
    min_main_len_lad_mm: float = 50.0
    min_main_len_lcx_mm: float = 25.0
    min_main_len_rca_mm: float = 50.0

@dataclass(frozen=True)
class DamageParams:
    n_damages_per_case: int = 6
    gap_len_mm: tuple[float, float] = (2.0, 8.0)
    shrink_len_mm: tuple[float, float] = (3.0, 8.0)
    shrink_factor: tuple[float, float] = (0.4, 0.7)
    false_len_mm: tuple[float, float] = (8.0, 25.0)
    false_radius_mm: tuple[float, float] = (0.8, 1.5)
    false_max_hu: float = 150.0
    occlusion_hu_mean: float = 50.0; occlusion_hu_sd: float = 15.0
    min_branch_radius_mm: float = 0.8
    min_branch_len_mm: float = 15.0
    type_weights: tuple[float, ...] = (0.2, 0.2, 0.15, 0.2, 0.15, 0.1)
    # order: gap, truncate, delete_branch, shrink, false_branch, occlusion

@dataclass(frozen=True)
class RegionParams:
    probe_step_mm: float = 5.0
    probe_min_branch_radius_mm: float = 1.0
    probe_rays: int = 8
    probe_inner_mm: float = 1.0; probe_outer_mm: float = 5.0   # added to local radius
    probe_min_hu: float = 250.0
    probe_nms_mm: float = 5.0
    probe_junction_exclusion_mm: float = 4.0
    min_terminal_branch_len_mm: float = 5.0
    match_radius_mm: float = 4.0       # damage record ↔ region matching

@dataclass(frozen=True)
class GapfillParams:
    patch_vox: int = 64
    patch_spacing_mm: float = 0.4
    forward_offset_mm: float = 8.0     # patch centre = seed + offset * direction (extend, add)
    seed_sigma_mm: float = 2.0
    hu_clip: tuple[float, float] = (-200.0, 800.0)
    hu_center: float = 300.0; hu_scale: float = 500.0
    prob_threshold: float = 0.5
    min_new_voxels: int = 20

@dataclass(frozen=True)
class CprParams:
    step_mm: float = 0.5
    size_px: int = 48
    pixel_mm: float = 0.2
    min_branch_len_mm: float = 10.0
    window_slices: int = 128
    window_stride: int = 96

@dataclass(frozen=True)
class RepairParams:
    max_iterations: int = 3
    decider_min_prob: float = 0.6

@dataclass(frozen=True)
class FlowParams:
    flow_mode: str = "auto"            # "auto" | "diameter_law" | "myocardial_mass"
    myo_density_g_ml: float = 1.05
    rest_flow_ml_min_per_g: float = 1.0   # UNCERTAIN, verify in S09
    ffr_uncertainty_radius_factor: float = 0.10

@dataclass(frozen=True)
class V2Params:                        # bundle passed to repair/eval code
    centerline: CenterlineParams = field(default_factory=CenterlineParams)   # v1 config.py
    lesion: LesionParams = field(default_factory=LesionParams)               # v1 config.py
    solver: SolverParams = field(default_factory=SolverParams)               # v1 config.py
    graph: GraphParams = field(default_factory=GraphParams)
    anatomy: AnatomyParams = field(default_factory=AnatomyParams)
    damage: DamageParams = field(default_factory=DamageParams)
    region: RegionParams = field(default_factory=RegionParams)
    gapfill: GapfillParams = field(default_factory=GapfillParams)
    cpr: CprParams = field(default_factory=CprParams)
    repair: RepairParams = field(default_factory=RepairParams)
    flow: FlowParams = field(default_factory=FlowParams)
```
Every function below that takes a `*Params` argument uses these classes; where a signature says
`P`, it means a `V2Params`.

---

## 4. Vessel graph (`src/twinloop/graph.py`)

```python
@dataclass
class Ending:
    tree_id: str; branch_id: str
    point_mm: np.ndarray        # (3,) last point of the branch
    direction: np.ndarray       # (3,) unit vector pointing distally
    r_end_mm: float; r_ref_mm: float; taper_ratio: float
    kind: str                   # "abrupt" | "taper"

@dataclass
class CoronaryGraph:
    trees: list[VesselTree]                 # the 2 largest components ("main trees")
    fragments: list[VesselTree]             # smaller components (e.g. distal piece after a gap)
    endings: list[Ending]                   # one per terminal branch of trees AND fragments
    labels: dict[str, str]                  # branch_id -> "LM"|"LAD"|"LCX"|"RCA"|"other"
    tree_side: dict[str, str]               # tree_id -> "left"|"right"|"unknown"
    warnings: list[str]

def build_graph(mask: sitk.Image, cp: CenterlineParams, gp: GraphParams) -> CoronaryGraph
def main_path(tree: VesselTree, start_branch_id: str) -> list[str]
def branch_length_mm(b: Branch) -> float            # b.s_mm[-1]
def graph_summary(g: CoronaryGraph) -> dict         # JSON-safe, see worked example
```
- `build_graph`: `all_t = extract_trees(mask, dataclasses.replace(cp, max_trees=gp.max_components))`
  (v1 function; it returns components sorted by volume, largest first, ids `tree_0, tree_1, …`).
  `trees = all_t[:2]`, `fragments = all_t[2:]`. Reason: a gap in a vessel splits it into two
  components; v1 would silently drop the distal piece. If `extract_trees` raises
  "No coronary tree found in mask", `build_graph` re-raises it. Labels start as all `"other"`,
  `tree_side` all `"unknown"` (anatomy fills them, §5). Endings for every terminal branch:
  `r_end` = median radius over samples with `s >= L - 2.0`; `r_ref` = median radius over
  `L-10 <= s <= L-5` (if the branch is shorter than 10 mm: median over the whole branch);
  `taper_ratio = r_end / r_ref`; `kind = "abrupt"` if `r_end >= 0.8 and taper_ratio >= 0.75`, else
  `"taper"`. Direction: `p_last - p_at(L-3)` normalised (`p_at` = linear interpolation by `s`;
  branch shorter than 3 mm → `p_last - p_first`).
- `main_path`: from `start_branch_id`, repeatedly append the child with the largest median radius
  over its first 5 mm (whole child if shorter); stop at a terminal branch. Returns branch ids.
- `graph_summary` example:
```json
{"n_trees": 2, "n_fragments": 0,
 "trees": [{"tree_id": "tree_0", "side": "left", "n_branches": 14, "total_length_mm": 412.5,
            "root_point_mm": [-21.4, 33.1, 102.5]}],
 "labels": {"tree_0_b0": "LM", "tree_0_b1": "LAD", "tree_0_b2": "LCX", "tree_1_b0": "RCA"},
 "endings": [{"branch_id": "tree_0_b7", "kind": "abrupt", "r_end_mm": 1.21, "taper_ratio": 0.93,
              "point_mm": [-5.2, 40.0, 61.3]}],
 "warnings": []}
```
(Round floats to 2 decimals in the summary.)

---

## 5. Anatomy (`src/twinloop/anatomy.py`)

### 5.1 TotalSegmentator
- Install: add `TotalSegmentator` to requirements.txt and install it with the existing constraint
  mechanism (Dockerfile pins torch). It depends on `nnunetv2`; the installed `nnunetv2` version must
  NOT change (the first-pass model is being trained with it). S00 gives the exact procedure.
- `scripts/dk` and `scripts/dkd` get `-e TOTALSEG_HOME_DIR=/data/totalseg`. Containers run with
  `HOME=/tmp`, so without this variable weights and the licence would be lost after every run.
- Licence: the owner has an academic licence key for `heartchambers_highres`. The owner (not you)
  runs `./scripts/dk totalseg_set_license -l <key>` once after S00. **Never print, log, read, `cat`
  or copy the key or `/data/totalseg/config.json`.** You may only test for the file's existence
  (`test -f /data/totalseg/config.json && echo present`).
- **UNCERTAIN:** that `totalseg_set_license` writes into `$TOTALSEG_HOME_DIR`. Verify by the
  existence test above after the owner runs it. If the file is absent but
  `/tmp/.totalsegmentator/config.json` was written instead inside that container, log it in
  DECISIONS_V2.md and ask the owner (WAITING ON OWNER) to rerun with
  `./scripts/dk bash -c 'mkdir -p /data/totalseg && totalseg_set_license -l <key> && cp -r /tmp/.totalsegmentator/. /data/totalseg/'`.
- GPU sharing: the first-pass nnU-Net training may be running on SPARK. Before any TotalSegmentator
  or GPU inference run, check `free -g`; if "available" < 20 GB, wait (do another task) instead.

```python
@dataclass
class Anatomy:
    aorta: sitk.Image | None          # UInt8 0/1 on the CCTA grid
    myocardium: sitk.Image | None
    lv_cavity: sitk.Image | None
    source: dict[str, str]            # structure -> "heartchambers_highres" | "total" | "missing"
    qc: dict[str, bool]               # see 5.2
    qc_messages: list[str]
    myo_volume_ml: float | None; aorta_volume_ml: float | None

def run_totalseg(image: sitk.Image, workdir: Path, task: str, timeout_s: int) -> dict[str, sitk.Image]
def segment_anatomy(image: sitk.Image, workdir: Path, ap: AnatomyParams) -> Anatomy
```
- `run_totalseg`: write the image to `workdir/ct.nii.gz`; subprocess
  `TotalSegmentator -i workdir/ct.nii.gz -o workdir/<task> -ta <task>` (no `-ml`; one file per
  class, named `<class>.nii.gz`). For task `total` add `--roi_subset aorta heart`. Timeout →
  `RuntimeError("TotalSegmentator timed out")`. Non-zero exit → `RuntimeError` with the last 20 lines
  of stderr. Read every `*.nii.gz` in the output folder into `{stem: image}`; for each: if its size
  equals the CCTA size, `CopyInformation(image)`; else resample onto the CCTA grid with
  `sitk.Resample(mask, image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)`.
  Binarise (`> 0`).
- `segment_anatomy`: try `heartchambers_highres` first (classes `myocardium`, `ventricle_left`,
  `aorta`, …). If it raises, record the message in `qc_messages` (e.g. licence missing) and fall back
  to `total` for `aorta` only (myocardium = None). Highres success → aorta from highres.
- **UNCERTAIN (verify in S01):** exact output file names of `heartchambers_highres`. Verify by
  running it once on one `v2_train` case and listing the folder; record the list in DECISIONS_V2.md.

### 5.2 Quality checks (`qc` keys; all must exist, value True = pass)
| Key | Pass rule |
|---|---|
| `aorta_present` | aorta not None and volume ≥ 15 mL and ≤ 400 mL |
| `aorta_connected` | largest 26-connected component ≥ 95 % of aorta voxels |
| `myo_present` | myocardium not None and 50 ≤ volume ≤ 350 mL |
| `ostia_ok` | filled by `find_ostia`: exactly 2 trees with an ostium within 3 mm of the aorta |
Volume mL = voxel count × prod(spacing) / 1000.

### 5.3 Ostia, LV axis, labels, coverage
```python
def find_ostia(g: CoronaryGraph, anat_or_aorta: "Anatomy | sitk.Image | None", ap) -> dict[str, float | None]
    # main trees only: tree_id -> distance (mm) from root branch points_mm[0] to the nearest aorta
    # voxel (scipy distance_transform_edt of ~aorta with sampling = spacing reversed, sampled at
    # that point with order=1); None for every tree if the aorta is None. When given an Anatomy,
    # also sets anat.qc["ostia_ok"] = (len(g.trees) == 2 and both distances <= 3.0).
def lv_axis(myo: sitk.Image, aorta: sitk.Image | None) -> tuple[np.ndarray, np.ndarray] | None
    # (base_point_mm, apex_point_mm). PCA of myocardium voxel physical coords; principal axis u.
    # Project voxels on u; the extreme (1st or 99th percentile of projection) farther from the
    # aorta centroid is the apex. Without aorta: return None.
def label_branches(g: CoronaryGraph, anat: Anatomy, ap) -> None   # fills g.labels, g.tree_side
def coverage_check(g: CoronaryGraph, ap) -> list[dict]
```
`label_branches` (coarse, **DEFAULT**):
1. Side: one tree → side by its ostium x (LPS x: larger = patient left) relative to the aorta
   centroid x (`>` → left). Two trees → the tree whose root point has the larger x is `left`, the
   other `right`. No aorta and one tree → `unknown`.
2. Right tree: `main_path(root)` branches labelled `RCA`.
3. Left tree: **LM rule** = the root branch is ≤ 25 mm long and has ≥ 2 children. If it holds,
   root = `LM`. If additionally an LV axis exists (myocardium present and qc `myo_present`): for
   each child c, `end_c = last point of main_path(c)`; LAD = the child whose `end_c` is nearest the
   apex (its main path labelled `LAD`); LCX = among the remaining children, the one with the largest
   median radius over its first 5 mm (main path labelled `LCX`); any further child (e.g. a ramus)
   stays `other`. No LV axis → children stay `other`, warning `"LAD/LCX not labelled: no myocardium"`.
   LM rule fails → warning `"LM not identified"` and every left-tree branch stays `other`.

`coverage_check` returns flags, each `{"kind", "branch_id", "point_mm", "direction", "message"}`
(`point_mm`/`direction` as lists or None):
| kind | Rule | point / direction | message |
|---|---|---|---|
| `missing_tree` | fewer than 2 main trees | None / None | `"Only {n} coronary tree(s) found"` |
| `missing_main_branch` | a tree has side `left` and the LM rule failed | root branch last point / its end direction | `"Left main bifurcation not found"` |
| `short_main_vessel` | for each of LAD, LCX, RCA that is labelled: summed length of its labelled branches < 50 / 25 / 50 mm | last point of the last labelled branch / its end direction | `"{name} main path {L:.1f} mm < {min:.0f} mm"` |
| `abrupt_end` | every `Ending` (trees and fragments) with `kind == "abrupt"` | ending point / direction | `"Vessel ends abruptly (r_end {r:.2f} mm)"` |

Worked example: an RCA main path of total length 31.2 mm whose last branch ends with r_end 1.4 mm
and taper ratio 0.95 → two flags: `short_main_vessel` (message `"RCA main path 31.2 mm < 50 mm"`) and
`abrupt_end` at the same point.

---

## 6. Damage generator (`src/twinloop/damage.py`)

```python
@dataclass
class DamageRecord:
    damage_id: str            # "d0".."d5"
    type: str                 # gap|truncate|delete_branch|shrink|false_branch|occlusion
    action: str               # extend|extend|add|restore|remove|stop (in the same order)
    branch_id: str            # GT branch that was damaged (false_branch: the branch the tube attaches to)
    point_mm: list[float]     # seed point (see table)
    direction: list[float]    # unit vector
    length_mm: float
    voxels_changed: int

def damage_case(image: sitk.Image, gt: sitk.Image, g: CoronaryGraph, dp: DamageParams,
                rng: np.random.Generator) -> tuple[sitk.Image, sitk.Image, list[DamageRecord]]
    # returns (image_out, mask_out, records). image_out differs from image only for "occlusion".
```
RNG: `np.random.default_rng(1000003 * seed + case_id)` created by the caller.
Eligible branch (branches of `g.trees` and `g.fragments`): median radius ≥ 0.8 mm and length
≥ 15 mm (`delete_branch`: non-root, median radius ≥ 0.8 mm, length ≥ 10 mm).
Procedure: for d in 0..5: draw a type with `rng.choice(6, p=type_weights)`; then up to 50 tries:
draw an eligible branch uniformly (`rng.integers`), draw the type's numbers in the order listed in
the table (uniforms with `rng.uniform`); reject the try if a range is empty (e.g. `L−5−len < 5`), if
the seed point is within 10 mm of an earlier record's seed point, or (false_branch) fewer than 50
voxels would be added. After 50 rejected tries the damage is skipped (no record). Record ids are
`d0`.. in the order kept.

"Voxels of a centerline interval": compute, for every mask voxel, the nearest centerline sample of
the GT graph (KD-tree on all branch points, `scipy.spatial.cKDTree`); a voxel belongs to interval
`[s0, s1]` of branch b if its nearest sample is on b with `s0 ≤ s ≤ s1` and its distance to that
sample ≤ 1.5 × that sample's radius + 0.5 mm.

| type | What changes | seed point / direction | action |
|---|---|---|---|
| gap | remove voxels of `[s, s+len]`, s ~ U(5, L−5−len), len ~ U(2,8) | point at s / tangent at s | extend |
| truncate | remove voxels of `[s, L]` on b and all descendants, s ~ U(0.3L, 0.8L) | point at s / tangent | extend |
| delete_branch | remove all voxels of b (s ≥ 1 mm) and descendants | b.points_mm[0] / unit(p(2 mm) − p(0)) | add |
| shrink | in `[s, s+len]` keep only voxels within f × r of the centerline, f ~ U(0.4,0.7) | point at s+len/2 / tangent | restore |
| false_branch | add a straight tube from point P at s ~ U(5, L−5) on an eligible branch: angle θ ~ U(60°, 120°) to the tangent T and azimuth φ ~ U(0, 360°) around it (direction = cos θ·T + sin θ·(cos φ·U + sin φ·V), U/V from `rmf` at that sample), length ~ U(8,25), radius ~ U(0.8,1.5); tube voxels = voxels whose centre is within `radius` of the segment P→P+len·direction and not already in the mask; add only those with image HU ≤ 150 | tube midpoint P + (len/2)·direction (matches the `branch` region point, §7) / that direction | remove |
| occlusion | as truncate, and in image_out set every removed voxel to N(50, 15) HU | point at s / tangent | stop |

---

## 7. Regions and features (`src/twinloop/regions.py`)

```python
@dataclass
class Region:
    region_id: str                     # "r0".. in proposal order
    kind: str                          # "ending" | "probe" | "narrowing" | "branch"
    tree_id: str; branch_id: str
    point_mm: np.ndarray; direction: np.ndarray
    idx: int                           # sample index on the branch

def propose_regions(g: CoronaryGraph, image: sitk.Image, mask: sitk.Image, P: V2Params) -> list[Region]
def region_features(regions: list[Region], g, image, mask, P: V2Params) -> np.ndarray  # (n, 31) float64
FEATURE_NAMES: list[str]   # fixed order, written to the model's config
ACTIONS = ["keep", "extend", "add", "remove", "restore", "stop"]
def label_synthetic(regions, records, P: V2Params) -> tuple[list[str], int]   # (labels, n_unmatched_records)
def label_real(regions, g_pred, pred_mask, gt_mask, g_gt, P: V2Params) -> list[str]
```
**Proposals** (order: endings, narrowings, branches, probes):
- `ending`: every `Ending` (point = last point, direction = end direction).
- `narrowing`: every v1 `detect_lesions` lesion (point = min_idx point, direction = tangent).
- `branch`: every terminal branch with length ≥ 5 mm (point = its middle sample).
- `probe`: on branches with median radius ≥ 1.0 mm, every 5 mm of arc length (skip first/last
  2 mm): build the local frame (§9.2 rotation-minimising frame), cast 8 rays at angles k·45°
  perpendicular to the tangent; sample the image (linear) along the ray from distance
  (r + 1) to (r + 5) mm in 0.5 mm steps; a ray qualifies if mean HU ≥ 250 and every sample point
  is outside the mask. Keep the qualifying ray with the highest mean HU. Drop a probe within 4 mm of
  any child-branch start point or within 5 mm of an already kept probe. Point = the centerline
  sample the ray starts from (not a point on the ray), direction = ray direction.

`region_features` builds once per call: the image and mask arrays (z,y,x), the Euclidean distance
(mm) of every non-mask voxel to the mask (`distance_transform_edt(~mask, sampling=spacing[::-1])`)
— for speed compute it on the mask's bounding box expanded by 10 mm — and the per-branch lesion
reference radii. Perpendicular basis for a direction d: u = normalise(cross(d, e)) with e the
coordinate axis least aligned with d, v = cross(d, u).

**Features** (all floats; missing → `nan`; HistGradientBoosting handles nan):
`kind_ending, kind_probe, kind_narrowing, kind_branch` (one-hot), `r_local_mm` (branch radius at
`idx`), `r_ref_mm` (v1 `reference_radius` of the branch at `idx`), `ds` (1 − r_local/r_ref),
`taper_ratio` (endings, else nan), `is_abrupt` (1 for an abrupt ending, else 0), `branch_len_mm`,
`dist_from_root_mm` (sum of ancestor lengths + s), `depth` (number of ancestors), `n_children`,
`is_terminal`, `in_fragment` (1 if the region's tree is a fragment),
`hu_ahead_1 … hu_ahead_10` (mean HU of 9 samples: the centre c_k = point + k mm · direction and
8 points c_k + 1 mm·(cos(j·45°)·u + sin(j·45°)·v), k = 1..10),
`mask_ahead_frac` (fraction of the 10 centres c_k inside the mask, nearest-voxel lookup),
`hu_lumen` (mean HU of mask voxels within 2 mm of point; nan if none), `hu_shell` (mean HU of
non-mask voxels with distance-to-mask 1–2 mm and within 3 mm of point; nan if none),
`hu_branch_mean` (branch regions: mean HU sampled at the branch's centerline points; else nan),
`calcium_frac` (fraction of voxels within 3 mm of point with HU ≥ 800), `probe_ray_hu` (probe
regions: mean HU along the chosen ray; else nan).
That is 31 features; `FEATURE_NAMES` lists them in exactly this order. **DEFAULT: the decider
uses no anatomy (no TotalSegmentator) features**, so training data can be generated on CLUSTER
without TotalSegmentator and the decider works when anatomy fails. `region_features` therefore
takes no `Anatomy` argument. Exact `FEATURE_NAMES`:
```python
FEATURE_NAMES = ["kind_ending", "kind_probe", "kind_narrowing", "kind_branch", "r_local_mm",
    "r_ref_mm", "ds", "taper_ratio", "is_abrupt", "branch_len_mm", "dist_from_root_mm", "depth",
    "n_children", "is_terminal", "in_fragment"] + [f"hu_ahead_{k}" for k in range(1, 11)] + [
    "mask_ahead_frac", "hu_lumen", "hu_shell", "hu_branch_mean", "calcium_frac"]   # 15+10+5 = 30
FEATURE_NAMES.append("probe_ray_hu")   # probe regions: mean HU of the chosen ray; else nan -> 31
```

HU sampling helper used everywhere in this file:
`sample_hu(image_arr_zyx, image, points_mm (K,3)) -> (K,)` = `map_coordinates(arr, phys_to_index_zyx(points, image).T, order=1, mode="nearest")`
(§9.1). "Mask voxels within d mm of point" = mask voxels whose physical centre is within d mm.

**Synthetic labels**: compatible pairs (damage action → region kind): extend→ending,
add→probe, restore→narrowing, remove→branch, stop→ending. For each record, the compatible region with
the nearest `point_mm` within 4 mm gets the record's action (each region at most one label; first
record wins). All other regions: `keep`. Records with no match are counted (`n_unmatched_records`).

**Real labels** (pred = first-pass mask, gt = GT mask, both graphs built; "GT points" = GT centerline
samples with radius ≥ 0.75 mm; "pred dilated" = pred dilated by 1 voxel):
- ending → `extend` if ≥ 4 GT points lie ahead (0 < (q−p)·d ≤ 10 mm and angle(q−p, d) ≤ 35°) and are
  outside pred dilated.
- probe → `add` if ≥ 4 GT points within 6 mm of point + 3 mm·direction are outside pred dilated.
- branch → `remove` if < 20 % of the branch's points are inside gt dilated.
- narrowing → `restore` if the nearest GT point (≤ 2 mm) has GT ds < 0.15 (GT ds from v1
  `reference_radius` on the GT branch).
- otherwise `keep`. (`stop` never comes from real data.)

---

## 8. Neural networks (`src/twinloop/nets.py`, plain PyTorch, no extra deps)

`nets.py` is written by CLUSTER (task C01). SPARK imports it (gap-filler and lumen inference) once
it has been pulled. Both classes take keyword `channels: tuple[int, ...] = (16, 32, 64, 128)`.

### 8.1 Gap-filler `GapfillNet`
3D U-Net. Input (B, 3, 64, 64, 64): [normalised image, current mask (0/1), seed heatmap].
Output (B, 1, 64, 64, 64) logits = full vessel mask in the patch. Encoder channels 16, 32, 64, 128
(each level: two `Conv3d(k=3, pad=1)` + `InstanceNorm3d(affine=True)` + `LeakyReLU(0.01)`;
downsample `Conv3d(k=2, stride=2)`); decoder mirrors with `ConvTranspose3d(k=2, stride=2)` and skip
concatenation; final `Conv3d(16, 1, 1)`.

Patch extraction (`src/twinloop/gapfill.py`):
```python
def extract_patch(image, mask, seed_mm, direction, offset_mm, gp) -> dict
    # centre = seed_mm + offset_mm * direction. Reference grid: size 64^3, spacing 0.4 mm,
    # identity direction, origin = centre - 0.4*(64-1)/2 per axis. Resample image (linear,
    # default -1000) and mask (nearest, default 0) onto it with sitk.Resample.
    # img_n = (clip(hu, -200, 800) - 300) / 500. Seed heatmap = exp(-d^2 / (2*2.0^2)), d = mm
    # distance of each voxel centre to seed_mm. Returns {"x": float32 (3,64,64,64),
    # "ref": sitk.Image (the reference grid)}.
def apply_gapfill(model, image, mask, seed_mm, direction, offset_mm, gp, device) -> tuple[sitk.Image, int]
    # p = extract_patch(...); if hasattr(model, "set_reference"): model.set_reference(p["ref"])
    # (test hook only); with torch.no_grad(): logits = model(torch.from_numpy(p["x"])[None].to(device));
    # prob > 0.5; new = pred & ~mask_patch; keep only new voxels 26-connected (within the
    # patch, together with mask_patch) to mask_patch; resample new back to the CCTA grid (nearest);
    # if count(new) < 20: return (mask, 0) ("nothing found"); else (mask | new, count).
```
offset: 8 mm for extend and add; 0 for restore.

### 8.2 Lumen model `LumenNet`
Input (B, T, 1, 48, 48) straightened slices (T ≤ 128); output (B, T, 1, 48, 48) lumen logits.
Per-slice 2D U-Net encoder (channels 16, 32, 64, 128; three stride-2 downsamples → 6×6×128
bottleneck). Bottleneck → global average pool → 128-d token per slice → add learned positional
embedding (max 128) → `nn.TransformerEncoder` (4 layers, d_model 128, 4 heads, ff 256, dropout 0.1,
`batch_first=True`) → token broadcast-added to the bottleneck map → per-slice 2D decoder with skips.
Image normalisation as the gap-filler.

### 8.3 Model files (both nets; exported by CLUSTER, read by SPARK)
`models_dir()/<name>/model.pt` = `state_dict`; `config.json`:
```json
{"name": "gapfill", "arch": "GapfillNet", "arch_kwargs": {"channels": [16, 32, 64, 128]},
 "params": {"patch_vox": 64, "patch_spacing_mm": 0.4, "hu_clip": [-200, 800], "hu_center": 300,
            "hu_scale": 500, "seed_sigma_mm": 2.0},
 "seed": 0, "git_commit": "abc1234", "trained_on": "v2_train synthetic",
 "val_metric_name": "new_voxel_dice", "val_metric": 0.71, "torch_version": "2.5.1+cu124"}
```
Lumen: same with `"name": "lumen"`, `"arch": "LumenNet"`, `"val_metric_name": "req_err_mm"`.
`metrics.json` = final validation metrics. Decider: `models_dir()/decider/decider.joblib`,
`config.json` = `{"name": "decider", "arch": "HistGradientBoostingClassifier", "feature_names":
[31 names], "actions": ["keep", …], "seed": 0, "sklearn_version": "1.x.y", "val_metric_name":
"macro_f1", "val_metric": 0.62, "trained_on": "...", "git_commit": "..."}`.
Loader `twinloop.modelio.load_model(name, device) -> nn.Module | sklearn model`: folder missing →
`RuntimeError("v2 model not available: <folder path>")`; nets: `getattr(twinloop.nets, cfg["arch"])(**cfg["arch_kwargs"])`,
`load_state_dict(torch.load(model.pt, map_location=device), strict=True)`, `.eval().to(device)`;
decider: `joblib.load`, then check `cfg["feature_names"] == FEATURE_NAMES` (else RuntimeError
`"decider feature names do not match"`).

---

## 9. CPR and lumen refinement (`src/twinloop/cpr.py`, `src/twinloop/lumen.py`)

### 9.1 Physical ↔ index
```python
def phys_to_index_zyx(points_mm: np.ndarray, img: sitk.Image) -> np.ndarray
    # idx_xyz = inv(D @ diag(spacing)) @ (p - origin) for each point; return idx[:, ::-1] (z,y,x)
```
D = `np.array(img.GetDirection()).reshape(3,3)`. Use with
`scipy.ndimage.map_coordinates(arr, idx.T, order=1, mode="nearest")` (masks: `order=0`).

### 9.2 Rotation-minimising frame (double reflection, Wang et al. 2008)
```python
def rmf(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]   # T, U, V each (N,3)
    T = tangents by central differences (one-sided at ends), normalised
    U0 = any unit vector ⟂ T0 (cross(T0, e) with e = the axis least aligned with T0, normalised)
    for i in 0..N-2:
        v1 = x[i+1]-x[i]; c1 = v1·v1
        rL = r[i] - (2/c1)(v1·r[i]) v1;  tL = t[i] - (2/c1)(v1·t[i]) v1
        v2 = t[i+1] - tL; c2 = v2·v2
        r[i+1] = rL - (2/c2)(v2·rL) v2       (if c2 < 1e-12: r[i+1] = rL)
    V = cross(T, U)
```

### 9.3 Straightening
```python
def straighten_points(image, mask_or_none, points_mm: np.ndarray, cp: CprParams) -> dict
    # arc length s of points; resample to s_k = 0, 0.5, ..., floor(L/0.5)*0.5 (np.interp per axis);
    # frame (T,U,V) = rmf(resampled); slice k pixel (a,b), a,b in 0..47:
    # p = x_k + ((a-23.5)*0.2) U_k + ((b-23.5)*0.2) V_k. Image sampled order=1 (map_coordinates,
    # mode="nearest"), mask order=0.
    # returns {"img": float32 (N,48,48) HU, "mask": uint8 (N,48,48) or None,
    #          "s_mm": (N,), "points_mm": (N,3)}
def straighten(image, mask_or_none, branch: Branch, cp: CprParams) -> dict
    # = straighten_points(image, mask_or_none, branch.points_mm, cp)
def center_component(mask_slice) -> np.ndarray   # 2D 8-connected component containing pixel (24,24),
                                                  # or the component nearest to it; empty if none
def r_eq_mm(lumen_slice) -> float                # sqrt(area_px * 0.2^2 / pi)
```
### 9.4 Refinement in the pipeline
```python
def refine_radii(trees, image, model, cp, device) -> list[str]   # returns warnings
```
For every branch ≥ 10 mm: straighten, normalise (`(clip(hu,-200,800)-300)/500`), run the model in
windows of 128 slices with stride 96 (a branch with ≤ 128 slices = one window; the last window is
aligned to the end; average overlapping logits), take `center_component(prob > 0.5)` per slice, r_eq per slice; slices
with an empty component keep the v1 radius. Smooth with Gaussian σ = 1 sample, interpolate back onto
the branch's own `s_mm`, replace `branch.radius_mm`. Branches < 10 mm keep v1 radii.

Worked check (test): tube phantom (r 1.5 mm), GT mask straightened → middle-slice `r_eq` within
1.5 ± 0.15 mm.

---

## 10. Repair loop (`src/twinloop/repair.py`)

```python
@dataclass
class RepairAction:
    iteration: int; region_id: str; kind: str; action: str; prob: float
    point_mm: list[float]; voxels_changed: int; result: str   # "applied"|"nothing_found"|"flagged"|"skipped"

def run_repair(image, mask, anat: Anatomy | None, models: dict, P: V2Params,
               device: str) -> tuple[sitk.Image, list[RepairAction], list[dict]]
    # returns (mask_out, actions, review_flags)
```
`models` keys: `"decider"` (object with `predict_proba(X: np.ndarray (n, 31)) -> (n, 6)`, columns in
`ACTIONS` order; either the trained sklearn model or `RulesDecider`), `"gapfill"` (GapfillNet or
None). If `"gapfill"` is None, extend/add/restore become `flagged` instead of applied.
`RulesDecider.predict_proba`: row = one-hot `extend` (prob 1.0) when `kind_ending == 1 and
is_abrupt == 1`, else one-hot `keep`.

Each iteration: `build_graph` → `propose_regions` → features → decider `predict_proba` → per
region: `a = argmax`, `p = max prob`.
- p < 0.6 → result `flagged`, add review flag `{"kind": "uncertain_region", ...}` (no change).
- keep → nothing (not logged). stop → `flagged`, review flag `possible_occlusion`.
- extend/add/restore → `apply_gapfill` (offset 8/8/0 mm) → `applied` or `nothing_found`.
- remove → delete the branch's voxels (interval rule of §6 with s ≥ 1 mm) → `applied`.
Apply actions in region order; skip a region within 4 mm of a region already changed this iteration
(`skipped`). Stop when an iteration changes no voxels or after 3 iterations. Review flags
(`uncertain_region`, `possible_occlusion`; fields `kind`, `region_id`, `point_mm`, `prob`,
`message`) are de-duplicated: a flag within 4 mm of an earlier flag of the same kind is dropped.
`actions` lists every non-keep region of every iteration. After the loop: build
the final graph; if `anat` is given, run `find_ostia`, `label_branches`, `coverage_check` on it and
append the coverage flags to review_flags.

---

## 11. Flow (`src/twinloop/flow.py`) and solver change

v1 `solve_tree(tree, lesions, p)` gains a keyword `q_rest_override_m3s: float | None = None`
(when set, it replaces `resting_flow_m3s(tree, p)`; all v1 tests must still pass).

```python
def tree_rest_flows(trees, anat: Anatomy | None, sp: SolverParams, fp: FlowParams) -> tuple[dict[str, float] | None, str]
    # returns ({tree_id: q_rest_m3s}, mode_used)
```
- mode `diameter_law`: return (None, "diameter_law") → v1 behaviour.
- mode `myocardial_mass`: requires `anat.qc["myo_present"]` (else raise
  `ValueError("myocardial_mass flow needs a valid myocardium")`); mass_g = myo_volume_ml × 1.05;
  q_total = 1.0 × mass_g mL/min; split between trees by Σ (outlet radius³) of each tree (outlet
  radius = median radius over the last 3 mm of each outlet branch, as v1 `outlet_weights`); convert
  to m³/s (× 1e-6 / 60).
- mode `auto` (**DEFAULT** in the pipeline): myocardial_mass if `myo_present`, else diameter_law and
  warning `"Myocardium unavailable; using diameter flow law"`.
- **UNCERTAIN:** 1.0 mL/min/g resting myocardial blood flow. Verify with WebSearch (e.g. PET resting
  MBF literature, typical 0.6–1.2 mL/min/g). If sources give a typical value in that range, keep 1.0;
  else use the cited value, update config_v2.py, log in DECISIONS_V2.md.

Worked example: myocardium 140 mL → 147 g → 147 mL/min; trees with Σr³ 30 and 15 (mm³) →
98.0 / 49.0 mL/min resting.

FFR uncertainty flag:
```python
def ffr_uncertainty_flags(trees, lesions_by_branch: dict[str, list[Lesion]], sp: SolverParams,
                          q_overrides: dict[str, float] | None, factor: float = 0.10,
                          min_diam_mm: float = 1.5) -> list[dict]
```
For each tree with ≥ 1 lesion: solve three times — baseline, and with every lesion's samples
`start_idx..end_idx` radius × (1 − factor) and × (1 + factor) (copies of the branches; the lesion's
`r_min_mm` scaled the same way, `r_ref_mm` unchanged). min FFR = minimum over samples with diameter
≥ 1.5 mm. If the three min FFRs are not all on the same side of 0.80 (≤ 0.80 vs > 0.80) → flag
`{"kind": "ffr_uncertain", "tree_id": ..., "message": "FFR near 0.80 is sensitive to lumen size (min FFR {lo:.2f}–{hi:.2f})"}`.

---

## 12. Pipeline integration (`pipeline.py`, `schema.py`)

`run_pipeline(..., v2: V2Options | None = None)`; `V2Options(anatomy: bool, repair: bool,
lumen: bool, flow_mode: str)`. **DEFAULT** for the CLI and the web app: `anatomy=True,
repair=False, lumen=False, flow_mode="auto"` until S16 decides otherwise. CLI flags:
`--no-anatomy`, `--repair`, `--lumen`, `--flow-mode {auto,diameter_law,myocardial_mass}`.
Status/step names stay v1's: TotalSegmentator runs inside `"segmenting"` (also when a mask is
supplied — then the step is still reported as `"segmenting"`), repair + lumen inside `"extracting"`.

`result.json` gets one new optional top-level field (allowed by CLAUDE.md, log it):
```json
"v2": {"anatomy_source": {"aorta": "heartchambers_highres", "myocardium": "heartchambers_highres"},
       "qc": {"aorta_present": true, "aorta_connected": true, "myo_present": true, "ostia_ok": true},
       "myocardium_ml": 141.2, "flow_mode": "myocardial_mass",
       "branch_labels": {"tree_0_b0": "LM", "tree_0_b1": "LAD"},
       "repair_actions": [{"iteration": 1, "region_id": "r4", "kind": "ending", "action": "extend",
                           "prob": 0.83, "point_mm": [1.2, 3.4, 5.6], "voxels_changed": 412,
                           "result": "applied"}],
       "review_flags": [{"kind": "abrupt_end", "branch_id": "tree_1_b3", "message": "..."}],
       "timings_s": {"anatomy": 74.2, "repair": 0.0, "lumen": 0.0}}
```
`CaseResult.v2: V2Out | None = None` (pydantic model mirroring this). Timings `anatomy` etc. are
included in the existing `segmentation`/`centerline` totals as well. v1 tests must still pass.

Robustness rules:
- Anatomy never fails a case: any exception inside `segment_anatomy` is caught; the pipeline uses an
  `Anatomy` with all masks None, every qc False, the error text in `qc_messages`, and adds warning
  `"Anatomy unavailable: <error>"`. Flow then falls back to `diameter_law` (mode `auto`).
- Files: anatomy masks saved as `case_dir/aorta.nii.gz` and `case_dir/myocardium.nii.gz` (when
  present); the TotalSegmentator workdir `case_dir/totalseg/` is deleted afterwards. With repair on,
  the input mask is saved as `case_dir/mask_firstpass.nii.gz` and `mask.nii.gz` is the repaired mask.
- `V2Options` is a dataclass with the defaults above. Tests of the app (v1 T06) and of the pipeline
  monkeypatch `twinloop.anatomy.segment_anatomy` (raise or return a synthetic `Anatomy`) so
  TotalSegmentator never runs in the CPU test suite.

---

## 13. Evaluation harness (`src/twinloop/metrics.py`, `scripts/eval_v2.py`)

```python
def dice(p, g) -> float                        # 2|P∩G|/(|P|+|G|); both empty -> 1.0
def cl_dice(p, g) -> float                     # Tprec=|S(P)∩G|/|S(P)|, Tsens=|S(G)∩P|/|S(G)|,
                                               # 2*Tprec*Tsens/(Tprec+Tsens); S = skeletonize
def branch_recall(g_gt, pred_mask) -> float    # GT branches with median r>=0.75 and length>=10 mm whose
                                               # >=80 % points lie in pred dilated 1 vox / all such
def false_branch_rate(g_pred, gt_mask) -> float  # pred branches (same filter) with <20 % points in
                                                 # gt dilated 1 vox / all such pred branches
def stenosis_metrics(g_gt, g_pred, lp) -> dict   # GT lesions (v1 detect on GT graph); for each, nearest
    # pred point within 2 mm: abs radius error at min point; "widened" if pred ds < 0.5 * gt ds or no
    # pred point. Returns {"n_lesions", "mean_abs_rmin_err_mm", "widened_rate"}
```
All masks numpy bool (z,y,x); dilation `scipy.ndimage.binary_dilation` with a 3×3×3 structure.

Regression set (`reports/v2_regression_set.json`, created once in S05, never edited): from
`v2_holdout`, compute per case `calcium = count of voxels with HU ≥ 800 inside (GT dilated by 4 vox)
minus GT`, and `n_lesions` (v1 detect on GT graph). Take the 10 highest calcium, then the 5 highest
n_lesions not yet chosen, then the 5 lowest remaining case ids. JSON: `{"case_ids": [...],
"rule": "<this rule text>"}`.

First-pass masks for evaluation (**DEFAULT**): nnU-Net writes fold-0 validation predictions at
the end of training to
`/data/nnUNet_results/Dataset501_ImageCAS/<TRAINER>__nnUNetPlans__3d_fullres/fold_0/validation/ImageCAS_XXXX.nii.gz`
(`<TRAINER>` = value of `TWINLOOP_NNUNET_TRAINER` after v1 T09). Every `v2_holdout` and
`v2_realfail` case is a fold-0 validation case, so it has such a file. Load with
`twinloop.io.load_mask(path, reference=image)`. If the folder has no `.nii.gz` files, run
`nnUNetv2_predict` (v1 `segment_nnunet`) on those cases instead and log it in DECISIONS_V2.md.

`scripts/eval_v2.py --cases regression|holdout --variants <comma list> --out CSV`:
CSV columns `case_id,variant,status,dice,cl_dice,branch_recall,false_branch_rate,n_lesions,
mean_abs_rmin_err_mm,widened_rate,min_ffr_tree0,min_ffr_tree1,n_review_flags,n_actions,t_total,error`.
Variants:
| variant | mask | pipeline |
|---|---|---|
| `gt_v1` | GT | v1 (`v2=None`) |
| `pred_v1` | first-pass | v1 (`v2=None`) |
| `pred_v2_rules` | first-pass | anatomy on, repair with `RulesDecider` + learned gap-filler, lumen off |
| `pred_v2_repair` | first-pass | anatomy on, repair with learned decider + gap-filler, lumen off |
| `pred_v2_repair_lumen` | first-pass | as above + lumen refinement |
Mask metrics (dice … false_branch_rate) compare the mask the pipeline ended with (after repair) to
GT. `stenosis_metrics` uses the trees the solver used (after lumen refinement when on), wrapped as
a `CoronaryGraph` (fragments empty). `g_gt` = `build_graph(GT)`. `min_ffr_tree0/1` = min FFR over
samples with diameter ≥ 1.5 mm of `tree_0`/`tree_1` in result.json (empty if absent).
`n_actions` = number of repair actions with result `applied`. `n_review_flags` = len(v2.review_flags)
(0 for v1 variants). Run the pipeline through `run_pipeline` with `mask` given (no nnU-Net call).
A failed case gets `status=failed`, the error text, and empty metric cells; it is never dropped.

### 13b. Stenosis error metric and guard (added in S20; supersedes `widened_rate` as a guard)

`widened_rate` is kept in `metrics.stenosis_metrics` (unchanged, for comparison with S16) but is
not fit for purpose as a guard: GT blurred by σ 0.25 mm (Dice 0.998) already scores 0.28 (S19).

**Radius definition.** The radius written to the tree is the **area-equivalent radius**
`r_eq = sqrt(A/π)` of the vessel cross-section (the solver's physics depends on lumen area).
`V2Options.radius_mode ∈ {"inscribed", "area_eq"}`, **DEFAULT "area_eq"**; `v2=None` is the v1
inscribed radius. `radius.area_eq_radii(graph, mask, spacing)` replaces every branch's `radius_mm`
by `r_eq` of the final mask from a CPR cross-section (§9 geometry: 48 px, 0.2 mm, 0.5 mm step)
perpendicular to the local tangent; component containing the slice centre, else the nearest one.
Lumen refinement (when on) writes `r_eq` itself and is applied after it.

```python
def stenosis_errors(g_gt, g_pred, lp) -> list[dict]   # one dict per GT lesion (v1 detect_lesions on g_gt)
def summarize_stenosis(rows) -> dict                  # per severity bin + down50 / down70
```
Per GT lesion: take the nearest `g_pred` point to the GT lesion's minimum point. Distance > 2 mm →
`missed=True`, other fields NaN. Otherwise, on the matched pred branch, the window of points within
±2 mm arc length of the matched point; `ds_pred = max over the window of (1 − r / r_ref)` with
`r_ref` = the pred branch's `reference_radius` (the window absorbs small position shifts).
The GT side uses the same rule on the GT branch around its lesion minimum (so `stenosis_errors(g, g)`
is exactly 0); the v1 `Lesion.ds` / `r_min_mm` are kept as `ds_lesion` / `rmin_lesion`.
Fields: `case_id` (caller), `lesion_id`, `ds_gt`, `ds_pred`, `ds_err_pp = 100·(ds_pred − ds_gt)`,
`rmin_gt`, `rmin_pred` (radius at the respective window's max-ds point), `rmin_err_mm` (signed, pred − gt),
`missed`, `bin` by `ds_gt`: mild [0.25, 0.5), moderate [0.5, 0.7), severe [0.7, 1.0]; below 0.25 →
"sub". `summarize_stenosis`: per bin n, n_missed, mean / median / mean-abs `ds_err_pp`, mean /
mean-abs `rmin_err_mm`; overall `down50` = #(ds_gt ≥ 0.5, matched, ds_pred < 0.5) with its
denominator (#(ds_gt ≥ 0.5)); `down70` likewise with 0.7.

**Guard for all FUTURE hypotheses** (replaces any widened_rate criterion): "mean `ds_err_pp` over
moderate + severe lesions does not decrease by more than 5 pp versus the comparator, and `down50`
does not increase." GT lesions for scoring are detected on the GT graph built in the same
`radius_mode` as the row being scored.

**Status of the data.** The 20-case regression set has been seen in S16, S18 and S19: results on it
are exploratory only. Confirmatory tests need a new, pre-selected, stenosis-enriched case set
(not chosen in S20).

### 13c. Hypothesis H3 (lumen v3) — pre-registered in S21, evaluated ONCE in S24

**Case set:** the 30 cases of `reports/v3_stenosis_set.json` (chosen from GT only by the rule stored
in that file; disjoint from the 20-case regression set and from the test split; never edited).
**Setup:** all radii inscribed (`radius_mode="inscribed"`), no repair, no anatomy, flow = v1
diameter law (`V2Options(anatomy=False, flow_mode="diameter_law")`). GT lesions are detected on the
GT graph built the same way (§13b `stenosis_errors`).

- Comparator `pred_v1`: first-pass mask, v1 pipeline. Treatment `pred_lumen3`: first-pass mask,
  lumen v3 refinement with the inference settings (`radius_out`, `fallback`, `sigma`) frozen in S23.
- **PRIMARY:** mean `ds_err_pp` over moderate + severe GT lesions is at least 10 pp less negative
  for `pred_lumen3` than for `pred_v1`.
- **GUARDS:** (a) `down50` does not increase; (b) false significant lesions do not increase by more
  than 5 over the 30 cases; (c) mean `|ds_err_pp|` in the mild bin does not increase by more than 5 pp.
- **SUPPORTED** only if the primary criterion and all guards hold; otherwise **FALSIFIED**.
- **Secondary (report only, no verdict):** per-case minimum FFR versus `gt_v1`; how many cases with
  `gt_v1` min FFR < 0.80 each variant also calls < 0.80; false calls (variant < 0.80 where `gt_v1` >= 0.80).

New metric `false_sig_lesions(g_gt, g_pred, lp) -> int` (`metrics.py`, S24): the number of predicted
lesions (v1 `detect_lesions` on `g_pred`) with `ds >= 0.5` whose minimum point has no GT lesion
(any `ds >= 0.25`, minimum point of a lesion detected on `g_gt`) within 5 mm.

Extra reference rows (no verdict): `pred_lumen_old` (deployed `lumen` model, old settings) and
`gt_lumen3` (GT mask + lumen v3 on GT centerlines: oracle for perfect centerlines).

---

## 14. Training (CLUSTER) — experiment section

**Hypothesis H1 (repair).** On the 20-case regression set, `pred_v2_repair` vs `pred_v1`: mean
`branch_recall` +≥ 0.02, mean `false_branch_rate` not more than +0.01, mean `widened_rate` not
higher. **Falsified** if any of the three fails.
**Hypothesis H2 (lumen).** `pred_v2_repair_lumen` has lower `mean_abs_rmin_err_mm` than
`pred_v2_repair` (mean over cases with ≥ 1 lesion) and mean `widened_rate` not higher.
**Falsified** otherwise.
Evaluation runs once (S16). Means over 20 cases; also report per-case paired differences
(median, number of cases improved/worse). Seeds: training uses 3 seeds per network; only the
deployed seed is evaluated end-to-end (deviation: evaluation cost), and the across-seed SD of the
validation metric is reported in `reports/v2_training.md`.

**Variables.** Independent: pipeline variant (§13). Dependent: §13 metrics. Controlled: data roles
(§2.2), all parameters (§3), hardware per stage.

**Baselines.** (1) Simplest: `pred_v1` (no repair). (2) "Rules-only": repair loop with a decider that
returns `extend` for every `abrupt` ending and `keep` otherwise (`RulesDecider` class in repair.py).

**Datasets** (generated on CLUSTER from `v2_train` only; `v2_train` with `case_id % 10 == 3` = val):
| Name | Generator | Content per sample |
|---|---|---|
| `gapfill` | `train/make_gapfill_data.py` | npz: `x` float16 (3,64,64,64), `y` uint8 (64,64,64), `case_id`, `type` |
| `decider_synth` | `train/make_decider_data.py` | parquet rows: `case_id, seed, region_id, kind, <31 features in FEATURE_NAMES order>, label` (`decider_real` uses `seed = -1`) |
| `lumen` | `train/make_lumen_data.py` | npz per branch: `img` float16 (N,48,48) normalised, `mask` uint8 (N,48,48) center component, `ds` float16 (N,) GT ds per slice, `case_id`, `jitter` |
| `decider_real` | `scripts/make_decider_real.py` (SPARK, S14) | parquet as decider_synth, labels from §7 real rules |

- gapfill samples (2 damage seeds per case, seed 0 and 1):
  - positive: every record of type gap, truncate, delete_branch (offset 8 mm along the record
    direction) and shrink (offset 0). Input = `extract_patch(image_out, mask_out, record point,
    record direction, offset)`; target = GT mask resampled onto the same reference grid (nearest).
  - negative: every `occlusion` record (input from image_out + mask_out, offset 8 mm; target =
    mask_out patch → nothing to add), plus per case and seed up to 2 GT `taper` endings chosen with
    the same rng (input = image + GT mask at the ending, offset 8 mm; target = GT patch).
  - `type` field: the damage type, or `"taper_neg"`.
- lumen: every GT branch ≥ 10 mm of the GT graph (`build_graph(GT)`), two versions: `clean`
  (branch points as is) and `jit` (each point + N(0, 0.3 mm) noise in U and V of the RMF, then
  `gaussian_filter1d(σ = 2 samples, axis 0, mode="nearest")`). Both: straighten image and GT mask
  along that path with `straighten_points`; `mask` = `center_component` of each GT slice; `ds` =
  1 − r/r_ref with r, r_ref from the GT branch's `radius_mm` and v1
  `reference_radius(branch.s_mm, branch.radius_mm, LesionParams())`, interpolated onto the
  straightened `s_mm` with `np.interp`.
- decider_synth: per case, seeds 0 and 1: `damage_case` → `build_graph(mask_out)` (skip the sample
  if it raises) → `propose_regions(g, image_out, mask_out, P)` → `region_features(...)` →
  `label_synthetic`. One parquet row per region.

**Training recipes** (seeds 0, 1, 2 as a Slurm array; deploy the seed with the best val metric):
| Model | Optimiser | Steps | Batch | Loss | Augment | Val metric |
|---|---|---|---|---|---|---|
| GapfillNet | AdamW lr 3e-4, wd 1e-4, cosine to 0, 500 warmup | 40 000 | 8 | weighted BCE (voxel weight 5 where target ≠ input-mask channel, else 1) + soft Dice | random flip of each axis (same flip for all 3 channels and target), intensity of channel 0 ×U(0.9,1.1) +U(−0.05,0.05) | **new-voxel Dice** on positive val samples: dice(pred∧¬inp, tgt∧¬inp) (both empty → 1), mean over samples; also log mean count of pred∧¬inp voxels on negative val samples. Every 2000 steps |
| LumenNet | same | 30 000 | 4 windows of 128 slices (pad shorter with zeros + mask out) | BCE + soft Dice; slices with GT ds ≥ 0.25 weighted ×3 | flip a/b axes, rot90 in slice plane (same for all slices), intensity as above | mean abs r_eq error (mm) on val, every 2000 steps |
| Decider | `sklearn.ensemble.HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed)` | — | — | log loss | — | macro-F1 on val cases |

**Decider fine-tuning (C08, after S14):** training rows = all `decider_synth` rows (all
`v2_train` cases, including the val ones) + `decider_real` rows. Per-row weight
`w = base × balance`, `base` = 5.0 for real rows, 1.0 for synthetic; `balance` = `n_rows / (n_labels_present × n_rows_with_that_label)`
computed over the training rows of that fit. Fit with `sample_weight=w` and `class_weight=None`
(same other hyper-parameters). Report: 5-fold `GroupKFold` over the `decider_real` rows
(groups = case_id): each fold trains on all synthetic rows + real rows of the other 4 folds and is
scored (macro-F1 over the labels present) on the held-out real rows. Deploy the model fit on all
rows with `random_state=0` as `decider` (move the previous one to `decider_synth_only/`).

**Sanity checks (must pass before full runs):** forward pass output shapes as in §8; loss at
init finite and between 0.5 and 3.0; overfit 4 fixed training samples: gap-filler new-voxel Dice
> 0.8 and lumen train Dice > 0.8 within 1000 steps (no augmentation, lr 1e-3); decider train
macro-F1 > val macro-F1 and val macro-F1 > 1/6 (better than chance). Any failure → stop that
model, BLOCKED.

**Kill criteria:** gap-filler val new-voxel Dice < 0.30 at step 10 000; lumen val r_eq error
> 0.5 mm at step 10 000 → `scancel` that seed, BLOCKED with the last 50 log lines.

**Compute budget:** per training job: 1 GPU, 16 CPUs, 64 GB RAM, wall time 24 h (or the partition
max if lower). Dataset generation: CPU array jobs, 20 shards, 4 h each.

**Logging:** `<CLUSTER_DATA>/v2/runs/<model>/seed<k>/train_log.csv` with columns
`step,loss,lr,val_metric,elapsed_s`; final `metrics.json`. Summary table in
`reports/v2_training.md`:
```markdown
| Model | Seed | Val metric | Steps | Hours | Deployed |
|---|---|---|---|---|---|
| gapfill | 0 | Dice 0.__ | 40000 | __ | yes/no |
Mean ± SD over seeds: __
```
Decider: also per-class precision/recall table on val.

**Honesty:** report every number as computed, including failed runs. Never use `v2_holdout`,
`v2_realfail` or `test` cases in training or model selection (except `v2_realfail` in C08 as
specified). Never re-run with changed settings to improve S16 numbers.

---

## 15. Owner actions summary
1. (done) TotalSegmentator academic licence obtained. After S00, run on SPARK:
   `./scripts/dk totalseg_set_license -l <your key>`. Only SPARK needs it.
2. Create a private git remote (e.g. GitHub private repo); push branch `v2` from the SPARK v2
   worktree; clone with `-b v2` on CLUSTER.
   After that, pull/push whenever a progress file says `WAITING ON OWNER: git ...`.
3. On CLUSTER, paste the cluster's Slurm instructions into `cluster/SLURM_NOTES.md` (template in
   the repo) and fill its "Required facts" table before starting the CLUSTER session.
4. Run the rsync commands in §1 when a progress file says `WAITING ON OWNER`.
5. Optional: if you want TotalSegmentator to stop sending anonymous usage statistics, edit
   `send_usage_stats` to `false` in `$TWINLOOP_DATA/totalseg/config.json` yourself (Claude must not
   open that file because it holds the licence).
