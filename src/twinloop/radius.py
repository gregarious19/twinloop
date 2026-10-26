"""Area-equivalent branch radii from the final mask (SPEC_V2 §13b)."""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from twinloop.centerline import VesselTree
from twinloop.config import CenterlineParams
from twinloop.config_v2 import CprParams
from twinloop.cpr import center_component, r_eq_mm, straighten_points

MIN_BRANCH_MM = 1.0  # shorter chains keep their inscribed radii
JUNCTION_RADIUS_FACTOR = 1.5  # exclusion zone = factor * local inscribed radius + margin
JUNCTION_MARGIN_MM = 1.0


def area_eq_radii(
    graph,
    mask: sitk.Image,
    cp: CprParams | None = None,
    cl: CenterlineParams | None = None,
    exclude_junctions: bool = False,
) -> int:
    """Replace every branch's ``radius_mm`` in place by sqrt(A/pi) of the mask cross-section.

    ``graph`` is a CoronaryGraph or a list of VesselTree; the grid (spacing, origin, direction)
    comes from ``mask``. Per centerline point the mask is sampled on a plane perpendicular to the
    local tangent (same geometry as the lumen model), the component containing the slice centre
    is taken (else the nearest one). Slices without any mask keep the inscribed radius. The
    result is smoothed like v1 smooths its radii (Gaussian, ``radius_sigma_mm``). Returns the
    number of branches changed.

    With ``exclude_junctions`` (S22) slices closer to a junction (branch start that has a parent,
    branch end that has children) than 1.5 x the local inscribed radius + 1.0 mm are not measured
    (the cross-section there includes the side branch); r is interpolated linearly across them
    from the remaining slices (inscribed radius if none remain).
    """
    cp = cp or CprParams()
    cl = cl or CenterlineParams()
    trees: list[VesselTree] = list(getattr(graph, "trees", graph))
    changed = 0
    for tree in trees:
        for b in tree.branches.values():
            if len(b.s_mm) < 2 or b.s_mm[-1] < MIN_BRANCH_MM:
                continue
            res = straighten_points(mask, mask, b.points_mm, cp)
            r_ins = np.interp(res["s_mm"], b.s_mm, b.radius_mm)
            r = r_ins.copy()
            for k, m in enumerate(res["mask"]):
                comp = center_component(m)
                if comp.any():
                    r[k] = r_eq_mm(comp, cp.pixel_mm)
            if exclude_junctions:
                s_k = res["s_mm"]
                d_junc = np.full(len(s_k), np.inf)
                if b.parent_id is not None:
                    d_junc = np.minimum(d_junc, s_k - s_k[0])
                if b.children:
                    d_junc = np.minimum(d_junc, s_k[-1] - s_k)
                keep = d_junc >= JUNCTION_RADIUS_FACTOR * r_ins + JUNCTION_MARGIN_MM
                if keep.any():
                    r = np.interp(s_k, s_k[keep], r[keep])
                else:
                    r = r_ins.copy()
            r = np.interp(b.s_mm, res["s_mm"], r)
            b.radius_mm = ndimage.gaussian_filter1d(
                r, cl.radius_sigma_mm / cl.resample_mm, mode="nearest"
            )
            changed += 1
    return changed
