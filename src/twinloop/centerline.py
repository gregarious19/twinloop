"""Mask -> trees of branches (SPEC §5.3).

Arrays are indexed (z, y, x); physical points are (x, y, z) mm in the image's LPS frame.
"""

import itertools
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.morphology import skeletonize

from twinloop.config import CenterlineParams

NO_TREE = "No coronary tree found in mask"
PAD = 3
# The 13 "forward" members of the 26-neighbourhood: each undirected edge is generated once.
_HALF_OFFSETS = np.array(
    [off for off in itertools.product((-1, 0, 1), repeat=3) if off > (0, 0, 0)]
)


@dataclass
class Branch:
    branch_id: str  # f"{tree_id}_b{k}", k in BFS order from root, root branch k=0
    parent_id: str | None
    points_mm: np.ndarray  # (N, 3) physical LPS coords, mm, ordered proximal -> distal
    radius_mm: np.ndarray  # (N,)
    s_mm: np.ndarray  # (N,) arc length from branch start, s[0] = 0
    children: list[str] = field(default_factory=list)


@dataclass
class VesselTree:
    tree_id: str  # "tree_0" (largest component by volume), "tree_1"
    branches: dict[str, Branch]
    root_branch_id: str
    volume_mm3: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class _RawBranch:
    """A voxel chain (graph node ids) between two critical nodes, before resampling."""

    vox: list[int]
    parent: int | None
    children: list[int] = field(default_factory=list)


def extract_trees(mask: sitk.Image, p: CenterlineParams) -> list[VesselTree]:
    arr = sitk.GetArrayFromImage(mask) > 0
    spacing = mask.GetSpacing()  # (x, y, z)
    sp = np.asarray(spacing[::-1], dtype=np.float64)  # (z, y, x) for scipy/skimage
    voxel_mm3 = float(np.prod(spacing))

    labels, n = ndimage.label(arr, structure=np.ones((3, 3, 3)))
    if n == 0:
        raise ValueError(NO_TREE)
    volumes = np.bincount(labels.ravel(), minlength=n + 1)[1:] * voxel_mm3
    order = [i for i in np.argsort(-volumes, kind="stable") if volumes[i] >= p.min_component_mm3]
    keep = [(int(i) + 1, float(volumes[i])) for i in order[: p.max_trees]]
    if not keep:
        raise ValueError(NO_TREE)

    slices = ndimage.find_objects(labels)
    to_physical = _physical_mapper(mask)
    trees: list[VesselTree] = []
    for label, volume in keep:
        sl = slices[label - 1]
        comp = np.pad(labels[sl] == label, PAD)  # bounding box + margin
        offset = np.array([s.start for s in sl]) - PAD
        tree = _build_tree(comp, offset, sp, to_physical, volume, f"tree_{len(trees)}", p)
        if tree is not None:
            trees.append(tree)
    if not trees:
        raise ValueError(NO_TREE)
    return trees


def _physical_mapper(img: sitk.Image):
    """Vectorised equivalent of TransformContinuousIndexToPhysicalPoint for (z, y, x) indices."""
    origin = np.asarray(img.GetOrigin())
    direction = np.asarray(img.GetDirection()).reshape(3, 3)
    spacing = np.asarray(img.GetSpacing())

    def to_physical(idx_zyx: np.ndarray) -> np.ndarray:
        xyz = np.asarray(idx_zyx, dtype=np.float64)[:, ::-1] * spacing
        return origin + xyz @ direction.T

    return to_physical


def _voxel_graph(coords: np.ndarray, shape: tuple[int, ...], sp: np.ndarray) -> nx.Graph:
    lin = np.ravel_multi_index(coords.T, shape)
    g = nx.Graph()
    g.add_nodes_from(range(len(lin)))
    for off in _HALF_OFFSETS:
        nb = np.ravel_multi_index((coords + off).T, shape)
        pos = np.minimum(np.searchsorted(lin, nb), len(lin) - 1)
        dist = float(np.linalg.norm(off * sp))
        for i in np.nonzero(lin[pos] == nb)[0]:
            g.add_edge(int(i), int(pos[i]), weight=dist)
    return g


def _pick_root(tree: nx.Graph, rad: np.ndarray) -> int:
    """Degree-1 voxel whose first 10 voxels along the path have the largest mean radius."""
    best, best_r = -1, -1.0
    for leaf in sorted(n for n in tree if tree.degree(n) == 1):
        path, prev, cur = [leaf], leaf, next(iter(tree[leaf]))
        while len(path) < 10 and tree.degree(cur) == 2:  # stop at a junction or leaf
            path.append(cur)
            prev, cur = cur, next(x for x in tree[cur] if x != prev)
        mean_r = float(np.mean(rad[path]))
        if mean_r > best_r:
            best, best_r = leaf, mean_r
    return best


def _contract_junctions(tree: nx.Graph, phys: np.ndarray) -> nx.Graph:
    """Merge adjacent degree>=3 voxels into one node at the voxel nearest the cluster centroid."""
    junctions = {n for n in tree if tree.degree(n) >= 3}
    rep_of: dict[int, int] = {}
    for cluster in nx.connected_components(tree.subgraph(junctions)):
        nodes = sorted(cluster)
        centroid = phys[nodes].mean(axis=0)
        rep = min(nodes, key=lambda n: (float(np.linalg.norm(phys[n] - centroid)), n))
        rep_of.update({n: rep for n in nodes})
    h = nx.Graph()
    for u, v, d in tree.edges(data=True):
        a, b = rep_of.get(u, u), rep_of.get(v, v)
        if a != b:
            h.add_edge(a, b, weight=d["weight"])
    return h


def _trace_branches(h: nx.Graph, root: int) -> dict[int, _RawBranch]:
    def chain(start: int, first: int) -> list[int]:
        path, prev, cur = [start, first], start, first
        while h.degree(cur) == 2:
            nxt = next(x for x in h[cur] if x != prev)
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    branches = {0: _RawBranch(chain(root, next(iter(h[root]))), None)}
    queue = [0]
    for bid in queue:  # the list grows while iterating (BFS)
        vox = branches[bid].vox
        end = vox[-1]
        if h.degree(end) < 3:
            continue
        for nb in sorted(x for x in h[end] if x != vox[-2]):
            cid = len(branches)
            branches[cid] = _RawBranch(chain(end, nb), bid)
            branches[bid].children.append(cid)
            queue.append(cid)
    return branches


def _chain_length(vox: list[int], phys: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(phys[vox], axis=0), axis=1).sum())


def _prune_spurs(branches: dict[int, _RawBranch], phys, rad, p: CenterlineParams) -> None:
    for _ in range(p.prune_passes):
        for bid in list(branches):
            b = branches[bid]
            if b.parent is None or b.children:
                continue
            limit = max(p.spur_min_mm, p.spur_radius_factor * float(rad[b.vox[0]]))
            if _chain_length(b.vox, phys) < limit:
                branches[b.parent].children.remove(bid)
                del branches[bid]
        merged = True
        while merged:
            merged = False
            for bid, b in list(branches.items()):
                if bid in branches and len(b.children) == 1:
                    child = branches.pop(b.children[0])
                    b.vox = b.vox + child.vox[1:]
                    b.children = child.children
                    for g in b.children:
                        branches[g].parent = bid
                    merged = True


def _resample(vox, phys, rad, p: CenterlineParams):
    """Smooth, resample by arc length, and return (points, radius, s) for one voxel chain."""
    pts, r = phys[vox], rad[vox]
    if len(pts) >= 3:
        step = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).mean())
        smooth = ndimage.gaussian_filter1d(pts, p.position_sigma_mm / step, axis=0, mode="nearest")
        smooth[0], smooth[-1] = pts[0], pts[-1]
        pts = smooth
    s_old = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total = float(s_old[-1])
    s_new = np.arange(0.0, total, p.resample_mm)
    if total - s_new[-1] > 1e-9:
        s_new = np.append(s_new, total)  # always include the last point
    else:
        s_new[-1] = total
    new_pts = np.stack([np.interp(s_new, s_old, pts[:, k]) for k in range(3)], axis=1)
    new_r = np.interp(s_new, s_old, r)
    new_r = ndimage.gaussian_filter1d(new_r, p.radius_sigma_mm / p.resample_mm, mode="nearest")
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(new_pts, axis=0), axis=1))])
    return new_pts, new_r, s


def _build_tree(comp, offset, sp, to_physical, volume, tree_id, p: CenterlineParams):
    warnings: list[str] = []
    edt = ndimage.distance_transform_edt(comp, sampling=sp)
    if p.radius_half_voxel_correction:
        radius = np.maximum(edt - 0.5 * sp.mean(), 0.5 * sp.min())
    else:
        radius = edt

    coords = np.argwhere(skeletonize(comp))  # row-major order == sorted by linear index
    if len(coords) < 2:
        return None
    graph = _voxel_graph(coords, comp.shape, sp)
    pieces = sorted(nx.connected_components(graph), key=lambda c: (-len(c), min(c)))
    if len(pieces) > 1:
        lost = sum(graph.subgraph(c).size(weight="weight") for c in pieces[1:])
        warnings.append(f"Discarded {len(pieces) - 1} disconnected skeleton pieces ({lost:.1f} mm)")
    tree = nx.minimum_spanning_tree(graph.subgraph(pieces[0]), weight="weight")
    if tree.number_of_nodes() < 2:
        return None

    rad = radius[tuple(coords.T)]
    phys = to_physical(coords + offset)
    root = _pick_root(tree, rad)
    h = _contract_junctions(tree, phys)
    if h.number_of_edges() == 0:
        return None
    raw = _trace_branches(h, root)
    _prune_spurs(raw, phys, rad, p)

    # Renumber in BFS order from the root branch and convert to physical, resampled branches.
    bfs = [min(raw, key=lambda k: (raw[k].parent is not None, k))]
    for k in bfs:
        bfs.extend(raw[k].children)
    names = {k: f"{tree_id}_b{i}" for i, k in enumerate(bfs)}
    branches: dict[str, Branch] = {}
    for k in bfs:
        pts, r, s = _resample(raw[k].vox, phys, rad, p)
        parent = raw[k].parent
        branches[names[k]] = Branch(
            names[k],
            names[parent] if parent is not None else None,
            pts,
            r,
            s,
            [names[c] for c in raw[k].children],
        )
    return VesselTree(tree_id, branches, names[bfs[0]], volume, warnings)
