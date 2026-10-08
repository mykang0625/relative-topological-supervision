"""Vessel topology metrics, extracted unchanged from the research evaluator."""
from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import numpy as np
import scipy.ndimage as ndi
import scipy.spatial as spatial
from skimage.morphology import skeletonize


def mask_to_skeleton_graph(binary_mask: np.ndarray) -> Tuple[nx.Graph, np.ndarray]:
    """Convert binary mask skeleton into networkx graph with Euclidean distance weights."""
    skel = skeletonize(binary_mask > 0)
    coords = np.argwhere(skel)  # (N, 2) [y, x]
    if len(coords) == 0:
        return nx.Graph(), np.empty((0, 2))

    g = nx.Graph()
    coord_to_id = {tuple(c): i for i, c in enumerate(coords)}
    for i, (y, x) in enumerate(coords):
        g.add_node(i, pos=(y, x))

    # Add 8-connectivity edges
    for i, (y, x) in enumerate(coords):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = (y + dy, x + dx)
                if neighbor in coord_to_id:
                    j = coord_to_id[neighbor]
                    if i < j:
                        weight = float(np.sqrt(dy * dy + dx * dx))
                        g.add_edge(i, j, weight=weight)

    return g, coords


def compute_one_way_apls(
    g_src: nx.Graph,
    coords_src: np.ndarray,
    g_tgt: nx.Graph,
    coords_tgt: np.ndarray,
    max_snap_dist: float = 6.0,
    num_sample_pairs: int = 150,
    seed: int = 42,
) -> float:
    """Compute one-way Average Path Length Similarity from src to tgt."""
    if len(coords_src) < 2:
        return 1.0 if len(coords_tgt) < 2 else 0.0
    if len(coords_tgt) < 2:
        return 0.0

    tree_tgt = spatial.cKDTree(coords_tgt)
    dists, nearest_tgt_ids = tree_tgt.query(coords_src)
    valid_snap = dists <= max_snap_dist

    rng = np.random.RandomState(seed)
    components = [list(c) for c in nx.connected_components(g_src) if len(c) > 1]
    if not components:
        return 1.0

    pairs = []
    for comp in components:
        comp_arr = np.array(comp)
        if len(comp_arr) < 2:
            continue
        n_pairs = min(num_sample_pairs // max(1, len(components)), len(comp_arr) * (len(comp_arr) - 1) // 2)
        n_pairs = max(10, n_pairs)
        for _ in range(n_pairs):
            idx1, idx2 = rng.choice(len(comp_arr), size=2, replace=False)
            u, v = comp_arr[idx1], comp_arr[idx2]
            pairs.append((u, v))

    if not pairs:
        return 1.0

    diffs = []
    for u, v in pairs:
        try:
            d_src = nx.shortest_path_length(g_src, source=u, target=v, weight="weight")
        except nx.NetworkXNoPath:
            continue

        if d_src < 1e-4:
            continue

        if not (valid_snap[u] and valid_snap[v]):
            diffs.append(1.0)
            continue

        u_tgt = int(nearest_tgt_ids[u])
        v_tgt = int(nearest_tgt_ids[v])

        if u_tgt == v_tgt:
            d_tgt = 0.0
        else:
            try:
                d_tgt = nx.shortest_path_length(g_tgt, source=u_tgt, target=v_tgt, weight="weight")
            except nx.NetworkXNoPath:
                d_tgt = np.inf

        if np.isinf(d_tgt):
            diffs.append(1.0)
        else:
            diff_ratio = min(1.0, abs(d_src - d_tgt) / d_src)
            diffs.append(diff_ratio)

    if not diffs:
        return 1.0
    return float(1.0 - np.mean(diffs))


def compute_apls(
    mask_pred: np.ndarray,
    mask_true: np.ndarray,
    max_snap_dist: float = 6.0,
    num_sample_pairs: int = 150,
) -> float:
    """Compute symmetric Average Path Length Similarity (APLS) in [0, 1]."""
    g_pred, coords_pred = mask_to_skeleton_graph(mask_pred)
    g_true, coords_true = mask_to_skeleton_graph(mask_true)

    s_true_to_pred = compute_one_way_apls(g_true, coords_true, g_pred, coords_pred, max_snap_dist, num_sample_pairs, seed=42)
    s_pred_to_true = compute_one_way_apls(g_pred, coords_pred, g_true, coords_true, max_snap_dist, num_sample_pairs, seed=84)

    return float(0.5 * (s_true_to_pred + s_pred_to_true))


def compute_cldice(v_pred: np.ndarray, v_true: np.ndarray) -> float:
    """Compute Centerline / Skeleton Dice (clDice)."""
    s_pred = skeletonize(v_pred > 0)
    s_true = skeletonize(v_true > 0)

    tprec = np.sum(v_pred[s_true]) / max(1e-8, np.sum(s_true))
    tsens = np.sum(v_true[s_pred]) / max(1e-8, np.sum(s_pred))

    if tprec + tsens == 0:
        return 0.0
    return float(2.0 * (tprec * tsens) / (tprec + tsens))


def compute_betti_numbers(binary_mask: np.ndarray) -> Tuple[int, int]:
    """Compute (beta_0, beta_1) under 8-connectivity foreground / 4-connectivity background."""
    labeled_fg, num_fg = ndi.label(binary_mask > 0, structure=np.ones((3, 3), dtype=int))
    padded = np.pad(binary_mask > 0, pad_width=1, mode="constant", constant_values=0)
    struct_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    labeled_bg, num_bg = ndi.label(~padded, structure=struct_4)
    num_holes = max(0, num_bg - 1)
    return int(num_fg), int(num_holes)


def compute_vessel_width_stratified_metrics(
    pred_vessel: np.ndarray,
    vessel_gt: np.ndarray,
    fov_mask: np.ndarray,
    radius_threshold: float = 1.5,
) -> Dict[str, float]:
    """Stratify evaluation into thin capillaries (radius <= threshold) and large vessels.
    
    Returns recall on thin vs thick vessel masks, and recall on thin vs thick skeletons.
    """
    valid_gt = vessel_gt & fov_mask
    if np.sum(valid_gt) == 0:
        return {
            "thin_recall": 0.0,
            "thick_recall": 0.0,
            "thin_skel_recall": 0.0,
            "thick_skel_recall": 0.0,
        }
    dist = ndi.distance_transform_edt(valid_gt)
    gt_thin = valid_gt & (dist <= radius_threshold)
    gt_thick = valid_gt & (dist > radius_threshold)

    skel_gt = skeletonize(valid_gt)
    skel_thin = skel_gt & (dist <= radius_threshold)
    skel_thick = skel_gt & (dist > radius_threshold)

    thin_rec = float(np.sum(pred_vessel & gt_thin) / max(1.0, np.sum(gt_thin)))
    thick_rec = float(np.sum(pred_vessel & gt_thick) / max(1.0, np.sum(gt_thick)))

    thin_skel_rec = float(np.sum(pred_vessel[skel_thin]) / max(1.0, np.sum(skel_thin))) if np.sum(skel_thin) > 0 else 0.0
    thick_skel_rec = float(np.sum(pred_vessel[skel_thick]) / max(1.0, np.sum(skel_thick))) if np.sum(skel_thick) > 0 else 0.0

    return {
        "thin_recall": thin_rec,
        "thick_recall": thick_rec,
        "thin_skel_recall": thin_skel_rec,
        "thick_skel_recall": thick_skel_rec,
    }


def find_skeleton_junctions(skel: np.ndarray) -> np.ndarray:
    """Find branch-point coordinates (centroids) on an 8-connected skeleton."""
    if np.sum(skel) == 0:
        return np.empty((0, 2), dtype=float)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    neighbors = ndi.convolve(skel.astype(int), kernel, mode="constant", cval=0)
    branch_mask = skel & (neighbors >= 3)
    if np.sum(branch_mask) == 0:
        return np.empty((0, 2), dtype=float)
    labeled, num_c = ndi.label(branch_mask, structure=np.ones((3, 3), dtype=int))
    centroids = []
    for i in range(1, num_c + 1):
        pts = np.argwhere(labeled == i)
        centroids.append(pts.mean(axis=0))
    return np.array(centroids, dtype=float)


def compute_junction_f1(
    pred_vessel: np.ndarray,
    vessel_gt: np.ndarray,
    fov_mask: np.ndarray,
    tolerance_px: float = 3.0,
) -> Dict[str, float]:
    """Compute branch-point / junction detection Precision, Recall, and F1 within tolerance."""
    skel_gt = skeletonize(vessel_gt & fov_mask)
    skel_pred = skeletonize(pred_vessel & fov_mask)

    pts_gt = find_skeleton_junctions(skel_gt)
    pts_pred = find_skeleton_junctions(skel_pred)

    if len(pts_gt) == 0 and len(pts_pred) == 0:
        return {"junction_prec": 1.0, "junction_rec": 1.0, "junction_f1": 1.0, "num_gt_junctions": 0, "num_pred_junctions": 0}
    if len(pts_gt) == 0:
        return {"junction_prec": 0.0, "junction_rec": 0.0, "junction_f1": 0.0, "num_gt_junctions": 0, "num_pred_junctions": len(pts_pred)}
    if len(pts_pred) == 0:
        return {"junction_prec": 0.0, "junction_rec": 0.0, "junction_f1": 0.0, "num_gt_junctions": len(pts_gt), "num_pred_junctions": 0}

    dists = spatial.distance.cdist(pts_gt, pts_pred)
    matched_gt = set()
    matched_pred = set()
    indices = np.argsort(dists, axis=None)
    for idx in indices:
        r, c = divmod(int(idx), dists.shape[1])
        if dists[r, c] > tolerance_px:
            break
        if r not in matched_gt and c not in matched_pred:
            matched_gt.add(r)
            matched_pred.add(c)

    tp = len(matched_gt)
    fp = len(pts_pred) - tp
    fn = len(pts_gt) - tp
    prec = float(tp / max(1e-8, tp + fp))
    rec = float(tp / max(1e-8, tp + fn))
    f1 = float((2 * prec * rec) / max(1e-8, prec + rec))
    return {
        "junction_prec": prec,
        "junction_rec": rec,
        "junction_f1": f1,
        "num_gt_junctions": len(pts_gt),
        "num_pred_junctions": len(pts_pred),
    }
