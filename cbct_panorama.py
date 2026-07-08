"""
CBCT → Panoramic (single-curve CMPR).

Fits one 2-D dental-arch spline in the axial plane through every tooth centroid
in the segmentation, then samples the raw CBCT along a tall vertical slab
following that spline. The segmentation is used only to place the spline and to
auto-size the vertical and slab extents — projection samples come from the raw
CBCT, unmasked. Output: one PNG covering both arches as a continuous image.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image
from scipy.interpolate import splev, splprep
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree


def _plt():
    """Lazily import matplotlib (Agg) — only the --debug plots need it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


UPPER_PERMANENT_LABELS = tuple(range(11, 19)) + tuple(range(21, 29))
LOWER_PERMANENT_LABELS = tuple(range(31, 39)) + tuple(range(41, 49))
# Deciduous (primary) FDI: upper-right 51-55, upper-left 61-65,
# lower-left 71-75, lower-right 81-85. Including these lets the arch fit work on
# mixed / primary dentition cases (e.g. bundled cases 44 and 60) instead of
# collapsing onto whatever few permanent teeth are present.
UPPER_DECIDUOUS_LABELS = tuple(range(51, 56)) + tuple(range(61, 66))
LOWER_DECIDUOUS_LABELS = tuple(range(71, 76)) + tuple(range(81, 86))
UPPER_TOOTH_LABELS = UPPER_PERMANENT_LABELS + UPPER_DECIDUOUS_LABELS
LOWER_TOOTH_LABELS = LOWER_PERMANENT_LABELS + LOWER_DECIDUOUS_LABELS
UPPER_PULP_LABELS = tuple(label + 100 for label in UPPER_PERMANENT_LABELS)
LOWER_PULP_LABELS = tuple(label + 100 for label in LOWER_PERMANENT_LABELS)
UPPER_JAW_LABELS = (2,)
LOWER_JAW_LABELS = (1,)
LOWER_CANAL_LABELS = (3, 4, 103, 104, 105)
UPPER_CONTEXT_LABELS = (5, 6)
ALL_TOOTH_LABELS = UPPER_TOOTH_LABELS + LOWER_TOOTH_LABELS

# Canonical arch position 0..15, patient-right molar → patient-left molar.
# Upper and lower teeth at the same position are averaged to produce the
# unified arch anchor for the spline.
ARCH_POSITION_ORDER: tuple[tuple[int, int], ...] = (
    (18, 48), (17, 47), (16, 46), (15, 45), (14, 44), (13, 43), (12, 42), (11, 41),
    (21, 31), (22, 32), (23, 33), (24, 34), (25, 35), (26, 36), (27, 37), (28, 38),
)
LABEL_TO_ARCH_POSITION: dict[int, int] = {
    label: index for index, pair in enumerate(ARCH_POSITION_ORDER) for label in pair
}
# Map each deciduous tooth onto the nearest permanent arch slot so it anchors
# the curve at the right place along the arch. A deciduous tooth and a permanent
# tooth sharing a slot are voxel-weighted-averaged (mixed dentition).
#   central(x1)->central slot, lateral(x2)->lateral, canine(x3)->canine,
#   1st molar(x4)->1st premolar slot, 2nd molar(x5)->2nd premolar slot.
_DECIDUOUS_TO_ARCH_POSITION: dict[int, int] = {
    51: 7, 52: 6, 53: 5, 54: 4, 55: 3,   # upper right
    61: 8, 62: 9, 63: 10, 64: 11, 65: 12,  # upper left
    71: 8, 72: 9, 73: 10, 74: 11, 75: 12,  # lower left
    81: 7, 82: 6, 83: 5, 84: 4, 85: 3,   # lower right
}
LABEL_TO_ARCH_POSITION.update(_DECIDUOUS_TO_ARCH_POSITION)
EXTENT_LABELS = (
    UPPER_JAW_LABELS
    + LOWER_JAW_LABELS
    + UPPER_CONTEXT_LABELS
    + LOWER_CANAL_LABELS
    + ALL_TOOTH_LABELS
    + UPPER_PULP_LABELS
    + LOWER_PULP_LABELS
)


@dataclass(frozen=True)
class PanoramaConfig:
    min_tooth_voxels: int = 50
    spline_resolution: int = 1200
    spline_smoothing: float | None = None   # None → auto from arc length
    plane_resolution_mm: float = 0.2
    slab_half_width_mm: float = 9.0
    vertical_margin_mm: float = 12.0
    endpoint_extension_mm: float = 30.0     # sweep back to the condyles/TMJ
    endpoint_slab_gain: float = 1.6         # trough widening at the posterior tips

    # -- compute backend (GPU only) --
    device: str = "auto"                    # auto | cuda — both require a visible GPU

    # -- HU → linear attenuation (mu) transfer --
    # Piecewise, continuous, monotonic. Air/soft tissue keep a small non-zero
    # floor so sinuses/airway read as grey (not black); the bone band is linear
    # to preserve trabecular texture; the dense tail (enamel/metal) is
    # log-compressed so it cannot saturate and erase cortical/bone contrast.
    mu_air: float = 0.002
    hu_soft_knee: float = 200.0             # soft-tissue → bone boundary
    mu_soft_knee: float = 0.020            # mu at hu_soft_knee
    hu_dense_knee: float = 2500.0          # bone → dense (enamel/metal) boundary
    mu_bone_hi: float = 0.85               # mu at hu_dense_knee
    dense_scale_hu: float = 1500.0         # log1p scale of the dense tail
    dense_gain: float = 0.15               # log1p gain of the dense tail

    # -- along-U path integral --
    # Accumulated attenuation A = sum_u w(u)·mu (film-positive: dense = bright).
    # w(u) = (1-beta) + beta·cos^2 : beta=1 → sharp/shallow, beta=0 → deep/flat.
    projection_beta: float = 0.55

    # -- tone mapping --
    intensity_clip_low_pct: float = 1.0
    intensity_clip_high_pct: float = 99.5
    tone_gamma: float = 0.75               # <1 lifts shadows/mid (main lever)
    local_contrast_sigma_px: float = 50.0
    local_contrast_target_std: float = 0.18
    local_contrast_gain_min: float = 0.5
    local_contrast_gain_max: float = 2.5
    apply_local_contrast: bool = True
    unsharp_amount: float = 0.5
    unsharp_sigma_px: float = 3.5

    # -- canal emphasis --
    # The mandibular canal is a ~2 mm lucent tube; averaged over an 18 mm slab it
    # vanishes. When on, its label modulates the grayscale μ (μ *= 1-strength
    # inside the projected canal mask) so it reads as a dark corticated tube.
    emphasize_canal: bool = True
    canal_strength: float = 0.5

    crop_oob_threshold: float = 0.75
    crop_margin_rows: int = 6
    flip_horizontal: bool = True
    dual_arch_offset: bool = False   # single unified focal trough (OPG-like) when False
    chunk_columns: int = 256
    terminal_gap_outlier_factor: float = 2.4
    terminal_gap_outlier_min_mm: float = 20.0
    save_debug: bool = False
    max_extent_samples: int = 300_000
    # Overlay: tint segmentation structures on the grayscale base.
    overlay: bool = False


@dataclass(frozen=True)
class ToothCentroid:
    label: int
    arch: str
    voxel_count: int
    voxel: tuple[float, float, float]
    world: tuple[float, float, float]


@dataclass
class ArchGeometry:
    # Column index ``c`` always refers to a position on the *unified* arch curve
    # (the same curve the original single-arch pipeline used), so columns remain
    # in correspondence across the whole panorama. The dual-arch behavior is
    # encoded as per-column *lateral offsets* along ``width_axis``: at column c,
    # the upper arch sits ``lateral_offset_upper_mm[c]`` away from the unified
    # curve and the lower arch sits ``lateral_offset_lower_mm[c]`` away — both
    # measured in the axial plane. ``sample_project_torch`` blends these two
    # offsets by V so upper-tooth-band rows sample the upper-arch position and
    # lower-tooth-band rows sample the lower-arch position.
    centerline_world: np.ndarray         # (N, 3) unified curve, in 3D
    tangent: np.ndarray                  # (N, 3) unit, in axial plane
    width_axis: np.ndarray               # (N, 3) unit, in axial plane (perpendicular)
    vertical_axis: np.ndarray            # (3,)   supero-inferior
    lateral_offset_upper_mm: np.ndarray  # (N,)
    lateral_offset_lower_mm: np.ndarray  # (N,)
    v_upper_mm: float                    # mean V of upper arch (world)
    v_lower_mm: float                    # mean V of lower arch
    residuals_mm: dict[str, Any]
    vertical_range_mm: tuple[float, float]
    # Per-column multiplier on the bucco-lingual trough half-width. 1.0 over the
    # dentition, ramping up toward the posterior tips so residual drift off the
    # curving mandible still captures the body/ramus.
    column_slab_gain: np.ndarray
    # Per-arch curves kept around so debug plots can show them; not used by the
    # main sampler.
    upper_centerline_world: np.ndarray
    lower_centerline_world: np.ndarray


# -- generic helpers ---------------------------------------------------------


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > eps)


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _as_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    flat = points.reshape(-1, 3)
    out = nib.affines.apply_affine(affine, flat)
    return out.reshape(points.shape)


def _spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(np.linalg.norm(affine[:3, axis])) for axis in range(3))


def _grid_values(min_value: float, max_value: float, step: float) -> np.ndarray:
    count = int(math.floor((max_value - min_value) / step + 0.5)) + 1
    return min_value + np.arange(count, dtype=np.float64) * step


# -- IO ----------------------------------------------------------------------


def load_nifti_pair(raw_path: Path, seg_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    raw_img = nib.load(str(raw_path))
    seg_img = nib.load(str(seg_path))

    if raw_img.shape != seg_img.shape:
        raise ValueError(f"Shape mismatch: raw {raw_img.shape}, segmentation {seg_img.shape}")
    if not np.allclose(raw_img.affine, seg_img.affine, atol=1e-4):
        raise ValueError("Raw and segmentation affines differ; resampling is required before reconstruction")

    raw = np.asanyarray(raw_img.dataobj).astype(np.float32, copy=False)
    seg = np.asanyarray(seg_img.dataobj)
    if not np.issubdtype(seg.dtype, np.integer):
        seg = np.rint(seg).astype(np.int16)
    else:
        seg = seg.astype(np.int16, copy=False)

    meta = {
        "raw_path": str(raw_path),
        "segmentation_path": str(seg_path),
        "shape": tuple(int(v) for v in raw.shape),
        "spacing_mm": _spacing_from_affine(raw_img.affine),
        "affine": raw_img.affine,
        "raw_dtype": str(raw_img.get_data_dtype()),
        "segmentation_dtype": str(seg_img.get_data_dtype()),
    }
    return raw, seg, raw_img.affine.astype(np.float64), meta


# -- segmentation-derived geometry ------------------------------------------


def extract_tooth_centroids(seg: np.ndarray, affine: np.ndarray, config: PanoramaConfig) -> list[ToothCentroid]:
    centroids: list[ToothCentroid] = []
    label_to_arch = {label: "upper" for label in UPPER_TOOTH_LABELS}
    label_to_arch.update({label: "lower" for label in LOWER_TOOTH_LABELS})
    for label, arch in sorted(label_to_arch.items()):
        coords = np.argwhere(seg == label)
        if coords.shape[0] < config.min_tooth_voxels:
            continue
        centroid_voxel = coords.mean(axis=0)
        centroid_world = _apply_affine(affine, centroid_voxel)
        centroids.append(
            ToothCentroid(
                label=int(label),
                arch=arch,
                voxel_count=int(coords.shape[0]),
                voxel=tuple(float(v) for v in centroid_voxel),
                world=tuple(float(v) for v in centroid_world),
            )
        )
    return centroids


def estimate_down_axis(
    centroids: list[ToothCentroid],
    affine: np.ndarray,
    seg: np.ndarray | None = None,
) -> np.ndarray:
    # Supero-inferior axis = normal of the occlusal plane. The tooth centroids
    # of both arches form a thin, roughly planar slab (arch width ~50 mm, depth
    # ~40 mm, SI thickness ~10 mm), so the smallest principal axis of that cloud
    # is the occlusal-plane normal. Using the PCA normal (rather than just
    # mean(lower) − mean(upper)) *levels* the reformat: teeth land at a
    # consistent image row and the panorama comes out horizontal instead of
    # tilted with the patient's head pose. The lower−upper vector only fixes the
    # sign (down points from maxilla toward mandible) and is the fallback.
    upper = np.array([c.world for c in centroids if c.arch == "upper"], dtype=np.float64)
    lower = np.array([c.world for c in centroids if c.arch == "lower"], dtype=np.float64)
    all_pts = np.array([c.world for c in centroids], dtype=np.float64)
    weights = np.array([max(c.voxel_count, 1) for c in centroids], dtype=np.float64)

    # Scanner z is the anatomical prior: CBCT patients are positioned upright, so
    # true supero-inferior sits within ~15° of it. mean(lower)−mean(upper) is a
    # weak sign hint but is itself tilted 20–47° by molar asymmetry, so it is NOT
    # used as the axis — only to orient the sign toward "inferior".
    z = _normalize(affine[:3, 2])
    sign_ref = None
    if len(upper) and len(lower):
        sign_ref = lower.mean(axis=0) - upper.mean(axis=0)

    def _orient(v: np.ndarray) -> np.ndarray:
        if sign_ref is not None and np.dot(v, sign_ref) < 0:
            return -v
        if sign_ref is None and np.dot(v, z) < 0:
            return -v
        return v

    if len(all_pts) >= 4:
        mean = np.average(all_pts, axis=0, weights=weights)
        centered = all_pts - mean
        cov = (centered * weights[:, None]).T @ centered / weights.sum()
        _evals, evecs = np.linalg.eigh(cov)
        normal = _normalize(evecs[:, 0])  # smallest eigenvalue → occlusal normal
        # Accept the PCA normal only if it agrees with the scanner axis (guards
        # against a degenerate cloud picking an in-plane direction). This levels
        # the reformat: the occlusal plane becomes horizontal.
        if abs(float(np.dot(normal, z))) > math.cos(math.radians(35.0)):
            return _orient(normal)

    if sign_ref is not None and np.linalg.norm(sign_ref) > 1e-6:
        return _normalize(sign_ref)
    return _orient(z)


def _project_to_axial(points: np.ndarray, down_axis: np.ndarray) -> np.ndarray:
    """Drop the supero-inferior component so that every point lies on the same axial plane."""
    along = points @ down_axis
    return points - along[:, None] * down_axis[None, :]


def _build_axial_basis(down_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fallback = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(fallback, down_axis))) > 0.9:
        fallback = np.array([0.0, 1.0, 0.0])
    e1 = _normalize(fallback - float(np.dot(fallback, down_axis)) * down_axis)
    e2 = _normalize(np.cross(down_axis, e1))
    return e1, e2


def _drop_terminal_gap_outliers(points: np.ndarray, config: PanoramaConfig) -> tuple[np.ndarray, int, int]:
    if len(points) < 5:
        return points, 0, 0
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if distances.size == 0:
        return points, 0, 0
    median_spacing = float(np.median(distances))
    if not np.isfinite(median_spacing) or median_spacing <= 1e-6:
        return points, 0, 0
    threshold = max(
        float(config.terminal_gap_outlier_min_mm),
        float(config.terminal_gap_outlier_factor) * median_spacing,
    )
    start = 1 if float(distances[0]) > threshold else 0
    end = len(points)
    if end - start >= 5 and float(distances[-1]) > threshold:
        end -= 1
    return points[start:end], start, len(points) - end


def _anchors_by_arch_position(
    centroids: list[ToothCentroid], down_axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build one voxel-weighted anchor per FDI position.

    For every position that has at least one segmented tooth, the anchor is the
    voxel-count-weighted mean of the present teeth's world centroids. Under-
    segmented teeth (e.g. peg-shaped remnants) therefore contribute less than
    full-size co-located teeth, instead of dragging the anchor toward their
    noisy centre. Returns (positions, anchors_world) already projected to the
    axial plane so the spline lives at the mean occlusal height set by
    ``fit_unified_arch``.
    """
    # Per slot, split entries by dentition. In mixed dentition a deciduous tooth
    # and its permanent successor (a bud deep in bone) share a slot but sit at
    # very different places — averaging them displaces the anchor. So prefer
    # permanent teeth where present and only fall back to deciduous otherwise.
    by_position: dict[int, dict[str, list[tuple[np.ndarray, float]]]] = {}
    for c in centroids:
        pos = LABEL_TO_ARCH_POSITION.get(int(c.label))
        if pos is None:
            continue
        kind = "permanent" if 11 <= int(c.label) <= 48 else "deciduous"
        by_position.setdefault(pos, {"permanent": [], "deciduous": []})[kind].append(
            (np.asarray(c.world, dtype=np.float64), float(max(c.voxel_count, 1)))
        )
    if not by_position:
        return np.empty((0,), dtype=np.int64), np.empty((0, 3), dtype=np.float64)
    positions = np.array(sorted(by_position.keys()), dtype=np.int64)
    anchors = []
    for pos in positions:
        groups = by_position[int(pos)]
        entries = groups["permanent"] if groups["permanent"] else groups["deciduous"]
        coords = np.stack([e[0] for e in entries], axis=0)
        weights = np.array([e[1] for e in entries], dtype=np.float64)
        anchors.append((coords * weights[:, None]).sum(axis=0) / weights.sum())
    anchors_world = _project_to_axial(np.array(anchors, dtype=np.float64), down_axis)
    return positions, anchors_world


def _patch_outlier_anchors(
    positions: np.ndarray, pts_2d: np.ndarray, k_mad: float = 4.0, min_threshold_mm: float = 4.0
) -> tuple[np.ndarray, list[int]]:
    """Replace anchors whose axial position deviates from a global smoothing
    spline through all anchors by more than ``k_mad`` × MAD of the residuals.

    A degree-3 smoothing spline (UnivariateSpline) is fit per axial dimension
    against the anchor *position index* (0..15 across the FDI sequence). Because
    every anchor is included in the fit, endpoints don't suffer the
    extrapolation pathology that leave-one-out parabolas show on U-shaped
    arches. With strong smoothing, a single mis-positioned anchor (e.g. F_001's
    tiny tooth 27 leaning lingually) shifts the curve only marginally, so its
    residual stands out and the rule patches it cleanly. ``min_threshold_mm``
    keeps clean fits from flagging false outliers when the global MAD is
    already small.
    """
    if len(positions) < 5:
        return pts_2d, []

    x = positions.astype(np.float64)
    span = float(x[-1] - x[0])
    # Strong smoothing target: the spline shouldn't track per-anchor noise.
    # s ≈ N × (1.5 mm)² keeps the fit close enough to the data while ignoring
    # millimeter-scale jitter; tuned empirically against the bundled cases.
    smoothing_s = max(1.0, len(x) * 1.5 ** 2)
    k = min(3, len(x) - 1)
    try:
        from scipy.interpolate import UnivariateSpline
        sp_e1 = UnivariateSpline(x, pts_2d[:, 0], k=k, s=smoothing_s)
        sp_e2 = UnivariateSpline(x, pts_2d[:, 1], k=k, s=smoothing_s)
        pred = np.column_stack([sp_e1(x), sp_e2(x)])
    except Exception:
        return pts_2d, []

    resid = np.linalg.norm(pts_2d - pred, axis=1)
    median_resid = float(np.median(resid))
    mad = float(np.median(np.abs(resid - median_resid))) * 1.4826
    threshold = max(float(k_mad) * mad, float(min_threshold_mm))

    new_pts = pts_2d.copy()
    replaced: list[int] = []
    for i in range(len(positions)):
        if resid[i] > threshold:
            new_pts[i] = pred[i]
            replaced.append(int(positions[i]))
    return new_pts, replaced


@dataclass
class _BaseArchFit:
    centerline: np.ndarray   # (N, 3) world, V offset already applied
    tangent: np.ndarray      # (N, 3) unit, in axial plane
    mean_step_mm: float
    mean_v_world: float
    residuals: dict[str, Any]
    centroid_world: np.ndarray
    # Fitted B-spline + axial basis, kept so the posterior sweep can be
    # extended *along the arch's own curvature* (splev past u∈[0,1]) instead of
    # flying off on a straight terminal tangent.
    tck: Any
    e1: np.ndarray
    e2: np.ndarray
    du: float                # u-step between interior samples (1/(res-1))


def _fit_arch_base(
    centroids_subset: list[ToothCentroid],
    down_axis: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    config: PanoramaConfig,
) -> _BaseArchFit | None:
    if len(centroids_subset) < 4:
        return None

    positions, anchors_world = _anchors_by_arch_position(centroids_subset, down_axis)
    if len(positions) < 4:
        return None

    pts_2d = np.column_stack([anchors_world @ e1, anchors_world @ e2])
    # Order anchors by sweep angle around the arch centre (robust to mixed
    # dentition, where FDI-slot order can zig-zag). Rotate the sequence so it
    # starts just after the largest angular gap — that gap is the open, poster-
    # ior side of the U, so the walk runs cleanly molar → incisors → molar.
    if len(pts_2d) >= 4:
        centre = pts_2d.mean(axis=0)
        ang = np.arctan2(pts_2d[:, 1] - centre[1], pts_2d[:, 0] - centre[0])
        order = np.argsort(ang)
        ang_sorted = ang[order]
        gaps = np.diff(np.concatenate([ang_sorted, ang_sorted[:1] + 2 * np.pi]))
        start = int(np.argmax(gaps)) + 1
        order = np.roll(order, -start)
        pts_2d = pts_2d[order]
        positions = positions[order]
    # Patch outliers against a monotonic arch-order rank (post-sort FDI slots are
    # no longer monotonic, and UnivariateSpline needs strictly increasing x).
    rank = np.arange(len(pts_2d), dtype=np.int64)
    pts_2d, _replaced_rank = _patch_outlier_anchors(rank, pts_2d)
    replaced_positions = [int(positions[r]) for r in _replaced_rank if r < len(positions)]

    kept_indices = [0]
    for i in range(1, len(pts_2d)):
        if np.linalg.norm(pts_2d[i] - pts_2d[kept_indices[-1]]) >= 0.5:
            kept_indices.append(i)
    pts_2d = pts_2d[kept_indices]
    kept_positions = positions[kept_indices]

    trimmed, dropped_start, dropped_end = _drop_terminal_gap_outliers(pts_2d, config)
    if len(trimmed) >= 5:
        pts_2d = trimmed
        kept_positions = kept_positions[dropped_start: len(kept_positions) - dropped_end]

    arc_length = float(np.sum(np.linalg.norm(np.diff(pts_2d, axis=0), axis=1)))
    smoothing = max(1.0, arc_length / 40.0) if config.spline_smoothing is None else float(config.spline_smoothing)

    k = min(3, len(pts_2d) - 1)
    tck, _ = splprep(pts_2d.T, s=smoothing, k=k)
    u_eval = np.linspace(0.0, 1.0, int(config.spline_resolution))
    curve_2d = np.array(splev(u_eval, tck), dtype=np.float64).T
    deriv_2d = np.array(splev(u_eval, tck, der=1), dtype=np.float64).T

    centerline = curve_2d[:, 0:1] * e1[None, :] + curve_2d[:, 1:2] * e2[None, :]
    centroid_world = np.array([c.world for c in centroids_subset], dtype=np.float64)
    mean_v = float(np.mean(centroid_world @ down_axis))
    centerline = centerline + mean_v * down_axis[None, :]
    tangent = _normalize(deriv_2d[:, 0:1] * e1[None, :] + deriv_2d[:, 1:2] * e2[None, :])

    mean_step = float(np.mean(np.linalg.norm(np.diff(centerline, axis=0), axis=1)))
    if not np.isfinite(mean_step) or mean_step <= 1e-6:
        mean_step = float(arc_length) / max(len(centerline) - 1, 1)

    residuals = {
        "dropped_terminal_outliers": int(dropped_start + dropped_end),
        "replaced_positions": [int(p) for p in replaced_positions],
        "anchor_positions": [int(p) for p in kept_positions],
        "arc_length_mm": arc_length,
        "smoothing": float(smoothing),
    }
    return _BaseArchFit(
        centerline=centerline,
        tangent=tangent,
        mean_step_mm=mean_step,
        mean_v_world=mean_v,
        residuals=residuals,
        centroid_world=centroid_world,
        tck=tck,
        e1=e1,
        e2=e2,
        du=1.0 / max(int(config.spline_resolution) - 1, 1),
    )


def _extend_with_count(
    base: _BaseArchFit, down_axis: np.ndarray, n_ext: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extend the arch past both terminal molars by continuing the *fitted
    B-spline's curvature* (evaluate ``splev`` at u<0 and u>1) rather than a
    straight terminal tangent. The posterior mandible curves lingually toward
    the ramus, so a straight extrapolation drifts off the bone and the body
    fades on the left/right; following the spline keeps the trough on it.
    """
    centerline, tangent = base.centerline, base.tangent
    if n_ext <= 0 or len(centerline) < 2 or base.du <= 0:
        return centerline, tangent

    vertical_axis = _normalize(down_axis)
    mean_v = base.mean_v_world
    du = base.du

    def _eval(u_ext: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c2d = np.array(splev(u_ext, base.tck), dtype=np.float64).T          # (n,2)
        d2d = np.array(splev(u_ext, base.tck, der=1), dtype=np.float64).T   # (n,2)
        c3d = c2d[:, 0:1] * base.e1[None, :] + c2d[:, 1:2] * base.e2[None, :]
        c3d = c3d + mean_v * vertical_axis[None, :]
        t3d = _normalize(d2d[:, 0:1] * base.e1[None, :] + d2d[:, 1:2] * base.e2[None, :])
        return c3d, t3d

    # u grows toward the terminal molar at u=1 and u=0; step outward by du.
    pre_u = (np.arange(n_ext, dtype=np.float64) + 1.0)[::-1] * (-du)        # -n·du … -du
    post_u = (np.arange(n_ext, dtype=np.float64) + 1.0) * du + 1.0          # 1+du … 1+n·du
    pre_c, pre_t = _eval(pre_u)
    post_c, post_t = _eval(post_u)
    centerline = np.vstack([pre_c, centerline, post_c])
    tangent = np.vstack([pre_t, tangent, post_t])
    return centerline, tangent


def _finalize_arch(
    base: _BaseArchFit,
    down_axis: np.ndarray,
    n_ext: int,
    step_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    centerline, tangent = _extend_with_count(base, down_axis, n_ext)
    vertical_axis = _normalize(down_axis)
    width_axis = _normalize(np.cross(vertical_axis[None, :].repeat(len(tangent), axis=0), tangent))

    centroid_axial = _project_to_axial(base.centroid_world, vertical_axis)
    centerline_axial = _project_to_axial(centerline, vertical_axis)
    tree = cKDTree(centerline_axial)
    distances = tree.query(centroid_axial, k=1)[0]
    residuals = dict(base.residuals)
    residuals["max_axial_mm"] = float(np.max(distances))
    residuals["mean_axial_mm"] = float(np.mean(distances))
    return centerline, tangent, width_axis, base.mean_v_world, residuals


def fit_unified_arch(
    centroids: list[ToothCentroid], down_axis: np.ndarray, config: PanoramaConfig
) -> ArchGeometry:
    if len(centroids) < 4:
        raise RuntimeError(f"Need at least 4 tooth centroids to fit an arch, got {len(centroids)}.")

    e1, e2 = _build_axial_basis(down_axis)
    upper_centroids = [c for c in centroids if c.arch == "upper"]
    lower_centroids = [c for c in centroids if c.arch == "lower"]

    unified_base = _fit_arch_base(centroids, down_axis, e1, e2, config)
    if unified_base is None:
        raise RuntimeError("Not enough arch positions for a stable spline.")
    upper_base = _fit_arch_base(upper_centroids, down_axis, e1, e2, config)
    lower_base = _fit_arch_base(lower_centroids, down_axis, e1, e2, config)

    n_ext = (
        max(1, int(math.ceil(config.endpoint_extension_mm / max(unified_base.mean_step_mm, 1e-6))))
        if config.endpoint_extension_mm > 0 else 0
    )
    centerline, tangent, width_axis, _v_mean, unified_resid = _finalize_arch(
        unified_base, down_axis, n_ext, unified_base.mean_step_mm
    )
    vertical_axis = _normalize(down_axis)

    # Posterior slab-width taper: the first/last n_ext columns are the condyle
    # sweep, where the trough can drift off the curving mandible — widen it there
    # (ramp 1.0 → endpoint_slab_gain toward each tip) so the body stays captured.
    column_slab_gain = np.ones(len(centerline), dtype=np.float64)
    gain = float(config.endpoint_slab_gain)
    if n_ext > 0 and gain != 1.0 and 2 * n_ext < len(centerline):
        ramp = np.linspace(gain, 1.0, n_ext, endpoint=False)
        column_slab_gain[:n_ext] = ramp
        column_slab_gain[len(centerline) - n_ext:] = ramp[::-1]

    # Project the unified centerline to the axial plane and find, for each
    # column, the nearest point on each per-arch curve. The signed projection
    # of that displacement onto width_axis is the lateral offset; the tangent-
    # direction component is folded into the column index via the nearest-
    # neighbor lookup (so anchors that move along the arch still align).
    unified_axial = _project_to_axial(centerline, vertical_axis)

    def lateral_offsets(base: _BaseArchFit | None, fallback_v: float) -> tuple[np.ndarray, float]:
        if base is None:
            return np.zeros(len(centerline), dtype=np.float64), fallback_v
        if not config.dual_arch_offset:
            # Single unified focal trough (the reference OPG look). We still
            # report the per-arch mean V for the row blend, but the slab follows
            # one curve — no per-arch lateral offset, which is fragile when a
            # per-arch spline misbehaves on mixed dentition.
            _, _, _, mean_v, _ = _finalize_arch(base, down_axis, n_ext, unified_base.mean_step_mm)
            return np.zeros(len(centerline), dtype=np.float64), float(mean_v)
        # Re-extend the per-arch base at the same shared step so its sampling
        # density matches the unified extension footprint.
        arch_centerline, _, _, mean_v, _ = _finalize_arch(base, down_axis, n_ext, unified_base.mean_step_mm)
        arch_axial = _project_to_axial(arch_centerline, vertical_axis)
        tree = cKDTree(arch_axial)
        _, idx = tree.query(unified_axial, k=1)
        displacement = arch_axial[idx] - unified_axial             # (N, 3) axial-plane
        offset = np.einsum("nd,nd->n", displacement, width_axis)   # signed scalar
        # Real buccolingual offset between matching upper/lower teeth is ≤4 mm;
        # tighter clamp prevents nearest-neighbor matches across the extension
        # from dragging the slab off the dentition at the posterior columns.
        return np.clip(offset, -4.0, 4.0), float(mean_v)

    fallback_v = unified_base.mean_v_world
    upper_offset, v_upper = lateral_offsets(upper_base, fallback_v)
    lower_offset, v_lower = lateral_offsets(lower_base, fallback_v)

    # Per-arch curves themselves (re-extended) for debug plots.
    if upper_base is not None:
        upper_curve, *_ = _finalize_arch(upper_base, down_axis, n_ext, unified_base.mean_step_mm)
    else:
        upper_curve = centerline.copy()
    if lower_base is not None:
        lower_curve, *_ = _finalize_arch(lower_base, down_axis, n_ext, unified_base.mean_step_mm)
    else:
        lower_curve = centerline.copy()

    residuals = {
        "unified": unified_resid,
        "max_axial_mm": float(unified_resid["max_axial_mm"]),
        "mean_axial_mm": float(unified_resid["mean_axial_mm"]),
        "replaced_positions": list(unified_resid["replaced_positions"]),
        "v_separation_mm": float(abs(v_lower - v_upper)),
        "lateral_offset_upper_max_mm": float(np.max(np.abs(upper_offset))),
        "lateral_offset_lower_max_mm": float(np.max(np.abs(lower_offset))),
    }

    return ArchGeometry(
        centerline_world=centerline,
        tangent=tangent,
        width_axis=width_axis,
        vertical_axis=vertical_axis,
        lateral_offset_upper_mm=upper_offset,
        lateral_offset_lower_mm=lower_offset,
        v_upper_mm=v_upper,
        v_lower_mm=v_lower,
        residuals_mm=residuals,
        vertical_range_mm=(0.0, 0.0),  # placeholder, filled by compute_vertical_range
        column_slab_gain=column_slab_gain,
        upper_centerline_world=upper_curve,
        lower_centerline_world=lower_curve,
    )


def _coords_for_labels(seg: np.ndarray, labels: tuple[int, ...], max_samples: int) -> np.ndarray:
    label_values = np.fromiter(labels, dtype=np.int16)
    if label_values.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    coords = np.argwhere(np.isin(seg, label_values))
    if coords.shape[0] > max_samples > 0:
        stride = int(math.ceil(coords.shape[0] / max_samples))
        coords = coords[::stride]
    return coords.astype(np.float64, copy=False)


def compute_vertical_range(
    seg: np.ndarray,
    affine: np.ndarray,
    geometry: ArchGeometry,
    raw_shape: tuple[int, int, int],
    config: PanoramaConfig,
) -> ArchGeometry:
    down = geometry.vertical_axis
    margin = float(config.vertical_margin_mm)

    # v offsets are measured from the unified centerline V (the mean tooth V).
    # Keeping this relative reference makes the slab span concentrate on the
    # dentition rather than the whole skull height — critical with a properly-
    # aligned down axis which otherwise lets EXTENT_LABELS span 100 mm.
    v_center = float(np.mean(geometry.centerline_world @ down))

    # Full-jaw, no-clip framing: drive the V band from the jaw labels that are
    # actually present (1=mandible, 2=maxilla), unioned with the tooth/canal
    # extent, using tolerant percentiles (0.1/99.9) that reject a stray voxel
    # while keeping the whole bone. Whichever jaws exist are shown complete; a
    # missing jaw (e.g. an upper-teeth-less scan) reserves no empty band because
    # its label contributes no coordinates.
    coords = _coords_for_labels(seg, EXTENT_LABELS, config.max_extent_samples)
    if coords.size:
        world = _apply_affine(affine, coords)
        rel = world @ down - v_center
        low = float(np.percentile(rel, 0.1)) - margin
        high = float(np.percentile(rel, 99.9)) + margin
    else:
        low, high = -40.0, 40.0

    corners = np.array(
        [[i, j, k] for i in (0, raw_shape[0] - 1) for j in (0, raw_shape[1] - 1) for k in (0, raw_shape[2] - 1)],
        dtype=np.float64,
    )
    corners_world = _apply_affine(affine, corners)
    v_corners = corners_world @ down - v_center
    low = max(low, float(np.min(v_corners)))
    high = min(high, float(np.max(v_corners)))
    # A full jaw spans ~70-90 mm; never let the band collapse below a floor that
    # would clip a whole arch. Clamped afterwards to the volume corners.
    min_span = 60.0
    if high - low < min_span:
        mid = 0.5 * (low + high)
        low, high = mid - 0.5 * min_span, mid + 0.5 * min_span
        low = max(low, float(np.min(v_corners)))
        high = min(high, float(np.max(v_corners)))

    return ArchGeometry(
        centerline_world=geometry.centerline_world,
        tangent=geometry.tangent,
        width_axis=geometry.width_axis,
        vertical_axis=geometry.vertical_axis,
        lateral_offset_upper_mm=geometry.lateral_offset_upper_mm,
        lateral_offset_lower_mm=geometry.lateral_offset_lower_mm,
        v_upper_mm=geometry.v_upper_mm,
        v_lower_mm=geometry.v_lower_mm,
        residuals_mm=geometry.residuals_mm,
        vertical_range_mm=(float(low), float(high)),
        column_slab_gain=geometry.column_slab_gain,
        upper_centerline_world=geometry.upper_centerline_world,
        lower_centerline_world=geometry.lower_centerline_world,
    )


# -- sampling ----------------------------------------------------------------


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# -- GPU backend (torch) -----------------------------------------------------
#
# Volume sampling uses torch.nn.functional.grid_sample (trilinear) — a standard
# op, no custom kernels — so the exact same code runs on AMD (ROCm) and NVIDIA
# (CUDA) GPUs and on CPU. The projection is an accumulated-attenuation path
# integral (film-positive: dense = bright) and the tone map is fully on-device.


def select_device(name: str = "auto") -> "Any":
    """GPU-only. ``auto`` and ``cuda`` both require a visible GPU (CUDA or ROCm,
    which torch exposes through the same ``torch.cuda`` API); there is no CPU
    fallback — this pipeline is GPU-only by design."""
    import torch

    name = (name or "auto").lower()
    if name not in ("auto", "cuda", "gpu"):
        raise ValueError(f"device must be 'auto' or 'cuda' (GPU-only); got {name!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU is visible to torch. This pipeline is GPU-only — install a "
            "CUDA/ROCm torch build and ensure a device is visible "
            "(torch.cuda.is_available() must be True)."
        )
    return torch.device("cuda")


def _row_blend(geometry: ArchGeometry, v_values: np.ndarray) -> np.ndarray:
    """Per-row upper→lower blend factor t (0 at upper band, 1 at lower band),
    matching the CPU sampler's ``_smoothstep`` behaviour."""
    v_axis = geometry.vertical_axis
    v_center = float(np.mean(geometry.centerline_world @ v_axis))
    v_upper_rel = float(geometry.v_upper_mm) - v_center
    v_lower_rel = float(geometry.v_lower_mm) - v_center
    denom = v_lower_rel - v_upper_rel
    if abs(denom) < 1e-6:
        return np.zeros(len(v_values), dtype=np.float64)
    return _smoothstep((v_values - v_upper_rel) / denom)


def _apply_affine_torch(pts: "Any", R: "Any", t: "Any") -> "Any":
    """world → voxel index = R @ pts + t, computed elementwise.

    NOTE: a batched 4-D matmul ``pts @ R.T`` is silently WRONG on some ROCm
    builds (rocBLAS mishandles the (…,3)@(3,3) contraction — verified off by
    O(100) voxels on gfx1201). A 2-D matmul and this elementwise form are both
    correct, so we do the transform componentwise with plain multiply/add.
    """
    import torch

    x = pts[..., 0] * R[0, 0] + pts[..., 1] * R[0, 1] + pts[..., 2] * R[0, 2] + t[0]
    y = pts[..., 0] * R[1, 0] + pts[..., 1] * R[1, 1] + pts[..., 2] * R[1, 2] + t[1]
    z = pts[..., 0] * R[2, 0] + pts[..., 1] * R[2, 1] + pts[..., 2] * R[2, 2] + t[2]
    return torch.stack((x, y, z), dim=-1)


def _hu_to_mu_torch(h: "Any", config: PanoramaConfig) -> "Any":
    """HU → linear-attenuation surrogate mu(h), piecewise + continuous + monotone.

    region 1 (h < hu_soft_knee):  mu_air .. mu_soft_knee, linear in (h+1000)
    region 2 (soft..dense knee):  linear ramp mu_soft_knee .. mu_bone_hi
    region 3 (h > hu_dense_knee):  mu_bone_hi + dense_gain·log1p((h-knee)/scale)
    """
    import torch

    soft_k = config.hu_soft_knee
    mu_soft = config.mu_soft_knee
    dense_k = config.hu_dense_knee
    mu_hi = config.mu_bone_hi

    # region 1: air/soft-tissue floor. At h=-1000 → mu_air; at soft_k → mu_soft.
    t1 = torch.clamp((h + 1000.0) / (soft_k + 1000.0), 0.0, 1.0)
    r1 = config.mu_air + (mu_soft - config.mu_air) * t1
    # region 2: bone linear ramp.
    t2 = torch.clamp((h - soft_k) / (dense_k - soft_k), 0.0, 1.0)
    r2 = mu_soft + (mu_hi - mu_soft) * t2
    # region 3: dense (enamel/metal) log-compressed tail.
    excess = torch.clamp(h - dense_k, min=0.0)
    r3 = mu_hi + config.dense_gain * torch.log1p(excess / config.dense_scale_hu)

    mu = torch.where(h < soft_k, r1, r2)
    mu = torch.where(h > dense_k, r3, mu)
    return mu


def sample_project_torch(
    raw: np.ndarray,
    affine: np.ndarray,
    geometry: ArchGeometry,
    config: PanoramaConfig,
    device: "Any | None" = None,
    seg: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample the curved slab and project it to an accumulated-attenuation
    panorama on the selected torch device. Returns (A[V,C] float32, qc).

    When ``config.emphasize_canal`` and ``seg`` is given, the mandibular-canal
    labels are sampled through the same trough and used to knock down μ inside
    the canal lumen, so the ~2 mm tube reads as a dark corticated channel instead
    of being averaged away across the slab."""
    import torch
    import torch.nn.functional as F

    if device is None:
        device = select_device(config.device)

    use_canal = bool(config.emphasize_canal) and seg is not None and float(config.canal_strength) > 0.0

    air_value = float(np.nanpercentile(raw, 0.5))
    u_values = _grid_values(-config.slab_half_width_mm, config.slab_half_width_mm, config.plane_resolution_mm)
    v_min, v_max = geometry.vertical_range_mm
    v_values = _grid_values(v_min, v_max, config.plane_resolution_mm)
    V, U, C = len(v_values), len(u_values), len(geometry.centerline_world)

    inv_affine = np.linalg.inv(affine)
    inv_R = torch.as_tensor(inv_affine[:3, :3], dtype=torch.float32, device=device)
    inv_t = torch.as_tensor(inv_affine[:3, 3], dtype=torch.float32, device=device)

    # Geometry → device (float32).
    centerline = torch.as_tensor(geometry.centerline_world, dtype=torch.float32, device=device)   # (C,3)
    width_axis = torch.as_tensor(geometry.width_axis, dtype=torch.float32, device=device)          # (C,3)
    v_axis = torch.as_tensor(geometry.vertical_axis, dtype=torch.float32, device=device)           # (3,)
    u_t = torch.as_tensor(u_values, dtype=torch.float32, device=device)                            # (U,)
    v_t = torch.as_tensor(v_values, dtype=torch.float32, device=device)                            # (V,)

    t_row = torch.as_tensor(_row_blend(geometry, v_values), dtype=torch.float32, device=device)    # (V,)
    upper_off = torch.as_tensor(geometry.lateral_offset_upper_mm, dtype=torch.float32, device=device)  # (C,)
    lower_off = torch.as_tensor(geometry.lateral_offset_lower_mm, dtype=torch.float32, device=device)  # (C,)
    offset_vc = (1.0 - t_row)[:, None] * upper_off[None, :] + t_row[:, None] * lower_off[None, :]   # (V,C)
    col_gain = torch.as_tensor(geometry.column_slab_gain, dtype=torch.float32, device=device)       # (C,)

    # Raised-cosine slab weight w(u), normalized (weighted mean over U).
    if U > 1:
        u_norm = torch.clamp(u_t / float(config.slab_half_width_mm), -1.0, 1.0)
        w = (1.0 - config.projection_beta) + config.projection_beta * torch.cos(0.5 * math.pi * u_norm) ** 2
    else:
        w = torch.ones(U, dtype=torch.float32, device=device)
    w = w / torch.clamp(w.sum(), min=1e-8)                                                         # (U,)

    # Volume as (1,1,I,J,K), shifted so out-of-bounds (zeros pad) reads as air.
    vol = torch.as_tensor(np.ascontiguousarray(raw, dtype=np.float32), device=device)
    vol = (vol - air_value)[None, None]                                                            # (1,1,I,J,K)
    I, J, K = raw.shape
    size = torch.as_tensor([I - 1, J - 1, K - 1], dtype=torch.float32, device=device)

    if use_canal:
        vol_seg = torch.as_tensor(np.ascontiguousarray(seg, dtype=np.float32), device=device)[None, None]
        canal_set = torch.as_tensor(LOWER_CANAL_LABELS, dtype=torch.float32, device=device)
        canal_k = float(config.canal_strength)

    A = torch.empty((V, C), dtype=torch.float32, device=device)
    row_oob = np.zeros(V, dtype=np.int64)
    row_total = np.zeros(V, dtype=np.int64)
    oob_total = 0
    total = 0

    chunk = max(1, int(config.chunk_columns))
    for start in range(0, C, chunk):
        end = min(C, start + chunk)
        cc = end - start
        centers = centerline[start:end]                # (cc,3)
        w_axis = width_axis[start:end]                 # (cc,3)
        g = col_gain[start:end]                         # (cc,) posterior trough widening
        u_total = offset_vc[:, None, start:end] + u_t[None, :, None] * g[None, None, :]   # (V,U,cc)
        # world point (V,U,cc,3)
        pts = (
            centers[None, None, :, :]
            + u_total[..., None] * w_axis[None, None, :, :]
            + v_t[:, None, None, None] * v_axis[None, None, None, :]
        )
        voxel = _apply_affine_torch(pts, inv_R, inv_t)                  # (V,U,cc,3) continuous voxel index

        # OOB bookkeeping (for crop_oob_rows parity).
        oob = ((voxel < 0.0) | (voxel > size)).any(dim=-1)   # (V,U,cc)
        row_oob += oob.sum(dim=(1, 2)).to("cpu").numpy()
        row_total += U * cc
        oob_total += int(oob.sum().item())
        total += V * U * cc

        # normalized coords, align_corners=True; grid last dim = (x=K, y=J, z=I),
        # reversed vs volume axes (I,J,K).
        norm = 2.0 * voxel / size - 1.0                # (V,U,cc,3) for axes (I,J,K)
        grid = torch.stack((norm[..., 2], norm[..., 1], norm[..., 0]), dim=-1)  # (V,U,cc,3)
        grid = grid[None]                              # (1,V,U,cc,3)

        sampled = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        sampled = sampled[0, 0] + air_value            # (V,U,cc) HU
        mu = _hu_to_mu_torch(sampled, config)          # (V,U,cc)

        if use_canal:
            # Nearest-sample the canal label through the same grid and drop μ
            # inside the lumen so it accumulates as a dark tube.
            lab = F.grid_sample(vol_seg, grid, mode="nearest", padding_mode="zeros", align_corners=True)[0, 0]
            is_canal = (torch.round(lab)[..., None] == canal_set).any(dim=-1)   # (V,U,cc) bool
            mu = mu * (1.0 - canal_k * is_canal.to(mu.dtype))

        A[:, start:end] = torch.einsum("vuc,u->vc", mu, w)

    A_cpu = A.to("cpu").numpy().astype(np.float32)
    qc = {
        "backend": "torch",
        "device": str(device),
        "height_pixels": int(V),
        "columns": int(C),
        "slab_width_pixels": int(U),
        "v_min_mm": float(v_values[0]),
        "v_max_mm": float(v_values[-1]),
        "u_min_mm": float(u_values[0]),
        "u_max_mm": float(u_values[-1]),
        "air_value": air_value,
        "projection_beta": float(config.projection_beta),
        "out_of_bounds_fraction": float(oob_total / max(total, 1)),
        "row_out_of_bounds_fraction": (row_oob / np.maximum(row_total, 1)).astype(float).tolist(),
    }
    return A_cpu, qc


def _gaussian_blur_torch(img: "Any", sigma: float) -> "Any":
    """Separable Gaussian blur of a 2-D image tensor via conv2d (reflect pad)."""
    import torch
    import torch.nn.functional as F

    if sigma <= 0:
        return img
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=img.dtype, device=img.device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    t = img[None, None]                                 # (1,1,H,W)
    kh = k.view(1, 1, -1, 1)
    kv = k.view(1, 1, 1, -1)
    t = F.pad(t, (0, 0, radius, radius), mode="reflect")
    t = F.conv2d(t, kh)
    t = F.pad(t, (radius, radius, 0, 0), mode="reflect")
    t = F.conv2d(t, kv)
    return t[0, 0]


def tone_map_torch(A: np.ndarray, config: PanoramaConfig, device: "Any | None" = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Accumulated-attenuation image → display [0,1]: percentile-normalize →
    gamma shadow-lift → local-contrast normalization → unsharp. All on device."""
    import torch

    if device is None:
        device = select_device(config.device)
    t = torch.as_tensor(np.ascontiguousarray(A, dtype=np.float32), device=device)
    finite = torch.isfinite(t)
    if not bool(finite.any()):
        return np.zeros_like(A, dtype=np.float32), {"low": 0.0, "high": 1.0}
    t = torch.where(finite, t, torch.zeros_like(t))

    vals = t[finite]
    lo = torch.quantile(vals, config.intensity_clip_low_pct / 100.0)
    hi = torch.quantile(vals, config.intensity_clip_high_pct / 100.0)
    if not bool(torch.isfinite(hi)) or float(hi) <= float(lo):
        lo, hi = vals.min(), vals.max()
    if float(hi) <= float(lo):
        return np.zeros_like(A, dtype=np.float32), {"low": float(lo), "high": float(hi)}
    t = torch.clamp((t - lo) / (hi - lo), 0.0, 1.0)

    # Gamma shadow-lift.
    if config.tone_gamma and config.tone_gamma > 0:
        t = t.clamp(0.0, 1.0) ** float(config.tone_gamma)

    # Local mean/variance contrast normalization (torch-native CLAHE substitute).
    if config.apply_local_contrast:
        sigma = float(config.local_contrast_sigma_px)
        m = _gaussian_blur_torch(t, sigma)
        v = torch.clamp(_gaussian_blur_torch(t * t, sigma) - m * m, min=0.0)
        s = torch.sqrt(v + 1e-6)
        gain = torch.clamp(
            config.local_contrast_target_std / (s + 1e-6),
            config.local_contrast_gain_min,
            config.local_contrast_gain_max,
        )
        t = torch.clamp(m + (t - m) * gain, 0.0, 1.0)

    # Unsharp mask.
    if config.unsharp_amount and config.unsharp_amount > 0:
        blur = _gaussian_blur_torch(t, float(config.unsharp_sigma_px))
        t = torch.clamp(t + float(config.unsharp_amount) * (t - blur), 0.0, 1.0)

    out = t.to("cpu").numpy().astype(np.float32)
    return out, {"low": float(lo), "high": float(hi)}


def crop_oob_rows(panorama: np.ndarray, qc: dict[str, Any], config: PanoramaConfig) -> tuple[np.ndarray, dict[str, Any]]:
    row_oob = np.asarray(qc.get("row_out_of_bounds_fraction", []), dtype=np.float64)
    if row_oob.size != panorama.shape[0]:
        qc.update({"cropped_top_rows": 0, "cropped_bottom_rows": 0})
        return panorama, qc

    keep = row_oob <= config.crop_oob_threshold
    if not np.any(keep):
        qc.update({"cropped_top_rows": 0, "cropped_bottom_rows": 0})
        return panorama, qc

    first = int(np.argmax(keep))
    last = int(len(keep) - 1 - np.argmax(keep[::-1]))
    first = max(0, first - int(config.crop_margin_rows))
    last = min(len(keep) - 1, last + int(config.crop_margin_rows))
    cropped = panorama[first : last + 1]
    step = float(config.plane_resolution_mm)
    original_v_min = float(qc["v_min_mm"])
    qc.update(
        {
            "cropped_top_rows": int(first),
            "cropped_bottom_rows": int(len(keep) - 1 - last),
            "cropped_v_min_mm": float(original_v_min + first * step),
            "cropped_v_max_mm": float(original_v_min + last * step),
            "height_pixels_after_crop": int(cropped.shape[0]),
        }
    )
    return cropped, qc


# -- output ------------------------------------------------------------------


OVERLAY_CANAL_LABELS = (3, 4, 103, 104, 105)


def project_segmentation_torch(
    seg: np.ndarray,
    affine: np.ndarray,
    geometry: ArchGeometry,
    config: PanoramaConfig,
    device: "Any | None" = None,
) -> dict[str, np.ndarray]:
    """Project label groups through the same curved trough as the grayscale
    image (nearest-neighbour along U, presence = any voxel in the slab). Returns
    (V,C) float masks in [0,1] registered to the un-cropped grayscale panorama."""
    import torch
    import torch.nn.functional as F

    if device is None:
        device = select_device(config.device)

    u_values = _grid_values(-config.slab_half_width_mm, config.slab_half_width_mm, config.plane_resolution_mm)
    v_min, v_max = geometry.vertical_range_mm
    v_values = _grid_values(v_min, v_max, config.plane_resolution_mm)
    V, U, C = len(v_values), len(u_values), len(geometry.centerline_world)

    inv_affine = np.linalg.inv(affine)
    inv_R = torch.as_tensor(inv_affine[:3, :3], dtype=torch.float32, device=device)
    inv_t = torch.as_tensor(inv_affine[:3, 3], dtype=torch.float32, device=device)
    centerline = torch.as_tensor(geometry.centerline_world, dtype=torch.float32, device=device)
    width_axis = torch.as_tensor(geometry.width_axis, dtype=torch.float32, device=device)
    v_axis = torch.as_tensor(geometry.vertical_axis, dtype=torch.float32, device=device)
    u_t = torch.as_tensor(u_values, dtype=torch.float32, device=device)
    v_t = torch.as_tensor(v_values, dtype=torch.float32, device=device)
    t_row = torch.as_tensor(_row_blend(geometry, v_values), dtype=torch.float32, device=device)
    upper_off = torch.as_tensor(geometry.lateral_offset_upper_mm, dtype=torch.float32, device=device)
    lower_off = torch.as_tensor(geometry.lateral_offset_lower_mm, dtype=torch.float32, device=device)
    offset_vc = (1.0 - t_row)[:, None] * upper_off[None, :] + t_row[:, None] * lower_off[None, :]
    col_gain = torch.as_tensor(geometry.column_slab_gain, dtype=torch.float32, device=device)

    vol = torch.as_tensor(np.ascontiguousarray(seg, dtype=np.float32), device=device)[None, None]
    I, J, K = seg.shape
    size = torch.as_tensor([I - 1, J - 1, K - 1], dtype=torch.float32, device=device)

    canal_set = torch.as_tensor(OVERLAY_CANAL_LABELS, dtype=torch.float32, device=device)
    tooth_set = torch.as_tensor(ALL_TOOTH_LABELS, dtype=torch.float32, device=device)
    canal = torch.zeros((V, C), dtype=torch.float32, device=device)
    teeth = torch.zeros((V, C), dtype=torch.float32, device=device)

    chunk = max(1, int(config.chunk_columns))
    for start in range(0, C, chunk):
        end = min(C, start + chunk)
        centers = centerline[start:end]
        w_axis = width_axis[start:end]
        g = col_gain[start:end]
        u_total = offset_vc[:, None, start:end] + u_t[None, :, None] * g[None, None, :]
        pts = (
            centers[None, None, :, :]
            + u_total[..., None] * w_axis[None, None, :, :]
            + v_t[:, None, None, None] * v_axis[None, None, None, :]
        )
        voxel = _apply_affine_torch(pts, inv_R, inv_t)
        norm = 2.0 * voxel / size - 1.0
        grid = torch.stack((norm[..., 2], norm[..., 1], norm[..., 0]), dim=-1)[None]
        lab = F.grid_sample(vol, grid, mode="nearest", padding_mode="zeros", align_corners=True)[0, 0]
        lab_r = torch.round(lab)
        is_canal = (lab_r[..., None] == canal_set).any(dim=-1)   # (V,U,cc)
        is_tooth = (lab_r[..., None] == tooth_set).any(dim=-1)
        canal[:, start:end] = is_canal.float().amax(dim=1)
        teeth[:, start:end] = is_tooth.float().amax(dim=1)

    return {"canal": canal.to("cpu").numpy(), "teeth": teeth.to("cpu").numpy()}


def _binary_edge(mask: np.ndarray) -> np.ndarray:
    """1-px outline of a binary mask (presence minus its 1-neighbour erosion)."""
    m = mask > 0.5
    er = m.copy()
    er[1:, :] &= m[:-1, :]; er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]; er[:, :-1] &= m[:, 1:]
    return (m & ~er).astype(np.float32)


def compose_overlay(base01: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    """Grayscale base (H,W)∈[0,1] → RGB with canals tinted red (filled, α) and
    tooth outlines in cyan. Masks are cropped/flipped to match ``base01``."""
    h, w = base01.shape
    rgb = np.repeat(base01[:, :, None], 3, axis=2).astype(np.float32)
    canal = masks.get("canal")
    teeth = masks.get("teeth")
    if teeth is not None and teeth.shape == base01.shape:
        edge = _binary_edge(teeth)
        cyan = np.array([0.15, 0.9, 0.9], dtype=np.float32)
        a = (edge * 0.6)[:, :, None]
        rgb = rgb * (1 - a) + cyan[None, None, :] * a
    if canal is not None and canal.shape == base01.shape:
        red = np.array([1.0, 0.15, 0.1], dtype=np.float32)
        a = (np.clip(canal, 0, 1) * 0.45)[:, :, None]
        rgb = rgb * (1 - a) + red[None, None, :] * a
    return np.clip(rgb, 0.0, 1.0)


def save_png8_rgb(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(out, mode="RGB").save(path)


def save_png16(image01: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.clip(image01, 0.0, 1.0)
    out = np.rint(out * 65535.0).astype(np.uint16)
    Image.fromarray(out).save(path)


def save_png8(image01: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.clip(image01, 0.0, 1.0)
    out = np.rint(out * 255.0).astype(np.uint8)
    Image.fromarray(out).save(path)


# -- debug plots -------------------------------------------------------------


def save_axial_mip_with_spline(
    raw: np.ndarray, affine: np.ndarray, geometry: ArchGeometry, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    down = geometry.vertical_axis
    axis_alignments = np.abs(affine[:3, :3].T @ down)
    axial_axis = int(np.argmax(axis_alignments))
    mip = raw.max(axis=axial_axis)
    inv_affine = np.linalg.inv(affine)
    upper_vox = _apply_affine(inv_affine, geometry.upper_centerline_world)
    lower_vox = _apply_affine(inv_affine, geometry.lower_centerline_world)
    plane_axes = [a for a in range(3) if a != axial_axis]

    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=140)
    ax.imshow(mip.T if plane_axes == [0, 1] else mip, cmap="gray", origin="lower")
    ax.plot(upper_vox[:, plane_axes[0]], upper_vox[:, plane_axes[1]], color="orange", lw=1.4, label="upper arch")
    ax.plot(lower_vox[:, plane_axes[0]], lower_vox[:, plane_axes[1]], color="cyan", lw=1.4, label="lower arch")
    ax.legend(loc="lower right", fontsize=8, facecolor="black", labelcolor="white")
    ax.set_title("axial MIP with upper/lower arch splines")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_sample_plane_montage(
    raw: np.ndarray, affine: np.ndarray, geometry: ArchGeometry, config: PanoramaConfig, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(geometry.centerline_world)
    columns = np.linspace(0, n - 1, min(5, n), dtype=int)
    inv_affine = np.linalg.inv(affine)
    air_value = float(np.nanpercentile(raw, 0.5))
    u_values = _grid_values(-config.slab_half_width_mm, config.slab_half_width_mm, config.plane_resolution_mm)
    v_min, v_max = geometry.vertical_range_mm
    v_values = _grid_values(v_min, v_max, config.plane_resolution_mm)

    v_axis = geometry.vertical_axis
    v_center = float(np.mean(geometry.centerline_world @ v_axis))
    v_upper_rel = float(geometry.v_upper_mm) - v_center
    v_lower_rel = float(geometry.v_lower_mm) - v_center
    denom = v_lower_rel - v_upper_rel
    if abs(denom) < 1e-6:
        t_row = np.zeros(len(v_values), dtype=np.float64)
    else:
        t_row = _smoothstep((v_values - v_upper_rel) / denom)

    planes = []
    for col in columns:
        idx = int(col)
        c = geometry.centerline_world[idx]
        w = geometry.width_axis[idx]
        u_off = float(geometry.lateral_offset_upper_mm[idx])
        l_off = float(geometry.lateral_offset_lower_mm[idx])
        offset_v = (1.0 - t_row) * u_off + t_row * l_off  # (V,)
        u_total = offset_v[:, None] + u_values[None, :]   # (V, U)
        pts = (
            c[None, None, :]
            + u_total[:, :, None] * w[None, None, :]
            + v_values[:, None, None] * v_axis[None, None, :]
        )
        vox = _apply_affine(inv_affine, pts.reshape(-1, 3))
        sampled = map_coordinates(raw, [vox[:, 0], vox[:, 1], vox[:, 2]], order=1, mode="constant", cval=air_value)
        planes.append(sampled.reshape(len(v_values), len(u_values)).astype(np.float32))

    values = np.concatenate([p.ravel() for p in planes])
    low, high = np.percentile(values, [1.0, 99.0])
    if high <= low:
        high = low + 1.0

    plt = _plt()
    fig, axes = plt.subplots(1, len(planes), figsize=(2.6 * len(planes), 5.0), dpi=140)
    if len(planes) == 1:
        axes = [axes]
    for ax, plane, col in zip(axes, planes, columns):
        ax.imshow(np.clip((plane - low) / (high - low), 0, 1), cmap="gray", aspect="auto")
        ax.set_title(f"col {int(col)}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("sampled planes along the arch")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# -- orchestrator ------------------------------------------------------------


def reconstruct_case(raw_path: Path, seg_path: Path, output_dir: Path, config: PanoramaConfig) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_id = seg_path.name.replace(".nii.gz", "").replace(".nii", "")
    raw, seg, affine, meta = load_nifti_pair(raw_path, seg_path)

    centroids = extract_tooth_centroids(seg, affine, config)
    if len(centroids) < 4:
        raise RuntimeError(f"{case_id}: only {len(centroids)} tooth centroids found — cannot fit a panoramic arch.")
    down_axis = estimate_down_axis(centroids, affine, seg=seg)
    geometry = fit_unified_arch(centroids, down_axis, config)
    geometry = compute_vertical_range(seg, affine, geometry, tuple(int(v) for v in raw.shape), config)

    import time

    device = select_device(config.device)
    t0 = time.time()
    panorama, sampling_qc = sample_project_torch(raw, affine, geometry, config, device, seg=seg)
    panorama, sampling_qc = crop_oob_rows(panorama, sampling_qc, config)
    top = int(sampling_qc.get("cropped_top_rows", 0))
    bottom = int(sampling_qc.get("cropped_bottom_rows", 0))
    normalized, intensity_qc = tone_map_torch(panorama, config, device)
    sampling_qc["reconstruct_seconds"] = round(time.time() - t0, 3)

    overlay_masks: dict[str, np.ndarray] | None = None
    if config.overlay:
        overlay_masks = project_segmentation_torch(seg, affine, geometry, config, device)
        # Match the grayscale crop.
        v_full = normalized.shape[0] + top + bottom
        for key, m in overlay_masks.items():
            if m.shape[0] == v_full:
                overlay_masks[key] = m[top: v_full - bottom] if bottom else m[top:]

    if config.flip_horizontal:
        normalized = normalized[:, ::-1]
        if overlay_masks is not None:
            overlay_masks = {k: m[:, ::-1] for k, m in overlay_masks.items()}

    main_path = output_dir / f"{case_id}_panoramic.png"
    preview_path = output_dir / f"{case_id}_panoramic_preview.png"
    save_png16(normalized, main_path)
    save_png8(normalized, preview_path)

    overlay_path_str: dict[str, str] = {}
    if overlay_masks is not None:
        overlay_rgb = compose_overlay(normalized, overlay_masks)
        overlay_path = output_dir / f"{case_id}_panoramic_overlay.png"
        save_png8_rgb(overlay_rgb, overlay_path)
        overlay_path_str = {"panoramic_overlay_png": str(overlay_path)}

    debug_paths: dict[str, str] = {}
    if config.save_debug:
        axial_path = output_dir / f"{case_id}_axial_spline.png"
        montage_path = output_dir / f"{case_id}_sample_planes.png"
        save_axial_mip_with_spline(raw, affine, geometry, axial_path)
        save_sample_plane_montage(raw, affine, geometry, config, montage_path)
        debug_paths = {"axial_spline_png": str(axial_path), "sample_planes_png": str(montage_path)}

    report = {
        "case_id": case_id,
        "metadata": {k: v for k, v in meta.items() if k != "affine"},
        "affine": meta["affine"],
        "config": asdict(config),
        "down_axis_world": down_axis,
        "centroid_count": len(centroids),
        "centroids": [asdict(c) for c in centroids],
        "geometry": {
            "vertical_range_mm": geometry.vertical_range_mm,
            "residuals_mm": geometry.residuals_mm,
            "centerline_points": int(len(geometry.centerline_world)),
        },
        "sampling_qc": sampling_qc,
        "intensity_qc": intensity_qc,
        "panoramic_png": str(main_path),
        "panoramic_preview_png": str(preview_path),
        **overlay_path_str,
        **debug_paths,
    }
    with (output_dir / f"{case_id}_qc.json").open("w", encoding="utf-8") as f:
        json.dump(_as_jsonable(report), f, indent=2)
    return report


# -- CLI ---------------------------------------------------------------------


def _config_from_args(args: argparse.Namespace) -> PanoramaConfig:
    return PanoramaConfig(
        device=args.device,
        spline_resolution=args.spline_resolution,
        spline_smoothing=args.spline_smoothing,
        plane_resolution_mm=args.plane_resolution_mm,
        slab_half_width_mm=args.slab_half_width_mm,
        vertical_margin_mm=args.vertical_margin_mm,
        endpoint_extension_mm=args.endpoint_extension_mm,
        endpoint_slab_gain=args.endpoint_slab_gain,
        emphasize_canal=not args.no_emphasize_canal,
        canal_strength=args.canal_strength,
        projection_beta=args.projection_beta,
        mu_air=args.mu_air,
        hu_dense_knee=args.hu_dense_knee,
        dense_scale_hu=args.dense_scale_hu,
        dense_gain=args.dense_gain,
        tone_gamma=args.tone_gamma,
        intensity_clip_low_pct=args.clip_low,
        intensity_clip_high_pct=args.clip_high,
        local_contrast_sigma_px=args.local_contrast_sigma,
        local_contrast_target_std=args.local_contrast_target_std,
        apply_local_contrast=not args.no_local_contrast,
        unsharp_amount=args.unsharp_amount,
        unsharp_sigma_px=args.unsharp_sigma,
        crop_oob_threshold=args.crop_oob_threshold,
        crop_margin_rows=args.crop_margin_rows,
        flip_horizontal=not args.no_flip,
        overlay=args.overlay,
        save_debug=args.debug,
    )


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("auto", "cuda"), default=PanoramaConfig.device,
                        help="compute backend (GPU only; both require a visible GPU)")
    parser.add_argument("--spline-resolution", type=int, default=PanoramaConfig.spline_resolution)
    parser.add_argument("--spline-smoothing", type=float, default=None, help="None → auto from arc length")
    parser.add_argument("--plane-resolution-mm", type=float, default=PanoramaConfig.plane_resolution_mm)
    parser.add_argument("--slab-half-width-mm", type=float, default=PanoramaConfig.slab_half_width_mm)
    parser.add_argument("--vertical-margin-mm", type=float, default=PanoramaConfig.vertical_margin_mm)
    parser.add_argument("--endpoint-extension-mm", type=float, default=PanoramaConfig.endpoint_extension_mm)
    parser.add_argument("--endpoint-slab-gain", type=float, default=PanoramaConfig.endpoint_slab_gain,
                        help="posterior trough widening toward the condyles (1=off)")
    parser.add_argument("--no-emphasize-canal", action="store_true",
                        help="disable canal-label μ modulation (darkening the mandibular canal)")
    parser.add_argument("--canal-strength", type=float, default=PanoramaConfig.canal_strength,
                        help="μ reduction inside the canal lumen (0..1); higher = darker tube")
    parser.add_argument("--projection-beta", type=float, default=PanoramaConfig.projection_beta,
                        help="U-weight peakiness: 1=sharp/shallow, 0=deep/flat")
    parser.add_argument("--mu-air", type=float, default=PanoramaConfig.mu_air,
                        help="attenuation floor: >0 keeps sinus/airway grey not black")
    parser.add_argument("--hu-dense-knee", type=float, default=PanoramaConfig.hu_dense_knee,
                        help="HU where enamel/metal log-compression starts")
    parser.add_argument("--dense-scale-hu", type=float, default=PanoramaConfig.dense_scale_hu)
    parser.add_argument("--dense-gain", type=float, default=PanoramaConfig.dense_gain)
    parser.add_argument("--tone-gamma", type=float, default=PanoramaConfig.tone_gamma,
                        help="<1 lifts shadows/mid — main 'show all features' lever")
    parser.add_argument("--clip-low", type=float, default=PanoramaConfig.intensity_clip_low_pct)
    parser.add_argument("--clip-high", type=float, default=PanoramaConfig.intensity_clip_high_pct)
    parser.add_argument("--local-contrast-sigma", type=float, default=PanoramaConfig.local_contrast_sigma_px)
    parser.add_argument("--local-contrast-target-std", type=float, default=PanoramaConfig.local_contrast_target_std)
    parser.add_argument("--no-local-contrast", action="store_true")
    parser.add_argument("--unsharp-amount", type=float, default=PanoramaConfig.unsharp_amount)
    parser.add_argument("--unsharp-sigma", type=float, default=PanoramaConfig.unsharp_sigma_px)
    parser.add_argument("--crop-oob-threshold", type=float, default=PanoramaConfig.crop_oob_threshold)
    parser.add_argument("--crop-margin-rows", type=int, default=PanoramaConfig.crop_margin_rows)
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--overlay", action="store_true",
                        help="also write an RGB overlay tinting canals (red) and tooth outlines (cyan)")
    parser.add_argument("--debug", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct a panoramic (OPG) from CBCT + tooth segmentation.")
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--seg", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    _add_config_args(parser)
    args = parser.parse_args()

    report = reconstruct_case(args.raw, args.seg, args.output_dir, _config_from_args(args))
    print(json.dumps(_as_jsonable({
        "case_id": report["case_id"],
        "panoramic_png": report["panoramic_png"],
        "geometry": report["geometry"],
        "sampling_qc": {k: report["sampling_qc"][k] for k in (
            "backend", "device", "reconstruct_seconds", "height_pixels", "columns",
            "v_min_mm", "v_max_mm", "out_of_bounds_fraction"
        ) if k in report["sampling_qc"]},
    }), indent=2))


if __name__ == "__main__":
    main()
