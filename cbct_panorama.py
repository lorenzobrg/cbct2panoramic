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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.interpolate import splev, splprep
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from skimage import exposure


UPPER_TOOTH_LABELS = tuple(range(11, 19)) + tuple(range(21, 29))
LOWER_TOOTH_LABELS = tuple(range(31, 39)) + tuple(range(41, 49))
UPPER_PULP_LABELS = tuple(label + 100 for label in UPPER_TOOTH_LABELS)
LOWER_PULP_LABELS = tuple(label + 100 for label in LOWER_TOOTH_LABELS)
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
    slab_half_width_mm: float = 7.0
    vertical_margin_mm: float = 6.0
    endpoint_extension_mm: float = 18.0
    projection_mode: str = "mip"            # mip | percentile | mean
    projection_percentile: float = 96.0
    intensity_clip_low_pct: float = 0.5
    intensity_clip_high_pct: float = 99.7
    apply_clahe: bool = True
    clahe_clip_limit: float = 0.01
    crop_oob_threshold: float = 0.75
    crop_margin_rows: int = 6
    flip_horizontal: bool = True
    chunk_columns: int = 32
    terminal_gap_outlier_factor: float = 2.4
    terminal_gap_outlier_min_mm: float = 20.0
    save_debug: bool = False
    max_extent_samples: int = 300_000


@dataclass(frozen=True)
class ToothCentroid:
    label: int
    arch: str
    voxel_count: int
    voxel: tuple[float, float, float]
    world: tuple[float, float, float]


@dataclass
class ArchGeometry:
    centerline_world: np.ndarray   # (N, 3)
    tangent: np.ndarray            # (N, 3) — unit, in axial plane
    width_axis: np.ndarray         # (N, 3) — unit, in axial plane (perpendicular)
    vertical_axis: np.ndarray      # (3,)   — constant, supero-inferior
    residuals_mm: dict[str, float]
    vertical_range_mm: tuple[float, float]


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


def estimate_down_axis(centroids: list[ToothCentroid], affine: np.ndarray) -> np.ndarray:
    upper = np.array([c.world for c in centroids if c.arch == "upper"], dtype=np.float64)
    lower = np.array([c.world for c in centroids if c.arch == "lower"], dtype=np.float64)
    if len(upper) and len(lower):
        down = lower.mean(axis=0) - upper.mean(axis=0)
        if np.linalg.norm(down) > 1e-6:
            return _normalize(down)
    # Single-arch fallback — affine axis 2 is the upper-to-lower direction
    # in the bundled ToothFairy NIfTIs.
    return _normalize(affine[:3, 2])


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


def _extend_centerline(centerline: np.ndarray, tangent: np.ndarray, extension_mm: float) -> tuple[np.ndarray, np.ndarray]:
    if extension_mm <= 0 or len(centerline) < 2:
        return centerline, tangent
    mean_step = float(np.mean(np.linalg.norm(np.diff(centerline, axis=0), axis=1)))
    if not np.isfinite(mean_step) or mean_step <= 1e-6:
        return centerline, tangent
    n_ext = max(1, int(math.ceil(extension_mm / mean_step)))
    distances = np.linspace(extension_mm, mean_step, n_ext)
    pre = centerline[0][None, :] - distances[:, None] * tangent[0][None, :]
    post = centerline[-1][None, :] + distances[::-1, None] * tangent[-1][None, :]
    pre_t = np.repeat(tangent[0][None, :], n_ext, axis=0)
    post_t = np.repeat(tangent[-1][None, :], n_ext, axis=0)
    return np.vstack([pre, centerline, post]), np.vstack([pre_t, tangent, post_t])


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
    by_position: dict[int, list[tuple[np.ndarray, float]]] = {}
    for c in centroids:
        pos = LABEL_TO_ARCH_POSITION.get(int(c.label))
        if pos is None:
            continue
        by_position.setdefault(pos, []).append(
            (np.asarray(c.world, dtype=np.float64), float(max(c.voxel_count, 1)))
        )
    if not by_position:
        return np.empty((0,), dtype=np.int64), np.empty((0, 3), dtype=np.float64)
    positions = np.array(sorted(by_position.keys()), dtype=np.int64)
    anchors = []
    for pos in positions:
        entries = by_position[int(pos)]
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


def fit_unified_arch(
    centroids: list[ToothCentroid], down_axis: np.ndarray, config: PanoramaConfig
) -> ArchGeometry:
    if len(centroids) < 4:
        raise RuntimeError(f"Need at least 4 tooth centroids to fit an arch, got {len(centroids)}.")

    positions, anchors_world = _anchors_by_arch_position(centroids, down_axis)
    if len(positions) < 4:
        raise RuntimeError(f"Only {len(positions)} arch positions present; need at least 4 for a stable spline.")

    e1, e2 = _build_axial_basis(down_axis)
    pts_2d = np.column_stack([anchors_world @ e1, anchors_world @ e2])

    # Patch outlier anchors (e.g. tiny mis-segmented teeth) by comparing each
    # anchor to a smoothing-spline trend through all the anchors. Tracked in
    # residuals for QC.
    pts_2d, replaced_positions = _patch_outlier_anchors(positions, pts_2d)

    # Drop near-duplicate adjacent points that would destabilise splprep.
    kept_indices = [0]
    for i in range(1, len(pts_2d)):
        if np.linalg.norm(pts_2d[i] - pts_2d[kept_indices[-1]]) >= 0.5:
            kept_indices.append(i)
    pts_2d = pts_2d[kept_indices]

    trimmed, dropped_start, dropped_end = _drop_terminal_gap_outliers(pts_2d, config)
    if len(trimmed) >= 5:
        pts_2d = trimmed

    arc_length = float(np.sum(np.linalg.norm(np.diff(pts_2d, axis=0), axis=1)))
    if config.spline_smoothing is None:
        smoothing = max(1.0, arc_length / 40.0)
    else:
        smoothing = float(config.spline_smoothing)

    # Fit the spline in 2-D (axial plane) — we restore the vertical
    # component as a constant offset (the mean V) so that the sampling
    # plane normal is purely horizontal.
    k = min(3, len(pts_2d) - 1)
    tck, _ = splprep(pts_2d.T, s=smoothing, k=k)
    u_eval = np.linspace(0.0, 1.0, int(config.spline_resolution))
    curve_2d = np.array(splev(u_eval, tck), dtype=np.float64).T   # (N, 2)
    deriv_2d = np.array(splev(u_eval, tck, der=1), dtype=np.float64).T  # (N, 2)

    centerline = curve_2d[:, 0:1] * e1[None, :] + curve_2d[:, 1:2] * e2[None, :]
    # Anchor the centerline at the mean V of the tooth centroids so the strip
    # is roughly centred on the dentition occlusal plane.
    centroid_world = np.array([c.world for c in centroids], dtype=np.float64)
    mean_v = float(np.mean(centroid_world @ down_axis))
    centerline = centerline + mean_v * down_axis[None, :]
    tangent = _normalize(deriv_2d[:, 0:1] * e1[None, :] + deriv_2d[:, 1:2] * e2[None, :])

    centerline, tangent = _extend_centerline(centerline, tangent, config.endpoint_extension_mm)
    # Vertical axis is constant (supero-inferior). U = V × T (in-axial perpendicular).
    vertical_axis = _normalize(down_axis)
    width_axis = _normalize(np.cross(vertical_axis[None, :].repeat(len(tangent), axis=0), tangent))

    # Residuals measured in the axial plane only — the spline lives at a single
    # vertical anchor by design, so a 3D distance would be dominated by the
    # upper/lower-arch height difference rather than any fit error.
    centroid_axial = _project_to_axial(centroid_world, vertical_axis)
    centerline_axial = _project_to_axial(centerline, vertical_axis)
    tree = cKDTree(centerline_axial)
    distances = tree.query(centroid_axial, k=1)[0]
    residuals = {
        "max_axial_mm": float(np.max(distances)),
        "mean_axial_mm": float(np.mean(distances)),
        "dropped_terminal_outliers": int(dropped_start + dropped_end),
        "replaced_positions": [int(p) for p in replaced_positions],
        "anchor_positions": [int(p) for p in positions],
        "arc_length_mm": arc_length,
        "smoothing": float(smoothing),
    }

    return ArchGeometry(
        centerline_world=centerline,
        tangent=tangent,
        width_axis=width_axis,
        vertical_axis=vertical_axis,
        residuals_mm=residuals,
        vertical_range_mm=(0.0, 0.0),  # placeholder, filled by compute_vertical_range
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
    coords = _coords_for_labels(seg, EXTENT_LABELS, config.max_extent_samples)
    down = geometry.vertical_axis

    # Project every relevant voxel into mm along the down axis; v=0 is the
    # mean tooth-centroid height (where the spline lives), so v<0 is above
    # the occlusal plane and v>0 is below.
    if coords.size:
        world = _apply_affine(affine, coords)
        v_values = world @ down
        v_center = float(np.mean(geometry.centerline_world @ down))
        rel = v_values - v_center
        low = float(np.percentile(rel, 0.5)) - float(config.vertical_margin_mm)
        high = float(np.percentile(rel, 99.5)) + float(config.vertical_margin_mm)
    else:
        low, high = -40.0, 40.0

    # Clamp to the volume's actual extent so we don't waste rows on air.
    corners = np.array(
        [[i, j, k] for i in (0, raw_shape[0] - 1) for j in (0, raw_shape[1] - 1) for k in (0, raw_shape[2] - 1)],
        dtype=np.float64,
    )
    corners_world = _apply_affine(affine, corners)
    v_corners = corners_world @ down
    v_center = float(np.mean(geometry.centerline_world @ down))
    v_min_vol = float(np.min(v_corners)) - v_center
    v_max_vol = float(np.max(v_corners)) - v_center
    low = max(low, v_min_vol)
    high = min(high, v_max_vol)
    if high - low < 30.0:
        # Failsafe: still produce a usable slab even if the seg-derived range is tiny.
        mid = 0.5 * (low + high)
        low, high = mid - 30.0, mid + 30.0

    return ArchGeometry(
        centerline_world=geometry.centerline_world,
        tangent=geometry.tangent,
        width_axis=geometry.width_axis,
        vertical_axis=geometry.vertical_axis,
        residuals_mm=geometry.residuals_mm,
        vertical_range_mm=(float(low), float(high)),
    )


# -- sampling ----------------------------------------------------------------


def _project_slab(sampled: np.ndarray, config: PanoramaConfig) -> np.ndarray:
    mode = config.projection_mode.lower()
    if mode == "mip":
        return sampled.max(axis=1)
    if mode == "percentile":
        return np.percentile(sampled, config.projection_percentile, axis=1)
    if mode == "mean":
        return sampled.mean(axis=1)
    raise ValueError(f"Unsupported projection_mode: {config.projection_mode}")


def sample_panorama(
    raw: np.ndarray,
    affine: np.ndarray,
    geometry: ArchGeometry,
    config: PanoramaConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    inv_affine = np.linalg.inv(affine)
    air_value = float(np.nanpercentile(raw, 0.5))
    u_values = _grid_values(-config.slab_half_width_mm, config.slab_half_width_mm, config.plane_resolution_mm)
    v_min, v_max = geometry.vertical_range_mm
    v_values = _grid_values(v_min, v_max, config.plane_resolution_mm)
    height = len(v_values)
    slab_width = len(u_values)
    columns = len(geometry.centerline_world)
    panorama = np.empty((height, columns), dtype=np.float32)

    shape_limit = np.array(raw.shape, dtype=np.float64) - 1.0
    total = 0
    oob_total = 0
    row_oob = np.zeros(height, dtype=np.int64)
    row_total = np.zeros(height, dtype=np.int64)

    v_axis = geometry.vertical_axis
    centerline_offset = v_axis * (geometry.centerline_world @ v_axis).mean()  # not used; kept for clarity
    del centerline_offset

    for start in range(0, columns, config.chunk_columns):
        end = min(columns, start + config.chunk_columns)
        centers = geometry.centerline_world[start:end]               # (B, 3)
        width_axis = geometry.width_axis[start:end]                  # (B, 3)

        # (V, U, B, 3): for each (v, u) and each column b, world position.
        points = (
            centers[None, None, :, :]
            + u_values[None, :, None, None] * width_axis[None, None, :, :]
            + v_values[:, None, None, None] * v_axis[None, None, None, :]
        )
        flat = points.reshape(-1, 3)
        voxel = _apply_affine(inv_affine, flat)
        oob = np.any((voxel < 0.0) | (voxel > shape_limit[None, :]), axis=1)
        total += oob.size
        oob_total += int(oob.sum())
        oob_grid = oob.reshape(height, slab_width, end - start)
        row_oob += oob_grid.sum(axis=(1, 2))
        row_total += slab_width * (end - start)

        sampled = map_coordinates(
            raw,
            [voxel[:, 0], voxel[:, 1], voxel[:, 2]],
            order=1,
            mode="constant",
            cval=air_value,
        ).reshape(height, slab_width, end - start)

        panorama[:, start:end] = _project_slab(sampled, config).astype(np.float32, copy=False)

    qc = {
        "height_pixels": int(height),
        "columns": int(columns),
        "slab_width_pixels": int(slab_width),
        "v_min_mm": float(v_values[0]),
        "v_max_mm": float(v_values[-1]),
        "u_min_mm": float(u_values[0]),
        "u_max_mm": float(u_values[-1]),
        "air_value": float(air_value),
        "out_of_bounds_fraction": float(oob_total / max(total, 1)),
        "row_out_of_bounds_fraction": (row_oob / np.maximum(row_total, 1)).astype(float).tolist(),
    }
    return panorama, qc


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


def normalize_panorama(panorama: np.ndarray, config: PanoramaConfig) -> tuple[np.ndarray, dict[str, float]]:
    finite = panorama[np.isfinite(panorama)]
    if finite.size == 0:
        return np.zeros_like(panorama, dtype=np.float32), {"low": 0.0, "high": 1.0}
    low, high = np.percentile(finite, [config.intensity_clip_low_pct, config.intensity_clip_high_pct])
    if not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        normalized = np.zeros_like(panorama, dtype=np.float32)
    else:
        clipped = np.clip(panorama, low, high)
        normalized = ((clipped - low) / (high - low)).astype(np.float32)
    if config.apply_clahe and normalized.size:
        normalized = exposure.equalize_adapthist(normalized, clip_limit=config.clahe_clip_limit).astype(np.float32)
    return normalized, {"low": float(low), "high": float(high)}


# -- output ------------------------------------------------------------------


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
    # Axial MIP — collapse along the axis that is closest to the down direction.
    down = geometry.vertical_axis
    axis_alignments = np.abs(affine[:3, :3].T @ down)
    axial_axis = int(np.argmax(axis_alignments))
    mip = raw.max(axis=axial_axis)
    # Project centerline into voxel coordinates and drop the axial axis.
    inv_affine = np.linalg.inv(affine)
    voxels = _apply_affine(inv_affine, geometry.centerline_world)
    plane_axes = [a for a in range(3) if a != axial_axis]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=140)
    ax.imshow(mip.T if plane_axes == [0, 1] else mip, cmap="gray", origin="lower")
    ax.plot(voxels[:, plane_axes[0]], voxels[:, plane_axes[1]], color="orange", lw=1.4)
    ax.set_title("axial MIP with fitted arch spline")
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

    planes = []
    for col in columns:
        c = geometry.centerline_world[int(col)]
        u_axis = geometry.width_axis[int(col)]
        pts = c[None, None, :] + u_values[None, :, None] * u_axis[None, None, :] + v_values[:, None, None] * geometry.vertical_axis[None, None, :]
        vox = _apply_affine(inv_affine, pts.reshape(-1, 3))
        sampled = map_coordinates(raw, [vox[:, 0], vox[:, 1], vox[:, 2]], order=1, mode="constant", cval=air_value)
        planes.append(sampled.reshape(len(v_values), len(u_values)).astype(np.float32))

    values = np.concatenate([p.ravel() for p in planes])
    low, high = np.percentile(values, [1.0, 99.0])
    if high <= low:
        high = low + 1.0

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
    down_axis = estimate_down_axis(centroids, affine)
    geometry = fit_unified_arch(centroids, down_axis, config)
    geometry = compute_vertical_range(seg, affine, geometry, tuple(int(v) for v in raw.shape), config)

    panorama, sampling_qc = sample_panorama(raw, affine, geometry, config)
    panorama, sampling_qc = crop_oob_rows(panorama, sampling_qc, config)
    normalized, intensity_qc = normalize_panorama(panorama, config)
    if config.flip_horizontal:
        normalized = normalized[:, ::-1]
        panorama = panorama[:, ::-1]

    main_path = output_dir / f"{case_id}_panoramic.png"
    preview_path = output_dir / f"{case_id}_panoramic_preview.png"
    save_png16(normalized, main_path)
    save_png8(normalized, preview_path)

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
        **debug_paths,
    }
    with (output_dir / f"{case_id}_qc.json").open("w", encoding="utf-8") as f:
        json.dump(_as_jsonable(report), f, indent=2)
    return report


# -- CLI ---------------------------------------------------------------------


def _config_from_args(args: argparse.Namespace) -> PanoramaConfig:
    return PanoramaConfig(
        spline_resolution=args.spline_resolution,
        spline_smoothing=args.spline_smoothing,
        plane_resolution_mm=args.plane_resolution_mm,
        slab_half_width_mm=args.slab_half_width_mm,
        vertical_margin_mm=args.vertical_margin_mm,
        endpoint_extension_mm=args.endpoint_extension_mm,
        projection_mode=args.projection_mode,
        projection_percentile=args.projection_percentile,
        apply_clahe=not args.no_clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        crop_oob_threshold=args.crop_oob_threshold,
        crop_margin_rows=args.crop_margin_rows,
        flip_horizontal=not args.no_flip,
        save_debug=args.debug,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct a panoramic image from CBCT + tooth segmentation.")
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--seg", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--spline-resolution", type=int, default=PanoramaConfig.spline_resolution)
    parser.add_argument("--spline-smoothing", type=float, default=None,
                        help="None → auto from arc length")
    parser.add_argument("--plane-resolution-mm", type=float, default=PanoramaConfig.plane_resolution_mm)
    parser.add_argument("--slab-half-width-mm", type=float, default=PanoramaConfig.slab_half_width_mm)
    parser.add_argument("--vertical-margin-mm", type=float, default=PanoramaConfig.vertical_margin_mm)
    parser.add_argument("--endpoint-extension-mm", type=float, default=PanoramaConfig.endpoint_extension_mm)
    parser.add_argument("--projection-mode", choices=("mip", "percentile", "mean"), default=PanoramaConfig.projection_mode)
    parser.add_argument("--projection-percentile", type=float, default=PanoramaConfig.projection_percentile)
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--clahe-clip-limit", type=float, default=PanoramaConfig.clahe_clip_limit)
    parser.add_argument("--crop-oob-threshold", type=float, default=PanoramaConfig.crop_oob_threshold)
    parser.add_argument("--crop-margin-rows", type=int, default=PanoramaConfig.crop_margin_rows)
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    report = reconstruct_case(args.raw, args.seg, args.output_dir, _config_from_args(args))
    print(json.dumps(_as_jsonable({
        "case_id": report["case_id"],
        "panoramic_png": report["panoramic_png"],
        "geometry": report["geometry"],
        "sampling_qc": {k: report["sampling_qc"][k] for k in (
            "height_pixels", "columns", "v_min_mm", "v_max_mm", "out_of_bounds_fraction"
        ) if k in report["sampling_qc"]},
    }), indent=2))


if __name__ == "__main__":
    main()
