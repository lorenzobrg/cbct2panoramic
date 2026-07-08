from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

from cbct_panorama import PanoramaConfig, _as_jsonable, reconstruct_case


def _pairs_raw_seg_dirs(root: Path) -> list[tuple[Path, Path]]:
    """Classic layout: raw/<case>_0000.nii.gz + seg/<case>.nii.gz."""
    cases: list[tuple[Path, Path]] = []
    for seg_path in sorted((root / "seg").glob("*.nii.gz")):
        case_id = seg_path.name.replace(".nii.gz", "")
        raw_path = root / "raw" / f"{case_id}_0000.nii.gz"
        if raw_path.exists():
            cases.append((raw_path, seg_path))
    return cases


def _pairs_flat(root: Path) -> list[tuple[Path, Path]]:
    """Flat ToothFairy layout: segmentation_<N>.nii.gz paired with the raw whose
    trailing case number (before _0000) matches <N> (zero-padding ignored)."""
    raws = [p for p in sorted(root.glob("*_0000.nii.gz"))]
    def raw_num(p: Path) -> str | None:
        m = re.search(r"_(\d+)_0000\.nii\.gz$", p.name)
        return str(int(m.group(1))) if m else None
    raw_by_num = {raw_num(p): p for p in raws if raw_num(p) is not None}
    cases: list[tuple[Path, Path]] = []
    for seg_path in sorted(root.glob("segmentation_*.nii.gz")):
        m = re.search(r"segmentation_(\d+)\.nii\.gz$", seg_path.name)
        if not m:
            continue
        num = str(int(m.group(1)))
        raw_path = raw_by_num.get(num)
        if raw_path is not None:
            cases.append((raw_path, seg_path))
    return cases


def local_cases(root: Path) -> list[tuple[Path, Path]]:
    cases = _pairs_raw_seg_dirs(root)
    if cases:
        return cases
    return _pairs_flat(root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run panoramic reconstruction on all local CBCT case pairs.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/panoramic_v3"))
    # forwarded config flags (subset most relevant to batch tuning)
    parser.add_argument("--device", choices=("auto", "cuda"), default=PanoramaConfig.device)
    parser.add_argument("--slab-half-width-mm", type=float, default=PanoramaConfig.slab_half_width_mm)
    parser.add_argument("--vertical-margin-mm", type=float, default=PanoramaConfig.vertical_margin_mm)
    parser.add_argument("--endpoint-extension-mm", type=float, default=PanoramaConfig.endpoint_extension_mm)
    parser.add_argument("--endpoint-slab-gain", type=float, default=PanoramaConfig.endpoint_slab_gain)
    parser.add_argument("--no-emphasize-canal", action="store_true")
    parser.add_argument("--canal-strength", type=float, default=PanoramaConfig.canal_strength)
    parser.add_argument("--projection-beta", type=float, default=PanoramaConfig.projection_beta)
    parser.add_argument("--mu-air", type=float, default=PanoramaConfig.mu_air)
    parser.add_argument("--hu-dense-knee", type=float, default=PanoramaConfig.hu_dense_knee)
    parser.add_argument("--dense-gain", type=float, default=PanoramaConfig.dense_gain)
    parser.add_argument("--tone-gamma", type=float, default=PanoramaConfig.tone_gamma)
    parser.add_argument("--clip-low", type=float, default=PanoramaConfig.intensity_clip_low_pct)
    parser.add_argument("--clip-high", type=float, default=PanoramaConfig.intensity_clip_high_pct)
    parser.add_argument("--local-contrast-sigma", type=float, default=PanoramaConfig.local_contrast_sigma_px)
    parser.add_argument("--local-contrast-target-std", type=float, default=PanoramaConfig.local_contrast_target_std)
    parser.add_argument("--no-local-contrast", action="store_true")
    parser.add_argument("--unsharp-amount", type=float, default=PanoramaConfig.unsharp_amount)
    parser.add_argument("--spline-resolution", type=int, default=PanoramaConfig.spline_resolution)
    parser.add_argument("--plane-resolution-mm", type=float, default=PanoramaConfig.plane_resolution_mm)
    parser.add_argument("--crop-oob-threshold", type=float, default=PanoramaConfig.crop_oob_threshold)
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    cases = local_cases(root)
    if not cases:
        raise SystemExit(f"No raw/seg case pairs found under {root}")

    config = PanoramaConfig(
        device=args.device,
        slab_half_width_mm=args.slab_half_width_mm,
        vertical_margin_mm=args.vertical_margin_mm,
        endpoint_extension_mm=args.endpoint_extension_mm,
        endpoint_slab_gain=args.endpoint_slab_gain,
        emphasize_canal=not args.no_emphasize_canal,
        canal_strength=args.canal_strength,
        projection_beta=args.projection_beta,
        mu_air=args.mu_air,
        hu_dense_knee=args.hu_dense_knee,
        dense_gain=args.dense_gain,
        tone_gamma=args.tone_gamma,
        intensity_clip_low_pct=args.clip_low,
        intensity_clip_high_pct=args.clip_high,
        local_contrast_sigma_px=args.local_contrast_sigma,
        local_contrast_target_std=args.local_contrast_target_std,
        apply_local_contrast=not args.no_local_contrast,
        unsharp_amount=args.unsharp_amount,
        spline_resolution=args.spline_resolution,
        plane_resolution_mm=args.plane_resolution_mm,
        crop_oob_threshold=args.crop_oob_threshold,
        flip_horizontal=not args.no_flip,
        overlay=args.overlay,
        save_debug=args.debug,
    )

    batch_summary = {
        "root": str(root),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "cases": [],
    }

    for raw_path, seg_path in cases:
        case_output_dir = output_dir / seg_path.name.replace(".nii.gz", "")
        print(f"Running {seg_path.name} + {raw_path.name} -> {case_output_dir}")
        report = reconstruct_case(raw_path, seg_path, case_output_dir, config)
        case_summary = {
            "case_id": report["case_id"],
            "panoramic_png": report["panoramic_png"],
            "centroid_count": report["centroid_count"],
            "geometry": report["geometry"],
            "sampling_qc": {
                k: report["sampling_qc"][k] for k in (
                    "backend", "device", "reconstruct_seconds", "height_pixels", "columns",
                    "v_min_mm", "v_max_mm", "out_of_bounds_fraction",
                    "cropped_top_rows", "cropped_bottom_rows", "height_pixels_after_crop",
                ) if k in report["sampling_qc"]
            },
            "intensity_qc": report["intensity_qc"],
        }
        batch_summary["cases"].append(case_summary)
        print(json.dumps(_as_jsonable(case_summary), indent=2))

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "batch_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_as_jsonable(batch_summary), f, indent=2)
    print(f"Wrote {output_dir / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
