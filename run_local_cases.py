from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from cbct_panorama import PanoramaConfig, _as_jsonable, reconstruct_case


def local_cases(root: Path) -> list[tuple[Path, Path]]:
    cases: list[tuple[Path, Path]] = []
    for seg_path in sorted((root / "seg").glob("*.nii.gz")):
        case_id = seg_path.name.replace(".nii.gz", "")
        raw_path = root / "raw" / f"{case_id}_0000.nii.gz"
        if raw_path.exists():
            cases.append((raw_path, seg_path))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Run panoramic reconstruction on bundled local CBCT cases.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/panoramic_v2"))
    parser.add_argument("--slab-half-width-mm", type=float, default=PanoramaConfig.slab_half_width_mm)
    parser.add_argument("--vertical-margin-mm", type=float, default=PanoramaConfig.vertical_margin_mm)
    parser.add_argument("--endpoint-extension-mm", type=float, default=PanoramaConfig.endpoint_extension_mm)
    parser.add_argument("--projection-mode", choices=("xray", "mip", "percentile", "mean"), default=PanoramaConfig.projection_mode)
    parser.add_argument("--projection-percentile", type=float, default=PanoramaConfig.projection_percentile)
    parser.add_argument("--spline-resolution", type=int, default=PanoramaConfig.spline_resolution)
    parser.add_argument("--spline-smoothing", type=float, default=None)
    parser.add_argument("--plane-resolution-mm", type=float, default=PanoramaConfig.plane_resolution_mm)
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--clahe-clip-limit", type=float, default=PanoramaConfig.clahe_clip_limit)
    parser.add_argument("--crop-oob-threshold", type=float, default=PanoramaConfig.crop_oob_threshold)
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    cases = local_cases(root)
    if not cases:
        raise SystemExit(f"No raw/seg case pairs found under {root}")

    config = PanoramaConfig(
        slab_half_width_mm=args.slab_half_width_mm,
        vertical_margin_mm=args.vertical_margin_mm,
        endpoint_extension_mm=args.endpoint_extension_mm,
        projection_mode=args.projection_mode,
        projection_percentile=args.projection_percentile,
        spline_resolution=args.spline_resolution,
        spline_smoothing=args.spline_smoothing,
        plane_resolution_mm=args.plane_resolution_mm,
        apply_clahe=not args.no_clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        crop_oob_threshold=args.crop_oob_threshold,
        flip_horizontal=not args.no_flip,
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
        print(f"Running {seg_path.name} -> {case_output_dir}")
        report = reconstruct_case(raw_path, seg_path, case_output_dir, config)
        case_summary = {
            "case_id": report["case_id"],
            "panoramic_png": report["panoramic_png"],
            "centroid_count": report["centroid_count"],
            "geometry": report["geometry"],
            "sampling_qc": {
                k: report["sampling_qc"][k] for k in (
                    "height_pixels", "columns", "v_min_mm", "v_max_mm",
                    "v_upper_rel_mm", "v_lower_rel_mm",
                    "out_of_bounds_fraction", "cropped_top_rows", "cropped_bottom_rows",
                    "height_pixels_after_crop",
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
