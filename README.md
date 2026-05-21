# cbct_v2 — CBCT → Panoramic

Synthesises a clinical-style panoramic (OPG) PNG from a CBCT volume and a
matching FDI-labelled tooth/jaw segmentation. The pipeline is a
**single-curve curved-MPR (CMPR)**: one 2-D dental-arch spline fitted in the
axial plane, swept across a tall vertical slab that covers maxilla through
mandible, projected through the slab into one continuous panoramic image.

```
                    sampling slab (≈14 mm thick, swept along the arch)
                  ┌────────────────────────────────────────────────┐
       maxilla    │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
                  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  ▲  V (down_axis)
       teeth      │░░██████████████████████████████████████████░░░░│  │
                  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
       mandible   │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
                  └────────────────────────────────────────────────┘
                       ↑ axial-plane spline (centerline), perpendicular U
```

The arch spline is anchored on the **mean position per FDI tooth slot** —
upper and lower teeth in the same slot are voxel-weighted-averaged into one
anchor, and obvious single-anchor outliers (e.g. tiny mis-segmented remnants)
are patched by a local-parabola fit before `scipy.interpolate.splprep` runs.

The segmentation is geometry input only: nothing about it is rasterised into
the final image. The slab is sampled directly from the raw CBCT and projected
with the configured `--projection-mode` (defaults to `xray`, a path-integrated
attenuation that reproduces clinical OPG contrast and preserves pulp/canal
visibility — see "Projection modes" below).

---

## Layout

```
cbct_panorama.py        # the pipeline (library + CLI)
run_local_cases.py      # batch driver for raw/*.nii.gz + seg/*.nii.gz pairs
requirements.txt        # pinned package floors
aim_result.png          # reference target image
cbct_panoramic_prompt.md  # original spec
raw/                    # CBCT NIfTIs (gitignored — keep your data here)
seg/                    # FDI-labelled segmentation NIfTIs (gitignored)
outputs/                # generated panoramics + QC (gitignored)
```

Drop matching pairs into `raw/<case_id>_0000.nii.gz` and
`seg/<case_id>.nii.gz` for the batch driver to pick them up.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

Three bundled local cases:

```bash
.venv/bin/python run_local_cases.py --output-dir outputs/panoramic_v3
```

One arbitrary case:

```bash
.venv/bin/python cbct_panorama.py \
    --raw  path/to/case_0000.nii.gz \
    --seg  path/to/case.nii.gz \
    --output-dir outputs/single_case
```

Add `--debug` to also write an axial-MIP-with-spline overlay and a 5-column
sample-plane montage alongside the panoramic.

## Outputs

For each case ID:

| File                              | Purpose                                          |
| --------------------------------- | ------------------------------------------------ |
| `<case_id>_panoramic.png`         | 16-bit panoramic (production)                    |
| `<case_id>_panoramic_preview.png` | 8-bit preview                                    |
| `<case_id>_qc.json`               | Spline residuals, sampling QC, intensity window  |
| `<case_id>_axial_spline.png`*     | Axial MIP with fitted arch overlay (`--debug`)   |
| `<case_id>_sample_planes.png`*    | 5 evenly-spaced perpendicular planes (`--debug`) |

## Configuration

All knobs live on `PanoramaConfig` in `cbct_panorama.py`. The CLI mirrors them.

| Field                       | Default   | Purpose                                                                         |
| --------------------------- | --------- | ------------------------------------------------------------------------------- |
| `slab_half_width_mm`        | 7.0       | Slab is `±slab_half_width_mm` thick along the perpendicular U axis              |
| `vertical_margin_mm`        | 6.0       | Extra mm above maxilla / below mandible in the V extent                         |
| `endpoint_extension_mm`     | 18.0      | Centerline extension past the terminal molars                                   |
| `projection_mode`           | `xray`    | `xray` (path-integrated attenuation), `mip`, `percentile`, `mean`               |
| `projection_percentile`     | 96.0      | Used only when `projection_mode="percentile"`                                   |
| `spline_resolution`         | 1200      | Number of sample columns along the arch                                         |
| `spline_smoothing`          | `None`    | `splprep` `s`; `None` → `arc_length/40`                                         |
| `plane_resolution_mm`       | 0.2       | Sampling step in both U and V                                                   |
| `apply_clahe` / `clahe_clip_limit` | True / 0.01 | Local contrast after percentile-clip normalisation                       |
| `crop_oob_threshold`        | 0.75      | Rows with ≥75 % out-of-bounds samples are auto-cropped                          |
| `flip_horizontal`           | True      | Patient-right on the image-left (clinical OPG convention)                       |
| `min_tooth_voxels`          | 50        | Minimum segmented voxels for a tooth to count as a centroid                     |

### Projection modes

- `xray` (default): each ray through the slab accumulates `max(HU + 1000, 0)/1000 × plane_resolution_mm`. Dense voxels add a lot; low-attenuation cavities (pulp lumen, canal lumen, marrow) add little — the integral preserves them. Closest to a real OPG.
- `mip`: maximum per ray. Sharpest enamel/cortex but **hides pulp and canal** because a single bright voxel along the ray dominates.
- `percentile` / `mean`: smoother variants; mostly useful for diagnostics.

## QC summary

The `_qc.json` per case includes:

- `geometry.residuals_mm.max_axial_mm` — max distance from any centroid to the
  fitted spline, projected to the axial plane. Should be < 4 mm for a clean
  fit; values much above that flag a missing or grossly mis-positioned tooth.
- `geometry.residuals_mm.replaced_positions` — FDI slots whose anchor was
  patched by the local-parabola outlier rule.
- `sampling_qc.out_of_bounds_fraction` — fraction of slab samples that fell
  outside the CBCT volume. < 0.30 after auto-crop is healthy.

## Version history

- `v3` (current) — robust anchors (voxel-weighted + local-parabola outlier
  patch); `xray` attenuation projection by default for visible pulps and
  canals.
- `v2` — single-curve CMPR; FDI-position anchoring; auto-cropped vertical
  extent. Replaces the original two-strip / segmentation-masked output.
