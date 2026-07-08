# cbct2panoramic — CBCT → Panoramic (OPG), GPU-accelerated

Synthesises a clinical-style panoramic radiograph (OPG / orthopantomogram) from a
CBCT volume and a matching FDI-labelled tooth/jaw segmentation.

The pipeline is a **curved planar reformat (CMPR)** along the dental arch, projected
as an **accumulated-attenuation path integral** (film-positive: dense structures are
bright). Sampling and projection run on the **GPU** via `torch.nn.functional.grid_sample`
— a standard op, **no custom CUDA kernels** — so the same code runs on **AMD (ROCm)**
and **NVIDIA (CUDA)** hardware. The pipeline is **GPU-only**: it requires a visible GPU
(`torch.cuda.is_available()`), there is no CPU fallback.

```
                    curved focal trough (swept along the dental arch)
                  ┌────────────────────────────────────────────────┐
   condyle/TMJ →  │ ░░░░░░░░░░░░░░ maxilla / sinuses ░░░░░░░░░░░░░░ │  ▲  V (supero-inferior)
       teeth      │░░██████████████ both arches █████████████████░░│  │
   mandible →     │ ░░░░░░░░░░░ body / canal / ramus ░░░░░░░░░░░░░░ │
                  └────────────────────────────────────────────────┘
        per column: integrate attenuation across the bucco-lingual (U) trough
```

## Why the segmentation is needed (and how it is used)

The segmentation is used **mostly for geometry** — grayscale intensities come from the
raw, unmasked CBCT. Specifically it drives:

1. the **dental-arch spline** — anchored on per-FDI-slot tooth centroids (permanent and
   deciduous, so mixed/primary dentition works);
2. the **supero-inferior axis** (upper- vs lower-tooth centroid difference);
3. the **vertical slab extent** — driven by the jaw labels present (mandible/maxilla),
   unioned with teeth/canals, so whichever arches exist are framed complete and a scan
   missing the upper teeth is not clipped;
4. the **canal emphasis** (`--canal-strength`, default on): the mandibular-canal label
   modulates μ so the ~2 mm lumen reads as a dark tube instead of averaging away — the one
   place the segmentation touches grayscale (disable with `--no-emphasize-canal`);
5. the optional **color overlay** (`--overlay`): canals and tooth outlines tinted.

Fitting the arch from raw CBCT alone (bone thresholding) is far less robust than using
a SOTA tooth segmentation's clean per-tooth centroids — hence the dependency.

## Install (uv)

Requires [`uv`](https://docs.astral.sh/uv/). PyTorch wheels are hardware-specific, so
install torch from the matching index:

**AMD (ROCm)** — developed against an AMD Radeon AI PRO R9700 (RDNA4) on ROCm 7.x:

```bash
uv venv --python 3.13
uv pip install numpy scipy nibabel scikit-image Pillow
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.2
# stable alternative that also works on RDNA4:
#   uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
```

If an older ROCm build doesn't recognise the GPU arch, force it:
`HSA_OVERRIDE_GFX_VERSION=12.0.0` (RDNA4) or `11.0.0` (RDNA3).

**NVIDIA (CUDA):**

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify the GPU is visible (required — the pipeline errors out with no GPU):

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## Run

Single case:

```bash
uv run python cbct_panorama.py \
  --raw  ToothFairy3F_060_0000.nii.gz \
  --seg  segmentation_60.nii.gz \
  --output-dir outputs/case_60 \
  --overlay
```

All matching pairs in a directory (classic `raw/`+`seg/` layout, **or** the flat
`ToothFairy3*_<N>_0000.nii.gz` + `segmentation_<N>.nii.gz` layout):

```bash
uv run python run_local_cases.py --root /srv/datasets/cbct --output-dir experiments/run1 --overlay
```

### Outputs (per case)

| file | description |
|------|-------------|
| `<case>_panoramic.png` | 16-bit grayscale panoramic (production) |
| `<case>_panoramic_preview.png` | 8-bit preview |
| `<case>_panoramic_overlay.png` | RGB overlay: canals red, tooth outlines cyan (`--overlay`) |
| `<case>_qc.json` | config, geometry residuals, device, timing, intensity window |

## Key tuning flags

Defaults are tuned for **feature clarity**. The knobs most worth adjusting:

| flag | default | effect |
|------|---------|--------|
| `--tone-gamma` | 0.75 | `<1` lifts shadows/mid — the main "show all features" lever |
| `--projection-beta` | 0.55 | U-weight peakiness: `1`=sharp/shallow, `0`=deep/flat |
| `--hu-dense-knee` / `--dense-gain` | 2500 / 0.15 | log-compress enamel/metal so it doesn't erase bone contrast |
| `--mu-air` | 0.002 | attenuation floor — keeps sinus/airway grey, not black |
| `--slab-half-width-mm` | 9.0 | focal-trough half-thickness (bucco-lingual) |
| `--endpoint-extension-mm` | 30.0 | how far the trough sweeps back toward the condyles |
| `--endpoint-slab-gain` | 1.6 | posterior trough widening toward the condyles (`1`=off) |
| `--canal-strength` / `--no-emphasize-canal` | 0.5 / on | darken the mandibular canal by modulating μ with the canal label |
| `--device` | auto | `auto` \| `cuda` (GPU only — no CPU fallback) |

## How it works (pipeline)

`reconstruct_case` (`cbct_panorama.py`):

1. **Geometry (CPU, cheap).** `extract_tooth_centroids` → `estimate_down_axis` →
   `fit_unified_arch` (spline through per-slot anchors, outlier-patched) →
   `compute_vertical_range`.
2. **Sampling + projection (GPU).** `sample_project_torch`: builds the `(V,U,C,3)`
   world-point grid on-device, `grid_sample`s the raw volume (trilinear; out-of-bounds
   reads as air via a +air shift + zero padding), applies the HU→μ transfer, and
   integrates `A = Σ_u w(u)·μ` across the trough.
3. **Tone map (GPU).** `tone_map_torch`: percentile-normalize → gamma shadow-lift →
   local mean/variance contrast (a torch-native CLAHE substitute) → unsharp.
4. **Overlay (optional).** `project_segmentation_torch` samples labels through the same
   trough; canals/tooth outlines are composited onto the grayscale base.

## Notes

- CBCT HU are uncalibrated ("HU-like"): air ≈ −1000, enamel/metal can exceed 15000. The
  air floor and percentile window are computed per-volume; the μ knees are heuristics.
- Volume axis convention is preserved end-to-end; `grid_sample`'s grid uses reversed
  axis order `(x=W, y=H, z=D)` with `align_corners=True` (see `sample_project_torch`).
