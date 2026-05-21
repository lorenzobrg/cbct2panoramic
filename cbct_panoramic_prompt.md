# CBCT to Panoramic Image Reconstruction — Full Implementation Prompt

---

## Context and Goal

You are helping implement a **fully automatic algorithm** that generates a synthetic dental panoramic image (OPG) from a CBCT (Cone-Beam Computed Tomography) volumetric scan. The pipeline starts from an existing, already-trained **tooth and jaw segmentation model** whose output is reliable and accurate. The panoramic image should faithfully represent all dental anatomy: teeth (crown and root), alveolar bone, maxilla/mandible, and surrounding soft tissue.

This is a research/engineering implementation task. Code should be written in **Python**, using libraries such as `numpy`, `scipy`, `SimpleITK` or `nibabel` for volume handling, and `matplotlib` or `Pillow` for output. The implementation should be clean, modular, and well-commented.

---

## Input

- A 3D CBCT volume: a numpy array of shape `(Z, Y, X)` (axial, coronal, sagittal), with known voxel spacing in mm (e.g. `[0.2, 0.2, 0.2]` mm).
- A corresponding 3D segmentation mask (same shape): integer labels where each tooth instance has a unique label (e.g. 1–32 for adult dentition), and the jaw bone has its own label. Background = 0.
- The algorithm must handle **both the upper arch (maxilla) and lower arch (mandible)** separately, producing two panoramic strips that can be combined.

---

## Algorithm Pipeline

Implement the following pipeline as separate, testable functions:

### Step 1 — Extract tooth centroids from segmentation

For each unique tooth label in the segmentation mask, compute its **3D centroid** in voxel coordinates, then convert to mm using the voxel spacing. Separate centroids into upper arch (maxilla) and lower arch (mandible) based on their Y coordinate (or Z coordinate depending on orientation — clarify based on the CBCT coordinate system). Output: two lists of 3D points, one per arch.

### Step 2 — Fit a 3D spline to each arch

For each arch:
- Sort the centroids along the arch (left molar → front incisors → right molar). The best sorting strategy is by angle around the centroid of all points projected onto the axial plane (top-down view). Compute the mean XZ position, then sort by `atan2(z - z_mean, x - x_mean)`.
- Fit a **parametric cubic spline** (using `scipy.interpolate.splprep`) through the sorted centroid positions in 3D space.
- Evaluate the spline at a high resolution (e.g. 1000 points along the parameter `t ∈ [0, 1]`).
- At each point, compute the **unit tangent vector** as the first derivative of the spline, normalized.

### Step 3 — Construct perpendicular sampling planes

At each point along the spline:
- The tangent vector `T` defines the direction the spline is moving.
- Construct a sampling plane whose **normal = T**.
- The plane has two in-plane axes:
  - `U = normalize(T × world_up)` where `world_up = [0, 1, 0]` (or the vertical axis of your CBCT). This gives the horizontal axis of the output panoramic image.
  - `V = normalize(U × T)`. This gives the vertical axis (height of the panoramic strip, from root tip to top of bone).
- The plane is centered at the spline point.
- The sampling region in the plane should extend:
  - ±`strip_half_width` mm along `U` (typically ±2–4 mm around the spline to capture tooth width and surrounding bone)
  - From `v_bottom` to `v_top` along `V` (e.g. −10 mm to +15 mm relative to the centroid height, to capture roots below and bone above)

### Step 4 — Sample the CBCT volume along each plane

For each plane:
- Generate a 2D grid of sample points in the plane (e.g. resolution: 0.2 mm steps in both U and V).
- Convert each sample point from mm to voxel coordinates.
- Trilinearly interpolate the CBCT intensity at each voxel coordinate (use `scipy.ndimage.map_coordinates` with `order=1`).
- For each column of the plane (i.e. fixed U), aggregate the intensities along U using **Maximum Intensity Projection (MIP)** — take the max value across the strip width. This produces one column of the final panoramic image.
- Alternatively, implement **mean projection** and **sum projection** as options and compare.

### Step 5 — Assemble the panoramic image

- Stack the projected columns (one per spline point) side by side → result is a 2D image of shape `(V_pixels, N_spline_points)`.
- Apply histogram equalization or CLAHE (`skimage.exposure.equalize_adapthist`) to improve contrast.
- Optionally flip/rotate so the image matches clinical OPG convention (patient's right on the left of the image, teeth pointing downward, bone at top).
- Save as 16-bit PNG or DICOM.

---

## Key Parameters to Expose

Make all of these configurable (e.g. via a config dict or dataclass):

```python
config = {
    "voxel_spacing": [0.2, 0.2, 0.2],       # mm
    "spline_smoothing": 0.0,                  # splprep s parameter; 0 = interpolating
    "spline_resolution": 800,                 # number of sample points along arch
    "strip_half_width_mm": 3.0,               # how far left/right of spline to sample
    "strip_bottom_mm": -12.0,                 # below centroid (roots)
    "strip_top_mm": 16.0,                     # above centroid (bone, maxilla)
    "plane_resolution_mm": 0.2,               # sampling resolution in the plane
    "projection_mode": "mip",                 # "mip", "mean", or "sum"
    "apply_clahe": True,
    "output_path": "panoramic_output.png"
}
```

---

## Output Checking and Validation

After generating the panoramic image, implement the following checks:

### Geometric validation
- Plot the fitted spline overlaid on the axial MIP of the CBCT. Verify it follows the arch curve correctly. Flag if any centroid is more than 5 mm from the fitted spline (possible segmentation error).
- Plot the centroid positions in 3D to verify correct upper/lower arch separation.

### Visual validation
- Render a series of 5 evenly-spaced sample planes as overlays on axial/coronal/sagittal slices to confirm they are perpendicular to the arch and correctly positioned.
- Compare the generated panoramic with: (a) a reference OPG if available, or (b) a manually inspected screenshot from clinical CBCT software.

### Quantitative metrics (if ground truth OPG is available)
- SSIM (Structural Similarity Index) using `skimage.metrics.structural_similarity`
- PSNR (Peak Signal-to-Noise Ratio)
- Tooth detection overlap: project the segmentation labels onto the panoramic image and verify each tooth region is visible and not clipped.

---

## Error Handling and Edge Cases

Handle the following explicitly:
- **Missing teeth**: if fewer than 4 centroids are found for one arch, warn and fall back to fitting a polynomial arch template (parabola) aligned to the available centroids.
- **Segmentation label noise**: filter out labels with volume < threshold (e.g. < 50 voxels) as likely artifacts.
- **Coordinate system ambiguity**: auto-detect CBCT orientation from metadata (SimpleITK direction cosines) and map to a canonical LPS or RAS frame before processing.
- **Out-of-bounds sampling**: clamp voxel coordinates to volume bounds and fill out-of-bounds regions with the minimum intensity value (air HU).

---

## Iteration Plan

Implement and test in the following order:

1. **Step 1 only**: visualize centroid positions on the volume. Confirm upper/lower separation is correct.
2. **Steps 1–2**: visualize fitted spline on axial MIP. Confirm it follows the arch.
3. **Steps 1–3**: visualize one sampling plane as a 2D image. Confirm anatomy is captured.
4. **Full pipeline**: generate the panoramic image. Compare visually with a known-good result.
5. **Parameter tuning**: adjust strip width and height until the panoramic matches clinical expectations.
6. **Validation**: run the quantitative checks above.

---

## Reference Literature

The following papers describe the established methods this algorithm is based on. Use them to validate design decisions:

1. **Arch detection via spline + perpendicular sampling (core method)**
   - *"Automatic Synthesis of Panoramic Radiographs from Dental Cone Beam Computed Tomography Data"*, PLOS One.
   - Key detail: uses Hermite cubic splines; takes the spline derivative to generate perpendicular lines at each point; projects CBCT intensities along those lines to build the panoramic.

2. **MIP-based arch detection (alternative arch extraction)**
   - *"A Fast Automatic Reconstruction Method for Panoramic Images Based on Cone Beam Computed Tomography"*
   - Key detail: detects dental arch from axial Maximum Intensity Projection images at different depth ranges; uses curved multi-planar reformatting (MPR).

3. **High-contrast panoramic via dental arch thickness**
   - *"Automatic reconstruction method for high-contrast panoramic image from dental cone-beam CT data"*, ScienceDirect.
   - Key detail: detects arch thickness (not just centerline); incorporates this thickness into the sampling strip width per point for more adaptive sampling.

4. **Deep learning segmentation as input (validates the segmentation-first approach)**
   - *"A fully automatic AI system for tooth and alveolar bone segmentation from cone-beam CT images"*, Nature Communications.
   - Key detail: demonstrates Dice scores of 91.5% (teeth) and 93.0% (alveolar bone), confirming that deep-learning segmentation is accurate enough to anchor panoramic reconstruction.

5. **3D spline + perpendicular planes from segmentation (closest to this implementation)**
   - *"Reconstruction of Panoramic view from CBCT through dental arch curve"* (ResearchGate / Scientific Diagram).
   - Key detail: 3D U-Net segments jaw and dentition; panoramic view reconstructed using 3D spline curves fitted to the arch; defines the 3D reconstruction zone from the segmentation output.

6. **B-spline optimisation for arch curve**
   - *"Reconstruction of Panoramic Dental Images Through Bézier Function Optimization"*, PMC.
   - Key detail: uses least-squares B-spline fitting to the arch; good reference for the spline fitting step specifically.

7. **Perpendicular line extraction from spline derivative**
   - *"CBCT-guided 3D Dental Structure Reconstruction from Panoramic X-ray: A Preprocessing Pipeline"*
   - Key detail: explicitly uses spline formula derivatives to find perpendicular lines on each spline point; applies this to both mandible and maxilla slices.

---

Explore the local directory and see the segmented and the raw CBCT scans. Produce output and check the actual quality of the output visually and numerical criteria if available. Reiterate until result is production grade quality.
Also, I have a separated model that produced the segmentation, so please use the segmentations as input for this step, I will next add the segmentation part to the complete algorithm.

You can open the segmentations and see all labels, these are the list:
"labels": { "background": 0, "Lower Jawbone": 1, "Upper Jawbone": 2, "Left Inferior Alveolar Canal": 3, "Right Inferior Alveolar Canal": 4, "Left Maxillary Sinus": 5, "Right Maxillary Sinus": 6, "Pharynx": 7, "Bridge": 8, "Crown": 9, "Implant": 10, "Upper Right Central Incisor": 11, "Upper Right Lateral Incisor": 12, "Upper Right Canine": 13, "Upper Right First Premolar": 14, "Upper Right Second Premolar": 15, "Upper Right First Molar": 16, "Upper Right Second Molar": 17, "Upper Right Third Molar (Wisdom Tooth)": 18, "Upper Left Central Incisor": 21, "Upper Left Lateral Incisor": 22, "Upper Left Canine": 23, "Upper Left First Premolar": 24, "Upper Left Second Premolar": 25, "Upper Left First Molar": 26, "Upper Left Second Molar": 27, "Upper Left Third Molar (Wisdom Tooth)": 28, "Lower Left Central Incisor": 31, "Lower Left Lateral Incisor": 32, "Lower Left Canine": 33, "Lower Left First Premolar": 34, "Lower Left Second Premolar": 35, "Lower Left First Molar": 36, "Lower Left Second Molar": 37, "Lower Left Third Molar (Wisdom Tooth)": 38, "Lower Right Central Incisor": 41, "Lower Right Lateral Incisor": 42, "Lower Right Canine": 43, "Lower Right First Premolar": 44, "Lower Right Second Premolar": 45, "Lower Right First Molar": 46, "Lower Right Second Molar": 47, "Lower Right Third Molar (Wisdom Tooth)": 48, "Left Mandibular Incisive Canal": 103, "Right Mandibular Incisive Canal": 104, "Lingual Canal": 105, "Upper Right Central Incisor Pulp": 111, "Upper Right Lateral Incisor Pulp": 112, "Upper Right Canine Pulp": 113, "Upper Right First Premolar Pulp": 114, "Upper Right Second Premolar Pulp": 115, "Upper Right First Molar Pulp": 116, "Upper Right Second Molar Pulp": 117, "Upper Right Third Molar (Wisdom Tooth) Pulp": 118, "Upper Left Central Incisor Pulp": 121, "Upper Left Lateral Incisor Pulp": 122, "Upper Left Canine Pulp": 123, "Upper Left First Premolar Pulp": 124, "Upper Left Second Premolar Pulp": 125, "Upper Left First Molar Pulp": 126, "Upper Left Second Molar Pulp": 127, "Upper Left Third Molar (Wisdom Tooth) Pulp": 128, "Lower Left Central Incisor Pulp": 131, "Lower Left Lateral Incisor Pulp": 132, "Lower Left Canine Pulp": 133, "Lower Left First Premolar Pulp": 134, "Lower Left Second Premolar Pulp": 135, "Lower Left First Molar Pulp": 136, "Lower Left Second Molar Pulp": 137, "Lower Left Third Molar (Wisdom Tooth) Pulp": 138, "Lower Right Central Incisor Pulp": 141, "Lower Right Lateral Incisor Pulp": 142, "Lower Right Canine Pulp": 143, "Lower Right First Premolar Pulp": 144, "Lower Right Second Premolar Pulp": 145, "Lower Right First Molar Pulp": 146, "Lower Right Second Molar Pulp": 147, "Lower Right Third Molar (Wisdom Tooth) Pulp": 148 }

## Summary of Design Rationale

This approach is preferred over alternatives (e.g. fixed vertical splines, manual arch tracing, raw MIP) for the following reasons:

- **Segmentation-anchored**: the spline is derived from actual tooth positions, not from image intensity heuristics, making it robust to low-contrast scans or unusual jaw orientations.
- **Patient-adaptive**: the arch shape is unique to each patient's anatomy, automatically captured by the centroid-fitted spline.
- **Anatomically complete**: the perpendicular sampling planes capture everything in the strip — tooth, root, alveolar bone, maxilla — without needing special handling for interproximal regions.
- **Literature-validated**: this exact approach (spline from arch points + perpendicular plane sampling) is the standard method reported in peer-reviewed literature.

The main dependency is segmentation quality. If the segmentation model produces accurate centroids, the downstream panoramic reconstruction is geometrically sound by construction.

