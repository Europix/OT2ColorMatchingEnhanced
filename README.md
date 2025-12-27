# OT2-based Plate Color-Matching

This project implements an computer vision pipeline for microplate images, focusing on robust geometric alignment and relative color consistency rather than absolute color accuracy.

---

## 1. Credentials / Dependencies

### Python
- Python **3.9 – 3.11** recommended  
(Tested on Windows 10/11)

### Required Libraries
```
pip install numpy opencv-python torch torchvision pillow pandas matplotlib pupil-apriltags
```

### Hardware / Setup Assumptions
- Fixed camera setup (strongly recommended)
- Custom **plate color ring** containing:
  - 4 AprilTags (`tag36h11`, IDs 0–3)
  - Top & bottom color bars
  - Left & right grayscale bars
- Plate approximately centered inside the ring

---

## 2. Project Structure

```
.
├── plate.py                             # Main end-to-end pipeline
├── plate_model_resnet18_colorRing.pth   # Trained CNN (plate type classifier)
├── vis_96well_from_csv.py               # CSV → 96-well visualization
├── Fixed*_well_colors.csv               # Example outputs
├── images/                              # Input images
└── README.md
```

---

## 3. Pipeline Overview (How It Works)

### Step 1 – Plate Type Classification (CNN)
- A **ResNet-18** model classifies the plate type from the **entire image**
- Output: probability distribution over  
  `nothing, 1, 6, 12, 24, 48, 96, 384`

This determines the expected **grid shape** (e.g. 96-well → 8×12).

---

### Step 2 – AprilTag Detection
- Detect four `tag36h11` AprilTags (IDs **0,1,2,3**)
- Extract each tag’s **center in pixel coordinates**

These tags provide a **stable reference frame**, independent of resolution, rotation, or scale.

---

### Step 3 – Homography Estimation
- Known **design-space tag layout** is matched to detected tag centers
- A homography is computed:

```
(design coordinates) → (image pixels)
```

This allows any logical plate coordinate (corners, grid points, color patches) to be projected into the image consistently.

---

### Step 4 – Plate Geometry & Grid Fitting
Using the homography:

- Project outer plate rectangle (debug: green)
- Compute inner “valid well” rectangle using **normalized margins**
- Generate a uniform grid inside the inner rectangle
- Project grid centers into image space (debug: blue dots)

This avoids brittle pixel-based heuristics.

---

### Step 5 – Color Calibration
The color ring provides reference patches:

- Top & bottom color bars
- Left & right grayscale bars

For each patch:
- Mean BGR is sampled
- A **linear affine color transform** is fitted:

```
[B, G, R, 1] → [B', G', R']
```

Goal:
- Reduce lighting bias
- Improve **relative color consistency** across wells

---

### Step 6 – Per-Well Color Extraction
- Sample each well center
- Apply color calibration
- Save calibrated BGR values to CSV

Example:
```
C3,42.1,97.3,211.8
```

---

### Step 7 – Visualization (Optional)
A helper script renders a **96-well plate map** from the CSV.

Used for:
- Alignment debugging
- Color trend validation
- Reporting / presentation

---

## 4. How to Run

### Run Full Pipeline
```
python plate.py path/to/image.jpg
```

Outputs:
- `*_debug_grid.jpg` – geometric alignment visualization
- `*_well_colors.csv` – calibrated per-well colors

---

### Visualize a 96-Well CSV
```
python output_debug.py your_csv
```

---

## 5. Known Limitations

- Physical margins vary across plate brands and types
- Side-wall reflections can dominate color in some wells
- Homography assumes tags lie on the same plane as the plate
- Absolute color accuracy is not guaranteed  
  (relative consistency is the primary goal)
