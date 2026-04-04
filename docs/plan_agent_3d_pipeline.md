# Plan: Agent-Based Pipeline + Full 3D Exploitation

## Context

We've improved F1 from 0.418 → 0.663 through fine-tuning + VLM classification + GT completion. The remaining bottlenecks are:
1. **Georeferencing error** (~10-15m) — causes TPs to be scored as FPs at tight radii
2. **Single-crop VLM classification** — only sees one angle, misses context
3. **Manual labeling/review** — humans spend 30-60 min verifying detections
4. **MASt3R underutilization** — we compute a full 3D scene but only use it as a binary multi-view filter

## Current System Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION                                │
│  EagleView API → 188 oblique images (N/E/S/W @ 45°)              │
│                → WMTS ortho tiles (zoom 19-23)                     │
│                → Camera metadata (azimuth, elevation, GSD, footprint)│
└─────────────────────────┬──────────────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────────────────┐
│                 DETECTION (per grid cell, 4 views)                 │
│                                                                    │
│  ┌─────────┐    ┌─────────────┐    ┌──────────────┐               │
│  │ GDino   │    │   MASt3R    │    │  Multi-View  │               │
│  │ Base    │───▶│ 3D Recon    │───▶│  Consensus   │               │
│  │ (0.9GB) │    │ (2.6GB)     │    │  (2+ views)  │               │
│  └─────────┘    └─────────────┘    └──────┬───────┘               │
│   10-17 dets     dense 3D pts              │                       │
│   per view       + camera poses            │                       │
│                                            │                       │
│                  ┌─────────────┐    ┌──────▼───────┐               │
│                  │   Height    │◀───│   3D Point   │               │
│                  │   Filter    │    │   Lookup     │               │
│                  │  (4-50m)    │    │  (per det)   │               │
│                  └──────┬──────┘    └──────────────┘               │
│                         │                                          │
│                  ┌──────▼──────┐                                   │
│                  │ 3D Cluster  │                                   │
│                  │ (0.05 units)│                                   │
│                  └──────┬──────┘                                   │
└─────────────────────────┼──────────────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────────────────┐
│                    GEOREFERENCING                                   │
│                                                                    │
│  pixel (cx,cy) ──▶ 4-point homography ──▶ GPS (lat,lon)          │
│                    (uses azimuth +                                  │
│                     crop_radius only)     ~10-15m error ← PROBLEM │
│                                                                    │
│  ⚠ IGNORES the 3D reconstruction we just computed!                │
└─────────────────────────┬──────────────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────────────────┐
│                   POST-PROCESSING                                  │
│                                                                    │
│  GPS dedup (15m) → Focus area filter → VLM classification         │
│                                        (single crop, 27B ollama)  │
│                                        pole/streetlight/fence/...  │
└─────────────────────────┬──────────────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────────────────┐
│                   EVALUATION                                       │
│  Detections vs GT at 10/15/20/30m match radius                    │
│  Current best: F1=0.663 @ 30m (P=65.6%, R=67.0%)                 │
└────────────────────────────────────────────────────────────────────┘
```

## What's Wrong: MASt3R is Massively Underutilized

MASt3R gives us per-pixel 3D point clouds + camera poses for all 4 views. Currently we:
- ✅ Use 3D projection for multi-view consensus (binary: visible in 2+ views?)
- ✅ Use 3D distance for within-cell clustering (0.05 unit threshold)
- ❌ DON'T use 3D coordinates for georeferencing (fall back to homography)
- ❌ DON'T use 3D shape for classification (vertical column vs spread tree vs flat building)
- ❌ DON'T use metric scale from known distances
- ❌ DON'T use 3D height measurement (use noisy bbox*GSD/sin formula instead)
- ❌ DON'T use 3D for cross-cell dedup (use GPS distance which has homography error)

## Proposed Architecture: 3 Improvements

### Improvement 1: 3D-Based Georeferencing (replaces homography)

**Current:** `pixel_to_gps(cx, cy, img_w, img_h, lat, lon, azimuth, crop_radius)` — 4-point homography with ~10-15m error.

**Proposed:** Use MASt3R's 3D reconstruction + known ground reference points to establish a metric 3D→GPS transform.

**How:**
1. MASt3R gives us 3D point cloud in its own coordinate system (arbitrary scale/rotation)
2. We know the cell center GPS `(lat, lon)` — this corresponds to the image center pixel in each view
3. We know the camera azimuth and crop_radius for each view
4. Using 4+ GCPs (image centers → known GPS), fit a 3D similarity transform (scale + rotation + translation) from MASt3R space → GPS space
5. For each detection's 3D point `p3d`, apply the transform to get GPS directly from 3D

**Why this is better:** The 3D point already accounts for oblique perspective, depth, and camera geometry. The homography assumes a flat ground plane; the 3D transform doesn't.

**Expected improvement:** Georeferencing error drops from 10-15m to 3-7m. This alone could push F1 at 15m from 0.618 to ~0.75+.

**Implementation:** Modify `run_grid_cell()` in `eval_testarea.py`:
```python
# After MASt3R alignment, fit 3D→GPS transform using image center GCPs
gcps_3d = []  # 3D coords of image centers in MASt3R space
gcps_gps = [] # Known GPS of image centers (cell lat,lon)
for i, d in enumerate(dir_list):
    h, w = pts3d[i].shape[:2]
    center_3d = pts3d[i][h//2, w//2]  # center pixel's 3D point
    gcps_3d.append(center_3d)
    gcps_gps.append((cell_lat, cell_lon))  # all views centered on same GPS

# Fit similarity transform: 3D → (east_m, north_m) offset from cell center
# Then for each detection: gps = transform(p3d) instead of homography(px, py)
```

### Improvement 2: Multi-View Verification Agent (replaces single-crop VLM)

**Current:** `classify_detections.py` sends ONE 384x384 crop to Qwen 3.5 27B. Misses context from other angles.

**Proposed:** For each detection, the agent:
1. Extracts the crop from the source view (already done)
2. Projects the 3D point into ALL other views using MASt3R poses → extracts crops from each
3. Loads an ortho tile at the approximate GPS
4. Sends ALL images (4 oblique + 1 ortho) to Qwen 3.5 in a single multi-image prompt
5. Qwen reasons across views: "I see a vertical wooden structure with crossarm in north and east views, wire shadows in ortho → pole"

**Tools for the agent:**
- `view_all_angles(det, pts3d, poses, focals, images)` — project 3D point into all views, extract crops
- `view_ortho(lat, lon, radius_m)` — load WMTS tiles around location
- `check_nearby_labels(lat, lon, radius_m)` — check for duplicate/nearby labels
- `classify_multiview(crops, ortho, prompt)` — send to Qwen 3.5 27B via ollama

**Key insight:** We already HAVE the 3D projection code (lines 130-147 in eval_testarea.py). It projects each detection into other views to check for nearby detections. We just need to also extract image crops at those projected locations.

**Expected improvement:** Precision from 65% → 80%+ by eliminating single-view ambiguity.

### Improvement 3: 3D Shape Analysis for Classification

**Current:** Height filter uses `bbox_h * GSD / sin(elevation)` — noisy, doesn't use 3D.

**Proposed:** Use the dense 3D point cloud within each detection bbox to analyze the object's 3D shape:
- **Vertical extent:** Measure actual 3D height by sampling the point cloud along the bbox vertical axis. Poles are tall and thin.
- **Aspect ratio in 3D:** Poles have high vertical-to-horizontal ratio. Trees are wider. Buildings are planar.
- **Vertical straightness:** Sample 3D points along the detection column. Poles follow a straight vertical line. Trees branch out.

**Implementation:**
```python
# Extract 3D points within the detection bbox
x1, y1, x2, y2 = det['bbox']
# Scale to MASt3R resolution
pts_in_box = pts3d[i][my1:my2, mx1:mx2]  # 3D points within bbox
# Analyze vertical spread
z_range = pts_in_box[:,:,2].max() - pts_in_box[:,:,2].min()  # height in 3D
xy_spread = ...  # horizontal spread
aspect_3d = z_range / max(xy_spread, 0.001)
# Poles: aspect_3d > 3.0, Trees: aspect_3d < 2.0, Buildings: aspect_3d < 1.0
```

**Expected improvement:** Better than the current noisy height formula. Could replace or supplement the VLM for some clear-cut cases.

## Implementation Plan (ordered by impact)

### Phase A: 3D Georeferencing (highest impact, moderate effort)
1. Create `src/georef_3d.py` — 3D similarity transform fitting
2. Modify `run_grid_cell()` to use 3D georeferencing instead of homography
3. Re-run eval to measure GPS error reduction
4. **Files:** `src/eval_testarea.py` (modify), `src/georef_3d.py` (new)

### Phase B: Multi-View Agent (high impact, moderate effort)
1. Create `src/agent_tools.py` — shared tools (ortho loading, oblique projection, ollama multi-image)
2. Create `src/multiview_agent.py` — multi-view verification replacing single-crop classify
3. Benchmark on the 192 current detections vs single-crop approach
4. **Files:** `src/agent_tools.py` (new), `src/multiview_agent.py` (new)

### Phase C: 3D Shape Features (moderate impact, low effort)
1. Add 3D shape extraction to `run_grid_cell()` — vertical extent, aspect ratio, straightness
2. Use as additional features alongside VLM classification
3. **Files:** `src/eval_testarea.py` (modify)

### Phase D: Coverage Gap Agent (lower impact, low effort)
1. Create `src/coverage_agent.py` — analyze pole spacing, find gaps
2. Verify gaps with multi-view agent
3. **Files:** `src/coverage_agent.py` (new)

## Verification
1. **Phase A:** Compare GPS error distribution before/after on known GT poles. Target: median error < 7m.
2. **Phase B:** Compare precision on the 192 detections: single-crop vs multi-view. Target: precision > 75%.
3. **Phase C:** Check if 3D shape features correctly separate poles/trees/buildings on known examples.
4. **Phase D:** Count newly discovered poles in gaps. Verify with human review.
5. **End-to-end:** F1 target > 0.75 at 15m match radius.

## Key Files
- `src/eval_testarea.py` — main pipeline, needs 3D georef + shape features
- `src/oblique_utils.py` — has `gps_to_pixel()`, `pixel_to_gps()` used by agents
- `src/classify_detections.py` — current single-crop VLM, replaced by multiview agent
- `src/download_wmts.py` — tile math for ortho loading
- `models/mast3r/` — MASt3R model and dust3r dependencies
