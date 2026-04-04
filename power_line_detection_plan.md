# Power Pole Detection from EagleView Imagery
## Project Plan for TerraByte Technical Challenge

---

## Project Goal
Build an end-to-end pipeline to identify and locate power poles in a target town using EagleView satellite imagery (orthomosaic + oblique views). Optimize for F1 balance. Identify where agent-based automation and human-in-the-loop can improve efficiency and model performance.

---

## Phase 1: Data Acquisition and Exploration

### 1.1 Retrieve EagleView Data
- **Input:** Target town (from available oblique coverage: Denver, LA, SF, Phoenix, Atlanta, Miami, NYC, Chicago, Boston)
- **API:** Use EagleView imagery API key to query bounding box
- **Expected Output:**
  - Orthomosaic GeoTIFF (3-12 inches/pixel, georeferenced)
  - Oblique views: 4 angles (N, S, E, W) at 3-6 inches/pixel
  - Camera pose metadata (intrinsics, extrinsics) for each oblique view
  - File sizes: ~50-200MB ortho, ~100-500MB oblique total
- **Deliverable:** Local directory with imagery files and metadata JSON

### 1.2 Data Exploration & Ground Truth
- Visually inspect imagery to understand power line visibility in oblique vs. nadir views
- Document where power lines are obvious vs. obscured (vegetation, shadows, occlusion)
- Identify 10-20 known power pole locations in the target area for validation (manual inspection)
- **Bottleneck Flag:** Manual ground truth collection is time-consuming
- **Agent Opportunity:** Use an LLM agent to identify utility poles, substations, and transmission towers directly — these are the primary detection targets

---

## Phase 2: Power Line Localization (Agent-Assisted)

### 2.1 The Core Problem
SAM3 segments power poles once you point at them — but you need to know **WHERE to look first**. Scanning the entire town is computationally wasteful and slow.

### 2.2 Agent-Based Region of Interest Detection
Use a vision-language model (e.g., GeoRSCLIP or CLIP) to scan oblique views and flag regions likely to contain power poles and utility infrastructure.

**What the agent looks for:**
- Utility poles (the primary target)
- Transmission towers
- Transformer boxes
- Pole clusters
- Substation boundaries

**Process:**
```
For each oblique view tile:
  1. Query vision-language agent: "Identify utility poles, transmission
     towers, and related infrastructure in this image"
  2. Agent returns bounding box coordinates of detected poles
  3. Expand bounding boxes by buffer (e.g., 20 pixels) around each pole
  4. Merge overlapping ROIs across N/S/E/W views
  5. Output: prioritized list of candidate pole locations for SAM3
```

**Output:** List of ROI bounding boxes (lat/lon + image pixel coordinates) centered on detected poles, sorted by confidence score

**Efficiency Gain:** 5-10x reduction in SAM3 processing area. Instead of tiling the entire town, you focus on high-probability pole locations.

**Validation Check:** Before proceeding, verify that your 10-20 manually identified ground truth pole locations fall within the agent-generated ROIs. If recall < 90%, widen the buffer or lower the confidence threshold.

---

## Phase 3: Power Line Segmentation with SAM3

### 3.1 SAM3 Inference on Oblique Views
- **Input:** ROI-cropped oblique image chips centered on candidate pole locations (from Phase 2)
- **Model:** SAM3 — use ViT-Base variant (~2.5GB GPU memory)
- **Why oblique views:** SAM3 was trained primarily on ground-level images. Oblique (45-degree) angles are geometrically closer to ground perspective than nadir, making SAM more effective at detecting vertical structures like utility poles
- **Prompt strategy:**
  - Primary: SAM automatic mask generation on each ROI chip centered on pole
  - Fallback: Point prompt (click on pole base or top) for ambiguous regions
- **Output:** Binary segmentation masks isolating the pole structure per oblique view per ROI
- **Inference time:** ~500ms per chip on GPU

### 3.2 Human-in-the-Loop Mask Quality Control
The agent flags uncertain or low-confidence masks for your review. You handle the final accept/reject decisions.

**Agent QA Criteria:**
- Mask is too large or too small to be a utility pole
- Mask doesn't have the expected vertical/cylindrical shape
- Segments that don't match pole-like aspect ratios
- Low SAM3 confidence scores
- Conflicting masks across views for the same location

**Your QA Interface (simple):**
```
For each flagged mask:
  - Display: original oblique image + SAM3 overlay
  - Show: confidence score, segment length, connectivity
  - Options:
    [A] Accept  [R] Reject  [E] Edit (re-run SAM with manual prompt)
```

**Output:** Validated binary masks with confidence scores, tagged as accepted/rejected/edited

---

## Phase 4: 3D Reconstruction and Fusion with MASt3R

### 4.1 MASt3R Dense Matching
- **Input:** Multiple EagleView views (oblique N/S/E/W + orthomosaic) with camera pose metadata
- **Model:** MASt3R (~8-10GB GPU memory) — dense pixel-to-pixel 3D matching
- **Output per view:**
  - Dense depth map (Z value per pixel)
  - Refined camera poses
  - 3D point cloud of scene geometry

**Note:** MASt3R expects camera intrinsics (focal length, principal point) and extrinsics (rotation, translation). Verify these are included in EagleView API metadata. If not, estimate from EXIF or focal length specs.

### 4.2 Project SAM3 Masks to 3D
For each validated mask from Phase 3.2:
```
For each power-pole pixel (u, v) in mask:
  1. Look up depth value Z = depth_map[u, v]
  2. Backproject to 3D: X = (u - cx) * Z / fx
                        Y = (v - cy) * Z / fy
  3. Apply camera extrinsics to get world coordinates (X_w, Y_w, Z_w)
  4. Store as 3D point tagged with source view and confidence
```

**Output:** 3D point cloud of power pole candidates (one cluster per pole per view)

### 4.3 Multi-View Fusion in 3D
Fuse pole detections from all views using voxel-based voting:

```
1. Define voxel grid over scene bounding box (e.g., 0.5m x 0.5m x 0.5m cells)
2. For each 3D pole point, increment the corresponding voxel counter
3. Compute per-voxel confidence = (# views detecting pole) / (# total views)
4. Keep voxels with confidence >= 0.5 (at least 2 of 4 views agree)
5. Apply morphological operations to clean noise (remove isolated voxels)
6. Find centroid of each surviving voxel cluster = pole location
```

**Output:** High-confidence 3D pole locations (one point per pole)

---

## Phase 5: Point Extraction and Georeferencing

### 5.1 Pole Location Extraction
```
1. Cluster 3D point cloud using DBSCAN (eps=1m, min_samples=5)
2. For each cluster:
   a. Compute centroid (X_w, Y_w, Z_w) = pole base location
   b. Estimate pole height from cluster vertical extent
   c. Assign confidence score from voxel fusion step
3. Filter out clusters smaller than minimum size threshold
```

**Output:** List of 3D pole centroids with confidence scores

### 5.2 GPS Coordinate Conversion
```
1. Load EagleView georeferencing transform (from GeoTIFF metadata or API)
2. Apply transform: (X_w, Y_w, Z_w) -> (lat, lon, elevation)
3. Export as GeoJSON Point features
4. Include metadata per pole: confidence score, estimated height, view count
```

**Output:** `power_poles.geojson` with GPS point locations and metadata

---

## Phase 6: Evaluation and Iteration

### 6.1 F1 Score Calculation
- **Ground Truth:** 10-20 manually verified power pole locations (from Phase 1.2)
- **Predicted:** GeoJSON output from Phase 5.2
- **Matching criterion:** Predicted pole within X meters of ground truth pole counts as true positive (suggest X = 3m)

**Metrics:**
```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * (Precision * Recall) / (Precision + Recall)
Target    = F1 > 0.85
```

### 6.2 Error Analysis

**False Positives (detected but wrong):**
- Common causes: trees, lamp posts, fence posts, chimneys
- Fix: Tighten SAM3 mask shape filtering (poles have specific aspect ratios), improve agent to distinguish utility poles from other vertical structures

**False Negatives (missed poles):**
- Common causes: occlusion by trees/buildings, oblique angle blind spots, agent missed the ROI
- Fix: Add more ROI buffer, lower agent confidence threshold, check if specific view angle is systematically missing poles

### 6.3 Iteration Loop
```
1. Calculate F1 on ground truth pole locations
2. If F1 < 0.85:
   a. Analyze false positive / false negative distribution
   b. If high FP: tighten mask shape filtering or fusion threshold
   c. If high FN: improve agent ROI coverage or add manual ROIs
   d. Re-run affected phases
   e. Recalculate F1
3. Repeat until F1 > 0.85 or diminishing returns
```

---

## Agent Integration Summary

| Phase | Agent Role | Human Role | Efficiency Gain |
|-------|-----------|------------|-----------------|
| Phase 2 | Detect utility poles, generate ROIs | Validate recall on ground truth | 5-10x less SAM3 work |
| Phase 3 | Flag uncertain/low-confidence pole masks | Accept / Reject / Edit flagged masks | 2-3x QA speedup |
| Phase 6 (future) | Learn from corrections, improve pole detection | Correct errors, retrain agent | Compounding improvement |

---

## GPU and Memory Budget

| Component | GPU Memory | Notes |
|-----------|-----------|-------|
| SAM3 (ViT-Base) | ~2.5 GB | Batch process ROI chips |
| MASt3R | ~8-10 GB | Dense matching is heaviest |
| Data buffers | ~2-3 GB | Intermediate arrays, point clouds |
| **Total** | **~13-16 GB** | Fits on 24GB GPU with headroom |

---

## Timeline Estimate (Single Developer)

| Phase | Time Estimate | Notes |
|-------|--------------|-------|
| Phase 1: Data | 1-2 hours | Mostly API download wait time |
| Phase 2: Agent ROI | 2-4 hours | Depends on agent implementation complexity |
| Phase 3: SAM3 + QA | 4-6 hours | Includes your manual review time |
| Phase 4: MASt3R fusion | 1-2 hours | Compute-heavy but automated |
| Phase 5: Vectorization | 1-2 hours | Mostly automated |
| Phase 6: Evaluation | 2-4 hours | Includes iteration loop |
| **Total** | **11-20 hours** | Reduced significantly with agent optimization |

---

## Implementation Checklist

### Phase 1
- [ ] Select target town with oblique coverage
- [ ] Set up EagleView API key
- [ ] Query bounding box and download imagery
- [ ] Extract and validate camera pose metadata
- [ ] Manually identify 10-20 ground truth power pole locations

### Phase 2
- [ ] Implement vision-language agent for utility pole detection
- [ ] Run agent across all oblique view tiles
- [ ] Generate and merge ROI bounding boxes centered on poles
- [ ] Validate agent recall against ground truth (target: >90%)

### Phase 3
- [ ] Set up SAM3 inference pipeline (ViT-Base)
- [ ] Run SAM3 on ROI-cropped oblique chips
- [ ] Implement agent mask QA (confidence scoring, flag criteria)
- [ ] Build simple accept/reject human review interface
- [ ] Store validated masks with metadata

### Phase 4
- [ ] Set up MASt3R with camera pose inputs
- [ ] Run dense matching across all views
- [ ] Implement 2D mask -> 3D backprojection
- [ ] Implement voxel-based fusion
- [ ] Output 3D power line point cloud

### Phase 5
- [ ] Implement DBSCAN clustering to extract pole centroids
- [ ] Implement confidence scoring per pole location
- [ ] Implement GPS coordinate conversion
- [ ] Export power_poles.geojson

### Phase 6
- [ ] Implement F1 evaluation against ground truth
- [ ] Analyze FP/FN distribution
- [ ] Iterate on weak phases
- [ ] Document final F1 and performance

---

## Key Open Questions to Resolve

1. **Target town selection:** Which city? Choose one with confirmed oblique coverage and good pole visibility
2. **Camera pose metadata:** Confirm EagleView API returns intrinsics + extrinsics per oblique view. If not, document estimation fallback
3. **Agent model choice:** GeoRSCLIP (remote sensing specific) vs generic CLIP for pole detection. GeoRSCLIP likely better for aerial context
4. **Fusion threshold:** Start at 0.5 (2/4 views agree), tune based on F1 results
5. **Human QA interface:** Command-line viewer vs simple web UI (Streamlit or Gradio)
6. **Matching radius for F1:** Start at 3m — a pole detected within 3m of ground truth counts as correct

---

## Future Work (Post Phase 1 Pipeline)

- **Feedback loop:** Agent learns from your accept/reject decisions to improve pole detection over iterations
- **Active learning:** Agent identifies most uncertain pole detections and prioritizes those for human review
- **Power line inference:** Once poles are located, optionally infer connecting power lines by linking nearby poles
- **Cross-town generalization:** Test pipeline on second town without retraining to measure generalization
- **Online adaptation:** Implement retrieval memory to adapt detections as new imagery arrives
