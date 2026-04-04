# Multi-View Consensus Power Pole Detection Pipeline

## Context
We have ~190 EagleView oblique + ortho images (PNG, max zoom, ~4cm GSD) of a 1.5 sq mile area in Omaha, NE. Single-view detectors (SAM2, GroundingDINO, Qwen3-VL) all struggle with aerial oblique imagery — high false positive rates because they're trained on ground-level photos. The key insight: **multi-view consensus using MASt3R 3D correspondences** dramatically reduces false positives.

## What We Have
- **188 oblique images** downloaded: 37 north, 57 east, 40 south, 51 west + 3 ortho (1.39GB)
- **1,333 WMTS ortho tiles** at zoom 19 for map background
- **SAM2** installed with `sam2.1_hiera_base_plus` checkpoint (309MB)
- **MASt3R** installed with `model.safetensors` checkpoint (2.6GB)
- **GroundingDINO-Tiny** (172M params via HF transformers, ~1-2s per image on MPS)
- **62 labeled detections** (28 pole, 15 streetlight, 9 fence, 7 other, 3 tree) — 45% precision baseline

## Experimental Results

### Single-View Detection (tested at 41.248644, -95.998878)
| Model | Detections | Time/img | Notes |
|-------|-----------|----------|-------|
| SAM2 auto + AR filter | 2-19 | 15-135s | Many false positives, slow |
| GroundingDINO-Tiny | 10-17 | 1-2s | Fast but noisy, low confidence |
| Qwen3-VL 2B | 0-2 | 5-50s | Unreliable, stuck in thinking loops |

### Multi-View Consensus Pipeline (10 locations, after NMS + 3D clustering + height filter)
| Metric | Value |
|--------|-------|
| Raw GDino detections | 196 (after NMS) |
| Multi-view confirmed | 105 |
| After height filter (4-25m) | 62 unique poles |
| **Precision** (human labeled) | **45%** |
| Top false positive | Streetlights (24%) |

---

## Current Pipeline (Working)

### Phase 1: Per-View Detection with GroundingDINO
**File:** `src/batch_multiview.py`

For each location:
1. Download 4 oblique crops (N/E/S/W) centered on same GPS point, 50m radius, max zoom
2. Run GroundingDINO on each view → candidate bounding boxes
3. Apply NMS (IoU > 0.3) to remove duplicate detections per view

### Phase 2: MASt3R 3D Correspondences
For each set of 4 views:
1. Load into MASt3R at 512px
2. Run inference on all 6 view pairs (~8s)
3. Global alignment → unified 3D coordinate frame + camera poses (~17s)

### Phase 3: Multi-View Consensus Voting
For each detection:
1. Map detection center to 3D point via MASt3R
2. Project 3D point into other views using camera poses
3. Check if projected point overlaps with a detection in that view
4. Keep detections with 2+ view agreement

### Phase 4: Height Filter
- Estimate real-world height from bbox pixel height using camera geometry:
  `height = h_px × GSD / sin(elevation_angle)`
- Reject anything < 4m or > 25m

### Phase 5: 3D Clustering
- Cluster confirmed detections by 3D proximity (0.05 MASt3R units)
- Merge overlapping detections of same physical pole
- Keep highest confidence, aggregate view counts

---

## Next: Fine-Tune GroundingDINO

### Step 1: Convert Labels to COCO Format
**File:** `src/prepare_training_data.py`

Convert `data/ground_truth.json` to COCO detection format:
- Crop source images around each detection bbox with padding
- Split 80/20 train/val (stratified by label)
- Categories: utility_pole, streetlight, tree, fence, other
- Output: `data/training/train.json`, `data/training/val.json`

### Step 2: Fine-Tune GroundingDINO-Tiny
**Files:** `training/train_gdino.py`, `training/requirements.txt`, `training/config.yaml`

Standalone Python scripts for CUDA machine:
- Base model: `IDEA-Research/grounding-dino-tiny`
- Learning rate: 1e-5, batch size: 2-4, epochs: 30-50
- Freeze text encoder, train vision backbone + decoder
- Text prompts: "utility pole", "streetlight", "tree", "fence"
- Save checkpoints every 5 epochs

### Step 3: Evaluate Fine-Tuned Model
**File:** `src/evaluate.py`

- Per-class AP at IoU=0.5
- Full pipeline precision/recall with fine-tuned weights
- Target: precision > 80%, recall > 70%, F1 > 0.75
- Error analysis: what's still being misclassified

### Step 4: Agentic Labeling System
**File:** `src/agentic_labeler.py`

Use fine-tuned model to scale labeling:
1. Run fine-tuned GDino on unlabeled images
2. Auto-label high confidence (> 0.5) detections
3. Queue low confidence for human review in labeler UI
4. Active learning: prioritize most uncertain images
5. Human reviews → add to training set → retrain

---

## File Structure
```
powerpolefinder/
  src/
    auth.py, discovery.py, download.py, main.py     # Data acquisition (done)
    ratelimit.py, download_wmts.py                   # Rate limiting, WMTS (done)
    compare_detection.py                              # Single-view comparison (done)
    test_multiview.py                                 # Single-location consensus test (done)
    batch_multiview.py                                # Batch consensus pipeline (done)
    prepare_training_data.py                          # COCO format conversion
    evaluate.py                                       # Evaluation metrics
    agentic_labeler.py                                # Automated labeling
  training/
    train_gdino.py                                    # Fine-tuning script (CUDA)
    requirements.txt
    config.yaml
  models/
    sam2/                                             # SAM2 repo + checkpoint
    mast3r/                                           # MASt3R repo + checkpoint
  data/
    metadata.json                                     # Image metadata
    ground_truth.json                                 # Human labels
    training/                                         # COCO format training data
    debug/batch/                                      # Batch test results
    oblique/, ortho/, wmts/                           # Downloaded imagery
  docs/
    plan_multiview_consensus.md                       # This plan
  viewer.html                                         # Map viewer
  comparison.html                                     # Detection comparison
  batch_results.html                                  # Batch results viewer
  labeler.html                                        # Labeling UI
```
