# Baseline Evaluation Results

**Date:** April 3, 2026
**Test area:** 400m × 360m around (41.248644, -95.998878), Omaha NE
**Ground truth:** 54 poles, 6 streetlights (labeled in labeler_v2)
**Grid:** 36 cells at 80m spacing, 144 pre-downloaded oblique images

## Pipeline
GDino-Tiny (0.15 threshold) → NMS (0.3 IoU) → MASt3R multi-view consensus (2+ views) → height filter (4-25m) → 3D clustering (0.05 units) → GPS dedup (15m)

## Results

| Match radius | TP | FP | FP lights | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 5m | 11 | 93 | 2 | 43 | 10.6% | 20.4% | 0.139 |
| **10m** | **26** | **78** | **4** | **28** | **25.0%** | **48.1%** | **0.329** |
| 15m | 30 | 74 | 4 | 24 | 28.8% | 55.6% | 0.380 |
| 20m | 31 | 73 | 4 | 23 | 29.8% | 57.4% | 0.392 |
| 30m | 33 | 71 | 6 | 21 | 31.7% | 61.1% | 0.418 |

**Raw detections:** 158 before dedup → 104 unique
**Processing time:** 1000s (36 cells × ~28s each on Apple MPS)

## Analysis

### Georeferencing
Homography-based pixel→GPS conversion using crop center + azimuth rotation. Average match distance ~10m. Previous approach (pixel × GSD) had ~50-100m error.

### Precision issues (25% at 10m)
- 104 detections for 54 GT poles = many false positives
- 4 streetlights detected (correctly classified in GT, but counted as FP since we're detecting "poles")
- Remaining ~74 FPs are trees, fences, building edges that survive multi-view consensus + height filtering

### Recall issues (48% at 10m)
- 28 poles missed out of 54
- Likely causes: some poles occluded in oblique views, GDino not detecting them at low confidence threshold, or grid coverage gaps

## Improvement targets
- **Fine-tune GDino** on aerial oblique pole data → should reduce FPs (streetlights, trees)
- **Better georeferencing** → more matches at tighter radius
- **GDino-first optimization** → only run MASt3R where detections exist
- **CUDA (2x 3090)** → 5-7x faster processing

## Target metrics
- Precision > 80%
- Recall > 70%
- F1 > 0.75
