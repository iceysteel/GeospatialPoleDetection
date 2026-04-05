# AutoResearch: Pole Detection Pipeline Optimization

## Objective
Maximize F1@10m for detecting utility poles in aerial imagery.
Current best: **F1@10m = 0.4876** (SAM3 threshold=0.20)
Previous best: F1@10m = 0.3348 (SAM3 threshold=0.10)
SAM3+MASt3R baseline: **F1@10m = 0.165**

## Hard Constraints
- MUST use SAM3 (or SAM3-LoRA) for detection in oblique views
- MUST use MASt3R for oblique→ortho cross-view mapping
- MUST produce pole locations as GPS coordinates
- 10 minute timeout per experiment
- Metric: F1 at 10m match radius against 94 verified GT poles

## What We Know Works
- SAM3 with "telephone pole" prompt finds poles but only ~2/image on full obliques
- SAM3-LoRA v2 (clean training data) finds ~10/image but many are false positives
- MASt3R AerialMegaDepth at 80m ortho crop radius gives 2.6m RMSE on known points
- Pole base projection (bottom of bbox) is more accurate than bbox center
- GPS dedup at 10m removes duplicate detections across views

## What Worked
- **SAM3 threshold 0.10→0.20**: F1 0.3348→0.4876. Precision nearly doubled (0.21→0.37) with modest recall drop (0.82→0.73). Many FPs were low-confidence — threshold is a high-leverage knob. Further threshold tuning likely still beneficial.

## What We've Tried and Failed
- AerialMegaDepth vs standard MASt3R for oblique↔oblique: identical performance
- Large ortho crops (200m): poles invisible at 512px MASt3R resolution
- Small ortho crops (30m): not enough context for MASt3R matching
- Per-detection MASt3R crops: too slow and unreliable
- 3D shape features from MASt3R at 512px: too coarse to distinguish poles from trees

## Promising Directions to Explore
1. **SAM3 prompt engineering**: try different prompts, visual exemplars
2. **SAM3 threshold tuning**: find optimal threshold for LoRA v2
3. **Ortho crop radius sweep**: find optimal balance between context and resolution
4. **Score-based filtering**: only keep high-confidence SAM3 detections
5. **Multi-view consensus**: require pole detected in 2+ oblique views before projecting
6. **VLM post-filtering**: use Qwen 3.5 to classify detection crops
7. **Ensemble**: combine SAM3 base + LoRA detections
8. **MASt3R iteration count**: try fewer/more alignment iterations

## Population (Top Configs)
See population.json

## Experiment Log
See autoresearch.jsonl
