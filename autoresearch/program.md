# AutoResearch: Pole Detection Pipeline Optimization

## Objective
Maximize F1@10m for detecting utility poles in aerial imagery.
Current best: **F1@10m = 0.5918** (SAM3 threshold=0.40, ortho_radius=60m)
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
- **SAM3 threshold 0.10→0.20**: F1 0.3348→0.4876. Precision nearly doubled (0.21→0.37) with modest recall drop (0.82→0.73). Many FPs were low-confidence — threshold is a high-leverage knob.
- **SAM3 threshold 0.20→0.30**: F1 0.4876→0.5182. Precision 36.5%→45.2%, recall 73.4%→60.6%. Still net positive — FP count halved (120→69) while losing 12 TPs (69→57). Threshold tuning may still have room but recall is getting thin.
- **SAM3 threshold 0.30→0.35**: F1 0.5182→0.5268. Precision 45.2%→48.6%, recall 60.6%→57.5%. Diminishing gains — only 12 fewer FPs (69→57) with 3 fewer TPs (57→54). Threshold approaching plateau; next iteration should try a different lever.
- **Ortho crop radius 80m→60m**: F1 0.5268→0.5592. Precision 48.6%→50.4%, recall 57.5%→62.8%. RMSE improved 5.8m→4.9m. Smaller crop gives MASt3R higher resolution, improving both projection accuracy and recall. Still room to explore (50m? 70m?).
- **SAM3 threshold 0.35→0.40**: F1 0.5592→0.5918. Precision 50.4%→56.9%, recall 62.8%→61.7%. Only lost 1 TP (59→58) but removed 14 FPs (58→44). Threshold tuning still has legs — diminishing returns but still net positive. RMSE 4.8m.

## What We've Tried and Failed
- **SAM3 prompt "utility pole" instead of "telephone pole"**: F1 0.5268→0.165. Catastrophic regression — SAM3 detects very different objects with "utility pole". The prompt "telephone pole" is critical and should not be changed.
- **Multi-view consensus (min 2 detections per cluster)**: F1 0.5268→0.4865. Precision soared (48.6%→66.7%) but recall cratered (57.5%→38.3%). Lost 18 TPs (54→36) — too many real poles are only visible from one direction. The filter is too aggressive at cluster_size>=2; could revisit with a softer version (e.g., boost score for multi-view but don't hard-filter).
- **Ortho crop radius 60m→50m**: F1 0.5592→0.5577. Marginal regression — 50m slightly too small, losing 1 TP and minor recall drop (62.8%→61.7%). 60m appears to be near optimal; 50m is past the sweet spot.
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
