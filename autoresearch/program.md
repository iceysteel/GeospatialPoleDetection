# AutoResearch: Pole Detection Pipeline Optimization

## Objective
Maximize F1@10m for detecting utility poles in aerial imagery.
Current best: **F1@10m = 0.714** (3-prompt SAM3 [telephone/wooden/power pole], power@0.65, thresh=0.40, ortho=60m, dedup=15m, two-tier sv_min=0.45, MASt3R PCO 100 iters)

## Progress So Far
- Baseline: F1=0.335 (SAM3 thresh=0.10, ortho=80m)
- Iteration 1: thresh→0.20 → F1=0.488 ✅
- Iteration 2: thresh→0.30 → F1=0.518 ✅
- Iteration 3: thresh→0.35 → F1=0.527 ✅
- Iteration 4: multi-view consensus → F1=0.487 ❌ (killed recall)
- Iteration 5: prompt→"utility pole" → F1=0.489 ❌ (regression)
- Iteration 6: ortho_crop 80m→60m → F1=0.559 ✅
- Iteration 7: ortho_crop 60m→50m → F1=0.558 ❌ (marginal)
- Iteration 8: thresh→0.40 → F1=0.592 ✅
- Iteration 9: VLM post-filter (strict) → F1=0.567 ❌ (lost too many TPs)
- Iteration 10: thresh→0.45 → F1=0.598 ✅ (new best)
- Iteration 11: thresh 0.35 + soft VLM → F1=0.559 ❌ (VLM barely filtered)
- Iteration 12: SAM3-LoRA v2 → F1=0.086 ❌ (catastrophic failure)
- Iteration 13: aspect ratio filter → F1=0.592 ❌ (no effect, filter too loose)
- Iteration 14: ortho crop 55m → F1=0.586 ❌ (smaller crop hurt recall)
- Iteration 15: MASt3R 768px → F1=0.536 ❌ (higher res hurt matching)
- Iteration 16: score-weighted centroid → F1=0.586 ❌ (hurt positioning)
- Iteration 17: threshold 0.40 + dedup 8m → F1=0.566 ❌ (more FP)
- Iteration 18: threshold 0.45 + dedup 8m → F1=0.564 ❌ (more FP)
- Iteration 19: threshold 0.42 → F1=0.596 ❌ (marginal, 0.45 still best)
- Iteration 20: VLM post-filter borderline → F1=0.559 ❌ (VLM classified everything as pole)
- Iteration 21: cluster scoring (thresh 0.40, cluster_score>=0.55) → F1=0.592 ❌ (killed 8 TPs)
- Iteration 22: GDino ensemble (thresh 0.30) → F1=0.452 ❌ (90 extra FPs)
- Iteration 23: GDino ensemble (thresh 0.50) → F1=0.582 ❌ (still too many FPs)
- Iteration 24: GDino cross-validation → F1=0.595 ❌ (GDino confirms everything)
- Iteration 25: multi-point projection → F1=0.591 ❌ (hurt localization, RMSE +0.5m)
- Iteration 26: threshold 0.48 + ortho 60m → F1=0.592 ❌ (recall drops too much)
- Iteration 27: dedup 12m + ortho 60m + thresh 0.45 → F1=0.616 ✅ NEW BEST!
- Iteration 28: two-tier (thresh 0.40, single-view min 0.55) → F1=0.614 ❌ (too aggressive)
- Iteration 29: two-tier (thresh 0.40, single-view min 0.45) → F1=0.624 ✅ NEW BEST!
- Iteration 30: two-tier single-view min 0.50 → F1=0.623 ❌ (marginal)
- Iteration 31: two-tier thresh 0.35 + sv min 0.45 → F1=0.617 ❌ (more FPs)
- Iteration 32: dedup 14m + two-tier → F1=0.651 ✅ NEW BEST!
- Iteration 33: dedup 16m + two-tier → F1=0.650 ❌ (merged 1 TP)
- Iteration 34: multi-prompt (telephone+wooden pole) → F1=0.667 ✅ NEW BEST!
- Iteration 35-36: 3 prompts — too many FPs, reverted
- Iteration 37: 2-prompt + dedup 16m → F1=0.647 ❌ (merged TPs)
- Iteration 38: per-prompt thresholds → F1=0.667 (no change, reverted)
- Iteration 39: MASt3R 200 iters → F1=0.667 (same, worse RMSE, reverted)
- Iteration 40: MASt3R 150 iters → F1=0.686 ✅ NEW BEST!
- Iteration 41: MASt3R 100 iters → F1=0.690 ✅ NEW BEST!
- Iteration 42-48: various (75 iters, median GPS, thresh 0.35, lr tuning, ortho 65m, dual proj) — all worse
- Iteration 49: PointCloudOptimizer (was Modular) → F1=0.702 ✅ NEW BEST!
- Iteration 50-55: PCO 150/50 iters, thresh tuning, sv_min tuning, pole prompt — all worse
- Iteration 56: PCO + dedup 15m → F1=0.707 ✅ NEW BEST!
- Iteration 57: VLM post-filter (Qwen 3.5 27B, conservative) → F1=0.675 ❌ (removed 4 TPs, 0 FPs)
- Iteration 58: direction-aware multi-view + thresh 0.35 + sv_min 0.50 → F1=0.675 ❌ (4 extra FPs)
- Iteration 59: trimmed mean GPS → F1=0.695 ❌ (lost 1 TP)
- Iteration 60: power pole prompt at 0.40 → F1=0.670 ❌ (8 more TPs but 25 more FPs)
- Iteration 61: power pole at 0.55 → F1=0.681 ❌ (5 more TPs but 16 more FPs)
- Iteration 62: power pole at 0.65 → F1=0.714 ✅ NEW BEST! (+1 TP, same FPs)
- Iteration 63-73: electric pole, sv_min 0.42, dedup 14m, linear schedule, utility pole@0.70, MASt3R 120 iters, 95% bbox height, quality-aware GPS, 448px, 3x3 median, GDino fallback — all ≤0.714

## Hard Constraints
- MUST use SAM3 (or SAM3-LoRA) for detection in oblique views
- MUST use MASt3R for oblique→ortho cross-view mapping
- MUST produce pole locations as GPS coordinates
- Metric: F1 at 10m match radius against 94 verified GT poles

## Available Resources
- 2x NVIDIA 3090 (24GB each), 256GB RAM
- SAM3 base model + LoRA v2 weights at models/sam3_finetuned/lora_v2/
- SAM3-LoRA training script at models/sam3_lora/train_sam3_lora_native.py
- MASt3R AerialMegaDepth checkpoint
- Qwen 3.5 27B VLM via ollama (localhost:11434, use think=False)
- Fine-tuned GDino-Base at models/gdino_finetuned/best/
- Verified GT at data/ground_truth_testarea.json (94 poles)
- Test area grid images at data/testarea_grid/

## What's Been Tried and Failed
- Multi-view consensus (min 2 views): killed recall too much
- Prompt "utility pole" vs "telephone pole": utility pole was worse
- Ortho crop < 50m: too little context for MASt3R
- 3D shape features at 512px MASt3R: too coarse
- VLM post-filtering: strict prompt loses TPs, soft prompt doesn't filter
- SAM3-LoRA v2: catastrophically worse than base SAM3
- Aspect ratio filtering: SAM3 detections already have reasonable ratios
- Ortho crop 55m: too little context, hurt recall
- MASt3R 768px: higher resolution degraded matching quality (RMSE +0.5m)
- Score-weighted centroid: worse than equal-weight averaging
- Dedup radius 8m: keeps too many FP duplicates separate
- Threshold 0.42: marginal, 0.45 is the sweet spot
- VLM (Qwen 3.5 27B) post-filtering on borderline SAM3 dets: VLM classifies everything as pole
- Cluster scoring (cluster_size * max_score): too aggressive, kills single-view TPs
- GDino ensemble (add GDino detections to SAM3): GDino adds far more FPs than TPs
- GDino cross-validation (keep SAM3 dets GDino confirms): GDino confirms everything including FPs
- Multi-point projection (3 points along pole, median GPS): hurts localization, RMSE +0.5m
- Threshold 0.48: slightly worse than 0.45, recall drops faster than precision gains
- Two-tier with single_view_min=0.55: too aggressive, removes legitimate single-view TPs
- VLM post-filter (Qwen 3.5 27B): removes TPs but not FPs — VLM can't distinguish at crop level
- Direction-aware multi-view: no benefit over cluster-size two-tier
- MASt3R linear schedule: cosine is strictly better
- MASt3R 448px or 120 iters: 512px and 100 iters are optimal
- Quality-aware GPS (reproj error weighting): no improvement
- GDino fallback for empty SAM3 images: GDino also finds nothing

## Promising Directions (DEEP CHANGES)

### HIGH PRIORITY — TRY THESE FIRST

7. **SAM3-LoRA v3 with CLEAN training data**: The v2 LoRA (iteration 12) failed
   catastrophically, BUT it was trained on homography-projected bboxes (~10-15m error).
   We have since rebuilt training data using MASt3R AerialMegaDepth projections (~2.6m
   accuracy). Retrain the LoRA with this clean data — it should perform MUCH better.
   - Training script: models/sam3_lora/train_sam3_lora_native.py
   - Config: models/sam3_lora/configs/pole_lora_v2.yaml
   - Clean training data: data/training/annotations.json (MASt3R-projected bboxes)
   - Build training set: python src/build_training_set.py
   - Fine-tune script: python src/finetune_sam3.py (or use sam3_lora repo)
   - The v2 failure was a DATA QUALITY issue, not a model issue. This is worth retrying.

8. **Seasonal/leaf-off imagery**: EagleView captures imagery at different times of year.
   Winter/leaf-off images have NO foliage occluding poles, making them far more visible.
   Explore whether we can pull leaf-off captures from the WMTS tile server (different
   capture dates or layer IDs). Even if we can't get seasonal data, try:
   - Checking if the WMTS server has a temporal dimension or alternate layers
   - Looking at the EagleView API docs for capture date parameters
   - Using image metadata to identify which existing tiles are leaf-off vs leaf-on
   - Tree canopy is our #1 source of false negatives — removing it would be huge

### Other directions

1. **VLM post-filtering**: After SAM3+MASt3R, classify each detection crop with
   Qwen 3.5 27B. Remove streetlights/trees/fences. This boosted precision in
   our GDino pipeline. Use ollama API with think=False.
2. **SAM3-LoRA v3**: (see #7 above — use clean data this time)
3. **Score-weighted dedup**: Instead of keeping max-score in cluster, weight
   GPS by score for more accurate positioning.
4. **Adaptive ortho crop**: Different crop size based on image direction/GSD.
5. **Detection ensemble**: Run SAM3 with multiple prompts, merge results.
6. **MASt3R iteration tuning**: Try 100 or 500 iterations instead of 300.

## Population (Top Configs)
See population.json

## Experiment Log
See autoresearch.jsonl
