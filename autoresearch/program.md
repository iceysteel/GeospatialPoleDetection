# AutoResearch: Pole Detection Pipeline Optimization

## Objective
Maximize F1@10m for detecting utility poles in aerial imagery.
Current best: **F1@10m = 0.598** (SAM3 threshold=0.45, ortho_crop=60m)

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

## Promising Directions (DEEP CHANGES)
1. **VLM post-filtering**: After SAM3+MASt3R, classify each detection crop with
   Qwen 3.5 27B. Remove streetlights/trees/fences. This boosted precision in
   our GDino pipeline. Use ollama API with think=False.
2. **SAM3-LoRA v3**: Retrain with better data or different hyperparams.
   Training script: models/sam3_lora/train_sam3_lora_native.py
   Config: models/sam3_lora/configs/pole_lora_v2.yaml
3. **Score-weighted dedup**: Instead of keeping max-score in cluster, weight
   GPS by score for more accurate positioning.
4. **Adaptive ortho crop**: Different crop size based on image direction/GSD.
5. **Detection ensemble**: Run SAM3 with multiple prompts, merge results.
6. **MASt3R iteration tuning**: Try 100 or 500 iterations instead of 300.

## Population (Top Configs)
See population.json

## Experiment Log
See autoresearch.jsonl
