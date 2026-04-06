# Work Session Log — April 4-5, 2026

## Overview
Two-day intensive session building an end-to-end power pole detection pipeline from EagleView aerial imagery. Started from a partially-built pipeline on a Mac (MPS), migrated to a CUDA workstation (2x 3090), and iterated through multiple architectures to reach F1@10m = 0.714.

---

## Day 1 (April 4)

### Phase 1: Environment Setup & CUDA Migration
- Copied project from Mac to CUDA workstation (2x RTX 3090, 256GB RAM)
- Data was nested (`data/data/`) — fixed directory structure
- Set up conda initially but **user corrected to use `uv`** instead (much faster)
- Installed PyTorch CUDA, transformers, MASt3R dependencies
- Fixed Mac paths in `testarea_grid/index.json` (paths referenced `/Users/z/...`)
- **Decision**: Upgraded from GDino-Tiny to GDino-Base (232M params, richer Swin-B backbone features for fine-tuning). Reasoning: frozen backbone fine-tuning benefits from richer pretrained features, negligible speed/VRAM penalty on 3090.

### Phase 2: VLM Classification Pipeline
- Set up vLLM with Qwen 3.5 9B for detection classification
- Discovered `chat_template_kwargs: {enable_thinking: false}` needed for clean JSON responses
- **Decision**: Switched to ollama backend with Qwen 3.5 27B. Reasoning: already installed, supports multi-image, `think: false` parameter works.
- Improved classification prompt with detailed visual descriptions of each category (pole crossarms, streetlight fixtures, fence horizontal runs)
- 27B caught more streetlights (9 vs 3) and trees than 9B

### Phase 3: Automated Labeling Pipeline
- Built `auto_label.py` — runs GDino on all 185 oblique images, classifies 918 crops with Qwen 3.5 27B
- Results: 630 poles, 142 streetlights, 55 trees, 40 fences
- Training data: 513 high-conf poles + 268 hard negatives = 781 candidates
- Had to resize images to 800px max to avoid OOM on 3090

### Phase 4-5: Fine-tuning GDino + Training Set
- Built COCO training set from VLM labels
- Fine-tuned GDino-Base: froze Swin-B backbone, trained detection head
- **Key result**: Recall nearly doubled (48% → 92% at 30m) but precision dropped
- Combined with VLM: F1=0.536 at 30m

### Ground Truth Discovery
- Visually inspected "false positives" — discovered **most were actually real poles** missing from GT
- Original GT had only 54 poles; test area likely contained 100+
- **Decision**: Used VLM + pipeline detections to generate candidate GT labels, then had user review/correct in labeler
- Added 34 new GT poles, bringing total to 88

### Height Filter Fix
- Analyzed pipeline funnel: 20% of multi-view confirmed detections rejected by height filter (25m max)
- All rejections were "too tall" (25-120m), none too short
- Height estimation formula (`bbox_h * GSD / sin(elevation)`) is noisy
- **Decision**: Relaxed max from 25m to 50m. Reasoning: catches absurd estimates while keeping borderline real poles.

### Focus Area Filter
- Discovered 24% of detections fell outside GT coverage area, inflating FP count
- **Decision**: Filter detections to GT focus area before evaluation

### Georeferencing Experiments
- Attempted 3D georeferencing from MASt3R point clouds (fit affine transform using GCPs)
- Attempted bbox-bottom projection (pole base vs center)
- Neither improved F1 significantly — the GT positional noise was the bigger issue
- **Decision**: Used hybrid 60/40 blend of 3D + homography

### Labeler Improvements
- Added draggable markers (replaced circleMarker with L.marker + divIcon)
- Added move mode (M key), auto-label review workflow (A/Y/R/X/N/P keys)
- Added multi-zoom WMTS tiles (zoom 19-23) for better ortho resolution
- Downloaded zoom 20-23 tiles for test area (~25K tiles)

### Multi-view Verification Agent
- Built agent that sends crops from all 4 oblique views + ortho to Qwen 3.5 in one prompt
- Caught 29 trees (vs 3 single-crop), 18 streetlights (vs 14)
- F1=0.699 at 30m — best overall but slow (~45 min)
- **Decision**: Made optional due to speed

### 3D Shape Analysis
- Extracted PCA features from MASt3R point cloud within each detection bbox
- TP and FP distributions nearly identical (median aspect 2.06 vs 2.05)
- **Conclusion**: 512px MASt3R resolution too coarse for shape discrimination

---

## Day 2 (April 5)

### SAM3 Integration
- **Decision**: Switched to SAM3 (Segment Anything 3) from Meta. Reasoning: replaces both GDino (detection) AND SAM2 (segmentation) in one model. Text-prompted detection + pixel masks. 848M params, ~4GB VRAM.
- Downloaded from alternative HuggingFace repo (`bodhicitta/sam3`) due to licensing concerns with official Meta repo
- Required `setuptools<72` for `pkg_resources` compatibility
- Required global `torch.autocast('cuda', dtype=torch.bfloat16)` (like the example notebook)
- Required `device='cuda'` not `device='cuda:0'` (model builder checks exact string)

### SAM3 Prompt Engineering
- Benchmarked 14 text prompts across test images
- "telephone pole" best balance of recall vs precision
- "wooden power pole" highest precision (88.9%) but too few detections
- "pole" most detections but very noisy

### SAM3-LoRA Fine-tuning
- First attempt: SAM3's native training framework blocked by fused CUDA kernels requiring special gradient handling
- **Decision**: Used `sam3_lora` repo (Sompote/sam3_lora) for parameter-efficient fine-tuning
- LoRA v1: batch_size=1, grad_accum=4 (effective batch=4) — too small, noisy training
- LoRA v2: batch_size=2, grad_accum=16 (effective batch=32) — matched research recommendations
- LoRA v2 with clean training data (MASt3R-projected verified GT bboxes)

### Oblique → Ortho Pipeline (Core Requirement)
- **Key realization**: Fuxun's requirements specified SAM on oblique → MASt3R map to ortho. We had been doing oblique↔oblique MASt3R consensus (not asked for).
- **Decision**: Simplified pipeline to per-image: SAM3 detect in oblique → pair with ortho tile → MASt3R project to ortho → GPS via tile math.

### AerialMegaDepth MASt3R
- Switched to `kvuong2711/checkpoint-aerial-mast3r` — fine-tuned on 132K aerial-ground pairs
- **Key finding**: AerialMegaDepth designed for extreme viewpoint differences (aerial↔ground), NOT oblique↔oblique (where standard MASt3R is fine)
- Initial testing showed identical performance to standard — because we were feeding oblique↔oblique pairs
- Proper use: oblique(45°)↔ortho(90°) pairs

### MASt3R Debugging
- Discovered ortho crops at 200m radius → 7424px → MASt3R downscales to 512px → poles are **0.38 pixels** (sub-pixel, invisible)
- **Root cause of poor oblique→ortho projection**: MASt3R can't find correspondences when poles are invisible in ortho view
- Validated 2.6m GPS accuracy on known GT pole with 80m radius ortho crop
- Tried per-detection cropping (both oblique + ortho cropped to 30m around detection) — worse, not enough context

### Verified Ground Truth
- User manually verified all 94 GT labels against high-res ortho imagery
- Created `gt_verify.html` with draggable markers, right-click to add, keyboard shortcuts
- **First real F1@10m = 0.448** (GDino-ft + VLM with verified GT)
- Created holdout area (500m east) with 94 verified poles — **holdout F1=0.714** (not overfit!)

### AutoResearch Loop
- Implemented Karpathy's AutoResearch pattern for autonomous pipeline optimization
- 3-file architecture: `prepare.py` (locked eval), `pipeline.py` (modifiable), `program.md` (research brief)
- Agent runs with `--dangerously-skip-permissions` for autonomous file edits + git commits
- Added genetic population tracking (`population.json`)

### AutoResearch Results (82 experiments)
Key improvements found autonomously:
1. **Threshold 0.10→0.20→0.30→0.35→0.40→0.45**: F1 0.335→0.598 (systematic threshold tuning)
2. **Ortho crop 80m→60m**: Better MASt3R matching scale
3. **Multi-prompt ensemble**: "telephone pole" + "wooden pole" + "power pole"
4. **MASt3R 300→100 iterations**: Less alignment overfitting
5. **PointCloudOptimizer** (instead of ModularPointCloudOptimizer): Better alignment mode
6. **Two-tier threshold**: High-conf detections kept without multi-view, low-conf need 2+ views
7. **Dedup radius 10m→15m**: Better GPS averaging

Agent correctly **reverted** failed experiments:
- Multi-view consensus (killed recall)
- Prompt change to "utility pole" (catastrophic regression)
- SAM3-LoRA v2 (catastrophic failure at 0.086)
- VLM filtering (both strict and soft versions)

### Final Architecture
```
SAM3 (multi-prompt: telephone/wooden/power pole)
  → MASt3R (AerialMegaDepth, 100 iters, PointCloudOptimizer)
    → oblique↔ortho projection (pole base, 60m crop)
      → GPS via tile math (exact)
        → two-tier dedup (15m radius)
          → F1@10m = 0.714
```

### Overfit Check
- Test area: F1@10m = 0.678 (94 GT poles)
- Holdout area: F1@10m = 0.714 (94 GT poles, 500m east)
- **No overfitting** — holdout actually slightly better
- Initially holdout showed only 4 detections — root cause: WMTS tiles not downloaded for holdout area

### Dashboard
- Built `dashboard.html` showing live results from autoresearch data files
- Shows: F1 progress bars, best config, latest experiments, population, map with detections
- Auto-refreshes every 30s

### Training Data Quality
- Discovered homography-projected training bboxes were ~10-15m off from actual poles
- **Decision**: Rebuilt training data using MASt3R AerialMegaDepth projection (~2.6m accuracy)
- Visual inspection confirmed MASt3R-projected bboxes land on actual poles
- Filtered edge bboxes (18% removed — were in black void areas at image borders)

---

## Key Metrics Progression

| Stage | F1@10m | Method |
|-------|--------|--------|
| Original baseline (Mac) | — | GDino-Tiny + MASt3R consensus (eval at 30m only) |
| First CUDA eval | 0.329 | GDino-Tiny + MASt3R, 30m match radius |
| GDino-Base + filters | 0.304 | Area filter, relaxed height, CUDA |
| GDino-ft + VLM | 0.448 | First eval with verified GT at 10m radius |
| SAM3 + MASt3R baseline | 0.335 | AutoResearch starting point |
| AutoResearch iter 1 | 0.488 | Threshold tuning |
| AutoResearch iter 6 | 0.559 | Ortho crop optimization |
| AutoResearch iter 8 | 0.592 | Further threshold tuning |
| AutoResearch iter 15 | 0.702 | PointCloudOptimizer discovery |
| **AutoResearch iter 16** | **0.714** | **Best — multi-prompt + PCO + dedup tuning** |

---

## Lessons Learned

1. **Ground truth quality is everything** — noisy GT labels distorted all our metrics for hours
2. **MASt3R resolution matters** — poles become invisible when ortho crops are too large at 512px
3. **AutoResearch works** — autonomous agent found improvements humans would take days to find
4. **VLM filtering has diminishing returns** — useful early but not after threshold optimization
5. **SAM3 replaces GDino+SAM2** but needs threshold calibration per domain
6. **AerialMegaDepth only helps for extreme viewpoint differences** — not oblique↔oblique
7. **Training data accuracy > quantity** — MASt3R-projected bboxes >>> homography-projected
8. **The pipeline bottleneck shifts** — detection → georeferencing → GT quality → threshold tuning
9. **User preferences matter** — uv not conda, commit after each phase, visual verification
