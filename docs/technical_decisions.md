# Technical Decisions Log

## Project: Power Pole Detection from EagleView Oblique Imagery
**Date:** April 3, 2026
**Area:** Omaha, NE sandbox (~1.5 sq miles, 2.5km × 1.7km)

---

## Phase 1: Data Acquisition

### Decision: Use EagleView Imagery API (not WMTS) for oblique views
**Reasoning:** WMTS only serves ortho (top-down) tiles. The Imagery API provides both ortho and oblique views (N/E/S/W at ~45° angle). Oblique views are critical because power poles are vertical structures — clearly visible from the side but nearly invisible from directly above.

### Decision: OAuth2 Client Credentials auth
**Reasoning:** The EagleView developer portal offers API Key auth (WMTS only) and Client Credentials (Imagery API). We created a second app ("powerlinefinder2") with Client Credentials to access the Imagery API. Token endpoint: `https://apicenter.eagleview.com/oauth2/v1/token`.

### Decision: PNG format over JPEG
**Reasoning:** PNG is lossless — no compression artifacts that could interfere with edge detection on thin structures like poles. Tradeoff: ~3-5x larger files (~7MB vs 2MB per crop). At 188 images = 1.4GB total, acceptable for our sandbox size.

### Decision: Max zoom per image with adaptive radius (50m → 35m → 25m)
**Reasoning:** We want the highest possible resolution (~4cm/pixel GSD) to detect thin poles (~30cm diameter = ~8 pixels wide). The EagleView API has a 10MB payload limit. At max zoom with 50m radius, some images exceed this. We try 50m first, fall back to 35m and 25m if 413 errors occur. Different images have different max zoom levels (5, 7, or 8 for obliques; 21-23 for ortho).

### Decision: Rate limiting at 90% of published limits
**Reasoning:** EagleView rate limits — Token: 4 rps, Discovery: 5 rps, Images: 5 rps, Tiles: 300 rps. We run at 3.5/4.5/4.5/270 rps respectively to leave headroom and avoid 429 errors. Exponential backoff with jitter on 429/5xx.

### Decision: WMTS tiles at zoom 19 for ortho map layer
**Reasoning:** 1,333 tiles covering the full sandbox area. Zoom 19 gives ~30cm/pixel — good enough for map background. Served locally via HTTP for the viewer. Total ~130MB.

### Decision: 130m grid spacing for discovery, 266 query points
**Reasoning:** The `rank/location` endpoint takes center + radius (max 75m). At 130m spacing with 75m radius, we get overlapping coverage. Found 190 unique images (37 north, 57 east, 40 south, 51 west, 5 ortho). 188 downloaded successfully.

---

## Phase 2: Model Selection & Detection Approach

### Decision: SAM2 (not SAM3 — doesn't exist)
**Reasoning:** SAM2 (`sam2.1_hiera_base_plus`, 309MB) is the latest from Meta as of 2026. Installed with `SAM2_BUILD_CUDA=0` for MPS compatibility. Supports automatic mask generation, point prompting, box prompting, and video tracking.

### Decision: MASt3R for 3D reconstruction
**Reasoning:** MASt3R (2.6GB checkpoint from HF) provides dense pixel-to-pixel 3D matching between image pairs. Runs on MPS with `PYTORCH_ENABLE_MPS_FALLBACK=1`. Processes images at 512px max dimension. ~8s inference + 17s alignment for 4 views on MPS. Used for multi-view consensus — establishing which detections in different views correspond to the same 3D point.

### Decision: GroundingDINO-Tiny (172M) as primary detector
**Reasoning:** We tested three detection approaches on the same location:

| Approach | Speed | Detections | Quality |
|----------|-------|-----------|---------|
| SAM2 auto-generate + AR filter | 15-135s/img | 2-19 candidates | Many false positives (fences, tree trunks, house edges) |
| GroundingDINO-Tiny | 1-2s/img | 10-17 detections | Fast, noisy, low confidence (0.15-0.3) |
| Qwen3-VL 2B (ollama) | 5-50s/img | 0-2 detections | Unreliable, stuck in thinking loops 3/4 views |

GroundingDINO won on speed (10-50x faster than SAM2) and reliability (works every time, unlike Qwen). Both GDino and SAM2 have high false positive rates on aerial oblique imagery because they're trained on ground-level photos.

We also tested **GroundingDINO-Base** (232M). It was more selective (fewer detections) but missed some real poles. Stuck with Tiny for better recall.

### Decision: Rejected Qwen3-VL for grounding
**Reasoning:**
- `qwen3.5:0.8b` (text model): vision is broken in ollama due to missing mmproj support. Responses were empty or stuck in thinking loops.
- `qwen3-vl:2b`: Correct grounding prompt format (`locate every instance...Report bbox coordinates in JSON format`) works and returns `bbox_2d` in 0-1000 normalized coords. BUT the 2B model gets stuck in repetitive thinking loops on 3/4 test images, making it unreliable.
- The correct prompt format was key: standard detection prompts fail, must use the specific grounding format.

### Decision: Rejected SAM2 auto-generate as primary detector
**Reasoning:** SAM2's automatic mask generator creates masks for everything — buildings, trees, cars, sidewalks. Filtering by aspect ratio (AR > 3) catches anything tall and thin, not just poles. Of ~12 "pole candidates" per image, visual inspection showed ~10-15% true positive rate. Too noisy as a standalone detector, and too slow (15-135s per image vs 1-2s for GDino).

---

## Phase 3: Multi-View Consensus Pipeline

### Decision: Multi-view consensus using MASt3R 3D correspondences
**Reasoning:** The key insight — all single-view detectors have high false positive rates on aerial oblique imagery. But a real pole is visible from multiple angles and maps to the same 3D point. A false positive (tree trunk, fence) looks different from each angle or maps to a different 3D location. By requiring 2+ views to agree on the same 3D point, we dramatically reduce false positives.

**Pipeline:**
1. GDino detects candidates in each of 4 oblique views (~3s)
2. MASt3R establishes 3D correspondences between views (~25s)
3. For each detection, project its 3D point into other views
4. If a detection in another view overlaps the projected point → agreement
5. Keep detections with 2+ view agreement

### Decision: NMS (IoU > 0.3) on GDino detections per view
**Reasoning:** GDino produces many overlapping bounding boxes on the same object. NMS deduplicates these before consensus, reducing 53 raw → 19 after NMS in one test case.

### Decision: Height filter (4-25m) using camera geometry
**Reasoning:** Estimated real-world height from bounding box pixel height:
`height = h_px × GSD / sin(elevation_angle)`

Camera elevation (~42-45°) and GSD (~4cm) are known from metadata. This filters out:
- Oversized detections (fences with 600px bboxes → 33-35m estimated, rejected)
- Very short objects (< 4m, not poles)
- Correctly estimates real poles at 8-15m

This removed the three fence false positives in the north view that survived multi-view consensus (because fences are real 3D structures visible from multiple angles).

### Decision: 3D clustering (0.05 MASt3R units) for deduplication
**Reasoning:** The same physical pole gets detected from multiple views, each as a separate entry. 3D clustering merges detections whose MASt3R 3D points are within 0.05 units. Reduced 14 confirmed → 5 unique poles in test case. Keeps highest confidence detection, averages height estimates, aggregates view counts.

### Decision: Rejected MASt3R-based height discrimination
**Reasoning:** We attempted to use MASt3R's 3D reconstruction to measure object height and distinguish poles (10m) from fences (2m). Failed because MASt3R processes images at 512px, giving insufficient 3D resolution — noise (~0.05-0.09 units) swamps the real height difference at that scale. The camera geometry approach (pixel height × GSD) works better.

### Decision: Rejected ortho cross-check for fence filtering  
**Reasoning:** Attempted to check each detection against the WMTS ortho tile — poles should appear as dots, fences as lines in top-down view. Failed because our GPS georeferencing was too inaccurate (~50-100m error) to look up the correct ortho tile pixel. The height filter (above) solved the fence problem more simply.

---

## Phase 4: Georeferencing

### Decision: Homography-based pixel-to-GPS conversion
**Reasoning:** Simple `pixel_offset × GSD` gives 50-100m errors because it ignores:
1. Oblique perspective distortion (top of image covers more ground per pixel than bottom)
2. Camera azimuth rotation (pixel-right doesn't always mean ground-east)

The homography approach maps 4 image edge midpoints to ground positions at crop_radius distance, rotated by camera azimuth. Uses `cv2.findHomography` + `cv2.perspectiveTransform`. Should reduce error to ~2-5m on flat terrain.

**Status:** Implemented but not yet validated — eval results pending.

---

## Phase 5: Evaluation

### Decision: Ground truth via ortho-based map labeler (labeler_v2.html)
**Reasoning:** The first labeler (labeler.html) worked per-detection in oblique views — couldn't see the spatial big picture or find poles the model missed. labeler_v2 uses the Leaflet map with WMTS ortho tiles, click-to-place labels, and oblique view sidebar for confirmation. Categories: pole (58), streetlight (6). Focus area: ~400m × 360m around test tile.

### Decision: 80m grid spacing for test area (36 cells, 144 images)
**Reasoning:** 50m radius crops at 80m spacing gives 20m overlap between adjacent crops, ensuring complete ground coverage. 36 cells × 4 directions = 144 images, ~1GB total. Downloaded once, reused for all eval runs.

### Decision: Baseline precision = 45% (from first 62 labeled detections)
**Reasoning:** Out of 62 confirmed detections across 10 locations:
- 28 real poles (45%)
- 15 streetlights (24%) — biggest confuser
- 9 fences (15%) — addressed by height filter
- 7 other (11%)
- 3 trees (5%)

Streetlights are the main remaining false positive — they look like poles from aerial oblique views and have similar height (8-15m).

---

## Phase 6: Infrastructure & Tooling

### Decision: Single-file HTML/JS viewers (no build step)
**Reasoning:** viewer.html, comparison.html, batch_results.html, labeler.html, labeler_v2.html, eval_map.html — all served via `python3 -m http.server 8080`. No React/webpack complexity. Leaflet.js for maps, vanilla JS for everything else.

### Decision: Model checkpoints from Hugging Face Hub
**Reasoning:** User preference — HF has better CDN, resume support, and caching vs direct download URLs. SAM2 from Meta's CDN (only 309MB, fast), MASt3R from `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` on HF.

### Decision: ollama for local vision models
**Reasoning:** User has ollama installed with various Qwen models. Used `qwen3-vl:2b` for grounding tests (not `qwen3.5:0.8b` which is text-only despite accepting images). Key finding: must use `/api/chat` endpoint with `think: false` at top level, and the grounding-specific prompt format.

---

## Next Steps (for CUDA machine with 2x 3090)

### Fine-tuning GroundingDINO
- Convert ground truth labels to COCO format
- Fine-tune GroundingDINO-Tiny on our labeled aerial pole data
- Focus on pole vs streetlight discrimination
- Need ~200+ labeled examples (have 64 + more from batch runs)

### Performance optimization
- GDino-first pipeline: run GDino on all cells, only MASt3R where detections found
- Multi-GPU: GDino on GPU 0, MASt3R on GPU 1 (pipeline parallel)
- Estimated: 2.5 min for test area (vs 18 min on MPS)
- Full sandbox at 100m grid: ~30 min (vs ~3.5 hours on MPS)

### Agentic labeling system
- Use fine-tuned model to auto-label high-confidence detections
- Queue low-confidence for human review
- Active learning loop: label → train → predict → review → label

---

## File Structure

```
powerpolefinder/
├── .env                          # EagleView API credentials (not committed)
├── .gitignore
├── requirements.txt
├── viewer.html                   # Map viewer with WMTS + oblique browser
├── comparison.html               # Side-by-side detection comparison
├── batch_results.html            # Batch run results viewer
├── labeler.html                  # Detection-based labeler (v1)
├── labeler_v2.html               # Map-based labeler (v2, ortho view)
├── eval_map.html                 # Eval sanity check map
├── docs/
│   ├── plan_multiview_consensus.md
│   └── technical_decisions.md    # This file
├── src/
│   ├── auth.py                   # OAuth2 token management
│   ├── discovery.py              # EagleView image discovery
│   ├── download.py               # Oblique image download (location endpoint)
│   ├── download_wmts.py          # WMTS ortho tile download
│   ├── download_testarea.py      # Test area grid download
│   ├── main.py                   # Data acquisition orchestrator
│   ├── ratelimit.py              # Rate limiting for API calls
│   ├── oblique_utils.py          # Pixel↔GPS conversion, image lookup
│   ├── compare_detection.py      # SAM2 vs GDino vs Qwen comparison
│   ├── test_multiview.py         # Single-location consensus test
│   ├── test_qwen_poles.py        # Qwen3-VL grounding test
│   ├── batch_multiview.py        # Batch consensus pipeline
│   └── eval_testarea.py          # Evaluation against ground truth
├── models/
│   ├── sam2/                     # SAM2 repo + checkpoint (309MB)
│   └── mast3r/                   # MASt3R repo + checkpoint (2.6GB)
└── data/
    ├── metadata.json             # Image metadata from discovery
    ├── ground_truth.json         # Detection-based labels (v1, 62 labels)
    ├── ground_truth_testarea.json # Map-based labels (v2, 64 labels)
    ├── oblique/                   # Downloaded oblique crops (188 images, 1.4GB)
    ├── ortho/                     # Downloaded ortho crops (3 images)
    ├── wmts/                      # WMTS ortho tiles (1333 tiles, ~130MB)
    ├── testarea_grid/             # Test area gridded crops (144 images)
    ├── debug/                     # Detection visualizations
    │   ├── batch/                 # Batch run results
    │   └── target/                # Single-location test results
    └── eval_testarea/             # Evaluation results
```
