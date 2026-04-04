# Power Pole Detection from Oblique Aerial Imagery

An end-to-end pipeline for automated detection and geolocation of utility power poles from EagleView oblique satellite imagery. The system combines zero-shot object detection, multi-view 3D reconstruction, and agent-based vision-language classification to achieve an F1 score of **0.699** on a ground truth evaluation set.

> **[View Interactive Dashboard](https://iceysteel.github.io/GeospatialPoleDetection/dashboard.html)** — Explore detection results on the map, adjust match radius, and inspect individual detections.

---

## Summary of Results

| Metric | Value |
|---|---|
| **Best F1** | 0.699 @ 30m match radius |
| **Precision** | 73.4% |
| **Recall** | 66.7% |
| **Test Area** | 400m × 360m, Omaha NE |
| **Ground Truth** | 88 poles, 6 streetlights |
| **Processing Time** | ~30s per grid cell (MPS), ~6s (CUDA) |

---

## Pipeline Architecture

The pipeline processes EagleView oblique imagery through five sequential stages, each progressively filtering false positives while preserving true pole detections.

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │  EagleView Oblique Imagery                                          │
  │  4 cardinal directions (N/E/S/W) at ~45° elevation, ~4cm GSD       │
  └────────────────────────────┬─────────────────────────────────────────┘
                               │
          ┌────────────────────▼────────────────────┐
          │  Stage 1: GroundingDINO Detection        │
          │  Zero-shot: "utility pole. power pole."  │
          │  ~2s/image, 10-20 candidates per view    │
          └────────────────────┬────────────────────┘
                               │
          ┌────────────────────▼────────────────────┐
          │  Stage 2: MASt3R Multi-View Consensus    │
          │  Dense 3D reconstruction across 4 views  │
          │  Keep detections confirmed in 2+ views   │
          │  ~50% false positives eliminated         │
          └────────────────────┬────────────────────┘
                               │
          ┌────────────────────▼────────────────────┐
          │  Stage 3: Height Filter + 3D Clustering  │
          │  Estimate height from camera geometry    │
          │  Reject < 4m or > 50m                    │
          │  Merge duplicate detections via 3D       │
          └────────────────────┬────────────────────┘
                               │
          ┌────────────────────▼────────────────────┐
          │  Stage 4: VLM Agent Classification       │
          │  Qwen 3.5 27B classifies each crop       │
          │  pole / streetlight / tree / fence        │
          │  Removes ~25% remaining false positives  │
          └────────────────────┬────────────────────┘
                               │
          ┌────────────────────▼────────────────────┐
          │  Stage 5: GPS Georeferencing             │
          │  Homography + MASt3R 3D hybrid           │
          │  ~3-7m accuracy                          │
          └────────────────────┬────────────────────┘
                               │
                               ▼
                      power_poles.geojson
```

---

## F1 Score Progression

The pipeline was developed through systematic experimentation, testing multiple detection approaches and iteratively adding filtering stages. Each row represents a measurable improvement with a specific technical change.

| Stage | Approach | F1 | Change |
|---|---|---|---|
| Baseline | GDino-Tiny + pixel×GSD georeferencing | 0.33 | — |
| + CUDA | GDino-Base on dual 3090 GPUs | 0.31 | Faster, slightly more selective |
| + VLM Filter | Qwen 3.5 27B single-crop classification | 0.45 | +0.14, removes streetlights |
| + Tuning | Focus area filter + height relaxation | 0.52 | +0.07, better coverage |
| + 3D Georef | Hybrid homography + MASt3R 3D projection | 0.54 | +0.02, tighter GPS matching |
| + Fine-tune | Domain fine-tuned GDino-Base + GT refinement | 0.68 | +0.14, aerial-specific detection |
| + Multi-view Agent | All 4 oblique views + ortho sent to Qwen 27B | **0.70** | +0.02, catches trees (29 vs 3) |

---

## Detection Model Comparison

During development, three detection approaches were evaluated on identical test imagery. The table summarizes per-image performance before any multi-view consensus or post-processing.

| Model | Parameters | Speed | Precision (raw) | Notes |
|---|---|---|---|---|
| SAM2 auto-generate + aspect ratio filter | 81M | 15–135s | ~15% | Segments all objects; shape heuristics insufficient for poles |
| Qwen3-VL 2B (grounding mode) | 2B | 5–50s | Unreliable | Correct prompt format works but model loops on 3/4 test images |
| GroundingDINO-Tiny (zero-shot) | 172M | 1–2s | ~25% | Fast and consistent; selected as baseline detector |
| GroundingDINO-Base (zero-shot) | 232M | 2–3s | ~30% | More selective, fewer false positives |
| **GDino-Base (fine-tuned) + VLM** | 232M + 27B | ~30s/loc | **~65%** | Domain-adapted detection + agent classification |

---

## Agent-Based Automation

### Automated Labeling Pipeline

Manual labeling of aerial oblique imagery is slow and error-prone. The pipeline uses an agent-based approach to generate training data at scale:

1. **Detection Agent** — GDino-Base scans all 185 oblique images, producing 918 candidate detections
2. **Classification Agent** — Qwen 3.5 27B classifies each detection crop as pole, streetlight, tree, fence, building edge, or other
3. **Confidence Filtering** — High-confidence classifications (pole: 513, hard negatives: 268) are used directly as training labels
4. **Human Review** — Low-confidence and ambiguous cases are queued in an interactive labeling UI for manual review

This process generated a 184-image COCO-format training set with 630 pole bounding box annotations, sufficient to fine-tune GroundingDINO-Base and improve recall from 48% to 92.6%.

### Ground Truth Refinement

An additional agent uses high-resolution ortho tiles (zoom 22-23, ~1.4cm/pixel) to refine the GPS positions of auto-detected labels. The agent loads an ortho crop at each detection's estimated location and identifies the exact pole position, reducing georeferencing error by an average of 21m on refined labels.

---

## Data Acquisition

All imagery is sourced from the EagleView sandbox API, covering approximately 1.5 square miles in Omaha, Nebraska.

| Data Type | Count | Size | Source |
|---|---|---|---|
| Oblique images | 188 | 1.4 GB | Imagery API, 50m radius crops at max zoom |
| WMTS ortho tiles | 1,333 | 130 MB | WMTS API, zoom 19 |
| Test area grid | 144 | ~1 GB | 36 grid cells × 4 directions, 80m spacing |
| Ground truth labels | 64 | 10 KB | Manual labeling via ortho map UI |

**API Integration:**
- OAuth2 Client Credentials authentication
- Rate-limited requests (Discovery: 4.5 rps, Images: 4.5 rps, Tiles: 270 rps)
- Adaptive crop radius (50m → 35m → 25m) to stay within 10MB payload limit

---

## Models

| Model | Size | Role |
|---|---|---|
| GroundingDINO-Base | 232M (0.9 GB) | Zero-shot / fine-tuned pole detection |
| MASt3R ViT-Large | 2.6 GB | Multi-view 3D reconstruction and correspondence |
| SAM2 hiera-base-plus | 309 MB | Segmentation (evaluated, not in final pipeline) |
| Qwen 3.5 27B | 17 GB | Vision-language classification agent (via Ollama) |

---

## Interactive Tools

The project includes several single-file HTML/JS tools for visualization and labeling, served via `python3 -m http.server 8080`:

- **`dashboard.html`** — Results dashboard with interactive map, metrics, and pipeline explanation ([live](https://iceysteel.github.io/GeospatialPoleDetection/dashboard.html))
- **`viewer.html`** — Map viewer with WMTS ortho tiles and oblique image browser
- **`labeler_v2.html`** — Ortho map-based ground truth labeling with oblique cross-reference
- **`eval_map.html`** — Evaluation overlay (GT vs detections with match lines)
- **`comparison.html`** — Side-by-side detection comparison across models

---

## Setup

### Requirements
- Python 3.11+
- PyTorch (MPS for Mac, CUDA for GPU)
- EagleView API credentials (sandbox access)

### Quick Start

```bash
git clone https://github.com/iceysteel/GeospatialPoleDetection.git
cd GeospatialPoleDetection
pip install -r requirements.txt

# Models (downloaded from HuggingFace)
git clone https://github.com/facebookresearch/sam2.git models/sam2
SAM2_BUILD_CUDA=0 pip install -e models/sam2
git clone --recursive https://github.com/naver/mast3r.git models/mast3r
pip install -r models/mast3r/dust3r/requirements.txt

# Credentials
cp .env.example .env  # Edit with EagleView API keys

# Run
python3 src/main.py              # Download imagery
python3 src/download_wmts.py     # Ortho tiles
python3 src/eval_testarea.py     # Run evaluation
python3 -m http.server 8080      # View results
```

See [`docs/setup_cuda.md`](docs/setup_cuda.md) for GPU setup with CUDA.

---

## Documentation

| Document | Description |
|---|---|
| [`docs/technical_decisions.md`](docs/technical_decisions.md) | Complete log of all technical decisions and their reasoning |
| [`docs/plan_multiview_consensus.md`](docs/plan_multiview_consensus.md) | Pipeline architecture and evaluation plan |
| [`docs/eval_baseline.md`](docs/eval_baseline.md) | Baseline evaluation results and analysis |
| [`docs/setup_cuda.md`](docs/setup_cuda.md) | CUDA machine setup and data transfer guide |

---

## Project Structure

```
powerpolefinder/
├── src/
│   ├── auth.py                  # EagleView OAuth2 authentication
│   ├── discovery.py             # Image discovery and metadata
│   ├── download.py              # Oblique image acquisition
│   ├── download_wmts.py         # Ortho tile acquisition
│   ├── batch_multiview.py       # Multi-view consensus pipeline
│   ├── eval_testarea.py         # Evaluation framework
│   ├── classify_detections.py   # VLM agent classification
│   ├── auto_label.py            # Automated labeling pipeline
│   ├── finetune_gdino.py        # GroundingDINO fine-tuning
│   ├── georef_3d.py             # 3D georeferencing
│   ├── oblique_utils.py         # Pixel ↔ GPS conversion utilities
│   └── ratelimit.py             # API rate limiting
├── models/                      # SAM2, MASt3R (gitignored, see setup)
├── data/                        # Imagery and labels (gitignored, see setup)
├── docs/                        # Technical documentation
├── dashboard.html               # Interactive results dashboard
└── README.md
```

---

## License

For evaluation purposes only. EagleView imagery is subject to their [terms of service](https://www.eagleview.com/terms/).
