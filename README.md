# Power Pole Detection from Oblique Aerial Imagery

An end-to-end pipeline for automated detection and geolocation of utility power poles from EagleView oblique aerial imagery. The system uses SAM3 for zero-shot detection, MASt3R for 3D oblique-to-ortho projection, and an autonomous research agent (AutoResearch) to optimize the pipeline — achieving **F1 = 0.727** at a 10m match radius across 121 automated experiments.

> **[View Interactive Dashboard](https://iceysteel.github.io/GeospatialPoleDetection/dashboard.html)** — Explore detection results on the map, track experiment progress, and inspect individual detections.

---

## Summary of Results

| Metric | Value |
|---|---|
| **Best F1@10m** | 0.727 |
| **Precision** | 78.0% |
| **Recall** | 68.1% |
| **RMSE** | ~4.6m |
| **Test Area** | 400m × 360m, Omaha NE (94 verified GT poles) |
| **Holdout Area** | Same size, 500m east (94 verified GT poles) |
| **Holdout F1@10m** | 0.714 (no overfitting) |
| **Experiments Run** | 121 autonomous iterations |

---

## Pipeline Architecture

The pipeline processes EagleView oblique imagery through SAM3 detection, MASt3R 3D projection, and multi-view aggregation to produce geolocated pole detections.

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │  EagleView Oblique Imagery                                          │
  │  4 cardinal directions (N/E/S/W) at ~45° elevation, ~4cm GSD       │
  │  36 grid cells × 4 directions = 144 images                         │
  └────────────────────────────┬─────────────────────────────────────────┘
                               │
          ┌────────────────────▼────────────────────┐
          │  Stage 1: SAM3 Text-Prompted Detection   │
          │  3 prompts: telephone / wooden / power   │
          │  Threshold: 0.40 (power pole: 0.65)      │
          │  + Horizontal flip TTA                   │
          └────────────────────┬────────────────────┘
                               │
          ┌────────────────────▼────────────────────┐
          │  Stage 2: MASt3R Oblique→Ortho Mapping   │
          │  AerialMegaDepth checkpoint              │
          │  PointCloudOptimizer, 100 iterations     │
          │  60m ortho crop per detection            │
          │  Projects bbox center → GPS via 3D       │
          └────────────────────┬────────────────────┘
                               │
          ┌────────────────────▼────────────────────┐
          │  Stage 3: Multi-View Aggregation          │
          │  Deduplicate within 15m radius           │
          │  Two-tier filter: single-view ≥ 0.45     │
          │  Keep highest-scoring detection           │
          └────────────────────┬────────────────────┘
                               │
                               ▼
                     Geolocated pole detections
                     (lat, lon, score) per pole
```

---

## F1 Score Progression

### Manual Development Phase

| Stage | Approach | F1@10m | Change |
|---|---|---|---|
| Baseline | GDino-Tiny + pixel×GSD georeferencing | 0.33 | — |
| + VLM Filter | Qwen 3.5 27B single-crop classification | 0.45 | +0.12 |
| + Fine-tune | Domain fine-tuned GDino-Base + GT refinement | 0.68 | +0.23 |
| + Multi-view | All 4 oblique views + ortho sent to VLM | 0.70 | +0.02 |

### AutoResearch Agent Phase (121 experiments)

| Milestone | Approach | F1@10m | Exp # |
|---|---|---|---|
| SAM3 baseline | SAM3 text-prompted, thresh=0.10 | 0.335 | 1 |
| Threshold tuning | thresh=0.40 | 0.592 | 8 |
| Ortho crop | 80m → 60m | 0.559 | 6 |
| Multi-prompt | "telephone pole" + "wooden pole" | 0.667 | 34 |
| MASt3R tuning | PointCloudOptimizer, 100 iters | 0.702 | 49 |
| Two-tier filter | Single-view min 0.45, dedup 15m | 0.714 | 56 |
| Horizontal flip TTA | Detect on original + flipped | 0.719 | 88 |
| Ortho-direct + tiled | High-res zoom-23 ortho detection | **0.727** | 116 |

---

## AutoResearch: Autonomous Pipeline Optimization

The project uses an autonomous research agent inspired by [Karpathy's AutoResearch pattern](https://x.com/karpathy/status/1915538839098286186) to systematically optimize the detection pipeline. The agent runs in a loop, modifying the pipeline, evaluating changes, and keeping or reverting based on F1 improvement.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│  run.sh — Loop Runner                               │
│  Launches Claude with 2-hour timeout per iteration  │
└──────────────────────┬──────────────────────────────┘
                       │
    ┌──────────────────▼──────────────────┐
    │  Agent Iteration Cycle               │
    │  1. Read program.md (history)        │
    │  2. Read pipeline.py (current code)  │
    │  3. Search web for techniques        │
    │  4. Implement ONE change             │
    │  5. git commit                       │
    │  6. Run prepare.py (eval)            │
    │  7. Keep if improved, revert if not  │
    │  8. Log to autoresearch.jsonl        │
    │  9. Update program.md               │
    └─────────────────────────────────────┘

Files:
  prepare.py        — LOCKED eval harness (F1@10m, auto holdout on new best)
  pipeline.py       — Agent-modifiable detection pipeline
  program.md        — Research brief: all experiments, what worked/failed
  autoresearch.jsonl — Append-only experiment log
  population.json   — Top-N config population (genetic)
```

### Key Design Decisions
- **Holdout protection**: Holdout eval only runs when test F1 is a new best, preventing overfitting
- **Git-based rollback**: Every experiment is a commit; failures are `git revert`ed
- **Web research**: Agent searches for papers and techniques when parameter tuning is exhausted
- **30-min eval timeout**: Prevents runaway experiments from blocking the loop

---

## Models

| Model | Size | Role |
|---|---|---|
| **SAM3** (Segment Anything 3) | 848M (4 GB) | Text-prompted pole detection in oblique views |
| **MASt3R** (AerialMegaDepth) | 2.6 GB | 3D oblique→ortho projection via dense reconstruction |
| Qwen 3.5 27B | 17 GB | VLM classification (optional post-filter, via Ollama) |
| GroundingDINO-Base | 232M | Earlier detector, replaced by SAM3 per requirements |

---

## Ground Truth & Evaluation

- **Test area**: 94 manually verified poles in a 400m × 360m area
- **Holdout area**: 94 poles, 500m east of test area (never seen during optimization)
- **Match radius**: 10m (industry standard for utility asset mapping)
- **Verification tool**: `gt_verify.html` — draggable markers on ortho imagery, right-click to add, keyboard shortcuts for flagging
- **Multi-radius analysis** (test area):

| Radius | F1 | Precision | Recall |
|---|---|---|---|
| 3m | 0.211 | 23.4% | 19.1% |
| 5m | 0.456 | 50.6% | 41.5% |
| 7m | 0.620 | 68.8% | 56.4% |
| **10m** | **0.727** | **78.0%** | **68.1%** |
| 15m | 0.702 | 77.9% | 63.8% |

---

## Data Acquisition

All imagery is sourced from the EagleView sandbox API, covering approximately 1.5 square miles in Omaha, Nebraska.

| Data Type | Count | Source |
|---|---|---|
| Oblique images | 188 (4 directions × ~47 locations) | Imagery API, 50m radius crops |
| WMTS ortho tiles | Zoom 19–23 (test + holdout) | WMTS API |
| Test area grid | 36 cells × 4 directions | 80m grid spacing |
| Ground truth | 94 test + 94 holdout poles | Manual verification via ortho + oblique |

---

## Interactive Tools

Served via `python3 -m http.server 8080`:

- **`dashboard.html`** — AutoResearch dashboard: F1 progress chart, experiment log, population, detection map ([live](https://iceysteel.github.io/GeospatialPoleDetection/dashboard.html))
- **`eval_map.html`** — Evaluation overlay with threshold sliders, multi-zoom ortho tiles, VLM-rejected detection layer
- **`gt_verify.html`** — Ground truth verification: draggable markers, right-click to add, supports `?area=holdout`
- **`labeler_v2.html`** — Map-based labeling with auto-label review workflow

---

## Setup

### Requirements
- Python 3.11+, [uv](https://github.com/astral-sh/uv) for package management
- 2× NVIDIA RTX 3090 (or equivalent, 24GB+ VRAM)
- EagleView API credentials (sandbox access)

### Quick Start

```bash
git clone https://github.com/iceysteel/GeospatialPoleDetection.git
cd GeospatialPoleDetection
uv pip install -r requirements.txt

# Models
# SAM3
git clone https://github.com/facebookresearch/sam3.git models/sam3
uv pip install -e models/sam3

# MASt3R + AerialMegaDepth checkpoint
git clone --recursive https://github.com/naver/mast3r.git models/mast3r
uv pip install -r models/mast3r/dust3r/requirements.txt

# Credentials
cp .env.example .env  # Edit with EagleView API keys

# Run pipeline
python3 autoresearch/pipeline.py

# Run evaluation
python3 autoresearch/prepare.py

# View results
python3 -m http.server 8080
```

### Run AutoResearch Loop

```bash
# Start autonomous optimization (runs indefinitely)
./autoresearch/run.sh

# Monitor progress
tail -f autoresearch/autoresearch.jsonl | python3 -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l)
    print(f'F1={d[\"f1\"]:.4f} P={d[\"precision\"]:.1%} R={d[\"recall\"]:.1%}')
"
```

---

## Project Structure

```
GeospatialPoleDetection/
├── autoresearch/
│   ├── prepare.py           # LOCKED evaluation harness
│   ├── pipeline.py          # Agent-modifiable detection pipeline
│   ├── program.md           # Research brief (121 experiments documented)
│   ├── run.sh               # Loop runner with 2h timeout
│   ├── autoresearch.jsonl   # Append-only experiment log
│   └── population.json      # Top-N config population
├── src/
│   ├── oblique_to_ortho.py  # Core SAM3 + MASt3R pipeline
│   ├── eval_testarea.py     # Original evaluation framework
│   ├── eval_sam3_mast3r.py  # SAM3 + MASt3R evaluation
│   ├── georef_3d.py         # 3D affine georeferencing
│   ├── classify_detections.py # VLM classification (Ollama/vLLM)
│   ├── auto_label.py        # Automated labeling pipeline
│   ├── finetune_sam3.py     # SAM3 fine-tuning
│   ├── build_training_set.py # COCO format training set builder
│   ├── agent_tools.py       # Shared tools: ortho loading, crops
│   ├── oblique_utils.py     # Pixel ↔ GPS conversion
│   ├── gpu_utils.py         # Dual-GPU memory management
│   └── download_grid.py     # Reusable grid downloader
├── models/                  # SAM3, MASt3R, LoRA weights (gitignored)
├── data/
│   ├── ground_truth_testarea.json   # 94 verified test poles
│   └── ground_truth_holdout.json    # 94 verified holdout poles
├── docs/                    # Technical documentation
├── dashboard.html           # AutoResearch dashboard
├── eval_map.html            # Interactive evaluation map
├── gt_verify.html           # Ground truth verification tool
└── labeler_v2.html          # Map-based labeling tool
```

---

## License

For evaluation purposes only. EagleView imagery is subject to their [terms of service](https://www.eagleview.com/terms/).
