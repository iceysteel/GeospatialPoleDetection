# Setup Guide: CUDA Machine (2x 3090)

## Prerequisites
- Python 3.11+
- CUDA 11.8+ / 12.x
- ~20GB disk for models, ~15GB for data

## 1. Clone and install dependencies

```bash
git clone <repo-url>
cd powerpolefinder

# Python deps
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install groundingdino-py supervision

# HuggingFace transformers (for GroundingDINO)
pip install transformers

# SAM2
git clone https://github.com/facebookresearch/sam2.git models/sam2
pip install -e models/sam2

# MASt3R
git clone --recursive https://github.com/naver/mast3r.git models/mast3r
pip install -r models/mast3r/requirements.txt
pip install -r models/mast3r/dust3r/requirements.txt
```

## 2. Download model checkpoints

```bash
# SAM2
mkdir -p models/sam2/checkpoints
cd models/sam2/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
cd ../../..

# MASt3R (from HuggingFace)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
    filename='model.safetensors',
    local_dir='models/mast3r/checkpoints',
)
"
```

## 3. Set up EagleView API credentials

```bash
cp .env.example .env
# Edit .env with your credentials:
# EAGLEVIEW_CLIENT_ID=0oa1aef9hzozVKFc92p8
# EAGLEVIEW_CLIENT_SECRET=<secret>
# EAGLEVIEW_TOKEN_URL=https://apicenter.eagleview.com/oauth2/v1/token
# EAGLEVIEW_BASE_URL=https://sandbox.apis.eagleview.com
```

## 4. Transfer data from Mac

The `data/` directory is gitignored. Transfer it separately:

```bash
# On Mac:
rsync -avz --progress data/ user@cuda-machine:/path/to/powerpolefinder/data/

# Key data files to transfer:
# data/metadata.json              - Image metadata
# data/ground_truth_testarea.json - Ground truth labels (64)
# data/oblique/                   - 188 oblique images (1.4GB)
# data/wmts/                      - 1333 ortho tiles (130MB)
# data/testarea_grid/             - 144 gridded test images
# data/debug/batch/               - Batch run results
```

## 5. Verify setup

```bash
# Test GDino
python3 -c "
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny').to('cuda')
print(f'GDino loaded on {next(model.parameters()).device}')
"

# Test MASt3R
python3 -c "
import sys; sys.path.insert(0, 'models/mast3r'); sys.path.insert(0, 'models/mast3r/dust3r')
from mast3r.model import AsymmetricMASt3R
model = AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric').to('cuda')
print(f'MASt3R loaded on cuda')
"

# Test SAM2
python3 -c "
import sys; sys.path.insert(0, 'models/sam2')
from sam2.build_sam import build_sam2
model = build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml', 'models/sam2/checkpoints/sam2.1_hiera_base_plus.pt', device='cuda')
print('SAM2 loaded on cuda')
"
```

## 6. Run evaluation

```bash
# Run the eval on the test area
python3 src/eval_testarea.py

# View results
python3 -m http.server 8080
# Open http://localhost:8080/eval_map.html
```

## 7. Key differences from Mac (MPS)

- Change `device = 'mps'` to `device = 'cuda'` in scripts
- Remove `PYTORCH_ENABLE_MPS_FALLBACK=1`
- MASt3R will be 3-5x faster on CUDA
- For multi-GPU: `device = 'cuda:0'` / `device = 'cuda:1'`

## 8. Where we left off

- **Eval running:** `src/eval_testarea.py` on 36 grid cells with homography georeferencing
- **Pending:** Validate georeferencing accuracy on eval_map.html
- **Next:** Fine-tune GroundingDINO on labeled data, implement GDino-first pipeline optimization
- **Ground truth:** 64 labels in `data/ground_truth_testarea.json` (54 poles, 6 streetlights in focus area)
- **Precision baseline:** 45% (before fine-tuning)
