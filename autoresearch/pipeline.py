#!/usr/bin/env python3
"""
AutoResearch Pipeline — MODIFIABLE BY AGENT

This file contains the full pole detection pipeline. The agent modifies this
file to improve F1@10m. Must use SAM3 + MASt3R as core components.

Current best F1@10m: 0.448 (GDino-ft + VLM, oblique consensus)
"""
import sys, os, json, math, time, tempfile

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'mast3r'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'mast3r', 'dust3r'))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
GRID_DIR = os.path.join(DATA_DIR, 'testarea_grid')
WMTS_DIR = os.path.join(DATA_DIR, 'wmts')

# ============================================================================
# CONFIGURABLE PARAMETERS — Agent can modify these
# ============================================================================

# Detection
DETECTOR = 'sam3'  # 'sam3' or 'sam3_lora_v2'
SAM3_PROMPT = 'telephone pole'
SAM3_THRESHOLD = 0.40
SAM3_CKPT = os.path.join(os.path.expanduser("~"),
    ".cache/huggingface/hub/models--bodhicitta--sam3/snapshots/"
    "cba430d22f6fdc3f06ad3841274ec7bb55885f2f/sam3.pt")

# MASt3R
MAST3R_CHECKPOINT = 'kvuong2711/checkpoint-aerial-mast3r'
ORTHO_CROP_RADIUS_M = 60
ORTHO_ZOOM = 21

# Projection
PROJECT_POLE_BASE = True  # True=bottom of bbox, False=center

# Dedup
DEDUP_RADIUS_M = 10

# ============================================================================
# PIPELINE FUNCTIONS
# ============================================================================

def load_sam3(device='cuda'):
    """Load SAM3 detector."""
    torch.autocast('cuda', dtype=torch.bfloat16).__enter__()
    import sam3 as sam3_module
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_root = os.path.join(os.path.dirname(sam3_module.__file__), '..')
    bpe_path = os.path.join(sam3_root, 'assets', 'bpe_simple_vocab_16e6.txt.gz')
    model = build_sam3_image_model(bpe_path=bpe_path, device=device,
                                    checkpoint_path=SAM3_CKPT, load_from_HF=False)

    if DETECTOR == 'sam3_lora_v2':
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'sam3_lora'))
        from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
        cfg = LoRAConfig(rank=16, alpha=32, dropout=0.05,
            target_modules=['q_proj','k_proj','v_proj','out_proj','qkv','proj','fc1','fc2','c_fc','c_proj'],
            apply_to_vision_encoder=True, apply_to_text_encoder=True,
            apply_to_geometry_encoder=True, apply_to_detr_encoder=True,
            apply_to_detr_decoder=True, apply_to_mask_decoder=True)
        model = apply_lora_to_model(model, cfg)
        lora_path = os.path.join(PROJECT_ROOT, 'models', 'sam3_finetuned', 'lora_v2', 'best_lora_weights.pt')
        load_lora_weights(model, lora_path)
        model = model.cuda()

    return Sam3Processor(model, confidence_threshold=SAM3_THRESHOLD)


def load_mast3r(device='cuda'):
    """Load MASt3R model."""
    from mast3r.model import AsymmetricMASt3R
    return AsymmetricMASt3R.from_pretrained(MAST3R_CHECKPOINT).to(device).eval()


def stitch_ortho(center_lat, center_lon, radius_m=ORTHO_CROP_RADIUS_M, zoom=ORTHO_ZOOM):
    """Stitch WMTS tiles into ortho crop."""
    from agent_tools import lat_lon_to_tile, tile_to_lat_lon
    n = 2 ** zoom
    tile_deg = 360 / n
    m_per_px = tile_deg * 111320 * math.cos(math.radians(center_lat)) / 256
    r_deg_lat = radius_m / 111320
    r_deg_lon = radius_m / (111320 * math.cos(math.radians(center_lat)))
    x1, y1 = lat_lon_to_tile(center_lat + r_deg_lat, center_lon - r_deg_lon, zoom)
    x2, y2 = lat_lon_to_tile(center_lat - r_deg_lat, center_lon + r_deg_lon, zoom)
    from PIL import Image
    w = (x2 - x1 + 1) * 256
    h = (y2 - y1 + 1) * 256
    stitched = Image.new('RGB', (w, h))
    loaded = 0
    for tx in range(x1, x2 + 1):
        for ty in range(y1, y2 + 1):
            path = os.path.join(WMTS_DIR, f'{zoom}_{tx}_{ty}.png')
            if os.path.exists(path):
                try:
                    tile = Image.open(path).convert('RGB')
                    stitched.paste(tile, ((tx - x1) * 256, (ty - y1) * 256))
                    loaded += 1
                except: pass
    tl_lat, tl_lon = tile_to_lat_lon(x1, y1, zoom)
    br_lat, br_lon = tile_to_lat_lon(x2 + 1, y2 + 1, zoom)
    return stitched, {
        'zoom': zoom, 'm_per_px': m_per_px, 'tiles': loaded,
        'tl_lat': tl_lat, 'tl_lon': tl_lon, 'br_lat': br_lat, 'br_lon': br_lon,
        'width': w, 'height': h,
    }


def ortho_pixel_to_gps(px, py, meta):
    lon = meta['tl_lon'] + px / meta['width'] * (meta['br_lon'] - meta['tl_lon'])
    lat = meta['tl_lat'] + py / meta['height'] * (meta['br_lat'] - meta['tl_lat'])
    return lat, lon


def match_and_project(oblique_path, ortho_img, mast3r, device, detections, oblique_size):
    """Run MASt3R on oblique↔ortho pair, project detections to ortho GPS."""
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    ortho_img.save(tmp.name)
    try:
        imgs = load_images([oblique_path, tmp.name], size=512)
        pairs = make_pairs(imgs, scene_graph='complete', symmetrize=True)
        output = inference(pairs, mast3r, device, batch_size=1)
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer)
        scene.compute_global_alignment(init='mst', niter=300, schedule='cosine', lr=0.01)
        pts3d = scene.get_pts3d()
        poses = scene.get_im_poses()
        focals = scene.get_focals()
    except:
        return []
    finally:
        os.unlink(tmp.name)

    oh, ow = oblique_size
    ortho_h, ortho_w = ortho_img.size[1], ortho_img.size[0]
    pv = pts3d[0]
    hm, wm = pv.shape[:2]

    results = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        if PROJECT_POLE_BASE:
            px, py = (x1 + x2) // 2, y2
        else:
            px, py = (x1 + x2) // 2, (y1 + y2) // 2

        sx, sy = wm / ow, hm / oh
        mx = min(max(int(round(px * sx)), 0), wm - 1)
        my = min(max(int(round(py * sy)), 0), hm - 1)
        p3d = pv[my, mx]
        if torch.isnan(p3d).any(): continue

        pi = torch.inverse(poses[1])
        pc = pi[:3, :3] @ p3d + pi[:3, 3]
        if pc[2] <= 0: continue

        th, tw = pts3d[1].shape[:2]
        u = focals[1] * pc[0] / pc[2] + tw / 2
        v = focals[1] * pc[1] / pc[2] + th / 2
        if not (0 <= u.item() < tw and 0 <= v.item() < th): continue

        uo = u.item() / (tw / ortho_w)
        vo = v.item() / (th / ortho_h)
        results.append({'ortho_px': (int(round(uo)), int(round(vo))), 'score': det['score']})

    return results


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline():
    """
    Run the full detection pipeline. Returns list of dicts with 'lat' and 'lon'.
    MUST use SAM3 for detection and MASt3R for oblique→ortho mapping.
    """
    device = 'cuda'
    from PIL import Image
    from oblique_utils import parse_footprint

    # Load models
    sam3_proc = load_sam3(device)
    mast3r = load_mast3r(device)

    # Load grid
    with open(os.path.join(GRID_DIR, 'index.json')) as f:
        grid = json.load(f)

    # Load metadata for footprint lookup
    with open(os.path.join(DATA_DIR, 'metadata.json')) as f:
        metadata = json.load(f)

    DIRECTIONS = ['north', 'east', 'south', 'west']
    all_points = []

    for ci, cell in enumerate(grid):
        lat, lon = cell['lat'], cell['lon']

        # Stitch ortho for this cell
        ortho, ortho_meta = stitch_ortho(lat, lon)
        if ortho_meta['tiles'] < 4: continue

        for d in DIRECTIONS:
            img_path = cell['images'].get(d)
            if not img_path or not os.path.exists(img_path): continue

            oblique = Image.open(img_path).convert('RGB')
            w, h = oblique.size

            # SAM3 detection
            state = sam3_proc.set_image(oblique)
            state = sam3_proc.set_text_prompt(state=state, prompt=SAM3_PROMPT)
            if len(state['boxes']) == 0: continue

            # Extract detections
            dets = []
            for i in range(len(state['boxes'])):
                box = state['boxes'][i].tolist()
                score = state['scores'][i].item()
                dets.append({
                    'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    'score': score,
                })

            # MASt3R project to ortho
            projected = match_and_project(img_path, ortho, mast3r, device, dets, (h, w))

            for proj in projected:
                ox, oy = proj['ortho_px']
                pt_lat, pt_lon = ortho_pixel_to_gps(ox, oy, ortho_meta)
                all_points.append({
                    'lat': round(pt_lat, 6),
                    'lon': round(pt_lon, 6),
                    'score': proj['score'],
                })

    # Dedup
    m = 111320 * math.cos(math.radians(41.249))
    used = [False] * len(all_points)
    deduped = []
    for i, p in enumerate(all_points):
        if used[i]: continue
        cluster = [p]; used[i] = True
        for j in range(i + 1, len(all_points)):
            if used[j]: continue
            dist = math.sqrt(((p['lat'] - all_points[j]['lat']) * 111320) ** 2 +
                           ((p['lon'] - all_points[j]['lon']) * m) ** 2)
            if dist < DEDUP_RADIUS_M:
                cluster.append(all_points[j]); used[j] = True
        best = max(cluster, key=lambda x: x['score'])
        best['lat'] = round(sum(c['lat'] for c in cluster) / len(cluster), 6)
        best['lon'] = round(sum(c['lon'] for c in cluster) / len(cluster), 6)
        deduped.append(best)

    return deduped
