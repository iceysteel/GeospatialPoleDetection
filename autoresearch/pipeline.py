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
SAM3_PROMPTS_EXTRA = [('wooden pole', 0.40), ('power pole', 0.65)]  # (prompt, threshold)
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
DEDUP_RADIUS_M = 15

# Two-tier confidence: single-view detections need higher score
SINGLE_VIEW_MIN_SCORE = 0.45

# Exemplar-guided second pass: use high-confidence detections as positive geometric
# prompts to help SAM3 find additional poles it missed with text-only prompting.
# Multi-exemplar: use up to N spatially diverse exemplars simultaneously for
# richer concept guidance (SAM3 paper: 3 exemplars >> 1 exemplar >> text-only).
EXEMPLAR_ENABLED = True
EXEMPLAR_MIN_SCORE = 0.70  # Only use high-confidence detections as exemplars
EXEMPLAR_MAX_COUNT = 3     # Use up to 3 diverse exemplars simultaneously
EXEMPLAR_PASS_THRESH = 0.35  # Lower threshold for exemplar pass (geometric guidance reduces ambiguity)

# SAHI-style tiling: run SAM3 on overlapping crops to catch small/distant poles
# Using large tiles (1400px) for 2-3 tiles/image — fast but helps edge/distant poles
TILE_ENABLED = False
TILE_SIZE = 1400          # large tiles = only 2-3 per image (fast!)
TILE_OVERLAP = 0.25       # moderate overlap
TILE_MIN_DIM = 1600       # only tile images wider/taller than this
TILE_SCORE_PENALTY = 0.0  # no penalty — let two-tier handle filtering

# ============================================================================
# PIPELINE FUNCTIONS
# ============================================================================

def generate_tiles(img_w, img_h, tile_size=TILE_SIZE, overlap=TILE_OVERLAP):
    """Generate overlapping tile coordinates for SAHI-style inference.
    Returns list of (x_offset, y_offset, crop_w, crop_h) tuples."""
    if img_w <= TILE_MIN_DIM and img_h <= TILE_MIN_DIM:
        return []
    stride = int(tile_size * (1 - overlap))
    tiles = []
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            x2 = min(x + tile_size, img_w)
            y2 = min(y + tile_size, img_h)
            # Ensure minimum tile size
            if x2 - x < tile_size // 2 or y2 - y < tile_size // 2:
                continue
            tiles.append((x, y, x2 - x, y2 - y))
    return tiles


def run_sam3_on_tile(sam3_proc, tile_img, prompt_configs, offset_x, offset_y):
    """Run SAM3 detection on a single tile and map bboxes back to full-image coords."""
    state = sam3_proc.set_image(tile_img)
    dets = []
    for prompt_cfg in prompt_configs:
        if isinstance(prompt_cfg, tuple):
            prompt, thresh = prompt_cfg
        else:
            prompt, thresh = prompt_cfg, SAM3_THRESHOLD
        state = sam3_proc.set_text_prompt(state=state, prompt=prompt)
        for i in range(len(state['boxes'])):
            box = state['boxes'][i].tolist()
            score = state['scores'][i].item()
            if score >= thresh:
                # Map bbox back to full-image coordinates
                dets.append({
                    'bbox': [
                        int(box[0] + offset_x),
                        int(box[1] + offset_y),
                        int(box[2] + offset_x),
                        int(box[3] + offset_y),
                    ],
                    'score': score - TILE_SCORE_PENALTY,
                    '_from_tile': True,
                })
    return dets


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
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        scene.compute_global_alignment(init='mst', niter=100, schedule='cosine', lr=0.01)
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

    # Flatten ortho 3D pointmap for nearest-neighbor lookup
    ortho_3d = pts3d[1]
    th, tw = ortho_3d.shape[:2]
    ortho_flat = ortho_3d.reshape(-1, 3)

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

        # Camera reprojection (original method)
        pi = torch.inverse(poses[1])
        pc = pi[:3, :3] @ p3d + pi[:3, 3]
        if pc[2] <= 0: continue

        u_cam = focals[1] * pc[0] / pc[2] + tw / 2
        v_cam = focals[1] * pc[1] / pc[2] + th / 2
        if not (0 <= u_cam.item() < tw and 0 <= v_cam.item() < th): continue

        # 3D nearest-neighbor projection (find closest ortho 3D point)
        dists_3d = torch.norm(ortho_flat - p3d.unsqueeze(0), dim=1)
        # Mask NaN points
        nan_mask = torch.isnan(dists_3d)
        dists_3d[nan_mask] = float('inf')
        nearest_idx = dists_3d.argmin().item()
        u_nn = nearest_idx % tw
        v_nn = nearest_idx // tw

        # Average camera reprojection and 3D-NN for more robust localization
        u_avg = (u_cam.item() + u_nn) / 2
        v_avg = (v_cam.item() + v_nn) / 2

        uo = u_avg / (tw / ortho_w)
        vo = v_avg / (th / ortho_h)
        results.append({'ortho_px': (int(round(uo)), int(round(vo))), 'score': det['score'], 'src_bbox': det['bbox'], '_from_tile': det.get('_from_tile', False)})

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

            # SAM3 detection — multi-prompt with per-prompt thresholds
            prompt_configs = [(SAM3_PROMPT, SAM3_THRESHOLD)] + SAM3_PROMPTS_EXTRA

            # Full-image detection (original)
            state = sam3_proc.set_image(oblique)
            dets = []
            for prompt_cfg in prompt_configs:
                if isinstance(prompt_cfg, tuple):
                    prompt, thresh = prompt_cfg
                else:
                    prompt, thresh = prompt_cfg, SAM3_THRESHOLD
                state = sam3_proc.set_text_prompt(state=state, prompt=prompt)
                for i in range(len(state['boxes'])):
                    box = state['boxes'][i].tolist()
                    score = state['scores'][i].item()
                    if score >= thresh:
                        dets.append({
                            'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                            'score': score,
                        })

            # Exemplar-guided second pass: re-run SAM3 with text + multiple
            # positive box prompts from high-confidence detections. Using multiple
            # spatially diverse exemplars gives SAM3 a richer "concept" of what
            # poles look like in THIS specific image, recovering more missed poles.
            if EXEMPLAR_ENABLED and len(dets) > 0:
                # Collect all high-confidence candidates
                candidates = [d for d in dets if d['score'] >= EXEMPLAR_MIN_SCORE]
                if len(candidates) > 0:
                    # Select spatially diverse exemplars using farthest-point sampling
                    def bbox_center(d):
                        b = d['bbox']
                        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

                    # Start with highest-scoring detection
                    candidates.sort(key=lambda x: x['score'], reverse=True)
                    selected = [candidates[0]]
                    remaining = candidates[1:]

                    while len(selected) < EXEMPLAR_MAX_COUNT and remaining:
                        # Pick the candidate farthest from all selected exemplars
                        best_dist = -1
                        best_idx = 0
                        for ri, r in enumerate(remaining):
                            rc = bbox_center(r)
                            min_dist = min(
                                math.sqrt((rc[0] - bbox_center(s)[0])**2 + (rc[1] - bbox_center(s)[1])**2)
                                for s in selected
                            )
                            if min_dist > best_dist:
                                best_dist = min_dist
                                best_idx = ri
                        selected.append(remaining.pop(best_idx))

                    first_pass_dets = list(dets)
                    # Reset and re-run with multi-exemplar guidance
                    sam3_proc.reset_all_prompts(state)
                    state = sam3_proc.set_text_prompt(state=state, prompt=SAM3_PROMPT)
                    # Add all selected exemplars as positive geometric prompts
                    for exm in selected:
                        bx1, by1, bx2, by2 = exm['bbox']
                        norm_box = [
                            (bx1 + bx2) / 2 / w,  # cx
                            (by1 + by2) / 2 / h,   # cy
                            (bx2 - bx1) / w,        # w
                            (by2 - by1) / h,         # h
                        ]
                        state = sam3_proc.add_geometric_prompt(norm_box, True, state)
                    exemplar_count = 0
                    for i in range(len(state['boxes'])):
                        box = state['boxes'][i].tolist()
                        score = state['scores'][i].item()
                        if score >= EXEMPLAR_PASS_THRESH:
                            dets.append({
                                'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                                'score': score,
                            })
                            exemplar_count += 1
                    if exemplar_count > 0:
                        print(f"  DIAG multi-exemplar pass ({len(selected)} exemplars): +{exemplar_count} dets (pre-NMS)", flush=True)

            # SAHI-style tiled detection to catch small/distant poles
            if TILE_ENABLED:
                tiles = generate_tiles(w, h)
                for tx, ty, tw, th_ in tiles:
                    tile_crop = oblique.crop((tx, ty, tx + tw, ty + th_))
                    tile_dets = run_sam3_on_tile(sam3_proc, tile_crop, prompt_configs, tx, ty)
                    dets.extend(tile_dets)

            # Dedup overlapping boxes from different prompts/tiles (IoU > 0.5)
            if len(dets) > 1:
                # Sort by score descending for NMS
                dets.sort(key=lambda x: x['score'], reverse=True)
                keep = [True] * len(dets)
                for i in range(len(dets)):
                    if not keep[i]: continue
                    for j in range(i + 1, len(dets)):
                        if not keep[j]: continue
                        bi, bj = dets[i]['bbox'], dets[j]['bbox']
                        ix1 = max(bi[0], bj[0]); iy1 = max(bi[1], bj[1])
                        ix2 = min(bi[2], bj[2]); iy2 = min(bi[3], bj[3])
                        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                        a1 = (bi[2]-bi[0]) * (bi[3]-bi[1])
                        a2 = (bj[2]-bj[0]) * (bj[3]-bj[1])
                        iou = inter / (a1 + a2 - inter + 1e-6)
                        if iou > 0.65:  # relaxed from 0.5 to keep adjacent poles
                            keep[j] = False
                dets = [d for d, k in zip(dets, keep) if k]

            # MASt3R project to ortho
            projected = match_and_project(img_path, ortho, mast3r, device, dets, (h, w))
            if dets:
                print(f"  DIAG {os.path.basename(os.path.dirname(img_path))}/{d}: {len(dets)} SAM3 dets → {len(projected)} projected", flush=True)

            for proj in projected:
                ox, oy = proj['ortho_px']
                pt_lat, pt_lon = ortho_pixel_to_gps(ox, oy, ortho_meta)
                all_points.append({
                    'lat': round(pt_lat, 6),
                    'lon': round(pt_lon, 6),
                    'score': proj['score'],
                    'src_img': img_path,
                    'src_bbox': proj.get('src_bbox', None),
                    '_from_tile': proj.get('_from_tile', False),
                })

    # Dedup with spatial-proximity-aware confidence filtering
    TILE_SINGLE_VIEW_MIN = 0.55  # tile-only single-view detections need higher score
    PROXIMITY_RESCUE_MIN = 0.38  # borderline dets rescued if near other poles
    PROXIMITY_RESCUE_RADIUS = 80  # meters — max distance to nearest accepted detection
    m = 111320 * math.cos(math.radians(41.249))
    used = [False] * len(all_points)
    clusters = []
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
        best['_cluster_size'] = len(cluster)
        has_fullimg = any(not c.get('_from_tile') for c in cluster)
        clusters.append((best, cluster, has_fullimg))

    # First pass: accept high-confidence detections
    deduped = []
    borderline = []  # candidates for proximity rescue
    for best, cluster, has_fullimg in clusters:
        if len(cluster) >= 2:
            deduped.append(best)
        elif has_fullimg and best['score'] >= SINGLE_VIEW_MIN_SCORE:
            deduped.append(best)
        elif not has_fullimg and best['score'] >= TILE_SINGLE_VIEW_MIN:
            deduped.append(best)
        elif has_fullimg and best['score'] >= PROXIMITY_RESCUE_MIN:
            borderline.append(best)
        else:
            src = "tile" if not has_fullimg else "full"
            print(f"  DIAG filtered {src}: score={best['score']:.3f} lat={best['lat']} lon={best['lon']}", flush=True)

    # Second pass: rescue borderline detections near accepted poles
    rescued = 0
    for b in borderline:
        near_accepted = False
        for d in deduped:
            dist = math.sqrt(((b['lat'] - d['lat']) * 111320) ** 2 +
                           ((b['lon'] - d['lon']) * m) ** 2)
            if dist < PROXIMITY_RESCUE_RADIUS:
                near_accepted = True
                break
        if near_accepted:
            deduped.append(b)
            rescued += 1
            print(f"  DIAG proximity-rescued: score={b['score']:.3f} lat={b['lat']} lon={b['lon']}", flush=True)
        else:
            print(f"  DIAG filtered isolated: score={b['score']:.3f} lat={b['lat']} lon={b['lon']}", flush=True)
    if rescued > 0:
        print(f"  DIAG proximity rescue: {rescued} borderline dets rescued", flush=True)

    # Dump diagnostic data
    diag = {'all_points': len(all_points), 'deduped': len(deduped),
            'detections': [{'lat': d['lat'], 'lon': d['lon'], 'score': d['score'],
                           'cluster_size': d.get('_cluster_size', 1)} for d in deduped]}
    with open(os.path.join(os.path.dirname(__file__), 'diag_detections.json'), 'w') as f:
        json.dump(diag, f)

    # VLM post-filter on OBLIQUE crops (not ortho!)
    # Previous VLM on ortho crops failed because poles look like dots from above.
    # Oblique crops clearly show what the object is — poles, trees, signs, etc.
    VLM_OBLIQUE_FILTER = False
    if not VLM_OBLIQUE_FILTER:
        return deduped

    import requests, base64, io

    # Free GPU memory from SAM3/MASt3R before VLM loads
    del sam3_proc, mast3r
    torch.cuda.empty_cache()
    import gc; gc.collect()

    vlm_filtered = []
    vlm_removed = 0
    for pt in deduped:
        src_img = pt.get('src_img')
        src_bbox = pt.get('src_bbox')
        if not src_img or not src_bbox or not os.path.exists(src_img):
            vlm_filtered.append(pt)
            continue

        # Crop oblique image around detection with 50% padding for context
        try:
            oblique = Image.open(src_img).convert('RGB')
            iw, ih = oblique.size
            x1, y1, x2, y2 = src_bbox
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = max(int(bw * 0.5), 50), max(int(bh * 0.5), 50)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(iw, x2 + pad_x)
            cy2 = min(ih, y2 + pad_y)
            crop = oblique.crop((cx1, cy1, cx2, cy2))

            buf = io.BytesIO()
            crop.save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode()

            resp = requests.post('http://localhost:11434/api/chat', json={
                'model': 'qwen3.5:27b',
                'messages': [{
                    'role': 'user',
                    'content': (
                        'This is a crop from an aerial photograph taken at an oblique angle. '
                        'Does this image show a utility pole, telephone pole, or power pole? '
                        'Utility poles are tall vertical wooden or concrete posts, usually with '
                        'crossarms, insulators, or electrical wires attached near the top. '
                        'Answer NO if you see: a tree, bush, street light, traffic sign, '
                        'building column, antenna, flagpole, or other non-pole object. '
                        'Answer ONLY "YES" or "NO".'
                    ),
                    'images': [b64],
                }],
                'stream': False,
                'options': {'temperature': 0, 'num_predict': 10},
                'think': False,
            }, timeout=30)
            answer = resp.json().get('message', {}).get('content', '').strip().lower()
            if 'no' in answer and 'yes' not in answer:
                vlm_removed += 1
                continue  # VLM says not a pole — filter out
        except:
            pass  # On error, keep detection

        vlm_filtered.append(pt)

    print(f"  VLM oblique filter: {len(deduped)} → {len(vlm_filtered)} (removed {vlm_removed})", flush=True)
    return vlm_filtered
