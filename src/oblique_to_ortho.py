#!/usr/bin/env python3
"""
Core pipeline: SAM3 → MASt3R(AerialMegaDepth) → Ortho Labels

For each oblique image:
  1. SAM3 detects poles (text prompt) → bboxes + masks
  2. Load matching ortho tiles for this image's ground footprint
  3. MASt3R (AerialMegaDepth) on [oblique, ortho] pair → dense 3D correspondences
  4. Project pole base pixels through MASt3R → ortho pixel → GPS via tile math
  5. Dedup across all images, evaluate, output labeled ortho image
"""
import sys, os, json, time, math, argparse, tempfile
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'mast3r'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'mast3r', 'dust3r'))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
import numpy as np
from PIL import Image, ImageDraw
from agent_tools import lat_lon_to_tile, tile_to_lat_lon
from gpu_utils import get_device, gpu_memory_report
from oblique_utils import parse_footprint

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
WMTS_DIR = os.path.join(DATA_DIR, 'wmts')
RESULTS_DIR = os.path.join(DATA_DIR, 'eval_testarea')

SAM3_PROMPT = "telephone pole"
SAM3_THRESHOLD = 0.10
MAST3R_CHECKPOINT = 'kvuong2711/checkpoint-aerial-mast3r'

SAM3_CKPT = os.path.join(os.path.expanduser("~"),
    ".cache/huggingface/hub/models--bodhicitta--sam3/snapshots/"
    "cba430d22f6fdc3f06ad3841274ec7bb55885f2f/sam3.pt")

FOCUS_LAT, FOCUS_LON = 41.248644, -95.998878
RADIUS_LAT, RADIUS_LON = 0.0018, 0.0024


# ---- Ortho tile handling ----

def stitch_ortho_for_footprint(center_lat, center_lon, radius_m=200, zoom=21):
    """Stitch WMTS tiles into an ortho reference image covering the area."""
    n = 2 ** zoom
    tile_deg = 360 / n
    m_per_px = tile_deg * 111320 * math.cos(math.radians(center_lat)) / 256

    r_deg_lat = radius_m / 111320
    r_deg_lon = radius_m / (111320 * math.cos(math.radians(center_lat)))

    x1, y1 = lat_lon_to_tile(center_lat + r_deg_lat, center_lon - r_deg_lon, zoom)
    x2, y2 = lat_lon_to_tile(center_lat - r_deg_lat, center_lon + r_deg_lon, zoom)

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
                except:
                    pass

    tl_lat, tl_lon = tile_to_lat_lon(x1, y1, zoom)
    br_lat, br_lon = tile_to_lat_lon(x2 + 1, y2 + 1, zoom)

    return stitched, {
        'zoom': zoom, 'm_per_px': m_per_px, 'tiles': loaded,
        'tl_lat': tl_lat, 'tl_lon': tl_lon,
        'br_lat': br_lat, 'br_lon': br_lon,
        'width': w, 'height': h,
    }


def ortho_pixel_to_gps(px, py, meta):
    """Convert pixel in stitched ortho → GPS. Exact via tile math."""
    lon = meta['tl_lon'] + px / meta['width'] * (meta['br_lon'] - meta['tl_lon'])
    lat = meta['tl_lat'] + py / meta['height'] * (meta['br_lat'] - meta['tl_lat'])
    return lat, lon


def gps_to_ortho_pixel(lat, lon, meta):
    """Convert GPS → pixel in stitched ortho."""
    x = (lon - meta['tl_lon']) / (meta['br_lon'] - meta['tl_lon']) * meta['width']
    y = (lat - meta['tl_lat']) / (meta['br_lat'] - meta['tl_lat']) * meta['height']
    return int(round(x)), int(round(y))


def footprint_center(meta):
    """Get center GPS of an oblique image's ground footprint."""
    ring = parse_footprint(meta)
    if ring is None:
        qp = meta.get('query_point', {})
        return qp.get('lat', 0), qp.get('lon', 0)
    lats = [p[1] for p in ring]
    lons = [p[0] for p in ring]
    return sum(lats) / len(lats), sum(lons) / len(lons)


# ---- MASt3R matching ----

def validate_mast3r_output(pts3d, poses, focals):
    """Check MASt3R output quality. Returns (ok, issues)."""
    issues = []
    # NaN rate
    for i, pv in enumerate(pts3d):
        nan_rate = torch.isnan(pv).any(dim=-1).float().mean().item()
        if nan_rate > 0.5:
            issues.append(f"view {i}: {nan_rate:.0%} NaN")
    # Pose determinant
    for i, pose in enumerate(poses):
        det = torch.det(pose[:3, :3]).item()
        if abs(det - 1.0) > 0.1 and abs(det + 1.0) > 0.1:
            issues.append(f"view {i}: bad pose det={det:.2f}")
    return len(issues) == 0, issues


def match_oblique_ortho(oblique_path, ortho_img, mast3r_model, device):
    """Run MASt3R on oblique↔ortho pair. Returns (pts3d, poses, focals) or None."""
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    # Save ortho to temp file for MASt3R's load_images
    ortho_tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    ortho_img.save(ortho_tmp.name)

    try:
        imgs = load_images([oblique_path, ortho_tmp.name], size=512)
        pairs = make_pairs(imgs, scene_graph='complete', symmetrize=True)
        output = inference(pairs, mast3r_model, device, batch_size=1)
        scene = global_aligner(output, device=device,
                               mode=GlobalAlignerMode.ModularPointCloudOptimizer)
        scene.compute_global_alignment(init='mst', niter=300, schedule='cosine', lr=0.01)

        pts3d = scene.get_pts3d()
        poses = scene.get_im_poses()
        focals = scene.get_focals()

        ok, issues = validate_mast3r_output(pts3d, poses, focals)
        if not ok:
            return None, f"MASt3R validation failed: {issues}"

        return (pts3d, poses, focals), None
    except Exception as e:
        return None, str(e)
    finally:
        os.unlink(ortho_tmp.name)


def project_to_ortho(px, py, oblique_size, ortho_size, pts3d, poses, focals):
    """Project a pixel from oblique (view 0) to ortho (view 1) via MASt3R 3D."""
    oh, ow = oblique_size  # (height, width)
    pv = pts3d[0]  # oblique point cloud
    hm, wm = pv.shape[:2]

    # Scale pixel to MASt3R resolution
    sx, sy = wm / ow, hm / oh
    mx = min(max(int(round(px * sx)), 0), wm - 1)
    my = min(max(int(round(py * sy)), 0), hm - 1)

    p3d = pv[my, mx]
    if torch.isnan(p3d).any():
        return None

    # Project into ortho view (view 1)
    pi = torch.inverse(poses[1])
    pc = pi[:3, :3] @ p3d + pi[:3, 3]
    if pc[2] <= 0:
        return None

    th, tw = pts3d[1].shape[:2]
    u = focals[1] * pc[0] / pc[2] + tw / 2
    v = focals[1] * pc[1] / pc[2] + th / 2

    if not (0 <= u.item() < tw and 0 <= v.item() < th):
        return None

    # Scale back to original ortho image coordinates
    oh2, ow2 = ortho_size  # (height, width)
    uo = u.item() / (tw / ow2)
    vo = v.item() / (th / oh2)

    return int(round(uo)), int(round(vo))


# ---- Evaluation ----

def evaluate(detections, gt_labels, match_radius_m=10):
    gt_poles = [l for l in gt_labels if l['label'] == 'pole']
    m_per_deg_lon = 111320 * math.cos(math.radians(41.249))
    gt_matched = set()
    tp = fp = 0
    match_dists = []
    for det in detections:
        best_dist, best_idx = float('inf'), -1
        for i, gt in enumerate(gt_poles):
            d = math.sqrt(((det['lat'] - gt['lat']) * 111320) ** 2 +
                         ((det['lon'] - gt['lon']) * m_per_deg_lon) ** 2)
            if d < best_dist:
                best_dist, best_idx = d, i
        if best_dist <= match_radius_m and best_idx not in gt_matched:
            tp += 1
            gt_matched.add(best_idx)
            match_dists.append(best_dist)
        else:
            fp += 1
    fn = len(gt_poles) - tp
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    rmse = math.sqrt(sum(d ** 2 for d in match_dists) / len(match_dists)) if match_dists else 0
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': round(p, 3), 'recall': round(r, 3), 'f1': round(f1, 3),
        'rmse': round(rmse, 1), 'gt_poles': len(gt_poles),
    }


# ---- Main pipeline ----

def run_pipeline(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast('cuda', dtype=torch.bfloat16).__enter__()

    device = args.device
    t_start = time.time()

    # Load metadata for all oblique images
    with open(os.path.join(DATA_DIR, 'metadata.json')) as f:
        metadata = json.load(f)
    obliques = {urn: m for urn, m in metadata.items()
                if m['type'] == 'oblique' and m.get('local_path') and os.path.exists(m['local_path'])}
    print(f"Oblique images: {len(obliques)}")

    # Load GT
    with open(os.path.join(DATA_DIR, 'ground_truth_testarea.json')) as f:
        gt = json.load(f)
    area = (FOCUS_LAT - RADIUS_LAT, FOCUS_LON - RADIUS_LON,
            FOCUS_LAT + RADIUS_LAT, FOCUS_LON + RADIUS_LON)
    gt_labels = [l for l in gt['labels']
                 if area[0] <= l['lat'] <= area[2] and area[1] <= l['lon'] <= area[3]]
    print(f"GT: {sum(1 for l in gt_labels if l['label'] == 'pole')} poles")

    # Build reference ortho for the full test area
    print("Stitching ortho reference...", flush=True)
    ortho_full, ortho_full_meta = stitch_ortho_for_footprint(FOCUS_LAT, FOCUS_LON, radius_m=300, zoom=21)
    print(f"  Ortho: {ortho_full.size}, {ortho_full_meta['tiles']} tiles")

    # Load SAM3
    print("Loading SAM3...", flush=True)
    import sam3 as sam3_module
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    sam3_root = os.path.join(os.path.dirname(sam3_module.__file__), '..')
    bpe_path = os.path.join(sam3_root, 'assets', 'bpe_simple_vocab_16e6.txt.gz')
    sam3_model = build_sam3_image_model(bpe_path=bpe_path, device=device,
                                         checkpoint_path=SAM3_CKPT, load_from_HF=False)
    sam3_proc = Sam3Processor(sam3_model, confidence_threshold=SAM3_THRESHOLD)

    # Load MASt3R (AerialMegaDepth)
    print(f"Loading MASt3R ({MAST3R_CHECKPOINT})...", flush=True)
    from mast3r.model import AsymmetricMASt3R
    mast3r_model = AsymmetricMASt3R.from_pretrained(MAST3R_CHECKPOINT).to(device).eval()
    gpu_memory_report()

    # Process each oblique image
    all_points = []
    n_processed = 0
    n_failed = 0
    n_detections = 0

    for idx, (urn, meta) in enumerate(obliques.items()):
        img_path = meta['local_path']
        direction = meta['direction']

        # Get footprint center for ortho crop
        fc_lat, fc_lon = footprint_center(meta)

        # Check if this image overlaps the focus area (rough check)
        if not (area[0] - 0.003 <= fc_lat <= area[2] + 0.003 and
                area[1] - 0.004 <= fc_lon <= area[3] + 0.004):
            continue

        # SAM3 detection on full oblique image
        oblique_img = Image.open(img_path).convert('RGB')
        state = sam3_proc.set_image(oblique_img)
        state = sam3_proc.set_text_prompt(state=state, prompt=SAM3_PROMPT)

        if len(state['boxes']) == 0:
            n_processed += 1
            continue

        # MASt3R ONCE on full oblique + 80m ortho crop (validated: 2.6m accuracy)
        ortho_crop, ortho_meta = stitch_ortho_for_footprint(fc_lat, fc_lon, radius_m=80, zoom=21)
        if ortho_meta['tiles'] < 4:
            n_processed += 1
            continue

        # MASt3R oblique↔ortho
        result, err = match_oblique_ortho(img_path, ortho_crop, mast3r_model, device)
        if result is None:
            n_failed += 1
            n_processed += 1
            continue

        pts3d, poses, focals = result
        w, h = oblique_img.size
        oblique_size = (h, w)
        ortho_size = (ortho_crop.size[1], ortho_crop.size[0])

        # Project each detection's base pixel to ortho
        for i in range(len(state['boxes'])):
            box = state['boxes'][i].tolist()
            score = state['scores'][i].item()
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            base_x = (x1 + x2) // 2
            base_y = y2  # pole base

            ortho_pt = project_to_ortho(base_x, base_y, oblique_size, ortho_size,
                                        pts3d, poses, focals)
            if ortho_pt is None:
                continue

            ox, oy = ortho_pt
            pt_lat, pt_lon = ortho_pixel_to_gps(ox, oy, ortho_meta)

            all_points.append({
                'lat': round(pt_lat, 6),
                'lon': round(pt_lon, 6),
                'score': round(score, 3),
                'direction': direction,
                'urn': urn,
                'oblique_bbox': [x1, y1, x2, y2],
                'ortho_pixel_local': [ox, oy],
            })
            n_detections += 1

        n_processed += 1
        if (n_processed) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  [{n_processed}/{len(obliques)}] {n_detections} dets, {n_failed} failed | {elapsed:.0f}s",
                  flush=True)

    print(f"\nProcessed {n_processed} images, {n_detections} detections, {n_failed} failed")

    # Dedup
    m_per_deg_lon = 111320 * math.cos(math.radians(FOCUS_LAT))
    used = [False] * len(all_points)
    deduped = []
    for i, p in enumerate(all_points):
        if used[i]: continue
        cluster = [p]; used[i] = True
        for j in range(i + 1, len(all_points)):
            if used[j]: continue
            dist = math.sqrt(((p['lat'] - all_points[j]['lat']) * 111320) ** 2 +
                           ((p['lon'] - all_points[j]['lon']) * m_per_deg_lon) ** 2)
            if dist < 10:
                cluster.append(all_points[j]); used[j] = True
        best = max(cluster, key=lambda x: x['score'])
        best['lat'] = round(sum(c['lat'] for c in cluster) / len(cluster), 6)
        best['lon'] = round(sum(c['lon'] for c in cluster) / len(cluster), 6)
        best['num_views'] = len(set(c['direction'] for c in cluster))
        best['cluster_size'] = len(cluster)
        deduped.append(best)

    # Filter to focus area
    in_area = [p for p in deduped
               if area[0] <= p['lat'] <= area[2] and area[1] <= p['lon'] <= area[3]]

    print(f"Raw: {len(all_points)} → Dedup: {len(deduped)} → In area: {len(in_area)}")

    # Draw on full ortho reference
    result_img = ortho_full.copy()
    draw = ImageDraw.Draw(result_img)
    for p in in_area:
        ox, oy = gps_to_ortho_pixel(p['lat'], p['lon'], ortho_full_meta)
        draw.ellipse([(ox - 6, oy - 6), (ox + 6, oy + 6)], outline='red', width=2)
        draw.ellipse([(ox - 2, oy - 2), (ox + 2, oy + 2)], fill='red')
    for l in gt_labels:
        if l['label'] == 'pole':
            gx, gy = gps_to_ortho_pixel(l['lat'], l['lon'], ortho_full_meta)
            draw.ellipse([(gx - 5, gy - 5), (gx + 5, gy + 5)], outline='lime', width=2)
    result_path = os.path.join(RESULTS_DIR, 'ortho_with_poles.png')
    result_img.save(result_path)
    print(f"Result image: {result_path}")

    # Evaluate at multiple radii
    print(f"\n{'='*60}")
    print("OBLIQUE → ORTHO RESULTS")
    print(f"{'='*60}")
    for radius in [3, 5, 7, 10, 15]:
        r = evaluate(in_area, gt_labels, match_radius_m=radius)
        print(f"  {radius}m: TP={r['tp']} FP={r['fp']} FN={r['fn']} | "
              f"P={r['precision']:.1%} R={r['recall']:.1%} F1={r['f1']:.3f} RMSE={r['rmse']:.1f}m")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.0f}s")

    # Save
    save_data = {
        'poles': in_area,
        'evaluation': {str(r): evaluate(in_area, gt_labels, match_radius_m=r) for r in [3, 5, 7, 10, 15]},
        'stats': {
            'images_processed': n_processed,
            'detections_raw': len(all_points),
            'detections_deduped': len(deduped),
            'detections_in_area': len(in_area),
            'mast3r_failures': n_failed,
        },
        'config': {
            'sam3_prompt': SAM3_PROMPT,
            'sam3_threshold': SAM3_THRESHOLD,
            'mast3r_checkpoint': MAST3R_CHECKPOINT,
        },
        'total_time': round(total_time, 1),
    }
    out_path = os.path.join(RESULTS_DIR, 'oblique_to_ortho_results.json')
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Results: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run_pipeline(args)
