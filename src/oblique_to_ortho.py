#!/usr/bin/env python3
"""
Map pole detections from oblique views → orthorectified (nadir) image using MASt3R.

This is the core pipeline requirement:
1. SAM3 detects poles in oblique views → pixel masks + bboxes
2. MASt3R matches oblique↔ortho → dense pixel correspondences
3. Detection centroids projected through MASt3R onto ortho image
4. Output: pole locations as pixel coordinates in the ortho image

The ortho image coordinates can then be converted to GPS using tile math.
"""
import sys, os, json, time, math, argparse, tempfile
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'mast3r'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'mast3r', 'dust3r'))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
import numpy as np
from PIL import Image, ImageDraw
from agent_tools import load_ortho_crop, lat_lon_to_tile, tile_to_lat_lon
from gpu_utils import get_device, gpu_memory_report

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
GRID_DIR = os.path.join(DATA_DIR, 'testarea_grid')
WMTS_DIR = os.path.join(DATA_DIR, 'wmts')
RESULTS_DIR = os.path.join(DATA_DIR, 'eval_testarea')
DIRECTIONS = ['north', 'east', 'south', 'west']

SAM3_CKPT = os.path.join(os.path.expanduser("~"),
    ".cache/huggingface/hub/models--bodhicitta--sam3/snapshots/"
    "cba430d22f6fdc3f06ad3841274ec7bb55885f2f/sam3.pt")

FOCUS_LAT, FOCUS_LON = 41.248644, -95.998878
RADIUS_LAT, RADIUS_LON = 0.0018, 0.0024


def stitch_ortho_for_area(center_lat, center_lon, radius_m=250, zoom=21):
    """Stitch WMTS tiles into a single ortho reference image for the test area."""
    n = 2 ** zoom
    tile_deg = 360 / n
    m_per_px = tile_deg * 111320 * math.cos(math.radians(center_lat)) / 256

    # Tiles covering the area
    radius_deg_lat = radius_m / 111320
    radius_deg_lon = radius_m / (111320 * math.cos(math.radians(center_lat)))

    x1, y1 = lat_lon_to_tile(center_lat + radius_deg_lat, center_lon - radius_deg_lon, zoom)
    x2, y2 = lat_lon_to_tile(center_lat - radius_deg_lat, center_lon + radius_deg_lon, zoom)

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

    # GPS bounds of the stitched image
    tl_lat, tl_lon = tile_to_lat_lon(x1, y1, zoom)
    br_lat, br_lon = tile_to_lat_lon(x2 + 1, y2 + 1, zoom)

    meta = {
        'zoom': zoom,
        'm_per_px': m_per_px,
        'tl_lat': tl_lat, 'tl_lon': tl_lon,
        'br_lat': br_lat, 'br_lon': br_lon,
        'width': w, 'height': h,
        'tiles_loaded': loaded,
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
    }
    return stitched, meta


def gps_to_ortho_pixel(lat, lon, ortho_meta):
    """Convert GPS to pixel in stitched ortho image."""
    x = (lon - ortho_meta['tl_lon']) / (ortho_meta['br_lon'] - ortho_meta['tl_lon']) * ortho_meta['width']
    y = (lat - ortho_meta['tl_lat']) / (ortho_meta['br_lat'] - ortho_meta['tl_lat']) * ortho_meta['height']
    return int(x), int(y)


def ortho_pixel_to_gps(px, py, ortho_meta):
    """Convert pixel in stitched ortho image to GPS."""
    lon = ortho_meta['tl_lon'] + px / ortho_meta['width'] * (ortho_meta['br_lon'] - ortho_meta['tl_lon'])
    lat = ortho_meta['tl_lat'] + py / ortho_meta['height'] * (ortho_meta['br_lat'] - ortho_meta['tl_lat'])
    return lat, lon


def match_oblique_to_ortho(oblique_path, ortho_path, mast3r_model, device):
    """
    Use MASt3R to find dense correspondences between an oblique image and ortho crop.
    Returns the 3D points and camera poses for both views.
    """
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    imgs = load_images([oblique_path, ortho_path], size=512)
    pairs = make_pairs(imgs, scene_graph='complete', symmetrize=True)
    output = inference(pairs, mast3r_model, device, batch_size=1)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer)
    scene.compute_global_alignment(init='mst', niter=300, schedule='cosine', lr=0.01)

    pts3d = scene.get_pts3d()
    poses = scene.get_im_poses()
    focals = scene.get_focals()

    return pts3d, poses, focals


def project_oblique_point_to_ortho(cx, cy, oblique_size, ortho_size, pts3d, poses, focals):
    """
    Project a point from the oblique image to the ortho image using MASt3R correspondences.
    oblique is view 0, ortho is view 1.
    """
    oh, ow = oblique_size
    pv = pts3d[0]  # oblique point cloud
    hm, wm = pv.shape[:2]

    # Scale detection pixel to MASt3R resolution
    sx = wm / ow
    sy = hm / oh
    mx = min(max(int(cx * sx), 0), wm - 1)
    my = min(max(int(cy * sy), 0), hm - 1)

    p3d = pv[my, mx]
    if torch.isnan(p3d).any():
        return None

    # Project 3D point into ortho view (view 1)
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
    oh2, ow2 = ortho_size
    uo = u.item() / (tw / ow2)
    vo = v.item() / (th / oh2)

    return int(uo), int(vo)


def run_pipeline(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast('cuda', dtype=torch.bfloat16).__enter__()

    device = args.device
    t_start = time.time()

    # Step 0: Stitch ortho reference image
    print("Stitching ortho reference image...", flush=True)
    ortho_img, ortho_meta = stitch_ortho_for_area(FOCUS_LAT, FOCUS_LON, radius_m=250, zoom=21)
    ortho_path = os.path.join(RESULTS_DIR, 'ortho_reference.png')
    ortho_img.save(ortho_path)
    print(f"  Ortho: {ortho_img.size}, {ortho_meta['tiles_loaded']} tiles, {ortho_meta['m_per_px']:.4f}m/px")

    # Load grid
    with open(os.path.join(GRID_DIR, 'index.json')) as f:
        grid = json.load(f)

    # Step 1: Build SAM3
    print("Loading SAM3...", flush=True)
    import sam3 as sam3_module
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_root = os.path.join(os.path.dirname(sam3_module.__file__), '..')
    bpe_path = os.path.join(sam3_root, 'assets', 'bpe_simple_vocab_16e6.txt.gz')
    sam3_model = build_sam3_image_model(bpe_path=bpe_path, device=device,
                                         checkpoint_path=SAM3_CKPT, load_from_HF=False)
    sam3_proc = Sam3Processor(sam3_model, confidence_threshold=0.10)

    # Step 2: Load MASt3R
    print("Loading MASt3R...", flush=True)
    from mast3r.model import AsymmetricMASt3R
    mast3r_model = AsymmetricMASt3R.from_pretrained(
        'kvuong2711/checkpoint-aerial-mast3r').to(device).eval()
    gpu_memory_report()

    # Step 3: Process each grid cell
    all_ortho_points = []

    for ci, cell in enumerate(grid):
        t0 = time.time()
        lat, lon = cell['lat'], cell['lon']

        # Get ortho crop for this cell area
        ortho_crop, crop_meta = load_ortho_crop(lat, lon, radius_m=60, preferred_zoom=21)
        if ortho_crop is None:
            print(f"  [{ci+1}/{len(grid)}] {cell['name']}: no ortho crop available")
            continue

        # Save ortho crop temporarily for MASt3R
        ortho_tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        ortho_crop.save(ortho_tmp.name)
        ortho_crop_size = (ortho_crop.size[1], ortho_crop.size[0])  # (h, w)

        cell_points = []

        for d in DIRECTIONS:
            img_path = cell['images'].get(d)
            if not img_path or not os.path.exists(img_path):
                continue

            oblique_img = Image.open(img_path).convert('RGB')
            oblique_size = (oblique_img.size[1], oblique_img.size[0])  # (h, w)

            # SAM3 detection on oblique
            state = sam3_proc.set_image(oblique_img)
            state = sam3_proc.set_text_prompt(state=state, prompt='telephone pole')

            if len(state['boxes']) == 0:
                continue

            # MASt3R matching: oblique ↔ ortho crop
            try:
                pts3d, poses, focals = match_oblique_to_ortho(
                    img_path, ortho_tmp.name, mast3r_model, device
                )
            except Exception as e:
                continue

            # Project each detection to ortho
            for i in range(len(state['boxes'])):
                box = state['boxes'][i].tolist()
                score = state['scores'][i].item()
                w, h = oblique_img.size
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                ortho_pt = project_oblique_point_to_ortho(
                    cx, cy, oblique_size, ortho_crop_size, pts3d, poses, focals
                )

                if ortho_pt is not None:
                    ox, oy = ortho_pt
                    # Convert ortho crop pixel to GPS
                    # The ortho crop is centered on (lat, lon) with known m_per_px
                    crop_w, crop_h = ortho_crop.size
                    offset_x_m = (ox - crop_w / 2) * crop_meta['m_per_px']
                    offset_y_m = (oy - crop_h / 2) * crop_meta['m_per_px']
                    m_per_deg_lat = 111320
                    m_per_deg_lon = 111320 * math.cos(math.radians(lat))
                    pt_lat = lat - offset_y_m / m_per_deg_lat  # y increases downward
                    pt_lon = lon + offset_x_m / m_per_deg_lon

                    # Also compute position in the full stitched ortho
                    full_ox, full_oy = gps_to_ortho_pixel(pt_lat, pt_lon, ortho_meta)

                    cell_points.append({
                        'cell': cell['name'],
                        'direction': d,
                        'score': round(score, 3),
                        'oblique_bbox': [x1, y1, x2, y2],
                        'ortho_pixel': [full_ox, full_oy],
                        'lat': round(pt_lat, 6),
                        'lon': round(pt_lon, 6),
                    })

        os.unlink(ortho_tmp.name)
        all_ortho_points.extend(cell_points)
        elapsed = time.time() - t0
        print(f"  [{ci+1}/{len(grid)}] {cell['name']}: {len(cell_points)} poles mapped to ortho in {elapsed:.1f}s",
              flush=True)

    # Step 4: Dedup points that map to same ortho location
    m_per_deg_lon = 111320 * math.cos(math.radians(FOCUS_LAT))
    used = [False] * len(all_ortho_points)
    deduped = []
    for i, p in enumerate(all_ortho_points):
        if used[i]: continue
        cluster = [p]; used[i] = True
        for j in range(i + 1, len(all_ortho_points)):
            if used[j]: continue
            dist = math.sqrt(((p['lat'] - all_ortho_points[j]['lat']) * 111320) ** 2 +
                           ((p['lon'] - all_ortho_points[j]['lon']) * m_per_deg_lon) ** 2)
            if dist < 15:
                cluster.append(all_ortho_points[j]); used[j] = True
        best = max(cluster, key=lambda x: x['score'])
        best['lat'] = round(sum(c['lat'] for c in cluster) / len(cluster), 6)
        best['lon'] = round(sum(c['lon'] for c in cluster) / len(cluster), 6)
        best['ortho_pixel'] = [
            int(sum(c['ortho_pixel'][0] for c in cluster) / len(cluster)),
            int(sum(c['ortho_pixel'][1] for c in cluster) / len(cluster)),
        ]
        best['num_views'] = len(set(c['direction'] for c in cluster))
        best['cluster_size'] = len(cluster)
        deduped.append(best)

    # Filter to focus area
    area = (FOCUS_LAT - RADIUS_LAT, FOCUS_LON - RADIUS_LON,
            FOCUS_LAT + RADIUS_LAT, FOCUS_LON + RADIUS_LON)
    in_area = [p for p in deduped if area[0] <= p['lat'] <= area[2] and area[1] <= p['lon'] <= area[3]]

    print(f"\nRaw: {len(all_ortho_points)} → Deduped: {len(deduped)} → In area: {len(in_area)}")

    # Step 5: Draw results on ortho image
    result_img = ortho_img.copy()
    draw = ImageDraw.Draw(result_img)

    for p in in_area:
        ox, oy = p['ortho_pixel']
        r = 6
        draw.ellipse([(ox - r, oy - r), (ox + r, oy + r)], outline='red', width=2)
        draw.ellipse([(ox - 2, oy - 2), (ox + 2, oy + 2)], fill='red')

    # Also draw GT poles
    with open(os.path.join(DATA_DIR, 'ground_truth_testarea.json')) as f:
        gt = json.load(f)
    for l in gt['labels']:
        if l['label'] == 'pole':
            gx, gy = gps_to_ortho_pixel(l['lat'], l['lon'], ortho_meta)
            draw.ellipse([(gx - 5, gy - 5), (gx + 5, gy + 5)], outline='lime', width=2)

    result_path = os.path.join(RESULTS_DIR, 'ortho_with_poles.png')
    result_img.save(result_path)
    print(f"Result image saved to {result_path}")

    # Step 6: Evaluate
    gt_labels = [l for l in gt['labels']
                 if area[0] <= l['lat'] <= area[2] and area[1] <= l['lon'] <= area[3]]
    gt_poles = [l for l in gt_labels if l['label'] == 'pole']

    gt_matched = set()
    tp = fp = 0
    for p in in_area:
        best_dist = float('inf')
        best_idx = -1
        for i, g in enumerate(gt_poles):
            d = math.sqrt(((p['lat'] - g['lat']) * 111320) ** 2 +
                         ((p['lon'] - g['lon']) * m_per_deg_lon) ** 2)
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_dist <= 30 and best_idx not in gt_matched:
            tp += 1; gt_matched.add(best_idx)
        else:
            fp += 1
    fn = len(gt_poles) - tp
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0

    print(f"\n{'='*60}")
    print(f"OBLIQUE → ORTHO MAPPING RESULTS")
    print(f"{'='*60}")
    print(f"Detections mapped to ortho: {len(in_area)}")
    print(f"GT poles: {len(gt_poles)}")
    print(f"TP={tp} FP={fp} FN={fn}")
    print(f"Precision: {prec:.1%} Recall: {rec:.1%} F1: {f1:.3f}")

    total_time = time.time() - t_start
    print(f"Total time: {total_time:.0f}s")

    # Save results
    save_data = {
        'poles_in_ortho': in_area,
        'ortho_meta': ortho_meta,
        'evaluation': {'tp': tp, 'fp': fp, 'fn': fn,
                       'precision': round(prec, 3), 'recall': round(rec, 3), 'f1': round(f1, 3)},
        'total_time': round(total_time, 1),
    }
    out_path = os.path.join(RESULTS_DIR, 'oblique_to_ortho_results.json')
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run_pipeline(args)
