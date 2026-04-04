#!/usr/bin/env python3
"""
Run multi-view consensus pipeline on multiple test locations.
Downloads target-centered crops, runs GDino + MASt3R + height check.
"""
import sys, os, time, json, math
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'mast3r'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'mast3r', 'dust3r'))
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
from PIL import Image, ImageDraw
import torch
import requests
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from auth import auth_headers
from ratelimit import images_limiter

BASE_URL = os.environ.get('EAGLEVIEW_BASE_URL', 'https://sandbox.apis.eagleview.com')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(DATA_DIR, 'debug', 'batch')
DIRECTIONS = ['north', 'east', 'south', 'west']
GDINO_TEXT = "utility pole. power pole. telephone pole."
GDINO_THRESHOLD = 0.15
MIN_POLE_HEIGHT = 4.0
MAX_POLE_HEIGHT = 25.0
CONSENSUS_DIST_THRESH = 15.0

# Test locations (all have 4-direction coverage)
TEST_LOCATIONS = [
    # Original 10
    (41.248644, -95.998878, "original_test"),
    (41.246000, -95.987000, "loc_1"),
    (41.251000, -95.993000, "loc_2"),
    (41.251000, -95.990000, "loc_3"),
    (41.251000, -96.001000, "loc_4"),
    (41.251000, -95.984000, "loc_5"),
    (41.246000, -96.004000, "loc_6"),
    (41.251000, -95.994000, "loc_7"),
    (41.243000, -96.004000, "loc_8"),
    (41.245000, -95.982000, "loc_9"),
    # Batch 2 — 20 more spread across the area
    (41.241404, -96.000548, "loc_10"),
    (41.241404, -95.993379, "loc_11"),
    (41.241404, -95.98621, "loc_12"),
    (41.241404, -95.979041, "loc_13"),
    (41.243201, -95.998158, "loc_14"),
    (41.243201, -95.990989, "loc_15"),
    (41.243201, -95.983821, "loc_16"),
    (41.244998, -96.002937, "loc_17"),
    (41.244998, -95.995769, "loc_18"),
    (41.244998, -95.9886, "loc_19"),
    (41.244998, -95.979041, "loc_20"),
    (41.246795, -96.000548, "loc_21"),
    (41.246795, -95.993379, "loc_22"),
    (41.246795, -95.983821, "loc_23"),
    (41.248592, -96.002937, "loc_24"),
    (41.248592, -95.990989, "loc_25"),
    (41.248592, -95.983821, "loc_26"),
    (41.250389, -96.000548, "loc_27"),
    (41.250389, -95.98621, "loc_28"),
    (41.250389, -95.979041, "loc_29"),
]


def find_best_obliques(meta, lat, lon):
    candidates = {}
    for urn, m in meta.items():
        if m['type'] != 'oblique' or not m.get('ground_footprint'):
            continue
        d = m['direction']
        gj = json.loads(m['ground_footprint']['geojson']['value'])
        feat = gj['features'][0] if gj['type'] == 'FeatureCollection' else gj
        geom = feat.get('geometry', feat)
        coords = geom['coordinates'][0] if geom['type'] == 'MultiPolygon' else geom['coordinates']
        ring = coords[0] if isinstance(coords[0][0], list) else coords
        inside = False
        for i in range(len(ring)):
            j = (i - 1) % len(ring)
            xi, yi = ring[i]; xj, yj = ring[j]
            if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
                inside = not inside
        if not inside:
            continue
        gsd = m.get('calculated_gsd', {}).get('value', 999)
        if d not in candidates or gsd < candidates[d]['gsd']:
            candidates[d] = {'urn': urn, 'gsd': gsd, 'meta': m}
    return candidates


def nms_detections(detections, iou_thresh=0.3):
    """Non-Maximum Suppression: merge overlapping detections, keep highest confidence."""
    if not detections:
        return []
    # Sort by confidence descending
    dets = sorted(detections, key=lambda d: d['conf'], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        remaining = []
        bx1, by1, bx2, by2 = best['bbox']
        for d in dets:
            dx1, dy1, dx2, dy2 = d['bbox']
            # Compute IoU
            ix1, iy1 = max(bx1, dx1), max(by1, dy1)
            ix2, iy2 = min(bx2, dx2), min(by2, dy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_b = (bx2 - bx1) * (by2 - by1)
            area_d = (dx2 - dx1) * (dy2 - dy1)
            union = area_b + area_d - inter
            iou = inter / union if union > 0 else 0
            if iou < iou_thresh:
                remaining.append(d)
        dets = remaining
    return keep


def download_crops(candidates, lat, lon, loc_dir):
    os.makedirs(loc_dir, exist_ok=True)
    paths = {}
    for d in DIRECTIONS:
        if d not in candidates:
            continue
        out_path = os.path.join(loc_dir, f'{d}.png')
        if os.path.exists(out_path):
            paths[d] = out_path
            continue
        c = candidates[d]
        urn = c['urn']
        max_zoom = c['meta'].get('zoom_range', {}).get('maximum_zoom_level')
        params = {
            'center.x': lon, 'center.y': lat,
            'center.radius': 50, 'epsg': 'EPSG:4326',
            'format': 'IMAGE_FORMAT_PNG',
        }
        if max_zoom:
            params['zoom'] = max_zoom
        url = f'{BASE_URL}/imagery/v3/images/{urn}/location'
        for radius in [50, 35, 25]:
            params['center.radius'] = radius
            images_limiter.wait()
            resp = requests.get(url, params=params, headers=auth_headers())
            if resp.status_code != 413:
                break
        if resp.status_code == 200:
            with open(out_path, 'wb') as f:
                f.write(resp.content)
            paths[d] = out_path
    return paths


def run_pipeline(images, image_paths, orig_sizes, view_metadata, gdino_proc, gdino_model, mast3r_model, device):
    """Run full pipeline: GDino → MASt3R → Consensus → Height check."""
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    # GDino
    gdino_dets = {}
    for d, img in images.items():
        w, h = img.size
        inputs = gdino_proc(images=img, text=GDINO_TEXT, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = gdino_model(**inputs)
        out = gdino_proc.post_process_grounded_object_detection(
            outputs, inputs['input_ids'],
            target_sizes=torch.tensor([[h, w]]).to(device),
            threshold=GDINO_THRESHOLD, text_threshold=GDINO_THRESHOLD
        )[0]
        dets = []
        for box, score, label in zip(out['boxes'], out['scores'], out['text_labels']):
            x1, y1, x2, y2 = box.int().tolist()
            dets.append({'bbox': [x1, y1, x2, y2], 'center': [(x1+x2)//2, (y1+y2)//2], 'conf': round(score.item(), 3), 'label': label})
        gdino_dets[d] = nms_detections(dets, iou_thresh=0.3)

    # MASt3R
    path_list = [image_paths[d] for d in DIRECTIONS if d in image_paths]
    dir_list = [d for d in DIRECTIONS if d in image_paths]
    imgs = load_images(path_list, size=512)
    pairs_list = make_pairs(imgs, scene_graph='complete', symmetrize=True)
    output = inference(pairs_list, mast3r_model, device, batch_size=1)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer)
    scene.compute_global_alignment(init='mst', niter=300, schedule='cosine', lr=0.01)
    pts3d = scene.get_pts3d()
    poses = scene.get_im_poses()
    focals = scene.get_focals()

    # Consensus
    all_results = []
    for i, d in enumerate(dir_list):
        dets = gdino_dets.get(d, [])
        if not dets:
            continue
        h_orig, w_orig = orig_sizes[d]
        pts3d_view = pts3d[i]
        h_m, w_m = pts3d_view.shape[0], pts3d_view.shape[1]
        sx, sy = w_m / w_orig, h_m / h_orig

        for det in dets:
            cx, cy = det['center']
            mx = min(max(int(cx * sx), 0), w_m - 1)
            my = min(max(int(cy * sy), 0), h_m - 1)
            point_3d = pts3d_view[my, mx]
            if torch.isnan(point_3d).any():
                continue

            agreeing_views = [d]
            for j, d2 in enumerate(dir_list):
                if j == i:
                    continue
                pose_inv = torch.inverse(poses[j])
                p_cam = pose_inv[:3, :3] @ point_3d + pose_inv[:3, 3]
                if p_cam[2] <= 0:
                    continue
                target_h, target_w = pts3d[j].shape[0], pts3d[j].shape[1]
                u = focals[j] * p_cam[0] / p_cam[2] + target_w / 2
                v = focals[j] * p_cam[1] / p_cam[2] + target_h / 2
                u, v = u.item(), v.item()
                if not (0 <= u < target_w and 0 <= v < target_h):
                    continue
                u_orig = u / (target_w / orig_sizes[d2][1])
                v_orig = v / (target_h / orig_sizes[d2][0])
                for det2 in gdino_dets.get(d2, []):
                    cx2, cy2 = det2['center']
                    dist = math.sqrt((cx2 - u_orig)**2 + (cy2 - v_orig)**2)
                    bbox_size = max(det2['bbox'][2] - det2['bbox'][0], det2['bbox'][3] - det2['bbox'][1])
                    if dist < max(CONSENSUS_DIST_THRESH, bbox_size * 0.5):
                        agreeing_views.append(d2)
                        break

            all_results.append({
                'source_view': d, 'bbox': det['bbox'], 'center': det['center'],
                'conf': det['conf'], 'label': det['label'],
                'agreeing_views': agreeing_views, 'num_views': len(agreeing_views),
                'point_3d': point_3d.detach().cpu().tolist(),
            })

    confirmed_raw = [r for r in all_results if r['num_views'] >= 2]

    # Height check
    confirmed = []
    rejected = []
    for r in confirmed_raw:
        x1, y1, x2, y2 = r['bbox']
        source = r['source_view']
        if source not in view_metadata:
            continue
        vm = view_metadata[source]
        h_px = y2 - y1
        sin_e = math.sin(math.radians(vm['elevation']))
        est_height = h_px * vm['gsd'] / sin_e if sin_e > 0.1 else 0
        r['est_height'] = round(est_height, 1)
        if MIN_POLE_HEIGHT <= est_height <= MAX_POLE_HEIGHT:
            confirmed.append(r)
        else:
            rejected.append(r)

    single_view = [r for r in all_results if r['num_views'] == 1]

    # Cluster confirmed poles by 3D proximity — merge detections of the same physical pole
    clustered = []
    used = [False] * len(confirmed)
    CLUSTER_DIST = 0.05  # MASt3R units

    for i, r in enumerate(confirmed):
        if used[i]:
            continue
        cluster = [r]
        used[i] = True
        p1 = r['point_3d']

        for j in range(i + 1, len(confirmed)):
            if used[j]:
                continue
            p2 = confirmed[j]['point_3d']
            dist = math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))
            if dist < CLUSTER_DIST:
                cluster.append(confirmed[j])
                used[j] = True

        # Merge cluster: keep highest confidence detection, aggregate views
        best = max(cluster, key=lambda r: r['conf'])
        all_views = set()
        for c in cluster:
            all_views.update(c['agreeing_views'])
        avg_height = sum(c['est_height'] for c in cluster) / len(cluster)

        merged = dict(best)
        merged['agreeing_views'] = sorted(all_views)
        merged['num_views'] = len(all_views)
        merged['est_height'] = round(avg_height, 1)
        merged['cluster_size'] = len(cluster)
        clustered.append(merged)

    return {
        'gdino_dets': gdino_dets,
        'all_detections': len(all_results),
        'multi_view_raw': len(confirmed_raw),
        'confirmed_before_cluster': len(confirmed),
        'confirmed': clustered,
        'rejected_height': rejected,
        'single_view': len(single_view),
    }


def draw_results(images, result, loc_dir):
    for d in DIRECTIONS:
        if d not in images:
            continue
        img = images[d].copy()
        draw = ImageDraw.Draw(img)
        for det in result['gdino_dets'].get(d, []):
            x1, y1, x2, y2 = det['bbox']
            draw.rectangle([x1, y1, x2, y2], outline='gray', width=1)
        for r in result.get('rejected_height', []):
            if r['source_view'] == d:
                x1, y1, x2, y2 = r['bbox']
                draw.rectangle([x1, y1, x2, y2], outline='orange', width=3)
                draw.text((x1, max(0, y1-16)), f"BAD {r.get('est_height','?')}m", fill='orange')
        for r in result['confirmed']:
            if r['source_view'] == d:
                x1, y1, x2, y2 = r['bbox']
                draw.rectangle([x1, y1, x2, y2], outline='lime', width=4)
                draw.text((x1, max(0, y1-16)), f"POLE {r.get('est_height','?')}m ({r['num_views']}v)", fill='lime')
        img.save(os.path.join(loc_dir, f'{d}_result.jpg'), quality=90)


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    with open(os.path.join(DATA_DIR, 'metadata.json')) as f:
        meta = json.load(f)

    # Load models once
    print("Loading GDino...")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    gdino_proc = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny')
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny').to(device)

    print("Loading MASt3R...")
    from mast3r.model import AsymmetricMASt3R
    mast3r_model = AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric').to(device).eval()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_loc_results = []

    for idx, (lat, lon, name) in enumerate(TEST_LOCATIONS):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(TEST_LOCATIONS)}] {name}: ({lat}, {lon})")
        print('='*60)

        loc_dir = os.path.join(RESULTS_DIR, name)

        # Find obliques
        candidates = find_best_obliques(meta, lat, lon)
        available_dirs = [d for d in DIRECTIONS if d in candidates]
        if len(available_dirs) < 2:
            print(f"  Skipping — only {len(available_dirs)} directions available")
            continue

        # Build view metadata
        view_metadata = {}
        for d in available_dirs:
            m = candidates[d]['meta']
            view_metadata[d] = {
                'gsd': m['calculated_gsd']['value'],
                'elevation': m['look_at']['elevation'],
                'azimuth': m['look_at']['azimuth'],
            }

        # Download crops
        print(f"  Downloading {len(available_dirs)} crops...")
        paths = download_crops(candidates, lat, lon, loc_dir)
        if len(paths) < 2:
            print(f"  Skipping — only {len(paths)} images downloaded")
            continue

        # Load images
        images = {}
        orig_sizes = {}
        for d in DIRECTIONS:
            if d in paths:
                img = Image.open(paths[d]).convert('RGB')
                images[d] = img
                orig_sizes[d] = (img.size[1], img.size[0])

        # Run pipeline
        t0 = time.time()
        result = run_pipeline(images, paths, orig_sizes, view_metadata, gdino_proc, gdino_model, mast3r_model, device)
        elapsed = time.time() - t0

        print(f"  GDino: {sum(len(v) for v in result['gdino_dets'].values())} detections (after NMS)")
        print(f"  Multi-view: {result['multi_view_raw']} confirmed raw")
        print(f"  Height filter: {result.get('confirmed_before_cluster', '?')} passed → {len(result['confirmed'])} unique poles (clustered)")
        print(f"  Rejected: {len(result['rejected_height'])}")
        print(f"  Time: {elapsed:.1f}s")

        for r in result['confirmed']:
            print(f"    POLE [{r['source_view']}] {r.get('est_height','?')}m ({r['num_views']}v)")

        # Draw and save
        draw_results(images, result, loc_dir)

        loc_summary = {
            'name': name, 'lat': lat, 'lon': lon,
            'directions': available_dirs,
            'total_gdino': sum(len(v) for v in result['gdino_dets'].values()),
            'multi_view_raw': result['multi_view_raw'],
            'confirmed_poles': len(result['confirmed']),
            'rejected_height': len(result['rejected_height']),
            'single_view': result['single_view'],
            'time': round(elapsed, 1),
            'confirmed': result['confirmed'],
        }
        all_loc_results.append(loc_summary)

    # Save summary
    with open(os.path.join(RESULTS_DIR, 'batch_results.json'), 'w') as f:
        json.dump(all_loc_results, f, indent=2)

    print(f"\n{'='*60}")
    print("BATCH SUMMARY")
    print(f"{'='*60}")
    print(f"{'Location':<20} {'GDino':<8} {'Multi':<8} {'Poles':<8} {'Rej':<8} {'Time':<8}")
    for r in all_loc_results:
        print(f"{r['name']:<20} {r['total_gdino']:<8} {r['multi_view_raw']:<8} {r['confirmed_poles']:<8} {r['rejected_height']:<8} {r['time']:<8}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
