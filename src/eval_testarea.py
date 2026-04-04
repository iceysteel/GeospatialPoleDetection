#!/usr/bin/env python3
"""
Evaluate multi-view consensus pipeline over the test area grid.
Uses pre-downloaded gridded oblique crops (data/testarea_grid/).
"""
import sys, os, time, json, math, argparse
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'mast3r'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'mast3r', 'dust3r'))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
from PIL import Image
import torch
from oblique_utils import pixel_to_gps

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
GRID_DIR = os.path.join(DATA_DIR, 'testarea_grid')
RESULTS_DIR = os.path.join(DATA_DIR, 'eval_testarea')
DIRECTIONS = ['north', 'east', 'south', 'west']
GDINO_TEXT = "utility pole. power pole. telephone pole."
GDINO_THRESHOLD = 0.15
MIN_POLE_HEIGHT = 4.0
MAX_POLE_HEIGHT = 50.0

FOCUS_LAT, FOCUS_LON = 41.248644, -95.998878
RADIUS_LAT, RADIUS_LON = 0.0018, 0.0024


def nms(dets, iou_thresh=0.3):
    if not dets: return []
    dets = sorted(dets, key=lambda d: d['conf'], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        remaining = []
        bx1, by1, bx2, by2 = best['bbox']
        for d in dets:
            dx1, dy1, dx2, dy2 = d['bbox']
            ix1, iy1 = max(bx1, dx1), max(by1, dy1)
            ix2, iy2 = min(bx2, dx2), min(by2, dy2)
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            union = (bx2-bx1)*(by2-by1) + (dx2-dx1)*(dy2-dy1) - inter
            if union > 0 and inter/union < iou_thresh:
                remaining.append(d)
            elif union == 0:
                remaining.append(d)
        dets = remaining
    return keep


def run_grid_cell(cell, gdino_proc, gdino_model, mast3r_model, device):
    """Run pipeline on one grid cell using its pre-downloaded images."""
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    lat, lon = cell['lat'], cell['lon']
    images = {}
    orig_sizes = {}
    path_list = []
    dir_list = []

    for d in DIRECTIONS:
        p = cell['images'].get(d)
        if not p or not os.path.exists(p):
            continue
        img = Image.open(p).convert('RGB')
        images[d] = img
        orig_sizes[d] = (img.size[1], img.size[0])
        path_list.append(p)
        dir_list.append(d)

    if len(dir_list) < 2:
        return []

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
            dets.append({'bbox': [x1, y1, x2, y2], 'center': [(x1+x2)//2, (y1+y2)//2],
                        'conf': round(score.item(), 3), 'label': label})
        gdino_dets[d] = nms(dets)

    # MASt3R
    imgs = load_images(path_list, size=512)
    pairs_list = make_pairs(imgs, scene_graph='complete', symmetrize=True)
    output = inference(pairs_list, mast3r_model, device, batch_size=1)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer)
    scene.compute_global_alignment(init='mst', niter=300, schedule='cosine', lr=0.01)
    pts3d = scene.get_pts3d()
    poses = scene.get_im_poses()
    focals = scene.get_focals()

    # Consensus + height + georeference
    all_results = []
    for i, d in enumerate(dir_list):
        meta_key = f'{d}_meta'
        dm = cell.get(meta_key, {})
        azimuth = dm.get('azimuth', 0)
        gsd = dm.get('gsd', 0.04)
        elevation = dm.get('elevation', 45)
        crop_radius = dm.get('crop_radius', 50)

        for det in gdino_dets.get(d, []):
            cx, cy = det['center']
            pv = pts3d[i]
            hm, wm = pv.shape[:2]
            sx, sy = wm / orig_sizes[d][1], hm / orig_sizes[d][0]
            mx = min(max(int(cx * sx), 0), wm - 1)
            my = min(max(int(cy * sy), 0), hm - 1)
            p3d = pv[my, mx]
            if torch.isnan(p3d).any():
                continue

            views = [d]
            for j, d2 in enumerate(dir_list):
                if j == i: continue
                pi = torch.inverse(poses[j])
                pc = pi[:3, :3] @ p3d + pi[:3, 3]
                if pc[2] <= 0: continue
                th, tw = pts3d[j].shape[:2]
                u = focals[j] * pc[0] / pc[2] + tw / 2
                v = focals[j] * pc[1] / pc[2] + th / 2
                if not (0 <= u.item() < tw and 0 <= v.item() < th): continue
                uo = u.item() / (tw / orig_sizes[d2][1])
                vo = v.item() / (th / orig_sizes[d2][0])
                for det2 in gdino_dets.get(d2, []):
                    c2x, c2y = det2['center']
                    dist = math.sqrt((c2x - uo)**2 + (c2y - vo)**2)
                    bs = max(det2['bbox'][2]-det2['bbox'][0], det2['bbox'][3]-det2['bbox'][1])
                    if dist < max(15, bs * 0.5):
                        views.append(d2)
                        break

            if len(views) >= 2:
                sine = math.sin(math.radians(elevation))
                hpx = det['bbox'][3] - det['bbox'][1]
                eh = hpx * gsd / sine if sine > 0.1 else 0

                if MIN_POLE_HEIGHT <= eh <= MAX_POLE_HEIGHT:
                    img_w, img_h = orig_sizes[d][1], orig_sizes[d][0]
                    det_lat, det_lon = pixel_to_gps(
                        cx, cy, img_w, img_h, lat, lon, azimuth, crop_radius
                    )
                    all_results.append({
                        'source_view': d, 'bbox': det['bbox'], 'conf': det['conf'],
                        'num_views': len(views), 'agreeing_views': views,
                        'est_height': round(eh, 1),
                        'approx_lat': float(det_lat), 'approx_lon': float(det_lon),
                        'point_3d': p3d.detach().cpu().tolist(),
                        'cell': cell['name'],
                    })

    # 3D cluster
    used = [False] * len(all_results)
    clustered = []
    for i, r in enumerate(all_results):
        if used[i]: continue
        cluster = [r]; used[i] = True
        for j in range(i+1, len(all_results)):
            if used[j]: continue
            dist = math.sqrt(sum((a-b)**2 for a, b in zip(r['point_3d'], all_results[j]['point_3d'])))
            if dist < 0.05:
                cluster.append(all_results[j]); used[j] = True
        best = max(cluster, key=lambda x: x['conf'])
        avs = set()
        for c in cluster: avs.update(c['agreeing_views'])
        best['agreeing_views'] = sorted(avs)
        best['num_views'] = len(avs)
        best['est_height'] = round(sum(c['est_height'] for c in cluster) / len(cluster), 1)
        best['approx_lat'] = float(round(sum(c['approx_lat'] for c in cluster) / len(cluster), 6))
        best['approx_lon'] = float(round(sum(c['approx_lon'] for c in cluster) / len(cluster), 6))
        best['cluster_size'] = len(cluster)
        clustered.append(best)

    return clustered


def evaluate(detections, gt_labels, match_radius_m=10):
    gt_poles = [l for l in gt_labels if l['label'] == 'pole']
    gt_lights = [l for l in gt_labels if l['label'] == 'streetlight']
    m_per_deg_lon = 111320 * math.cos(math.radians(41.249))
    gt_matched = set()
    tp = fp = fp_light = 0
    for det in detections:
        best_dist, best_idx, best_type = float('inf'), None, None
        for i, gt in enumerate(gt_poles):
            d = math.sqrt(((det['approx_lat']-gt['lat'])*111320)**2 + ((det['approx_lon']-gt['lon'])*m_per_deg_lon)**2)
            if d < best_dist: best_dist, best_idx, best_type = d, i, 'pole'
        for i, gt in enumerate(gt_lights):
            d = math.sqrt(((det['approx_lat']-gt['lat'])*111320)**2 + ((det['approx_lon']-gt['lon'])*m_per_deg_lon)**2)
            if d < best_dist: best_dist, best_idx, best_type = d, i, 'streetlight'
        if best_dist <= match_radius_m and best_type == 'pole' and best_idx not in gt_matched:
            tp += 1; gt_matched.add(best_idx)
        elif best_dist <= match_radius_m and best_type == 'streetlight':
            fp += 1; fp_light += 1
        else:
            fp += 1
    fn = len(gt_poles) - tp
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f1 = 2*p*r/(p+r) if p+r else 0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'fp_streetlight': fp_light,
            'precision': round(p, 3), 'recall': round(r, 3), 'f1': round(f1, 3),
            'gt_poles': len(gt_poles), 'gt_streetlights': len(gt_lights), 'detections': len(detections)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--gdino-model', default='IDEA-Research/grounding-dino-base', help='GDino model ID')
    args = parser.parse_args()
    device = args.device
    t_start = time.time()
    print(f"Device: {device}")

    # Load ground truth
    with open(os.path.join(DATA_DIR, 'ground_truth_testarea.json')) as f:
        gt = json.load(f)
    area = (FOCUS_LAT - RADIUS_LAT, FOCUS_LON - RADIUS_LON,
            FOCUS_LAT + RADIUS_LAT, FOCUS_LON + RADIUS_LON)
    gt_labels = [l for l in gt['labels']
                 if area[0] <= l['lat'] <= area[2] and area[1] <= l['lon'] <= area[3]]
    n_poles = sum(1 for l in gt_labels if l['label'] == 'pole')
    n_lights = sum(1 for l in gt_labels if l['label'] == 'streetlight')
    print(f"GT: {n_poles} poles, {n_lights} streetlights (in focus area)")

    # Load grid index
    with open(os.path.join(GRID_DIR, 'index.json')) as f:
        grid = json.load(f)
    print(f"Grid: {len(grid)} cells")

    # Load models
    print(f"Loading GDino ({args.gdino_model})...", flush=True)
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    gdino_proc = AutoProcessor.from_pretrained(args.gdino_model)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino_model).to(device)

    print("Loading MASt3R...", flush=True)
    from mast3r.model import AsymmetricMASt3R
    mast3r_model = AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric').to(device).eval()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Process each grid cell
    all_detections = []
    for i, cell in enumerate(grid):
        t0 = time.time()
        dets = run_grid_cell(cell, gdino_proc, gdino_model, mast3r_model, device)
        elapsed = time.time() - t0
        all_detections.extend(dets)
        remaining = (len(grid) - i - 1) * elapsed
        print(f"  [{i+1}/{len(grid)}] {cell['name']}: {len(dets)} dets in {elapsed:.1f}s | ETA: {remaining:.0f}s", flush=True)

    # Dedup across cells
    print(f"\nDeduplicating: {len(all_detections)} raw...", flush=True)
    m_per_deg_lon = 111320 * math.cos(math.radians(41.249))
    deduped = []
    used = [False] * len(all_detections)
    for i, d in enumerate(all_detections):
        if used[i]: continue
        cluster = [d]; used[i] = True
        for j in range(i+1, len(all_detections)):
            if used[j]: continue
            dist = math.sqrt(((d['approx_lat']-all_detections[j]['approx_lat'])*111320)**2 +
                           ((d['approx_lon']-all_detections[j]['approx_lon'])*m_per_deg_lon)**2)
            if dist < 15:
                cluster.append(all_detections[j]); used[j] = True
        best = max(cluster, key=lambda x: x['conf'])
        best['approx_lat'] = float(round(sum(c['approx_lat'] for c in cluster)/len(cluster), 6))
        best['approx_lon'] = float(round(sum(c['approx_lon'] for c in cluster)/len(cluster), 6))
        best['cluster_size'] = len(cluster)
        deduped.append(best)
    print(f"After dedup: {len(deduped)} unique detections")

    # Filter to focus area (only evaluate detections inside the GT coverage region)
    in_area = [d for d in deduped
               if area[0] <= d['approx_lat'] <= area[2] and area[1] <= d['approx_lon'] <= area[3]]
    print(f"In focus area: {len(in_area)} / {len(deduped)} (dropped {len(deduped)-len(in_area)} outside GT coverage)")

    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    for radius in [5, 10, 15, 20, 30]:
        result = evaluate(in_area, gt_labels, match_radius_m=radius)
        print(f"\n  {radius}m: TP={result['tp']} FP={result['fp']} (lights={result['fp_streetlight']}) FN={result['fn']} | P={result['precision']:.1%} R={result['recall']:.1%} F1={result['f1']:.3f}")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.0f}s")

    # Save
    save_data = {
        'detections': [{k: v for k, v in d.items() if k != 'point_3d'} for d in in_area],
        'evaluation': {str(r): evaluate(in_area, gt_labels, match_radius_m=r) for r in [5, 10, 15, 20, 30]},
        'gt_summary': {'poles': n_poles, 'streetlights': n_lights},
        'grid_cells': len(grid),
        'total_time': round(total_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'eval_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {RESULTS_DIR}/eval_results.json")


if __name__ == '__main__':
    main()
