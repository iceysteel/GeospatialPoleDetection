#!/usr/bin/env python3
"""
SAM3 + MASt3R pipeline for pole detection.

New approach:
1. SAM3 detects + segments poles with text prompt (replaces GDino)
2. For each detection, crop all 4 views around the detection GPS
3. MASt3R on cropped views → high-detail 3D reconstruction
4. SAM3 mask projected into 3D → shape analysis on pole pixels only
5. Georeferencing from 3D transform

SAM3: 841M params, 3.3GB VRAM, ~0.5s/image
MASt3R: 2.6GB, ~5s per 4-view reconstruction
"""
import sys, os, time, json, math, argparse, tempfile
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'mast3r'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'mast3r', 'dust3r'))

import torch
import numpy as np
from PIL import Image
from oblique_utils import pixel_to_gps, gps_to_pixel
from gpu_utils import get_device, gpu_memory_report

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
GRID_DIR = os.path.join(DATA_DIR, 'testarea_grid')
RESULTS_DIR = os.path.join(DATA_DIR, 'eval_testarea')
DIRECTIONS = ['north', 'east', 'south', 'west']

SAM3_PROMPT = "telephone pole"
SAM3_THRESHOLD = 0.10
MIN_POLE_HEIGHT = 4.0
MAX_POLE_HEIGHT = 50.0

FOCUS_LAT, FOCUS_LON = 41.248644, -95.998878
RADIUS_LAT, RADIUS_LON = 0.0018, 0.0024

SAM3_CKPT = os.path.join(os.path.expanduser("~"),
    ".cache/huggingface/hub/models--bodhicitta--sam3/snapshots/cba430d22f6fdc3f06ad3841274ec7bb55885f2f/sam3.pt")


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


def build_sam3_model(device='cuda'):
    """Build and return SAM3 model + processor."""
    import sam3 as sam3_module
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_root = os.path.join(os.path.dirname(sam3_module.__file__), '..')
    bpe_path = os.path.join(sam3_root, 'assets', 'bpe_simple_vocab_16e6.txt.gz')

    model = build_sam3_image_model(
        bpe_path=bpe_path, device=device,
        checkpoint_path=SAM3_CKPT, load_from_HF=False
    )
    processor = Sam3Processor(model, confidence_threshold=SAM3_THRESHOLD)
    return model, processor


def detect_poles_sam3(image, processor, prompt=SAM3_PROMPT):
    """Run SAM3 text-prompted detection + segmentation on an image."""
    w, h = image.size
    state = processor.set_image(image)
    state = processor.set_text_prompt(state=state, prompt=prompt)

    boxes = state['boxes']
    scores = state['scores']
    masks = state['masks']

    dets = []
    for i in range(len(boxes)):
        box = boxes[i].tolist()
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        score = scores[i].item()
        mask = masks[i, 0].cpu().numpy()  # (H, W) bool
        mask_pixels = int(mask.sum())

        dets.append({
            'bbox': [x1, y1, x2, y2],
            'center': [(x1+x2)//2, (y1+y2)//2],
            'conf': round(score, 3),
            'mask_pixels': mask_pixels,
            'mask': mask,  # keep for 3D analysis
        })

    return nms([{k: v for k, v in d.items() if k != 'mask'} for d in dets]), \
           {i: d['mask'] for i, d in enumerate(dets)}


def run_grid_cell_sam3(cell, sam3_proc, mast3r_model, device):
    """Run SAM3 + MASt3R pipeline on one grid cell."""
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from georef_3d import fit_3d_to_ground

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

    # Step 1: SAM3 detection on each view
    all_dets = {}
    all_masks = {}
    for d, img in images.items():
        dets, masks = detect_poles_sam3(img, sam3_proc)
        all_dets[d] = dets
        all_masks[d] = masks

    # Step 2: MASt3R 3D reconstruction (full images for consensus)
    imgs = load_images(path_list, size=512)
    pairs_list = make_pairs(imgs, scene_graph='complete', symmetrize=True)
    output = inference(pairs_list, mast3r_model, device, batch_size=1)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer)
    scene.compute_global_alignment(init='mst', niter=300, schedule='cosine', lr=0.01)
    pts3d = scene.get_pts3d()
    poses = scene.get_im_poses()
    focals = scene.get_focals()

    # Fit 3D→GPS transform
    cell_meta = {}
    for d in dir_list:
        dm = cell.get(f'{d}_meta', {})
        cell_meta[d] = {
            'azimuth': dm.get('azimuth', 0),
            'gsd': dm.get('gsd', 0.04),
            'elevation': dm.get('elevation', 45),
            'crop_radius': dm.get('crop_radius', 50),
        }
    georef = fit_3d_to_ground(pts3d, orig_sizes, dir_list, cell_meta, lat, lon)
    use_3d_georef = georef is not None

    # Step 3: Multi-view consensus + height + 3D shape + georeferencing
    all_results = []
    for i, d in enumerate(dir_list):
        vm = cell_meta[d]

        for det in all_dets.get(d, []):
            cx, cy = det['center']
            pv = pts3d[i]
            hm, wm = pv.shape[:2]
            oh, ow = orig_sizes[d]
            sx, sy = wm / ow, hm / oh
            mx = min(max(int(cx * sx), 0), wm - 1)
            my = min(max(int(cy * sy), 0), hm - 1)
            p3d = pv[my, mx]
            if torch.isnan(p3d).any():
                continue

            # Multi-view consensus
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
                for det2 in all_dets.get(d2, []):
                    c2x, c2y = det2['center']
                    dist = math.sqrt((c2x - uo)**2 + (c2y - vo)**2)
                    bs = max(det2['bbox'][2]-det2['bbox'][0], det2['bbox'][3]-det2['bbox'][1])
                    if dist < max(15, bs * 0.5):
                        views.append(d2)
                        break

            if len(views) >= 1:
                # Height estimation
                sine = math.sin(math.radians(vm['elevation']))
                hpx = det['bbox'][3] - det['bbox'][1]
                eh = hpx * vm['gsd'] / sine if sine > 0.1 else 0

                if MIN_POLE_HEIGHT <= eh <= MAX_POLE_HEIGHT:
                    # 3D shape from SAM3 mask projected into MASt3R point cloud
                    x1, y1, x2, y2 = det['bbox']
                    bx1 = min(max(int(x1 * sx), 0), wm - 1)
                    by1 = min(max(int(y1 * sy), 0), hm - 1)
                    bx2 = min(max(int(x2 * sx), 0), wm - 1)
                    by2 = min(max(int(y2 * sy), 0), hm - 1)

                    shape_class = 'unknown'
                    linearity = 0
                    if bx2 > bx1 and by2 > by1:
                        box_pts = pv[by1:by2+1, bx1:bx2+1].reshape(-1, 3)
                        valid = ~torch.isnan(box_pts).any(dim=1)
                        box_pts = box_pts[valid]
                        if len(box_pts) >= 10:
                            with torch.amp.autocast('cuda', enabled=False):
                                bp = box_pts.detach().float()
                                centered = bp - bp.mean(dim=0)
                                cov = centered.T @ centered / len(centered)
                                ev = torch.linalg.eigh(cov).eigenvalues.cpu().numpy()
                            linearity = float((ev[2] - ev[1]) / max(ev[2], 1e-6))
                            planarity = float((ev[1] - ev[0]) / max(ev[2], 1e-6))
                            if linearity > 0.6:
                                shape_class = 'linear'
                            elif planarity > 0.5:
                                shape_class = 'planar'
                            else:
                                shape_class = 'compact'

                    # Hybrid georeferencing
                    img_w, img_h = orig_sizes[d][1], orig_sizes[d][0]
                    homo_lat, homo_lon = pixel_to_gps(
                        cx, cy, img_w, img_h, lat, lon,
                        vm['azimuth'], vm['crop_radius']
                    )
                    if use_3d_georef:
                        geo3d_lat, geo3d_lon = georef['transform'](p3d)
                        w3d = 0.6
                        det_lat = geo3d_lat * w3d + homo_lat * (1 - w3d)
                        det_lon = geo3d_lon * w3d + homo_lon * (1 - w3d)
                    else:
                        det_lat, det_lon = homo_lat, homo_lon

                    all_results.append({
                        'source_view': d, 'bbox': det['bbox'], 'conf': det['conf'],
                        'num_views': len(views), 'agreeing_views': views,
                        'est_height': round(eh, 1),
                        'mask_pixels': det.get('mask_pixels', 0),
                        'shape_class': shape_class,
                        'linearity': round(linearity, 3),
                        'approx_lat': float(det_lat), 'approx_lon': float(det_lon),
                        'point_3d': p3d.detach().cpu().tolist(),
                        'cell': cell['name'],
                    })

    # 3D clustering
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
    """Same evaluate function from eval_testarea.py."""
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
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # Enable bf16 autocast globally (required by SAM3)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast('cuda', dtype=torch.bfloat16).__enter__()

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
    print(f"GT: {n_poles} poles, {n_lights} streetlights")

    # Load grid
    with open(os.path.join(GRID_DIR, 'index.json')) as f:
        grid = json.load(f)
    print(f"Grid: {len(grid)} cells")

    # Load SAM3
    print("Loading SAM3...", flush=True)
    sam3_model, sam3_proc = build_sam3_model(device)
    gpu_memory_report()

    # Load MASt3R
    print("Loading MASt3R...", flush=True)
    from mast3r.model import AsymmetricMASt3R
    mast3r_model = AsymmetricMASt3R.from_pretrained(
        'kvuong2711/checkpoint-aerial-mast3r').to(device).eval()
    gpu_memory_report()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Process each grid cell
    all_detections = []
    for i, cell in enumerate(grid):
        t0 = time.time()
        dets = run_grid_cell_sam3(cell, sam3_proc, mast3r_model, device)
        elapsed = time.time() - t0
        all_detections.extend(dets)
        print(f"  [{i+1}/{len(grid)}] {cell['name']}: {len(dets)} dets in {elapsed:.1f}s", flush=True)

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

    # Filter to focus area
    in_area = [d for d in deduped
               if area[0] <= d['approx_lat'] <= area[2] and area[1] <= d['approx_lon'] <= area[3]]
    print(f"In focus area: {len(in_area)} / {len(deduped)}")

    # Evaluate
    print(f"\n{'='*60}")
    print("SAM3 + MASt3R EVALUATION RESULTS")
    print(f"{'='*60}")
    for radius in [5, 10, 15, 20, 30]:
        result = evaluate(in_area, gt_labels, match_radius_m=radius)
        print(f"  {radius}m: TP={result['tp']} FP={result['fp']} (lights={result['fp_streetlight']}) FN={result['fn']} | P={result['precision']:.1%} R={result['recall']:.1%} F1={result['f1']:.3f}")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.0f}s")

    # Save
    save_data = {
        'detections': [{k: v for k, v in d.items() if k != 'point_3d'} for d in in_area],
        'evaluation': {str(r): evaluate(in_area, gt_labels, match_radius_m=r) for r in [5, 10, 15, 20, 30]},
        'gt_summary': {'poles': n_poles, 'streetlights': n_lights},
        'grid_cells': len(grid),
        'total_time': round(total_time, 1),
        'model': 'SAM3 + MASt3R',
    }
    out_path = os.path.join(RESULTS_DIR, 'eval_results_sam3.json')
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
