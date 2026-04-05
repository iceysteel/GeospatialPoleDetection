#!/usr/bin/env python3
"""
High-resolution 3D shape analysis using cropped MASt3R.

Instead of feeding full oblique images to MASt3R (which downscales to 512px
covering ~100m ground), crops a small region around each detection (~30m)
from all 4 views, then runs MASt3R on those crops. This gives ~13x higher
3D resolution, enough to distinguish poles (thin vertical) from trees
(wide branching) and fences (short horizontal).

Run after the main eval pipeline as a refinement step.
"""
import sys, os, json, time, math, argparse, tempfile
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'mast3r'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'mast3r', 'dust3r'))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
import numpy as np
from PIL import Image
from oblique_utils import gps_to_pixel
from gpu_utils import get_device, gpu_memory_report

DIRECTIONS = ['north', 'east', 'south', 'west']


def extract_view_crop(image_path, lat, lon, cell_lat, cell_lon, azimuth, crop_radius,
                      target_radius_m=30, min_crop_px=200):
    """
    Extract a crop from an oblique image centered on (lat, lon).
    Returns (crop_image, temp_path) or (None, None).
    """
    img = Image.open(image_path).convert('RGB')
    img_w, img_h = img.size

    px, py = gps_to_pixel(lat, lon, img_w, img_h, cell_lat, cell_lon, azimuth, crop_radius)

    if px < 0 or px >= img_w or py < 0 or py >= img_h:
        return None, None

    # Calculate crop size in pixels from target_radius_m
    # GSD ≈ 2 * crop_radius / max(img_w, img_h) meters per pixel
    gsd = 2 * crop_radius / max(img_w, img_h)
    radius_px = int(target_radius_m / gsd)
    radius_px = max(radius_px, min_crop_px)

    x1 = max(0, px - radius_px)
    y1 = max(0, py - radius_px)
    x2 = min(img_w, px + radius_px)
    y2 = min(img_h, py + radius_px)

    crop = img.crop((x1, y1, x2, y2))
    if crop.size[0] < min_crop_px or crop.size[1] < min_crop_px:
        return None, None

    return crop, (px - x1, py - y1)  # return crop + detection position within crop


def analyze_3d_shape(pts3d, center_y, center_x, radius_px=30):
    """
    Analyze 3D shape of the object at (center_y, center_x) in the point cloud.
    Returns shape features dict.
    """
    h, w = pts3d.shape[:2]

    # Extract points in a column around the detection center (narrow horizontal, full vertical)
    col_half = max(3, radius_px // 8)  # narrow column
    cx1 = max(0, center_x - col_half)
    cx2 = min(w, center_x + col_half)

    # Full vertical strip
    col_pts = pts3d[:, cx1:cx2+1].reshape(-1, 3)
    valid = ~torch.isnan(col_pts).any(dim=1)
    col_pts = col_pts[valid]

    if len(col_pts) < 10:
        return {'shape_class': 'unknown', 'aspect_hires': 0, 'thinness_hires': 0, 'straightness': 0}

    # Overall bbox of object region
    y1 = max(0, center_y - radius_px)
    y2 = min(h, center_y + radius_px)
    x1 = max(0, center_x - radius_px)
    x2 = min(w, center_x + radius_px)
    box_pts = pts3d[y1:y2+1, x1:x2+1].reshape(-1, 3)
    valid_box = ~torch.isnan(box_pts).any(dim=1)
    box_pts = box_pts[valid_box]

    if len(box_pts) < 10:
        return {'shape_class': 'unknown', 'aspect_hires': 0, 'thinness_hires': 0, 'straightness': 0}

    # PCA to find principal axes
    centered = box_pts - box_pts.mean(dim=0)
    cov = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    ev = eigenvalues.detach().cpu().numpy()

    # Shape features
    # Poles: one dominant axis (large eigenvalue), two small → high linearity
    # Trees: two large, one small → planar/spread
    # Buildings: one large, one medium, one small → planar
    linearity = (ev[2] - ev[1]) / max(ev[2], 1e-6)
    planarity = (ev[1] - ev[0]) / max(ev[2], 1e-6)
    sphericity = ev[0] / max(ev[2], 1e-6)

    # Aspect ratio from ranges
    ranges = box_pts.detach().max(dim=0).values - box_pts.detach().min(dim=0).values
    sorted_ranges = ranges.sort(descending=True).values
    aspect_hires = (sorted_ranges[0] / max(sorted_ranges[1], 1e-4)).item()
    thinness_hires = (sorted_ranges[0] / max(sorted_ranges[2], 1e-4)).item()

    # Straightness: how well the vertical column follows a line
    # Fit a line through the column points, measure R²
    if len(col_pts) >= 5:
        col_centered = col_pts - col_pts.mean(dim=0)
        col_cov = col_centered.T @ col_centered / len(col_centered)
        col_ev, col_evec = torch.linalg.eigh(col_cov)
        straightness = (col_ev[2] / max(col_ev.sum().item(), 1e-6)).item()
    else:
        straightness = 0.5

    # Classify shape
    if linearity > 0.7 and aspect_hires > 3.0:
        shape_class = 'linear'  # pole-like
    elif planarity > 0.5:
        shape_class = 'planar'  # building/fence-like
    elif sphericity > 0.3:
        shape_class = 'compact'  # tree/bush-like
    else:
        shape_class = 'mixed'

    return {
        'shape_class': shape_class,
        'linearity': round(linearity, 3),
        'planarity': round(planarity, 3),
        'sphericity': round(sphericity, 3),
        'aspect_hires': round(aspect_hires, 2),
        'thinness_hires': round(thinness_hires, 2),
        'straightness': round(straightness, 3),
    }


def run_hires_3d_analysis(eval_results_path, grid_index_path, device='cuda:0'):
    """
    For each detection, crop all views around it and run MASt3R at high res.
    """
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from mast3r.model import AsymmetricMASt3R

    with open(eval_results_path) as f:
        eval_data = json.load(f)
    with open(grid_index_path) as f:
        grid = json.load(f)
    cell_map = {c['name']: c for c in grid}

    print(f"Loading MASt3R...")
    mast3r = AsymmetricMASt3R.from_pretrained(
        'kvuong2711/checkpoint-aerial-mast3r').to(device).eval()
    gpu_memory_report()

    dets = eval_data['detections']
    print(f"Running hi-res 3D analysis on {len(dets)} detections...")

    results = []
    t0 = time.time()

    for idx, det in enumerate(dets):
        cell = cell_map.get(det.get('cell'))
        if not cell:
            results.append({**det, 'shape_3d': {'shape_class': 'unknown'}})
            continue

        lat, lon = det['approx_lat'], det['approx_lon']

        # Extract crops from all views around this detection
        crops = {}
        crop_centers = {}
        tmp_paths = []
        for d in DIRECTIONS:
            img_path = cell['images'].get(d)
            if not img_path or not os.path.exists(img_path):
                continue
            dm = cell.get(f'{d}_meta', {})
            crop, center_in_crop = extract_view_crop(
                img_path, lat, lon,
                cell['lat'], cell['lon'],
                dm.get('azimuth', 0), dm.get('crop_radius', 50),
                target_radius_m=25
            )
            if crop is not None:
                # Save crop to temp file for MASt3R's load_images
                tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                crop.save(tmp.name)
                crops[d] = tmp.name
                crop_centers[d] = center_in_crop
                tmp_paths.append(tmp.name)

        if len(crops) < 2:
            results.append({**det, 'shape_3d': {'shape_class': 'unknown', 'n_views': len(crops)}})
            for p in tmp_paths:
                os.unlink(p)
            continue

        # Run MASt3R on cropped views
        try:
            path_list = [crops[d] for d in DIRECTIONS if d in crops]
            dir_list = [d for d in DIRECTIONS if d in crops]

            imgs = load_images(path_list, size=512)
            pairs_list = make_pairs(imgs, scene_graph='complete', symmetrize=True)
            output = inference(pairs_list, mast3r, device, batch_size=1)
            scene = global_aligner(output, device=device,
                                   mode=GlobalAlignerMode.ModularPointCloudOptimizer)
            scene.compute_global_alignment(init='mst', niter=200, schedule='cosine', lr=0.01)
            pts3d = scene.get_pts3d()

            # Analyze shape from the source view's point cloud
            source_dir = det.get('source_view', dir_list[0])
            if source_dir in dir_list:
                si = dir_list.index(source_dir)
            else:
                si = 0

            pv = pts3d[si]
            h, w = pv.shape[:2]

            # Detection center in the crop → scale to MASt3R resolution
            if source_dir in crop_centers:
                ccx, ccy = crop_centers[source_dir]
                crop_img = Image.open(crops[source_dir] if source_dir in crops else path_list[si])
                cw, ch = crop_img.size
                mx = int(ccx / cw * w)
                my = int(ccy / ch * h)
            else:
                mx, my = w // 2, h // 2

            mx = min(max(mx, 0), w - 1)
            my = min(max(my, 0), h - 1)

            shape = analyze_3d_shape(pv, my, mx, radius_px=min(w, h) // 4)
            shape['n_views'] = len(dir_list)

            results.append({**det, 'shape_3d': shape})

        except Exception as e:
            results.append({**det, 'shape_3d': {'shape_class': 'error', 'error': str(e)[:100]}})

        # Cleanup temp files
        for p in tmp_paths:
            try:
                os.unlink(p)
            except:
                pass

        if (idx + 1) % 10 == 0 or idx == len(dets) - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(dets) - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{len(dets)}] {rate:.1f} det/s | ETA: {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    # Save results
    out_path = os.path.join(os.path.dirname(eval_results_path), 'eval_results_hires3d.json')
    save_data = {**eval_data, 'detections': [{k: v for k, v in r.items() if k != 'point_3d'} for r in results]}
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved to {out_path}")

    # Quick summary of shape distributions
    from collections import Counter
    shapes = Counter(r['shape_3d']['shape_class'] for r in results)
    print(f"Shape distribution: {dict(shapes)}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-results', default='data/eval_testarea/eval_results.json')
    parser.add_argument('--grid-index', default='data/testarea_grid/index.json')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run_hires_3d_analysis(args.eval_results, args.grid_index, args.device)
