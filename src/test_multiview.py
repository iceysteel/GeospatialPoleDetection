#!/usr/bin/env python3
"""
Test multi-view consensus: GDino detections + MASt3R 3D correspondences.
Runs on the 4 target-centered oblique images at 41.248644, -95.998878.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'mast3r'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'mast3r', 'dust3r'))
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
from PIL import Image, ImageDraw
import torch

TARGET_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'debug', 'target')
DIRECTIONS = ['north', 'east', 'south', 'west']
GDINO_TEXT = "utility pole. power pole. telephone pole."
GDINO_THRESHOLD = 0.15
CONSENSUS_DIST_THRESH = 15.0  # pixels in projected space


def run_gdino_all(images, processor, model, device):
    """Run GroundingDINO on all views, return detections per view."""
    all_dets = {}
    for d, img in images.items():
        w, h = img.size
        inputs = processor(images=img, text=GDINO_TEXT, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        out = processor.post_process_grounded_object_detection(
            outputs, inputs['input_ids'],
            target_sizes=torch.tensor([[h, w]]).to(device),
            threshold=GDINO_THRESHOLD, text_threshold=GDINO_THRESHOLD
        )[0]
        dets = []
        for box, score, label in zip(out['boxes'], out['scores'], out['text_labels']):
            x1, y1, x2, y2 = box.int().tolist()
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            dets.append({'bbox': [x1, y1, x2, y2], 'center': [cx, cy], 'conf': round(score.item(), 3), 'label': label})
        all_dets[d] = dets
        print(f"  GDino {d}: {len(dets)} detections")
    return all_dets


def run_mast3r(image_paths, device):
    """Run MASt3R on all view pairs, return 3D points and scene."""
    from mast3r.model import AsymmetricMASt3R
    from mast3r.fast_nn import fast_reciprocal_NNs
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    print("  Loading MASt3R model...")
    model = AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
    model = model.to(device)
    model.eval()

    print("  Loading images...")
    imgs = load_images(image_paths, size=512)
    pairs = make_pairs(imgs, scene_graph='complete', symmetrize=True)
    print(f"  {len(imgs)} images, {len(pairs)} pairs")

    print("  Running inference...")
    t0 = time.time()
    output = inference(pairs, model, device, batch_size=1)
    print(f"  Inference done in {time.time()-t0:.1f}s")

    print("  Global alignment...")
    t1 = time.time()
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer)
    loss = scene.compute_global_alignment(init='mst', niter=300, schedule='cosine', lr=0.01)
    print(f"  Alignment done in {time.time()-t1:.1f}s, final loss={loss:.4f}")

    # Get results
    imgs_tensor = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    print(f"  Poses shape: {poses.shape}")
    print(f"  Points3D: {len(pts3d)} views, first shape: {pts3d[0].shape}")

    return {
        'pts3d': pts3d,
        'poses': poses,
        'focals': focals,
        'masks': confidence_masks,
        'img_shapes': [(im.shape[1], im.shape[2]) for im in imgs_tensor],
    }


def project_3d_to_view(point_3d, pose, focal, img_shape):
    """Project a 3D point into a view's pixel coordinates using camera pose and focal."""
    # pose is 4x4 world-to-camera (or camera-to-world, need to check)
    # For MASt3R, poses are camera-to-world, so we need inverse
    pose_inv = torch.inverse(pose)
    R = pose_inv[:3, :3]
    t = pose_inv[:3, 3]

    # Transform to camera frame
    p_cam = R @ point_3d + t

    if p_cam[2] <= 0:
        return None  # Behind camera

    # Project
    h, w = img_shape
    fx = fy = focal
    cx, cy = w / 2, h / 2

    u = fx * p_cam[0] / p_cam[2] + cx
    v = fy * p_cam[1] / p_cam[2] + cy

    u, v = u.item(), v.item()
    if 0 <= u < w and 0 <= v < h:
        return (u, v)
    return None


def consensus_vote(gdino_dets, mast3r_result, directions, orig_sizes):
    """
    For each detection in each view, project its 3D point to other views
    and check if there's a matching detection there.
    """
    pts3d = mast3r_result['pts3d']
    poses = mast3r_result['poses']
    focals = mast3r_result['focals']
    img_shapes = mast3r_result['img_shapes']

    results = []

    for i, d in enumerate(directions):
        dets = gdino_dets.get(d, [])
        if not dets:
            continue

        h_orig, w_orig = orig_sizes[d]
        pts3d_view = pts3d[i]  # (H, W, 3)
        h_mast, w_mast = pts3d_view.shape[0], pts3d_view.shape[1]

        # Scale factors from original to MASt3R resolution
        sx = w_mast / w_orig
        sy = h_mast / h_orig

        for det in dets:
            cx, cy = det['center']
            # Map detection center to MASt3R pixel coords
            mx = min(max(int(cx * sx), 0), w_mast - 1)
            my = min(max(int(cy * sy), 0), h_mast - 1)

            # Get 3D point at detection center
            point_3d = pts3d_view[my, mx]  # (3,)

            if torch.isnan(point_3d).any():
                continue

            # Project to other views and check for matches
            agreeing_views = [d]  # This view agrees by definition

            for j, d2 in enumerate(directions):
                if j == i:
                    continue

                target_shape = (pts3d[j].shape[0], pts3d[j].shape[1])
                projected = project_3d_to_view(point_3d, poses[j], focals[j], target_shape)
                if projected is None:
                    continue

                pu, pv = projected
                # Scale back to original image coords
                pu_orig = pu / (target_shape[1] / orig_sizes[d2][1])
                pv_orig = pv / (target_shape[0] / orig_sizes[d2][0])

                # Check if any detection in view j is near this projected point
                for det2 in gdino_dets.get(d2, []):
                    cx2, cy2 = det2['center']
                    dist = ((cx2 - pu_orig)**2 + (cy2 - pv_orig)**2)**0.5
                    # Use a threshold relative to bbox size
                    bbox_size = max(det2['bbox'][2] - det2['bbox'][0], det2['bbox'][3] - det2['bbox'][1])
                    if dist < max(CONSENSUS_DIST_THRESH, bbox_size * 0.5):
                        agreeing_views.append(d2)
                        break

            results.append({
                'source_view': d,
                'bbox': det['bbox'],
                'center': det['center'],
                'conf': det['conf'],
                'label': det['label'],
                'point_3d': point_3d.tolist(),
                'agreeing_views': agreeing_views,
                'num_views': len(agreeing_views),
            })

    return results


def height_check(confirmed_results, directions, orig_sizes, view_metadata):
    """
    Estimate real-world height of each detection using camera geometry
    across ALL views where the detection was confirmed.

    real_height = bbox_vertical_pixels * GSD / sin(elevation_angle)

    Poles: 8-20m. Anything > 25m or < 4m is rejected.
    """
    import math

    MIN_POLE_HEIGHT = 4.0   # meters
    MAX_POLE_HEIGHT = 25.0  # meters

    checked = []
    for r in confirmed_results:
        x1, y1, x2, y2 = r['bbox']
        source = r['source_view']
        h_px = y2 - y1

        # Estimate height from the source view
        vm = view_metadata[source]
        gsd = vm['gsd']
        elevation = vm['elevation']
        sin_e = math.sin(math.radians(elevation))

        est_height_source = h_px * gsd / sin_e if sin_e > 0.1 else 0

        # Also check from agreeing views if we had their bbox sizes
        # For now use the source view estimate
        est_height = est_height_source

        is_pole = MIN_POLE_HEIGHT <= est_height <= MAX_POLE_HEIGHT
        reason = f"est_height={est_height:.1f}m (h_px={h_px}, GSD={gsd:.4f}, elev={elevation:.1f}deg)"

        r_copy = dict(r)
        r_copy['is_pole'] = is_pole
        r_copy['est_height'] = round(est_height, 1)
        r_copy['height_reason'] = reason

        status = "POLE" if is_pole else "REJECTED"
        print(f"  [{source}] bbox=[{x1},{y1},{x2},{y2}] → {status}: {reason}")

        checked.append(r_copy)

    return checked


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load images
    images = {}
    orig_sizes = {}
    image_paths = []
    for d in DIRECTIONS:
        path = os.path.join(TARGET_DIR, f'{d}.png')
        img = Image.open(path).convert('RGB')
        images[d] = img
        orig_sizes[d] = (img.size[1], img.size[0])  # (h, w)
        image_paths.append(path)

    # Step 1: GDino detections
    print("\n--- Step 1: GroundingDINO Detections ---")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    gdino_proc = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny')
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny').to(device)

    t0 = time.time()
    gdino_dets = run_gdino_all(images, gdino_proc, gdino_model, device)
    gdino_time = time.time() - t0
    print(f"  Total GDino time: {gdino_time:.1f}s")

    # Free GDino memory
    del gdino_model, gdino_proc
    torch.mps.empty_cache() if device == 'mps' else None

    # Step 2: MASt3R 3D correspondences
    print("\n--- Step 2: MASt3R 3D Reconstruction ---")
    t1 = time.time()
    mast3r_result = run_mast3r(image_paths, device)
    mast3r_time = time.time() - t1
    print(f"  Total MASt3R time: {mast3r_time:.1f}s")

    # Step 3: Consensus voting
    print("\n--- Step 3: Multi-View Consensus ---")
    all_results = consensus_vote(gdino_dets, mast3r_result, DIRECTIONS, orig_sizes)

    # Deduplicate — group by 3D point proximity
    confirmed_raw = [r for r in all_results if r['num_views'] >= 2]
    single_view = [r for r in all_results if r['num_views'] == 1]

    print(f"\n  Total detections across all views: {len(all_results)}")
    print(f"  Multi-view confirmed: {len(confirmed_raw)}")
    print(f"  Single view only: {len(single_view)}")

    # Step 4: Height check — filter by estimated real-world height using camera geometry
    print("\n--- Step 4: Height Check (camera geometry) ---")

    # Build view metadata from image metadata
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'metadata.json')) as f:
        full_meta = json.load(f)

    from math import sqrt
    view_metadata = {}
    for d in DIRECTIONS:
        # Find the best matching oblique URN for this direction
        best_urn = None
        best_gsd = 999
        for urn, m in full_meta.items():
            if m['type'] != 'oblique' or m['direction'] != d or not m.get('ground_footprint'):
                continue
            gsd = m.get('calculated_gsd', {}).get('value', 999)
            if gsd < best_gsd:
                # Check footprint containment
                gj = json.loads(m['ground_footprint']['geojson']['value'])
                feat = gj['features'][0] if gj['type'] == 'FeatureCollection' else gj
                geom = feat.get('geometry', feat)
                coords = geom['coordinates'][0] if geom['type'] == 'MultiPolygon' else geom['coordinates']
                ring = coords[0] if isinstance(coords[0][0], list) else coords
                inside = False
                for i in range(len(ring)):
                    j = (i - 1) % len(ring)
                    xi, yi = ring[i]; xj, yj = ring[j]
                    if ((yi > 41.248644) != (yj > 41.248644)) and (-95.998878 < (xj - xi) * (41.248644 - yi) / (yj - yi) + xi):
                        inside = not inside
                if inside:
                    best_gsd = gsd
                    best_urn = urn

        if best_urn:
            m = full_meta[best_urn]
            view_metadata[d] = {
                'gsd': m['calculated_gsd']['value'],
                'elevation': m['look_at']['elevation'],
                'azimuth': m['look_at']['azimuth'],
            }
            print(f"  {d}: GSD={view_metadata[d]['gsd']:.4f}m, elevation={view_metadata[d]['elevation']:.1f}deg")

    checked = height_check(confirmed_raw, DIRECTIONS, orig_sizes, view_metadata)

    confirmed = [r for r in checked if r.get('is_pole', True)]
    rejected_ortho = [r for r in checked if not r.get('is_pole', True)]

    print(f"\n  After height check:")
    print(f"    Confirmed poles: {len(confirmed)}")
    print(f"    Rejected (height out of range): {len(rejected_ortho)}")
    for r in confirmed:
        print(f"    POLE [{r['source_view']}] bbox={r['bbox']} views={r['agreeing_views']} h={r.get('est_height','?')}m")
    for r in rejected_ortho:
        print(f"    REJECTED [{r['source_view']}] bbox={r['bbox']} reason={r['height_reason']}")

    # Step 5: Visualize
    print("\n--- Step 5: Saving visualizations ---")
    for d in DIRECTIONS:
        img = images[d].copy()
        draw = ImageDraw.Draw(img)

        # Draw all GDino dets in dim gray
        for det in gdino_dets.get(d, []):
            x1, y1, x2, y2 = det['bbox']
            draw.rectangle([x1, y1, x2, y2], outline='gray', width=1)

        # Draw height-rejected in orange
        for r in rejected_ortho:
            if r['source_view'] == d:
                x1, y1, x2, y2 = r['bbox']
                draw.rectangle([x1, y1, x2, y2], outline='orange', width=3)
                draw.text((x1, max(0, y1-16)), f"BAD {r.get('est_height','?')}m", fill='orange')

        # Draw confirmed poles in bright green
        for r in confirmed:
            if r['source_view'] == d:
                x1, y1, x2, y2 = r['bbox']
                draw.rectangle([x1, y1, x2, y2], outline='lime', width=4)
                draw.text((x1, max(0, y1-16)), f"POLE {r.get('est_height','?')}m ({r['num_views']}v)", fill='lime')

        # Draw single-view detections in red
        for r in single_view:
            if r['source_view'] == d:
                x1, y1, x2, y2 = r['bbox']
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

        img.save(os.path.join(TARGET_DIR, f'{d}_consensus.jpg'), quality=90)

    # Save results
    save_results = {
        'confirmed': [{k: v for k, v in r.items() if k != 'point_3d'} for r in confirmed],
        'single_view_count': len(single_view),
        'total_detections': len(all_results),
        'gdino_time': round(gdino_time, 1),
        'mast3r_time': round(mast3r_time, 1),
    }
    # Add 3d points separately (not JSON serializable as tensors)
    for r in save_results['confirmed']:
        match = next((c for c in confirmed if c['bbox'] == r['bbox'] and c['source_view'] == r['source_view']), None)
        if match:
            r['point_3d'] = match['point_3d']

    with open(os.path.join(TARGET_DIR, 'consensus_results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nDone! Results saved to data/debug/target/")
    print(f"View: http://localhost:8080/data/debug/target/north_consensus.jpg")


if __name__ == '__main__':
    main()
