#!/usr/bin/env python3
"""
Multi-View Verification Agent.

Replaces single-crop VLM classification with multi-angle verification.
For each detection, gathers crops from all oblique views (via MASt3R 3D
projection) + ortho tile, sends all to Qwen 3.5 in one prompt.
"""
import sys, os, json, time, math, argparse, asyncio
sys.path.insert(0, os.path.dirname(__file__))

from PIL import Image
from agent_tools import (
    load_ortho_crop, extract_multiview_crops,
    query_ollama_multiview, image_to_b64
)

MULTIVIEW_PROMPT = """You are verifying whether a detected object is a UTILITY/POWER POLE in oblique aerial imagery (45° angle).

You are shown the SAME LOCATION from multiple oblique angles and a top-down ortho view.
The detected object is at the CENTER of each oblique crop.

Analyze systematically:
1. In how many oblique views do you see a tall, narrow, vertical structure at the center?
2. KEY POLE FEATURES: crossarms (horizontal bars near top), insulators, wires attached, wooden/metal texture, guy wires
3. KEY STREETLIGHT FEATURES: curved arm with light fixture at top, decorative design, shorter
4. KEY NON-POLE: tree (organic/branching), fence (horizontal run, short), building edge (flat/geometric)
5. In the ortho view: look for a small dark dot with cross-shaped shadow pattern or radiating wire shadows

Respond with ONLY a JSON object:
{"class": "pole"|"streetlight"|"fence"|"tree"|"building_edge"|"other", "confidence": 0.0-1.0, "views_with_object": <number of views where you see the object>, "reasoning": "<brief explanation>"}"""


def classify_detection_multiview(det, images, orig_sizes, pts3d, poses, focals, dir_list, model='qwen3.5:27b'):
    """Classify a single detection using multi-view evidence."""
    # Get crops from all views
    crops = extract_multiview_crops(det, images, orig_sizes, pts3d, poses, focals, dir_list)
    if not crops:
        return {'vlm_class': 'other', 'vlm_confidence': 0.0, 'views_checked': 0}

    # Get ortho crop
    ortho, _ = load_ortho_crop(det.get('approx_lat', 0), det.get('approx_lon', 0), radius_m=15)

    # Query VLM with all images
    response = query_ollama_multiview(crops, ortho, MULTIVIEW_PROMPT, model)
    if response is None:
        return {'vlm_class': 'other', 'vlm_confidence': 0.0, 'views_checked': len(crops)}

    # Parse response
    text = response.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[-1].rsplit('```', 1)[0].strip()

    categories = ['pole', 'streetlight', 'fence', 'tree', 'building_edge', 'other']
    try:
        result = json.loads(text)
        cls = result.get('class', 'other').lower().strip()
        if cls not in categories:
            cls = 'other'
        conf = float(result.get('confidence', 0.5))
        return {
            'vlm_class': cls,
            'vlm_confidence': min(max(conf, 0.0), 1.0),
            'views_checked': int(result.get('views_with_object', len(crops))),
            'reasoning': result.get('reasoning', ''),
            'vlm_raw': text,
        }
    except (json.JSONDecodeError, ValueError):
        return {'vlm_class': 'other', 'vlm_confidence': 0.0, 'views_checked': len(crops), 'vlm_raw': text}


def run_multiview_classification(eval_results_path, grid_index_path, model='qwen3.5:27b',
                                  output_path=None):
    """
    Run multi-view verification on all detections from eval results.
    Needs the grid cell images + MASt3R data, so we reload and process per-cell.
    """
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'mast3r'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'mast3r', 'dust3r'))

    with open(eval_results_path) as f:
        eval_data = json.load(f)
    with open(grid_index_path) as f:
        grid = json.load(f)

    cell_map = {c['name']: c for c in grid}
    DIRECTIONS = ['north', 'east', 'south', 'west']

    # Group detections by cell
    dets_by_cell = {}
    for det in eval_data['detections']:
        cell_name = det.get('cell', '')
        dets_by_cell.setdefault(cell_name, []).append(det)

    print(f"Multi-view verification on {len(eval_data['detections'])} detections across {len(dets_by_cell)} cells")

    # Load MASt3R for 3D projection
    from gpu_utils import get_device
    device = get_device(0)

    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from mast3r.model import AsymmetricMASt3R
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    print("Loading MASt3R...")
    mast3r_model = AsymmetricMASt3R.from_pretrained(
        'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric').to(device).eval()

    all_classified = []
    t_total = time.time()

    for cell_idx, (cell_name, cell_dets) in enumerate(dets_by_cell.items()):
        cell = cell_map.get(cell_name)
        if not cell:
            for det in cell_dets:
                all_classified.append({**det, 'vlm_class': 'other', 'vlm_confidence': 0})
            continue

        # Load cell images
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
            for det in cell_dets:
                all_classified.append({**det, 'vlm_class': 'other', 'vlm_confidence': 0})
            continue

        # Run MASt3R for this cell
        imgs = load_images(path_list, size=512)
        pairs_list = make_pairs(imgs, scene_graph='complete', symmetrize=True)
        output = inference(pairs_list, mast3r_model, device, batch_size=1)
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer)
        scene.compute_global_alignment(init='mst', niter=300, schedule='cosine', lr=0.01)
        pts3d = scene.get_pts3d()
        poses = scene.get_im_poses()
        focals = scene.get_focals()

        # Classify each detection in this cell
        for det in cell_dets:
            result = classify_detection_multiview(
                det, images, orig_sizes, pts3d, poses, focals, dir_list, model
            )
            classified = {**det, **result}
            all_classified.append(classified)

        n_done = sum(len(v) for v in list(dets_by_cell.values())[:cell_idx+1])
        print(f"  [{cell_idx+1}/{len(dets_by_cell)}] {cell_name}: {len(cell_dets)} dets | "
              f"total {n_done}/{len(eval_data['detections'])} | {time.time()-t_total:.0f}s", flush=True)

    elapsed = time.time() - t_total

    # Summary
    from collections import Counter
    counts = Counter(r['vlm_class'] for r in all_classified)
    print(f"\nMulti-view classification complete in {elapsed:.0f}s")
    print(f"Results: {dict(counts)}")

    if output_path is None:
        output_path = os.path.join(os.path.dirname(eval_results_path),
                                    'classifications_multiview.json')
    with open(output_path, 'w') as f:
        json.dump({
            'classifications': [{k: v for k, v in r.items() if k != 'vlm_raw'}
                                 for r in all_classified],
            'summary': dict(counts),
            'model': model,
            'method': 'multiview',
            'time': round(elapsed, 1),
        }, f, indent=2)
    print(f"Saved to {output_path}")

    return all_classified


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-results', default='data/eval_testarea/eval_results.json')
    parser.add_argument('--grid-index', default='data/testarea_grid/index.json')
    parser.add_argument('--model', default='qwen3.5:27b')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    run_multiview_classification(args.eval_results, args.grid_index, args.model, args.output)
