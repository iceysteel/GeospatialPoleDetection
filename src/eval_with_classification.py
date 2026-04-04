#!/usr/bin/env python3
"""
Evaluation pipeline with VLM classification stage.
Runs: GDino+MASt3R detection → unload → vLLM classification → filtered evaluation.
"""
import sys, os, time, json, math, argparse, subprocess, signal
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'mast3r'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'mast3r', 'dust3r'))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
from gpu_utils import get_device, clear_gpu, unload_model, gpu_memory_report
from eval_testarea import run_grid_cell, evaluate, nms

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
GRID_DIR = os.path.join(DATA_DIR, 'testarea_grid')
RESULTS_DIR = os.path.join(DATA_DIR, 'eval_testarea')

FOCUS_LAT, FOCUS_LON = 41.248644, -95.998878
RADIUS_LAT, RADIUS_LON = 0.0018, 0.0024


def wait_for_vllm(url="http://localhost:8000/v1/models", timeout=300):
    """Wait for vLLM server to be ready."""
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                data = json.loads(resp.read())
                model = data['data'][0]['id'] if data.get('data') else 'unknown'
                print(f"  vLLM ready with model: {model}")
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gdino-model', default='IDEA-Research/grounding-dino-base')
    parser.add_argument('--vlm-model', default='Qwen/Qwen3.5-9B')
    parser.add_argument('--vlm-confidence', type=float, default=0.7, help='Min VLM confidence to keep as pole')
    parser.add_argument('--skip-detection', action='store_true', help='Reuse existing eval_results.json')
    parser.add_argument('--skip-vllm-start', action='store_true', help='Assume vLLM is already running')
    parser.add_argument('--concurrency', type=int, default=16)
    args = parser.parse_args()

    device = args.device
    t_total = time.time()

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

    # ===== STEP 1: Detection (GDino + MASt3R) =====
    if args.skip_detection:
        print("\nSkipping detection, loading existing results...")
        with open(os.path.join(RESULTS_DIR, 'eval_results.json')) as f:
            saved = json.load(f)
        all_deduped = saved['detections']
        print(f"Loaded {len(all_deduped)} detections")
    else:
        print(f"\n{'='*60}")
        print("STEP 1: Detection (GDino + MASt3R)")
        print(f"{'='*60}")

        with open(os.path.join(GRID_DIR, 'index.json')) as f:
            grid = json.load(f)

        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        print(f"Loading GDino ({args.gdino_model})...")
        gdino_proc = AutoProcessor.from_pretrained(args.gdino_model)
        gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino_model).to(device)

        print("Loading MASt3R...")
        from mast3r.model import AsymmetricMASt3R
        mast3r_model = AsymmetricMASt3R.from_pretrained(
            'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric').to(device).eval()

        gpu_memory_report()
        os.makedirs(RESULTS_DIR, exist_ok=True)

        all_detections = []
        for i, cell in enumerate(grid):
            t0 = time.time()
            dets = run_grid_cell(cell, gdino_proc, gdino_model, mast3r_model, device)
            elapsed = time.time() - t0
            all_detections.extend(dets)
            print(f"  [{i+1}/{len(grid)}] {cell['name']}: {len(dets)} dets in {elapsed:.1f}s", flush=True)

        # Dedup
        m_per_deg_lon = 111320 * math.cos(math.radians(41.249))
        all_deduped = []
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
            all_deduped.append(best)

        print(f"\nDetection: {len(all_detections)} raw → {len(all_deduped)} after dedup")

        # Unload detection models
        print("Unloading detection models...")
        unload_model(gdino_model)
        unload_model(mast3r_model)
        del gdino_proc
        clear_gpu()
        gpu_memory_report()

    # ===== Baseline eval (before classification) =====
    print(f"\n{'='*60}")
    print("BASELINE (before VLM classification)")
    print(f"{'='*60}")
    for radius in [10, 15, 30]:
        r = evaluate(all_deduped, gt_labels, match_radius_m=radius)
        print(f"  {radius}m: P={r['precision']:.1%} R={r['recall']:.1%} F1={r['f1']:.3f} (TP={r['tp']} FP={r['fp']} FN={r['fn']})")

    # ===== STEP 2: VLM Classification =====
    print(f"\n{'='*60}")
    print("STEP 2: VLM Classification")
    print(f"{'='*60}")

    vllm_proc = None
    if not args.skip_vllm_start:
        print(f"Starting vLLM server with {args.vlm_model} (TP=2)...")
        script_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
        vllm_proc = subprocess.Popen(
            [os.path.join(os.path.dirname(__file__), '..', '.venv', 'bin', 'python'),
             '-m', 'vllm.entrypoints.openai.api_server',
             '--model', args.vlm_model,
             '--tensor-parallel-size', '2',
             '--gpu-memory-utilization', '0.85',
             '--max-model-len', '4096',
             '--trust-remote-code',
             '--port', '8000'],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        print("  Waiting for vLLM to load model...")
        if not wait_for_vllm():
            print("  ERROR: vLLM failed to start")
            vllm_proc.kill()
            return
    else:
        print("Assuming vLLM is already running on :8000")
        if not wait_for_vllm(timeout=10):
            print("  ERROR: vLLM not reachable")
            return

    try:
        # Build detection list with image paths for classification
        with open(os.path.join(GRID_DIR, 'index.json')) as f:
            grid = json.load(f)
        cell_map = {c['name']: c for c in grid}

        classify_input = []
        for det in all_deduped:
            cell = cell_map.get(det.get('cell'))
            if not cell: continue
            direction = det.get('source_view')
            img_path = cell['images'].get(direction)
            if not img_path or not os.path.exists(img_path): continue
            classify_input.append({
                'image_path': img_path,
                'bbox': det['bbox'],
                'conf': det['conf'],
                'source_view': direction,
                'cell': det['cell'],
                'approx_lat': det.get('approx_lat'),
                'approx_lon': det.get('approx_lon'),
                'est_height': det.get('est_height'),
                'num_views': det.get('num_views'),
            })

        from classify_detections import classify_detections_sync
        print(f"Classifying {len(classify_input)} detections...")
        t0 = time.time()
        classified = classify_detections_sync(classify_input, args.vlm_model, args.concurrency)
        classify_time = time.time() - t0
        print(f"Classification done in {classify_time:.1f}s")

        # Filter to poles only
        poles = [r for r in classified if r['vlm_class'] == 'pole' and r['vlm_confidence'] >= args.vlm_confidence]
        non_poles = [r for r in classified if r not in poles]

        from collections import Counter
        all_counts = Counter(r['vlm_class'] for r in classified)
        print(f"\nClassification breakdown: {dict(all_counts)}")
        print(f"Kept as poles (conf >= {args.vlm_confidence}): {len(poles)} / {len(classified)}")

    finally:
        if vllm_proc:
            print("\nStopping vLLM server...")
            vllm_proc.send_signal(signal.SIGTERM)
            vllm_proc.wait(timeout=30)

    # ===== Eval with classification =====
    print(f"\n{'='*60}")
    print("WITH VLM CLASSIFICATION")
    print(f"{'='*60}")
    for radius in [10, 15, 30]:
        r = evaluate(poles, gt_labels, match_radius_m=radius)
        print(f"  {radius}m: P={r['precision']:.1%} R={r['recall']:.1%} F1={r['f1']:.3f} (TP={r['tp']} FP={r['fp']} FN={r['fn']})")

    total_time = time.time() - t_total
    print(f"\nTotal time: {total_time:.0f}s")

    # Save full results
    save_data = {
        'baseline': {str(r): evaluate(all_deduped, gt_labels, match_radius_m=r) for r in [5, 10, 15, 20, 30]},
        'with_vlm': {str(r): evaluate(poles, gt_labels, match_radius_m=r) for r in [5, 10, 15, 20, 30]},
        'classifications': [{k: v for k, v in r.items() if k != 'vlm_raw'} for r in classified],
        'vlm_model': args.vlm_model,
        'vlm_confidence_threshold': args.vlm_confidence,
        'classification_breakdown': dict(all_counts),
        'total_time': round(total_time, 1),
    }
    out_path = os.path.join(RESULTS_DIR, 'eval_with_vlm.json')
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
