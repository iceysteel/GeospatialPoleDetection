#!/usr/bin/env python3
"""
AutoResearch Evaluation Harness — LOCKED, DO NOT MODIFY

Runs the pipeline from pipeline.py and computes F1@10m against verified GT.
Returns a single metric to optimize. The agent modifies pipeline.py, not this file.
"""
import sys, os, json, math, time, signal

# Timeout: 10 minutes max per experiment
TIMEOUT_SECONDS = 600

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'mast3r'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'mast3r', 'dust3r'))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
GT_FILE = os.path.join(DATA_DIR, 'ground_truth_testarea.json')

FOCUS_LAT, FOCUS_LON = 41.248644, -95.998878
RADIUS_LAT, RADIUS_LON = 0.0018, 0.0024
MATCH_RADIUS_M = 10  # Industry standard


def load_ground_truth():
    """Load verified GT poles in the focus area."""
    with open(GT_FILE) as f:
        gt = json.load(f)
    area = (FOCUS_LAT - RADIUS_LAT, FOCUS_LON - RADIUS_LON,
            FOCUS_LAT + RADIUS_LAT, FOCUS_LON + RADIUS_LON)
    poles = [l for l in gt['labels']
             if l['label'] == 'pole' and area[0] <= l['lat'] <= area[2] and area[1] <= l['lon'] <= area[3]]
    return poles, area


def compute_f1(detections, gt_poles, match_radius_m=MATCH_RADIUS_M):
    """Compute F1 at given match radius. Returns (f1, precision, recall, tp, fp, fn, rmse)."""
    m_per_deg_lon = 111320 * math.cos(math.radians(FOCUS_LAT))
    gt_matched = set()
    tp = fp = 0
    match_dists = []

    for det in detections:
        lat = det.get('lat', det.get('approx_lat', 0))
        lon = det.get('lon', det.get('approx_lon', 0))
        best_dist, best_idx = float('inf'), -1
        for i, gt in enumerate(gt_poles):
            d = math.sqrt(((lat - gt['lat']) * 111320) ** 2 +
                         ((lon - gt['lon']) * m_per_deg_lon) ** 2)
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

    return f1, p, r, tp, fp, fn, rmse


def timeout_handler(signum, frame):
    raise TimeoutError(f"Experiment exceeded {TIMEOUT_SECONDS}s timeout")


def run_experiment():
    """Run one experiment: execute pipeline.py, compute F1@10m, return results."""
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    t_start = time.time()
    gt_poles, area = load_ground_truth()

    try:
        # Import and run the pipeline (this is what the agent modifies)
        # Force reimport to pick up changes
        if 'pipeline' in sys.modules:
            del sys.modules['pipeline']
        sys.path.insert(0, os.path.dirname(__file__))
        from pipeline import run_pipeline

        detections = run_pipeline()

        # Filter to focus area
        in_area = [d for d in detections
                   if area[0] <= d.get('lat', d.get('approx_lat', 0)) <= area[2]
                   and area[1] <= d.get('lon', d.get('approx_lon', 0)) <= area[3]]

    except TimeoutError:
        signal.alarm(0)
        return {
            'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
            'tp': 0, 'fp': 0, 'fn': len(gt_poles),
            'rmse': 0.0, 'detections': 0, 'time': TIMEOUT_SECONDS,
            'status': 'timeout', 'error': f'Exceeded {TIMEOUT_SECONDS}s'
        }
    except Exception as e:
        signal.alarm(0)
        return {
            'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
            'tp': 0, 'fp': 0, 'fn': len(gt_poles),
            'rmse': 0.0, 'detections': 0, 'time': time.time() - t_start,
            'status': 'crash', 'error': str(e)[:200]
        }

    signal.alarm(0)
    elapsed = time.time() - t_start

    # Compute F1@10m
    f1, p, r, tp, fp, fn, rmse = compute_f1(in_area, gt_poles)

    return {
        'f1': round(f1, 4),
        'precision': round(p, 4),
        'recall': round(r, 4),
        'tp': tp, 'fp': fp, 'fn': fn,
        'rmse': round(rmse, 1),
        'detections': len(in_area),
        'detections_raw': len(detections),
        'gt_poles': len(gt_poles),
        'time': round(elapsed, 1),
        'status': 'ok',
    }


if __name__ == '__main__':
    result = run_experiment()
    print(f"\n{'='*50}")
    print(f"F1@10m: {result['f1']:.4f}")
    print(f"P={result['precision']:.1%} R={result['recall']:.1%}")
    print(f"TP={result['tp']} FP={result['fp']} FN={result['fn']} RMSE={result['rmse']}m")
    print(f"Detections: {result['detections']} (raw: {result.get('detections_raw', '?')})")
    print(f"Time: {result['time']}s | Status: {result['status']}")
    print(f"{'='*50}")

    # Write result to jsonl
    import datetime
    result['timestamp'] = datetime.datetime.now().isoformat()
    with open(os.path.join(os.path.dirname(__file__), 'autoresearch.jsonl'), 'a') as f:
        f.write(json.dumps(result) + '\n')
