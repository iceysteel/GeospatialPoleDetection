#!/usr/bin/env python3
"""
AutoResearch Evaluation Harness — LOCKED, DO NOT MODIFY

Runs the pipeline from pipeline.py and computes F1@10m against verified GT.
Returns a single metric to optimize. The agent modifies pipeline.py, not this file.
"""
import sys, os, json, math, time, signal

# Timeout: 20 minutes (VLM filtering enabled temporarily)
# TODO: If VLM hasn't improved F1 by 2 hours from now, disable it and revert to 600s
TIMEOUT_SECONDS = 1200

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'mast3r'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'mast3r', 'dust3r'))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
GT_FILE_TEST = os.path.join(DATA_DIR, 'ground_truth_testarea.json')
GT_FILE_HOLDOUT = os.path.join(DATA_DIR, 'ground_truth_holdout.json')

# Test area
TEST_LAT, TEST_LON = 41.248644, -95.998878
# Holdout area (500m east, unseen during development)
HOLDOUT_LAT, HOLDOUT_LON = 41.2486, -95.9929

RADIUS_LAT, RADIUS_LON = 0.0018, 0.0024
MATCH_RADIUS_M = 10  # Industry standard


def load_ground_truth(area_name='test'):
    """Load verified GT poles for test or holdout area."""
    if area_name == 'holdout':
        gt_file = GT_FILE_HOLDOUT
        center_lat, center_lon = HOLDOUT_LAT, HOLDOUT_LON
    else:
        gt_file = GT_FILE_TEST
        center_lat, center_lon = TEST_LAT, TEST_LON

    with open(gt_file) as f:
        gt = json.load(f)
    area = (center_lat - RADIUS_LAT, center_lon - RADIUS_LON,
            center_lat + RADIUS_LAT, center_lon + RADIUS_LON)
    poles = [l for l in gt['labels']
             if l['label'] == 'pole' and area[0] <= l['lat'] <= area[2] and area[1] <= l['lon'] <= area[3]]
    return poles, area


def compute_f1(detections, gt_poles, match_radius_m=MATCH_RADIUS_M):
    """Compute F1 at given match radius. Returns (f1, precision, recall, tp, fp, fn, rmse)."""
    m_per_deg_lon = 111320 * math.cos(math.radians(TEST_LAT))
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
    """Run one experiment: execute pipeline.py on BOTH areas, compute F1@10m, return results."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    t_start = time.time()

    try:
        # Import and run pipeline on TEST area only
        if 'pipeline' in sys.modules:
            del sys.modules['pipeline']
        sys.path.insert(0, os.path.dirname(__file__))
        from pipeline import run_pipeline
        import pipeline as pipeline_mod
        pipeline_mod.GRID_DIR = os.path.join(PROJECT_ROOT, 'data', 'testarea_grid')

        test_dets = run_pipeline()

    except TimeoutError:
        signal.alarm(0)
        return {
            'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
            'tp': 0, 'fp': 0, 'fn': 0,
            'detections': 0, 'time': TIMEOUT_SECONDS,
            'status': 'timeout', 'error': f'Exceeded {TIMEOUT_SECONDS}s'
        }
    except Exception as e:
        signal.alarm(0)
        return {
            'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
            'tp': 0, 'fp': 0, 'fn': 0,
            'detections': 0, 'time': time.time() - t_start,
            'status': 'crash', 'error': str(e)[:200]
        }

    signal.alarm(0)
    elapsed = time.time() - t_start

    # Eval on test area
    gt_test, area_test = load_ground_truth('test')
    test_in = [d for d in test_dets
               if area_test[0] <= d.get('lat', d.get('approx_lat', 0)) <= area_test[2]
               and area_test[1] <= d.get('lon', d.get('approx_lon', 0)) <= area_test[3]]
    f1_t, p_t, r_t, tp_t, fp_t, fn_t, rmse_t = compute_f1(test_in, gt_test)

    result = {
        'f1': round(f1_t, 4),
        'precision': round(p_t, 4),
        'recall': round(r_t, 4),
        'tp': tp_t, 'fp': fp_t, 'fn': fn_t,
        'rmse': round(rmse_t, 1),
        'detections': len(test_in),
        'detections_raw': len(test_dets),
        'gt_poles': len(gt_test),
        'time': round(elapsed, 1),
        'status': 'ok',
    }

    # Only run holdout if this is a new best (read best from jsonl)
    best_f1 = 0.0
    jsonl_path = os.path.join(os.path.dirname(__file__), 'autoresearch.jsonl')
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get('f1', 0) > best_f1:
                        best_f1 = d['f1']
                except:
                    pass

    if f1_t > best_f1:
        print(f"  NEW BEST {f1_t:.4f} > {best_f1:.4f} — running holdout check...", flush=True)
        try:
            pipeline_mod.GRID_DIR = os.path.join(PROJECT_ROOT, 'data', 'holdout_grid')
            holdout_dets = run_pipeline()
            pipeline_mod.GRID_DIR = os.path.join(PROJECT_ROOT, 'data', 'testarea_grid')

            gt_hold, area_hold = load_ground_truth('holdout')
            hold_in = [d for d in holdout_dets
                       if area_hold[0] <= d.get('lat', d.get('approx_lat', 0)) <= area_hold[2]
                       and area_hold[1] <= d.get('lon', d.get('approx_lon', 0)) <= area_hold[3]]
            f1_h, p_h, r_h, tp_h, fp_h, fn_h, rmse_h = compute_f1(hold_in, gt_hold)

            result['f1_holdout'] = round(f1_h, 4)
            result['holdout_precision'] = round(p_h, 4)
            result['holdout_recall'] = round(r_h, 4)
            result['holdout_tp'] = tp_h
            result['holdout_fp'] = fp_h
            result['holdout_fn'] = fn_h
            result['holdout_detections'] = len(hold_in)
            result['overfit_gap'] = round(abs(f1_t - f1_h), 4)
        except Exception as e:
            result['holdout_error'] = str(e)[:100]

    return result


if __name__ == '__main__':
    result = run_experiment()
    print(f"\n{'='*60}")
    print(f"F1@10m:  {result['f1']:.4f}  P={result['precision']:.1%} R={result['recall']:.1%}")
    print(f"TP={result['tp']} FP={result['fp']} FN={result['fn']} RMSE={result['rmse']}m")
    if 'f1_holdout' in result:
        print(f"HOLDOUT: F1={result['f1_holdout']:.4f} P={result['holdout_precision']:.1%} R={result['holdout_recall']:.1%}")
        print(f"  Overfit gap: {result.get('overfit_gap', 0):.4f}")
    print(f"Time: {result['time']}s | Status: {result['status']}")
    print(f"{'='*60}")

    # Write result to jsonl
    import datetime
    result['timestamp'] = datetime.datetime.now().isoformat()
    with open(os.path.join(os.path.dirname(__file__), 'autoresearch.jsonl'), 'a') as f:
        f.write(json.dumps(result) + '\n')
