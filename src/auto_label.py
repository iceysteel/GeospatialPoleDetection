#!/usr/bin/env python3
"""
Automated labeling pipeline:
1. Run GDino-Base on all 185 oblique images → candidate detections
2. Classify each detection crop with Qwen 3.5 27B via ollama
3. Tier results by confidence for training data

Output: data/auto_labels/
  - detections.json (all GDino detections with crops)
  - classifications.json (VLM-classified detections)
  - training_poles.json (high-confidence pole labels for fine-tuning)
  - crops/ (saved detection crops)
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
from PIL import Image
from gpu_utils import get_device, gpu_memory_report, unload_model, clear_gpu
from classify_detections import (
    extract_crop, image_to_base64, parse_vlm_response,
    classify_detections_sync, CLASSIFICATION_PROMPT, CATEGORIES
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'auto_labels')
GDINO_TEXT = "utility pole. power pole. telephone pole."
GDINO_THRESHOLD = 0.15
NMS_IOU = 0.3


def nms(dets, iou_thresh=NMS_IOU):
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


def run_gdino_all(metadata, device, gdino_model_id):
    """Run GDino on all oblique images, return detections with image paths."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    print(f"Loading GDino ({gdino_model_id})...")
    proc = AutoProcessor.from_pretrained(gdino_model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_model_id).to(device)
    gpu_memory_report()

    obliques = {urn: m for urn, m in metadata.items()
                if m['type'] == 'oblique' and m.get('local_path') and os.path.exists(m['local_path'])}
    print(f"Processing {len(obliques)} oblique images...")

    all_detections = []
    t0 = time.time()
    for idx, (urn, m) in enumerate(obliques.items()):
        img_orig = Image.open(m['local_path']).convert('RGB')
        orig_w, orig_h = img_orig.size

        # Resize to limit processor input — prevents OOM on large obliques
        max_dim = 800
        scale = min(1.0, max_dim / max(orig_w, orig_h))
        if scale < 1.0:
            img_resized = img_orig.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
        else:
            img_resized = img_orig
        w, h = img_resized.size

        inputs = proc(images=img_resized, text=GDINO_TEXT, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        out = proc.post_process_grounded_object_detection(
            outputs, inputs['input_ids'],
            target_sizes=torch.tensor([[h, w]]).to(device),
            threshold=GDINO_THRESHOLD, text_threshold=GDINO_THRESHOLD
        )[0]
        del inputs, outputs
        torch.cuda.empty_cache()

        # Scale bboxes back to original image coordinates
        sx_back = orig_w / w
        sy_back = orig_h / h
        dets = []
        for box, score, label in zip(out['boxes'], out['scores'], out['text_labels']):
            x1, y1, x2, y2 = box.int().tolist()
            dets.append({
                'bbox': [int(x1*sx_back), int(y1*sy_back), int(x2*sx_back), int(y2*sy_back)],
                'center': [int((x1+x2)/2*sx_back), int((y1+y2)/2*sy_back)],
                'conf': round(score.item(), 3),
                'label': label,
            })
        dets = nms(dets)

        for det in dets:
            det['image_path'] = m['local_path']
            det['urn'] = urn
            det['direction'] = m['direction']
            det['image_size'] = [orig_w, orig_h]
        all_detections.extend(dets)

        if (idx + 1) % 20 == 0 or idx == len(obliques) - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(obliques) - idx - 1) / rate
            print(f"  [{idx+1}/{len(obliques)}] {len(all_detections)} dets so far | {rate:.1f} img/s | ETA: {eta:.0f}s",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\nGDino complete: {len(all_detections)} detections from {len(obliques)} images in {elapsed:.0f}s")

    # Unload GDino
    print("Unloading GDino...")
    unload_model(model)
    del proc
    clear_gpu()

    return all_detections


def save_crops(detections, crops_dir, max_crops=None):
    """Extract and save crops for all detections."""
    os.makedirs(crops_dir, exist_ok=True)
    for i, det in enumerate(detections):
        if max_crops and i >= max_crops:
            break
        crop = extract_crop(det['image_path'], det['bbox'])
        crop_path = os.path.join(crops_dir, f"crop_{i:05d}.jpg")
        crop.save(crop_path, quality=90)
        det['crop_path'] = crop_path
    print(f"Saved {min(len(detections), max_crops or len(detections))} crops to {crops_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gdino-model', default='IDEA-Research/grounding-dino-base')
    parser.add_argument('--vlm-model', default='qwen3.5:27b')
    parser.add_argument('--vlm-backend', default='ollama')
    parser.add_argument('--vlm-concurrency', type=int, default=2)
    parser.add_argument('--skip-detection', action='store_true', help='Reuse existing detections.json')
    parser.add_argument('--skip-classification', action='store_true', help='Reuse existing classifications')
    parser.add_argument('--save-crops', action='store_true', help='Save crop images to disk')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_total = time.time()

    # ===== STEP 1: GDino detection on all images =====
    det_path = os.path.join(OUTPUT_DIR, 'detections.json')
    if args.skip_detection and os.path.exists(det_path):
        print("Loading existing detections...")
        with open(det_path) as f:
            all_detections = json.load(f)
        print(f"Loaded {len(all_detections)} detections")
    else:
        with open(os.path.join(DATA_DIR, 'metadata.json')) as f:
            metadata = json.load(f)
        all_detections = run_gdino_all(metadata, args.device, args.gdino_model)

        # Save detections (without image_path for portability)
        with open(det_path, 'w') as f:
            json.dump(all_detections, f, indent=2)
        print(f"Saved detections to {det_path}")

    # Optionally save crops
    if args.save_crops:
        save_crops(all_detections, os.path.join(OUTPUT_DIR, 'crops'))

    # ===== STEP 2: VLM classification =====
    cls_path = os.path.join(OUTPUT_DIR, 'classifications.json')
    if args.skip_classification and os.path.exists(cls_path):
        print("\nLoading existing classifications...")
        with open(cls_path) as f:
            cls_data = json.load(f)
        classified = cls_data['classifications']
        print(f"Loaded {len(classified)} classifications")
    else:
        print(f"\n{'='*60}")
        print(f"STEP 2: Classifying {len(all_detections)} detections with {args.vlm_model}")
        print(f"{'='*60}")

        t0 = time.time()
        classified = classify_detections_sync(
            all_detections, args.vlm_model, args.vlm_backend, args.vlm_concurrency
        )
        elapsed = time.time() - t0

        from collections import Counter
        counts = Counter(r['vlm_class'] for r in classified)
        print(f"\nClassification complete in {elapsed:.0f}s")
        print(f"Breakdown: {dict(counts)}")

        with open(cls_path, 'w') as f:
            json.dump({
                'classifications': [{k: v for k, v in r.items() if k != 'vlm_raw'}
                                     for r in classified],
                'summary': dict(counts),
                'model': args.vlm_model,
                'time': round(elapsed, 1),
            }, f, indent=2)
        print(f"Saved to {cls_path}")

    # ===== STEP 3: Tier results =====
    print(f"\n{'='*60}")
    print("STEP 3: Tiering results for training data")
    print(f"{'='*60}")

    high_conf_poles = []
    medium_conf_poles = []
    hard_negatives = []

    for r in classified:
        conf = r.get('vlm_confidence', 0)
        cls = r.get('vlm_class', 'other')

        if cls == 'pole' and conf >= 0.9:
            high_conf_poles.append(r)
        elif cls == 'pole' and conf >= 0.6:
            medium_conf_poles.append(r)
        elif cls in ('streetlight', 'fence', 'tree', 'building_edge') and conf >= 0.8:
            hard_negatives.append(r)

    print(f"High-confidence poles (>=0.9):    {len(high_conf_poles)}")
    print(f"Medium-confidence poles (0.6-0.9): {len(medium_conf_poles)}")
    print(f"Hard negatives (non-pole, >=0.8): {len(hard_negatives)}")
    print(f"Total training candidates:        {len(high_conf_poles) + len(hard_negatives)}")

    # Save training-ready data
    training = {
        'high_conf_poles': [{k: v for k, v in r.items() if k != 'vlm_raw'} for r in high_conf_poles],
        'medium_conf_poles': [{k: v for k, v in r.items() if k != 'vlm_raw'} for r in medium_conf_poles],
        'hard_negatives': [{k: v for k, v in r.items() if k != 'vlm_raw'} for r in hard_negatives],
        'summary': {
            'high_poles': len(high_conf_poles),
            'medium_poles': len(medium_conf_poles),
            'hard_negatives': len(hard_negatives),
        },
    }
    training_path = os.path.join(OUTPUT_DIR, 'training_data.json')
    with open(training_path, 'w') as f:
        json.dump(training, f, indent=2)

    total_time = time.time() - t_total
    print(f"\nTotal time: {total_time:.0f}s")
    print(f"Training data saved to {training_path}")


if __name__ == '__main__':
    main()
