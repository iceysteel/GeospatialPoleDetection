#!/usr/bin/env python3
"""
Compare SAM2 auto-generate vs GroundingDINO vs Qwen3-VL for pole detection.
Downloads fresh oblique crops centered on a target location, runs all 3 detectors,
and saves annotated images to data/debug/target/.

Usage:
    python src/compare_detection.py [lat] [lon]
    Default: 41.248644, -95.998878 (known pole location)
"""
import sys, os, time, json, base64, io

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'sam2'))
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import numpy as np
from PIL import Image, ImageDraw
import torch
import requests
from math import sqrt
from auth import auth_headers

BASE_URL = os.environ.get('EAGLEVIEW_BASE_URL', 'https://sandbox.apis.eagleview.com')
QWEN_URL = "http://localhost:11434/api/chat"
QWEN_MODEL = "qwen3-vl:2b"
QWEN_PROMPT = 'locate every instance that belongs to the following categories: "utility pole". Report bbox coordinates in JSON format.'
GDINO_TEXT = "utility pole. power pole. telephone pole."
GDINO_THRESHOLD = 0.15
SAM_AR_THRESHOLD = 3
SAM_AREA_RANGE = (100, 50000)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'debug', 'target')


def find_best_obliques(meta, target_lat, target_lon):
    """Find best oblique image per direction whose footprint covers the target."""
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
            if ((yi > target_lat) != (yj > target_lat)) and (target_lon < (xj - xi) * (target_lat - yi) / (yj - yi) + xi):
                inside = not inside
        if not inside:
            continue

        gsd = m.get('calculated_gsd', {}).get('value', 999)
        if d not in candidates or gsd < candidates[d]['gsd']:
            candidates[d] = {'urn': urn, 'gsd': gsd, 'meta': m}

    return candidates


def download_target_crops(candidates, target_lat, target_lon):
    """Download oblique crops centered on the target location."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = {}

    for d in ['north', 'east', 'south', 'west']:
        if d not in candidates:
            print(f'  {d}: no coverage')
            continue

        out_path = os.path.join(OUTPUT_DIR, f'{d}.png')
        if os.path.exists(out_path):
            print(f'  {d}: cached')
            paths[d] = out_path
            continue

        c = candidates[d]
        urn = c['urn']
        max_zoom = c['meta'].get('zoom_range', {}).get('maximum_zoom_level')

        params = {
            'center.x': target_lon, 'center.y': target_lat,
            'center.radius': 50, 'epsg': 'EPSG:4326',
            'format': 'IMAGE_FORMAT_PNG',
        }
        if max_zoom:
            params['zoom'] = max_zoom

        url = f'{BASE_URL}/imagery/v3/images/{urn}/location'

        for radius in [50, 35, 25]:
            params['center.radius'] = radius
            resp = requests.get(url, params=params, headers=auth_headers())
            if resp.status_code != 413:
                break

        if resp.status_code == 200:
            with open(out_path, 'wb') as f:
                f.write(resp.content)
            print(f'  {d}: saved {len(resp.content)//1024}KB')
            paths[d] = out_path
        else:
            print(f'  {d}: FAILED {resp.status_code}')

    return paths


def run_sam2(generator, image_np):
    """Run SAM2 auto-generate and filter for pole-like masks."""
    t0 = time.time()
    masks = generator.generate(image_np)
    elapsed = time.time() - t0

    candidates = []
    for m in masks:
        x, y, bw, bh = m['bbox']
        ar = bh / bw if bw > 0 else 0
        if ar > SAM_AR_THRESHOLD and SAM_AREA_RANGE[0] < m['area'] < SAM_AREA_RANGE[1]:
            candidates.append({
                'bbox': [int(x), int(y), int(x + bw), int(y + bh)],
                'ar': round(ar, 1),
                'iou': round(m['predicted_iou'], 3),
            })

    return {'total_masks': len(masks), 'candidates': candidates, 'time': round(elapsed, 1)}


def run_gdino(processor, model, image_pil, device):
    """Run GroundingDINO zero-shot detection."""
    w, h = image_pil.size
    inputs = processor(images=image_pil, text=GDINO_TEXT, return_tensors='pt').to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    elapsed = time.time() - t0

    out = processor.post_process_grounded_object_detection(
        outputs, inputs['input_ids'],
        target_sizes=torch.tensor([[h, w]]).to(device),
        threshold=GDINO_THRESHOLD, text_threshold=GDINO_THRESHOLD
    )[0]

    dets = []
    for box, score, label in zip(out['boxes'], out['scores'], out['text_labels']):
        x1, y1, x2, y2 = box.int().tolist()
        dets.append({'bbox': [x1, y1, x2, y2], 'conf': round(score.item(), 3), 'label': label})

    return {'detections': dets, 'time': round(elapsed, 1)}


def run_qwen(image_pil):
    """Run Qwen3-VL grounding via ollama."""
    w, h = image_pil.size
    thumb = image_pil.copy()
    thumb.thumbnail((768, 768))
    buf = io.BytesIO()
    thumb.save(buf, format='JPEG', quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    t0 = time.time()
    try:
        resp = requests.post(QWEN_URL, json={
            'model': QWEN_MODEL,
            'messages': [{'role': 'user', 'content': QWEN_PROMPT, 'images': [img_b64]}],
            'think': False, 'stream': False,
            'options': {'temperature': 0, 'num_predict': 4096, 'top_k': 20, 'top_p': 0.8}
        }, timeout=120)
        elapsed = time.time() - t0
        data = resp.json()
        content = data.get('message', {}).get('content', '')
        thinking = data.get('message', {}).get('thinking', '')
        text = content if content.strip() else thinking

        clean = text.strip()
        if clean.startswith('```json'): clean = clean[7:]
        if clean.startswith('```'): clean = clean[3:]
        if clean.endswith('```'): clean = clean[:-3]
        parsed = json.loads(clean.strip())

        dets = []
        for det in parsed:
            bbox = det.get('bbox_2d', det.get('bbox', []))
            if len(bbox) == 4:
                dets.append({
                    'bbox': [int(bbox[0]/1000*w), int(bbox[1]/1000*h), int(bbox[2]/1000*w), int(bbox[3]/1000*h)],
                    'label': det.get('label', 'pole'),
                })
        return {'detections': dets, 'time': round(elapsed, 1)}
    except Exception:
        return {'detections': [], 'time': round(time.time() - t0, 1)}


def draw_boxes(image_pil, boxes, color, label_key='label'):
    """Draw bounding boxes on a copy of the image."""
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b['bbox']
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = b.get(label_key, b.get('conf', b.get('ar', '')))
        if 'ar' in b:
            label = f"SAM#{i} AR={b['ar']}"
        elif 'conf' in b:
            label = f"{b.get('label', '')} {b['conf']}"
        draw.text((x1, max(0, y1 - 14)), str(label), fill=color)
    return img


def main():
    target_lat = float(sys.argv[1]) if len(sys.argv) > 1 else 41.248644
    target_lon = float(sys.argv[2]) if len(sys.argv) > 2 else -95.998878

    print(f"Target: ({target_lat}, {target_lon})")

    # Load metadata
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'metadata.json')) as f:
        meta = json.load(f)

    # Find and download target-centered crops
    print("\nFinding obliques covering target...")
    candidates = find_best_obliques(meta, target_lat, target_lon)
    print(f"Found {len(candidates)} directions")

    print("\nDownloading target-centered crops...")
    paths = download_target_crops(candidates, target_lat, target_lon)

    # Load models
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    print("Loading SAM2...")
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    sam2 = build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',
                      os.path.join(os.path.dirname(__file__), '..', 'models', 'sam2', 'checkpoints', 'sam2.1_hiera_base_plus.pt'),
                      device=device)
    generator = SAM2AutomaticMaskGenerator(sam2, points_per_side=32, pred_iou_thresh=0.7, stability_score_thresh=0.9)

    print("Loading GroundingDINO...")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    gdino_proc = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny')
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny').to(device)

    # Run detectors
    results = {}
    for d in ['north', 'east', 'south', 'west']:
        if d not in paths:
            continue
        path = paths[d]
        img_pil = Image.open(path).convert('RGB')
        img_np = np.array(img_pil)
        w, h = img_pil.size
        print(f"\n=== {d.upper()} ({w}x{h}) ===")

        sam_result = run_sam2(generator, img_np)
        print(f"  SAM2: {sam_result['total_masks']} masks, {len(sam_result['candidates'])} candidates, {sam_result['time']}s")

        gdino_result = run_gdino(gdino_proc, gdino_model, img_pil, device)
        print(f"  GDino: {len(gdino_result['detections'])} detections, {gdino_result['time']}s")

        qwen_result = run_qwen(img_pil)
        print(f"  Qwen: {len(qwen_result['detections'])} detections, {qwen_result['time']}s")

        # Save annotated images
        img_pil.save(os.path.join(OUTPUT_DIR, f'{d}_original.jpg'), quality=90)
        draw_boxes(img_pil, sam_result['candidates'], 'yellow').save(os.path.join(OUTPUT_DIR, f'{d}_sam.jpg'), quality=90)
        draw_boxes(img_pil, gdino_result['detections'], 'lime').save(os.path.join(OUTPUT_DIR, f'{d}_gdino.jpg'), quality=90)
        draw_boxes(img_pil, qwen_result['detections'], 'cyan').save(os.path.join(OUTPUT_DIR, f'{d}_qwen.jpg'), quality=90)

        results[d] = {
            'sam': {'candidates': len(sam_result['candidates']), 'time': sam_result['time']},
            'gdino': {'detections': len(gdino_result['detections']), 'time': gdino_result['time']},
            'qwen': {'detections': len(qwen_result['detections']), 'time': qwen_result['time']},
        }

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Dir':<10} {'SAM2':<15} {'GDino':<15} {'Qwen':<15}")
    for d in ['north', 'east', 'south', 'west']:
        if d not in results:
            continue
        r = results[d]
        print(f"{d:<10} {r['sam']['candidates']} ({r['sam']['time']}s){'':<5} "
              f"{r['gdino']['detections']} ({r['gdino']['time']}s){'':<5} "
              f"{r['qwen']['detections']} ({r['qwen']['time']}s)")

    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
