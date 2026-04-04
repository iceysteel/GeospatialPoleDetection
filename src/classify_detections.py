#!/usr/bin/env python3
"""
Classify detection crops using Qwen 3.5 VLM via vLLM OpenAI-compatible API.
Takes GDino detections, extracts crops, classifies each as:
  pole / streetlight / fence / tree / building_edge / other
"""
import os, json, time, base64, asyncio, argparse
from io import BytesIO
from PIL import Image

VLLM_URL = "http://localhost:8000/v1/chat/completions"
CATEGORIES = ["pole", "streetlight", "fence", "tree", "building_edge", "other"]

CLASSIFICATION_PROMPT = """You are analyzing a crop from an oblique aerial photograph taken at approximately 45 degrees.
Classify the main object in the center of this image into exactly one category:
- "pole": wooden or metal utility/power pole (typically vertical, brown/dark, with crossarms or wires attached)
- "streetlight": decorative street lamp or light fixture (typically has a light head, shorter, may be curved)
- "fence": fence post or fence section (typically part of a horizontal run, short)
- "tree": tree trunk or branch (organic shape, bark texture)
- "building_edge": building corner, roof edge, or architectural feature
- "other": none of the above

Respond with ONLY a JSON object: {"class": "<category>", "confidence": <0.0-1.0>}"""


def extract_crop(image_path, bbox, padding=0.3):
    """Extract a crop around a detection bbox with padding for context."""
    img = Image.open(image_path).convert('RGB')
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = int(w * padding), int(h * padding)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(img.width, x2 + pad_x)
    cy2 = min(img.height, y2 + pad_y)
    crop = img.crop((cx1, cy1, cx2, cy2))
    crop = crop.resize((384, 384), Image.LANCZOS)
    return crop


def image_to_base64(img):
    """Convert PIL Image to base64 data URI."""
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def parse_vlm_response(text):
    """Parse JSON response from VLM, handling common formatting issues."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith('```'):
        text = text.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    try:
        result = json.loads(text)
        cls = result.get('class', 'other').lower().strip()
        conf = float(result.get('confidence', 0.5))
        if cls not in CATEGORIES:
            cls = 'other'
        return cls, min(max(conf, 0.0), 1.0)
    except (json.JSONDecodeError, ValueError, KeyError):
        return 'other', 0.0


async def classify_single(session, crop_b64, model, semaphore):
    """Classify a single crop via the vLLM API."""
    import aiohttp
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a classifier. Reply with ONLY a single JSON object, no explanation."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": crop_b64}},
                {"type": "text", "text": CLASSIFICATION_PROMPT}
            ]}
        ],
        "max_tokens": 50,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with semaphore:
        try:
            async with session.post(VLLM_URL, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    return 'other', 0.0, f"HTTP {resp.status}: {err[:200]}"
                data = await resp.json()
                text = data['choices'][0]['message']['content']
                cls, conf = parse_vlm_response(text)
                return cls, conf, text
        except Exception as e:
            return 'other', 0.0, str(e)


async def classify_batch(detections, model, concurrency=16):
    """Classify a batch of detections concurrently."""
    import aiohttp
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        for det in detections:
            crop = extract_crop(det['image_path'], det['bbox'])
            crop_b64 = image_to_base64(crop)
            tasks.append(classify_single(session, crop_b64, model, semaphore))

        responses = await asyncio.gather(*tasks)
        for det, (cls, conf, raw) in zip(detections, responses):
            results.append({
                **det,
                'vlm_class': cls,
                'vlm_confidence': conf,
                'vlm_raw': raw,
            })

    return results


def classify_detections_sync(detections, model, concurrency=16):
    """Synchronous wrapper for classify_batch."""
    return asyncio.run(classify_batch(detections, model, concurrency))


def classify_from_eval_results(eval_results_path, model, output_path=None, concurrency=16):
    """Load eval results, extract crops, classify, and save."""
    with open(eval_results_path) as f:
        data = json.load(f)

    # Build detection list with image paths
    grid_dir = os.path.join(os.path.dirname(eval_results_path), '..', 'testarea_grid')
    with open(os.path.join(grid_dir, 'index.json')) as f:
        grid = json.load(f)
    cell_map = {c['name']: c for c in grid}

    detections = []
    for det in data['detections']:
        cell = cell_map.get(det.get('cell'))
        if not cell:
            continue
        direction = det.get('source_view')
        img_path = cell['images'].get(direction)
        if not img_path or not os.path.exists(img_path):
            continue
        detections.append({
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

    print(f"Classifying {len(detections)} detections with {model}...")
    t0 = time.time()
    results = classify_detections_sync(detections, model, concurrency)
    elapsed = time.time() - t0

    # Summary
    from collections import Counter
    counts = Counter(r['vlm_class'] for r in results)
    print(f"\nClassification complete in {elapsed:.1f}s")
    print(f"Results: {dict(counts)}")
    poles = [r for r in results if r['vlm_class'] == 'pole' and r['vlm_confidence'] >= 0.7]
    print(f"Poles (conf >= 0.7): {len(poles)} / {len(results)}")

    if output_path is None:
        output_path = os.path.join(os.path.dirname(eval_results_path), 'classifications.json')
    with open(output_path, 'w') as f:
        json.dump({'classifications': results, 'summary': dict(counts), 'time': round(elapsed, 1)}, f, indent=2)
    print(f"Saved to {output_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-results', default='data/eval_testarea/eval_results.json')
    parser.add_argument('--model', default='Qwen/Qwen3.5-9B')
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    classify_from_eval_results(args.eval_results, args.model, args.output, args.concurrency)
