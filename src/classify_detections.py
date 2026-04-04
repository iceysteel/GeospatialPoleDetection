#!/usr/bin/env python3
"""
Classify detection crops using Qwen 3.5 VLM via ollama or vLLM API.
Takes GDino detections, extracts crops, classifies each as:
  pole / streetlight / fence / tree / building_edge / other
"""
import os, json, time, base64, asyncio, argparse
from io import BytesIO
from PIL import Image

OLLAMA_URL = "http://localhost:11434/api/chat"
VLLM_URL = "http://localhost:8000/v1/chat/completions"
CATEGORIES = ["pole", "streetlight", "fence", "tree", "building_edge", "other"]

CLASSIFICATION_PROMPT = """Classify the central object in this oblique aerial photograph (taken at ~45° angle from above).

Look carefully at the object's shape, material, and context:
- POLE: A tall, narrow, vertical wooden or metal structure. Usually brown/grey, with crossarms, insulators, or wires attached near the top. Often stands alone near roads. Utility poles are typically 8-15m tall with a consistent cylindrical shape.
- STREETLIGHT: A pole with a distinct light fixture/lamp head at the top. Often curved or has a horizontal arm with a light. The light head is the key distinguishing feature — utility poles do NOT have light fixtures.
- FENCE: A horizontal structure with vertical posts. Usually forms a line/boundary. Posts are short (1-2m). Often has horizontal rails or mesh between posts.
- TREE: Organic shape with branches, leaves, or bark texture. Irregular outline, often wider at top.
- BUILDING_EDGE: Part of a building — roof edge, corner, wall, chimney. Geometric, flat surfaces, often with uniform color/texture.
- OTHER: None of the above.

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
    """Convert PIL Image to base64 string (no data URI prefix)."""
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def image_to_data_uri(img):
    """Convert PIL Image to base64 data URI for vLLM."""
    return f"data:image/jpeg;base64,{image_to_base64(img)}"


def parse_vlm_response(text):
    """Parse JSON response from VLM, handling common formatting issues."""
    text = text.strip()
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


# ---- Ollama backend ----

async def classify_single_ollama(session, crop_b64, model, semaphore):
    """Classify a single crop via ollama API."""
    import aiohttp
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an aerial imagery classifier. Reply ONLY with a JSON object."},
            {"role": "user", "content": CLASSIFICATION_PROMPT, "images": [crop_b64]},
        ],
        "stream": False,
        "think": False,
        "options": {"num_predict": 60, "temperature": 0.0},
    }
    async with semaphore:
        try:
            async with session.post(OLLAMA_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    return 'other', 0.0, f"HTTP {resp.status}: {err[:200]}"
                data = await resp.json()
                text = data['message']['content']
                cls, conf = parse_vlm_response(text)
                return cls, conf, text
        except Exception as e:
            return 'other', 0.0, str(e)


# ---- vLLM backend ----

async def classify_single_vllm(session, crop_b64, model, semaphore):
    """Classify a single crop via vLLM OpenAI-compatible API."""
    import aiohttp
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an aerial imagery classifier. Reply ONLY with a JSON object."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
                {"type": "text", "text": CLASSIFICATION_PROMPT}
            ]}
        ],
        "max_tokens": 60,
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


# ---- Batch processing ----

async def classify_batch(detections, model, backend='ollama', concurrency=4):
    """Classify a batch of detections concurrently."""
    import aiohttp
    semaphore = asyncio.Semaphore(concurrency)
    classify_fn = classify_single_ollama if backend == 'ollama' else classify_single_vllm

    async with aiohttp.ClientSession() as session:
        tasks = []
        for det in detections:
            crop = extract_crop(det['image_path'], det['bbox'])
            crop_b64 = image_to_base64(crop)
            tasks.append(classify_fn(session, crop_b64, model, semaphore))

        responses = await asyncio.gather(*tasks)
        results = []
        for det, (cls, conf, raw) in zip(detections, responses):
            results.append({
                **det,
                'vlm_class': cls,
                'vlm_confidence': conf,
                'vlm_raw': raw,
            })

    return results


def classify_detections_sync(detections, model, backend='ollama', concurrency=4):
    """Synchronous wrapper for classify_batch."""
    return asyncio.run(classify_batch(detections, model, backend, concurrency))


def classify_from_eval_results(eval_results_path, model, backend='ollama',
                                output_path=None, concurrency=4):
    """Load eval results, extract crops, classify, and save."""
    with open(eval_results_path) as f:
        data = json.load(f)

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

    print(f"Classifying {len(detections)} detections with {model} via {backend}...")
    t0 = time.time()
    results = classify_detections_sync(detections, model, backend, concurrency)
    elapsed = time.time() - t0

    from collections import Counter
    counts = Counter(r['vlm_class'] for r in results)
    print(f"\nClassification complete in {elapsed:.1f}s")
    print(f"Results: {dict(counts)}")
    poles = [r for r in results if r['vlm_class'] == 'pole' and r['vlm_confidence'] >= 0.7]
    print(f"Poles (conf >= 0.7): {len(poles)} / {len(results)}")

    if output_path is None:
        output_path = os.path.join(os.path.dirname(eval_results_path),
                                    f'classifications_{model.replace("/", "_")}.json')
    with open(output_path, 'w') as f:
        json.dump({'classifications': results, 'summary': dict(counts),
                    'model': model, 'backend': backend, 'time': round(elapsed, 1)}, f, indent=2)
    print(f"Saved to {output_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-results', default='data/eval_testarea/eval_results.json')
    parser.add_argument('--model', default='qwen3.5:27b')
    parser.add_argument('--backend', choices=['ollama', 'vllm'], default='ollama')
    parser.add_argument('--concurrency', type=int, default=4,
                        help='Concurrent requests (ollama: keep low, vllm: can go higher)')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    classify_from_eval_results(args.eval_results, args.model, args.backend,
                                args.output, args.concurrency)
