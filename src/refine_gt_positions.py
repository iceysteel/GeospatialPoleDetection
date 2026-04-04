#!/usr/bin/env python3
"""
Auto-refine ground truth label positions using high-res ortho tiles + VLM.

For each GT label with source='auto_detected', loads a high-res ortho crop
centered on the approximate GPS, asks Qwen 3.5 to locate the exact pole
position, and updates the GPS coordinates.
"""
import sys, os, json, math, argparse, base64, time
sys.path.insert(0, os.path.dirname(__file__))

from io import BytesIO
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
WMTS_DIR = os.path.join(DATA_DIR, 'wmts')


def lat_lon_to_tile(lat, lon, zoom):
    """Convert GPS to tile coordinates."""
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    lat_rad = math.radians(lat)
    y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    return x, y


def tile_to_lat_lon(x, y, zoom):
    """Convert tile top-left corner to GPS."""
    n = 2 ** zoom
    lon = x / n * 360 - 180
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def load_ortho_crop(lat, lon, radius_m=20, preferred_zoom=23):
    """
    Load an ortho crop centered on (lat, lon) from WMTS tiles.
    Returns (PIL.Image, metadata_dict) or (None, None).
    """
    # Find best available zoom
    zoom = preferred_zoom
    while zoom >= 19:
        tx, ty = lat_lon_to_tile(lat, lon, zoom)
        tile_path = os.path.join(WMTS_DIR, f'{zoom}_{tx}_{ty}.png')
        if os.path.exists(tile_path):
            break
        zoom -= 1
    else:
        return None, None

    # Calculate pixel size at this zoom
    n = 2 ** zoom
    tile_size_deg_lon = 360 / n
    tile_size_deg_lat = tile_size_deg_lon  # approximate at this latitude
    m_per_pixel_lon = tile_size_deg_lon * 111320 * math.cos(math.radians(lat)) / 256
    m_per_pixel_lat = tile_size_deg_lon * 111320 / 256  # approximate

    # How many tiles we need to cover radius_m
    radius_tiles = max(1, int(math.ceil(radius_m / (256 * m_per_pixel_lon))) + 1)

    # Load and stitch tiles
    center_tx, center_ty = lat_lon_to_tile(lat, lon, zoom)
    tile_imgs = {}
    for dx in range(-radius_tiles, radius_tiles + 1):
        for dy in range(-radius_tiles, radius_tiles + 1):
            tx, ty = center_tx + dx, center_ty + dy
            path = os.path.join(WMTS_DIR, f'{zoom}_{tx}_{ty}.png')
            if os.path.exists(path):
                try:
                    tile_imgs[(dx, dy)] = Image.open(path).convert('RGB')
                except:
                    pass

    if not tile_imgs:
        return None, None

    # Stitch into one image
    min_dx = min(dx for dx, dy in tile_imgs.keys())
    min_dy = min(dy for dx, dy in tile_imgs.keys())
    max_dx = max(dx for dx, dy in tile_imgs.keys())
    max_dy = max(dy for dx, dy in tile_imgs.keys())

    w = (max_dx - min_dx + 1) * 256
    h = (max_dy - min_dy + 1) * 256
    stitched = Image.new('RGB', (w, h))
    for (dx, dy), tile in tile_imgs.items():
        px = (dx - min_dx) * 256
        py = (dy - min_dy) * 256
        stitched.paste(tile, (px, py))

    # Calculate pixel position of our target GPS in the stitched image
    # Top-left corner of stitched image in GPS
    tl_lat, tl_lon = tile_to_lat_lon(center_tx + min_dx, center_ty + min_dy, zoom)
    br_lat, br_lon = tile_to_lat_lon(center_tx + max_dx + 1, center_ty + max_dy + 1, zoom)

    # Target pixel in stitched image
    px_x = (lon - tl_lon) / (br_lon - tl_lon) * w
    px_y = (lat - tl_lat) / (br_lat - tl_lat) * h

    # Crop around target with radius_m
    radius_px = int(radius_m / m_per_pixel_lon)
    crop_x1 = max(0, int(px_x - radius_px))
    crop_y1 = max(0, int(px_y - radius_px))
    crop_x2 = min(w, int(px_x + radius_px))
    crop_y2 = min(h, int(px_y + radius_px))

    crop = stitched.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    # Center of target in the crop
    center_in_crop_x = px_x - crop_x1
    center_in_crop_y = px_y - crop_y1

    meta = {
        'zoom': zoom,
        'm_per_pixel': m_per_pixel_lon,
        'crop_origin_lat': tl_lat + crop_y1 / h * (br_lat - tl_lat),
        'crop_origin_lon': tl_lon + crop_x1 / w * (br_lon - tl_lon),
        'crop_w': crop.size[0],
        'crop_h': crop.size[1],
        'center_x': center_in_crop_x,
        'center_y': center_in_crop_y,
        'tl_lat': tl_lat, 'tl_lon': tl_lon,
        'br_lat': br_lat, 'br_lon': br_lon,
        'stitch_w': w, 'stitch_h': h,
        'crop_x1': crop_x1, 'crop_y1': crop_y1,
    }

    return crop, meta


def pixel_to_gps_ortho(px, py, meta):
    """Convert pixel in crop back to GPS."""
    # Pixel in stitched image
    sx = px + meta['crop_x1']
    sy = py + meta['crop_y1']
    # GPS
    lon = meta['tl_lon'] + sx / meta['stitch_w'] * (meta['br_lon'] - meta['tl_lon'])
    lat = meta['tl_lat'] + sy / meta['stitch_h'] * (meta['br_lat'] - meta['tl_lat'])
    return lat, lon


def refine_one_label(label, model='qwen3.5:27b'):
    """Ask VLM to find exact pole position in ortho crop."""
    import requests

    crop, meta = load_ortho_crop(label['lat'], label['lon'], radius_m=25)
    if crop is None:
        return None

    # Draw a small red crosshair at the current estimated position
    from PIL import ImageDraw
    annotated = crop.copy()
    draw = ImageDraw.Draw(annotated)
    cx, cy = int(meta['center_x']), int(meta['center_y'])
    r = 8
    draw.line([(cx-r, cy), (cx+r, cy)], fill='red', width=2)
    draw.line([(cx, cy-r), (cx, cy+r)], fill='red', width=2)
    draw.ellipse([(cx-r, cy-r), (cx+r, cy+r)], outline='red', width=2)

    # Encode image
    buf = BytesIO()
    annotated.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = """This is a top-down aerial photograph at very high resolution (~1-3cm per pixel).

There is a utility/power pole somewhere near the RED CROSSHAIR marker in this image. The pole appears from above as:
- A small circular or square dark dot (the pole top, ~30-50cm diameter)
- Often with a cross-shaped shadow extending from it (from crossarms)
- Thin dark line shadows extending outward (from power wires)
- The pole shadow is a long thin line pointing away from the sun

The red crosshair shows the APPROXIMATE location — the pole is likely within 15 meters of it.

Find the exact pixel coordinates of the pole's center (the dark dot at the top of the pole).

Look carefully for:
1. Small dark circular dots with radiating shadows
2. Cross-shaped patterns from crossarms
3. Wire shadow lines converging on a point

Respond with ONLY a JSON object: {"x": <pixel_x>, "y": <pixel_y>, "confidence": <0.0-1.0>, "description": "<what you see at that location>"}

If you cannot find a pole, respond: {"x": null, "y": null, "confidence": 0, "description": "no pole found"}"""

    resp = requests.post('http://localhost:11434/api/chat', json={
        'model': model,
        'messages': [
            {'role': 'system', 'content': 'You are an expert aerial image analyst. Respond only with JSON.'},
            {'role': 'user', 'content': prompt, 'images': [b64]},
        ],
        'stream': False,
        'think': False,
        'options': {'num_predict': 100, 'temperature': 0.0},
    }, timeout=120)

    if resp.status_code != 200:
        return None

    text = resp.json()['message']['content'].strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[-1].rsplit('```', 1)[0].strip()

    try:
        result = json.loads(text)
        if result.get('x') is None:
            return None

        px, py = float(result['x']), float(result['y'])
        conf = float(result.get('confidence', 0.5))

        # Clamp to crop bounds
        px = max(0, min(meta['crop_w'] - 1, px))
        py = max(0, min(meta['crop_h'] - 1, py))

        refined_lat, refined_lon = pixel_to_gps_ortho(px, py, meta)

        # Distance moved
        m_per_deg_lon = 111320 * math.cos(math.radians(label['lat']))
        dist = math.sqrt(((refined_lat - label['lat']) * 111320) ** 2 +
                         ((refined_lon - label['lon']) * m_per_deg_lon) ** 2)

        return {
            'lat': refined_lat,
            'lon': refined_lon,
            'confidence': conf,
            'distance_moved_m': round(dist, 1),
            'description': result.get('description', ''),
            'zoom_used': meta['zoom'],
        }
    except (json.JSONDecodeError, ValueError, KeyError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-file', default='data/ground_truth_testarea.json')
    parser.add_argument('--model', default='qwen3.5:27b')
    parser.add_argument('--max-move-m', type=float, default=25,
                        help='Max distance to move a label (reject larger moves)')
    parser.add_argument('--min-confidence', type=float, default=0.5)
    parser.add_argument('--dry-run', action='store_true', help='Show results without modifying GT')
    args = parser.parse_args()

    with open(args.gt_file) as f:
        gt = json.load(f)

    auto_labels = [l for l in gt['labels'] if l.get('source') == 'auto_detected']
    print(f"Refining {len(auto_labels)} auto-detected labels using ortho + {args.model}")

    refined = 0
    skipped = 0
    failed = 0
    distances = []

    for i, label in enumerate(auto_labels):
        result = refine_one_label(label, args.model)

        if result is None:
            failed += 1
            print(f"  [{i+1}/{len(auto_labels)}] FAILED — no result")
            continue

        dist = result['distance_moved_m']
        conf = result['confidence']

        if conf < args.min_confidence:
            skipped += 1
            print(f"  [{i+1}/{len(auto_labels)}] SKIP — low confidence ({conf:.2f})")
            continue

        if dist > args.max_move_m:
            skipped += 1
            print(f"  [{i+1}/{len(auto_labels)}] SKIP — too far ({dist:.1f}m > {args.max_move_m}m)")
            continue

        distances.append(dist)
        old = f"({label['lat']:.6f},{label['lon']:.6f})"
        new = f"({result['lat']:.6f},{result['lon']:.6f})"
        print(f"  [{i+1}/{len(auto_labels)}] REFINE {old} → {new} ({dist:.1f}m, conf={conf:.2f}) {result['description'][:60]}")

        if not args.dry_run:
            label['lat'] = round(result['lat'], 6)
            label['lon'] = round(result['lon'], 6)
            label['refined'] = True
            label['refine_distance_m'] = dist
            label['refine_confidence'] = conf
        refined += 1

    print(f"\nResults: {refined} refined, {skipped} skipped, {failed} failed")
    if distances:
        distances.sort()
        print(f"Move distances: min={min(distances):.1f}m median={distances[len(distances)//2]:.1f}m max={max(distances):.1f}m mean={sum(distances)/len(distances):.1f}m")

    if not args.dry_run and refined > 0:
        with open(args.gt_file, 'w') as f:
            json.dump(gt, f, indent=2)
        print(f"Saved updated GT to {args.gt_file}")
    elif args.dry_run:
        print("(dry run — no changes saved)")


if __name__ == '__main__':
    main()
