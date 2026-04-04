#!/usr/bin/env python3
"""
Shared tools for AI agents: ortho tile loading, multi-view crop extraction,
VLM querying via ollama.
"""
import os, json, math, base64
from io import BytesIO
from PIL import Image, ImageDraw

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
WMTS_DIR = os.path.join(DATA_DIR, 'wmts')


# ---- Ortho tile tools ----

def lat_lon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    lat_rad = math.radians(lat)
    y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    return x, y


def tile_to_lat_lon(x, y, zoom):
    n = 2 ** zoom
    lon = x / n * 360 - 180
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    return math.degrees(lat_rad), lon


def load_ortho_crop(lat, lon, radius_m=20, preferred_zoom=22):
    """Load an ortho crop centered on (lat, lon). Returns (PIL.Image, meta) or (None, None)."""
    zoom = preferred_zoom
    while zoom >= 19:
        tx, ty = lat_lon_to_tile(lat, lon, zoom)
        if os.path.exists(os.path.join(WMTS_DIR, f'{zoom}_{tx}_{ty}.png')):
            break
        zoom -= 1
    else:
        return None, None

    n = 2 ** zoom
    tile_deg = 360 / n
    m_per_px = tile_deg * 111320 * math.cos(math.radians(lat)) / 256

    radius_tiles = max(1, int(math.ceil(radius_m / (256 * m_per_px))) + 1)
    center_tx, center_ty = lat_lon_to_tile(lat, lon, zoom)

    tiles = {}
    for dx in range(-radius_tiles, radius_tiles + 1):
        for dy in range(-radius_tiles, radius_tiles + 1):
            path = os.path.join(WMTS_DIR, f'{zoom}_{center_tx+dx}_{center_ty+dy}.png')
            if os.path.exists(path):
                try:
                    tiles[(dx, dy)] = Image.open(path).convert('RGB')
                except:
                    pass
    if not tiles:
        return None, None

    min_dx = min(dx for dx, _ in tiles)
    min_dy = min(dy for _, dy in tiles)
    max_dx = max(dx for dx, _ in tiles)
    max_dy = max(dy for _, dy in tiles)
    w = (max_dx - min_dx + 1) * 256
    h = (max_dy - min_dy + 1) * 256

    stitched = Image.new('RGB', (w, h))
    for (dx, dy), tile in tiles.items():
        stitched.paste(tile, ((dx - min_dx) * 256, (dy - min_dy) * 256))

    tl_lat, tl_lon = tile_to_lat_lon(center_tx + min_dx, center_ty + min_dy, zoom)
    br_lat, br_lon = tile_to_lat_lon(center_tx + max_dx + 1, center_ty + max_dy + 1, zoom)

    px_x = (lon - tl_lon) / (br_lon - tl_lon) * w
    px_y = (lat - tl_lat) / (br_lat - tl_lat) * h

    radius_px = int(radius_m / m_per_px)
    crop = stitched.crop((
        max(0, int(px_x - radius_px)), max(0, int(px_y - radius_px)),
        min(w, int(px_x + radius_px)), min(h, int(px_y + radius_px)),
    ))

    return crop, {'zoom': zoom, 'm_per_px': m_per_px}


# ---- Multi-view crop extraction ----

def extract_multiview_crops(det, images, orig_sizes, pts3d, poses, focals, dir_list, crop_size=192):
    """
    Extract crops of a detection from ALL views using MASt3R 3D projection.
    Returns dict of {direction: PIL.Image} for views where the detection is visible.
    """
    import torch

    source_dir = det['source_view']
    source_idx = dir_list.index(source_dir) if source_dir in dir_list else -1
    if source_idx < 0:
        return {}

    # Get 3D point for the detection — compute center from bbox if not stored
    if 'center' in det:
        cx, cy = det['center']
    else:
        x1, y1, x2, y2 = det['bbox']
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    pv = pts3d[source_idx]
    hm, wm = pv.shape[:2]
    oh, ow = orig_sizes[source_dir]
    sx, sy = wm / ow, hm / oh
    mx = min(max(int(cx * sx), 0), wm - 1)
    my = min(max(int(cy * sy), 0), hm - 1)
    p3d = pv[my, mx]
    if torch.isnan(p3d).any():
        return {}

    crops = {}

    # Source view crop (original detection)
    img = images[source_dir]
    x1, y1, x2, y2 = det['bbox']
    w, h = x2 - x1, y2 - y1
    pad = int(max(w, h) * 0.5)
    crop = img.crop((max(0, x1 - pad), max(0, y1 - pad),
                     min(img.width, x2 + pad), min(img.height, y2 + pad)))
    crops[source_dir] = crop.resize((crop_size, crop_size), Image.LANCZOS)

    # Project into other views
    for j, d2 in enumerate(dir_list):
        if d2 == source_dir:
            continue

        pi = torch.inverse(poses[j])
        pc = pi[:3, :3] @ p3d + pi[:3, 3]
        if pc[2] <= 0:
            continue

        th, tw = pts3d[j].shape[:2]
        u = focals[j] * pc[0] / pc[2] + tw / 2
        v = focals[j] * pc[1] / pc[2] + th / 2

        if not (0 <= u.item() < tw and 0 <= v.item() < th):
            continue

        # Scale to original image coords
        oh2, ow2 = orig_sizes[d2]
        uo = u.item() / (tw / ow2)
        vo = v.item() / (th / oh2)

        # Extract crop around projected point
        img2 = images[d2]
        half = crop_size
        crop2 = img2.crop((
            max(0, int(uo - half)), max(0, int(vo - half)),
            min(img2.width, int(uo + half)), min(img2.height, int(vo + half)),
        ))
        if crop2.size[0] > 10 and crop2.size[1] > 10:
            crops[d2] = crop2.resize((crop_size, crop_size), Image.LANCZOS)

    return crops


# ---- VLM tools ----

def image_to_b64(img):
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def query_ollama_multiview(images_dict, ortho_img, prompt, model='qwen3.5:27b'):
    """
    Send multiple images to Qwen 3.5 via ollama in one request.
    images_dict: {direction: PIL.Image}
    ortho_img: PIL.Image or None
    Returns: raw response text
    """
    import requests

    # Build annotated images with direction labels
    all_images = []
    labels = []
    for d in ['north', 'east', 'south', 'west']:
        if d in images_dict:
            # Add direction label to image
            img = images_dict[d].copy()
            draw = ImageDraw.Draw(img)
            draw.rectangle([(0, 0), (60, 16)], fill='black')
            draw.text((4, 2), d.upper(), fill='white')
            all_images.append(image_to_b64(img))
            labels.append(f"Oblique {d.upper()}")

    if ortho_img is not None:
        draw = ImageDraw.Draw(ortho_img)
        draw.rectangle([(0, 0), (50, 16)], fill='black')
        draw.text((4, 2), 'ORTHO', fill='white')
        all_images.append(image_to_b64(ortho_img))
        labels.append("Top-down ortho")

    if not all_images:
        return None

    # Prepend image list to prompt
    image_desc = "Images provided: " + ", ".join(f"[{i+1}] {l}" for i, l in enumerate(labels))
    full_prompt = f"{image_desc}\n\n{prompt}"

    resp = requests.post('http://localhost:11434/api/chat', json={
        'model': model,
        'messages': [
            {'role': 'system', 'content': 'You are an expert aerial imagery analyst. Respond only with JSON.'},
            {'role': 'user', 'content': full_prompt, 'images': all_images},
        ],
        'stream': False,
        'think': False,
        'options': {'num_predict': 150, 'temperature': 0.0},
    }, timeout=120)

    if resp.status_code != 200:
        return None
    return resp.json()['message']['content']
