#!/usr/bin/env python3
"""Download WMTS ortho tiles for the sandbox bounding box."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import requests
from auth import auth_headers
from ratelimit import tiles_limiter

BASE_URL = os.environ.get("EAGLEVIEW_BASE_URL", "https://sandbox.apis.eagleview.com")
TILE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "wmts")

BBOX = (-96.00532698173473, 41.24140396772262, -95.97589954958912, 41.25672882015283)
ZOOM = 19
LAYER = "Latest"
TMS = "GoogleMapsCompatible_9-23"


def lat_lon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    lat_rad = math.radians(lat)
    y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    return x, y


def download_tiles():
    os.makedirs(TILE_DIR, exist_ok=True)

    x1, y1 = lat_lon_to_tile(BBOX[3], BBOX[0], ZOOM)  # top-left
    x2, y2 = lat_lon_to_tile(BBOX[1], BBOX[2], ZOOM)   # bottom-right
    total = (x2 - x1 + 1) * (y2 - y1 + 1)

    print(f"[wmts] Downloading {total} tiles at zoom {ZOOM}")
    print(f"  x: {x1}-{x2}, y: {y1}-{y2}")

    downloaded = 0
    skipped = 0
    failed = 0

    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            path = os.path.join(TILE_DIR, f"{ZOOM}_{x}_{y}.png")
            if os.path.exists(path):
                skipped += 1
                continue

            url = f"{BASE_URL}/imagery/wmts/v1/visual/tile/{LAYER}/default/{TMS}/{ZOOM}/{x}/{y}.png"
            tiles_limiter.wait()

            try:
                resp = requests.get(url, headers=auth_headers())
                if resp.status_code == 200:
                    with open(path, "wb") as f:
                        f.write(resp.content)
                    downloaded += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1

            done = downloaded + skipped + failed
            if done % 50 == 0 or done == total:
                pct = done / total * 100
                print(f"  [{done}/{total}] {pct:.0f}% - {downloaded} new, {skipped} cached, {failed} failed")

    # Write tile index for viewer
    index = {
        "zoom": ZOOM,
        "x_range": [x1, x2],
        "y_range": [y1, y2],
        "tile_dir": "wmts",
        "format": "png",
        "total": total,
        "downloaded": downloaded + skipped,
    }
    import json
    with open(os.path.join(TILE_DIR, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n[wmts] Done: {downloaded} downloaded, {skipped} cached, {failed} failed")
    return index


if __name__ == "__main__":
    download_tiles()
