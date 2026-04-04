#!/usr/bin/env python3
"""Test qwen3.5:0.8b on all oblique images at a specific location for pole detection."""
import base64
import io
import json
import os
import sys
import time

from PIL import Image
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:0.8b"
TARGET_SIZE = 768  # resize longest side
PROMPT = """Look at this aerial oblique photograph carefully.
Identify any utility poles, power poles, or electrical poles visible in the image.
For each pole found, describe:
1. Its approximate position in the image (left/center/right, top/middle/bottom)
2. What makes you think it is a utility/power pole

If you see no poles, say "No poles detected."
Be concise."""


def encode_image(path, max_size=TARGET_SIZE):
    img = Image.open(path)
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def query_qwen(image_b64):
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": PROMPT,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 500}
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()


def main():
    lat, lon = 41.248644, -95.998878

    with open(os.path.join(os.path.dirname(__file__), "..", "data", "metadata.json")) as f:
        meta = json.load(f)

    # Find obliques at this location (same logic as viewer)
    matches = []
    for urn, m in meta.items():
        if m["type"] != "oblique" or not m.get("local_path") or not m.get("ground_footprint"):
            continue
        gj = json.loads(m["ground_footprint"]["geojson"]["value"])
        feat = gj["features"][0] if gj["type"] == "FeatureCollection" else gj
        geom = feat.get("geometry", feat)
        coords = geom["coordinates"][0] if geom["type"] == "MultiPolygon" else geom["coordinates"]
        ring = coords[0] if isinstance(coords[0][0], list) else coords

        inside = False
        for i in range(len(ring)):
            j = (i - 1) % len(ring)
            xi, yi = ring[i]
            xj, yj = ring[j]
            if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
                inside = not inside
        if inside:
            matches.append(m)

    print(f"Testing {len(matches)} oblique images at ({lat}, {lon})")
    print(f"Model: {MODEL}, resize: {TARGET_SIZE}px")
    print("=" * 70)

    results = []
    for i, m in enumerate(matches):
        path = m["local_path"]
        fname = os.path.basename(path)
        direction = m["direction"]

        print(f"\n[{i+1}/{len(matches)}] {direction:6s} | {fname[:60]}")

        t0 = time.time()
        try:
            img_b64 = encode_image(path)
            resp = query_qwen(img_b64)
            elapsed = time.time() - t0
            response_text = resp.get("response", "")

            has_poles = "no poles" not in response_text.lower()
            print(f"  Time: {elapsed:.1f}s | Poles: {'YES' if has_poles else 'NO'}")
            print(f"  Response: {response_text[:200]}")

            results.append({
                "urn": m["urn"],
                "direction": direction,
                "filename": fname,
                "has_poles": has_poles,
                "response": response_text,
                "elapsed": elapsed,
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED ({elapsed:.1f}s): {e}")
            results.append({
                "urn": m["urn"],
                "direction": direction,
                "filename": fname,
                "has_poles": None,
                "response": None,
                "error": str(e),
                "elapsed": elapsed,
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    detected = sum(1 for r in results if r.get("has_poles") is True)
    not_detected = sum(1 for r in results if r.get("has_poles") is False)
    failed = sum(1 for r in results if r.get("has_poles") is None)
    avg_time = sum(r["elapsed"] for r in results) / len(results) if results else 0
    print(f"  Poles detected: {detected}/{len(results)}")
    print(f"  No poles: {not_detected}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    print(f"  Avg time: {avg_time:.1f}s per image")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "test_qwen_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
