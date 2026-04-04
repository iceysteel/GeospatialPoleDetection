import json
import os
import re
import requests
from auth import auth_headers
from ratelimit import images_limiter, retry_with_backoff

BASE_URL = os.environ.get("EAGLEVIEW_BASE_URL", "https://sandbox.apis.eagleview.com")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Max payload is ~10MB. Try radii from large to small at max zoom.
RADIUS_ATTEMPTS = [50, 35, 25]


def _safe_filename(urn):
    """Convert a URN to a filesystem-safe filename."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", urn)


def _download_image_at_location(image_meta):
    """Download an image at max zoom, reducing radius if payload too large."""
    urn = image_meta["urn"]
    img_type = image_meta["type"]
    direction = image_meta["direction"]

    # Determine save path
    if img_type == "oblique":
        save_dir = os.path.join(DATA_DIR, "oblique", direction)
    else:
        save_dir = os.path.join(DATA_DIR, "ortho")
    os.makedirs(save_dir, exist_ok=True)

    filename = _safe_filename(urn) + ".png"
    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path):
        return save_path  # already downloaded

    # Build base request params
    query_point = image_meta.get("query_point", {})
    lon = query_point.get("lon", -95.98608)
    lat = query_point.get("lat", 41.25056)

    zoom_range = image_meta.get("zoom_range", {})
    max_zoom = zoom_range.get("maximum_zoom_level") if zoom_range else None

    url = f"{BASE_URL}/imagery/v3/images/{urn}/location"

    # Try max zoom with decreasing radius to stay under payload limit
    for radius in RADIUS_ATTEMPTS:
        params = {
            "center.x": lon,
            "center.y": lat,
            "center.radius": radius,
            "epsg": "EPSG:4326",
            "format": "IMAGE_FORMAT_PNG",
        }
        if max_zoom is not None:
            params["zoom"] = max_zoom

        images_limiter.wait()
        resp = requests.get(url, params=params, headers=auth_headers())

        if resp.status_code == 413:
            continue  # payload too large, try smaller radius
        if resp.status_code == 429 or resp.status_code >= 500:
            # Use retry logic for transient errors
            resp = retry_with_backoff(
                lambda: requests.get(url, params=params, headers=auth_headers())
            )
        resp.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(resp.content)
        image_meta["download_radius"] = radius
        image_meta["download_zoom"] = max_zoom
        return save_path

    # All radii failed at max zoom — shouldn't happen but raise if it does
    raise RuntimeError(f"All radius attempts failed for {urn}")


def download_all(images, on_progress=None):
    """
    Download all discovered images.
    images: dict of urn -> metadata from discovery.
    Returns updated metadata with local file paths.
    """
    total = len(images)
    print(f"[download] Downloading {total} images at max zoom (PNG)...")

    results = {}
    for i, (urn, meta) in enumerate(images.items()):
        if on_progress:
            on_progress(i + 1, total)

        try:
            path = _download_image_at_location(meta)
            meta["local_path"] = path
            results[urn] = meta
        except Exception as e:
            print(f"\n  [{i+1}/{total}] {meta['type']}/{meta['direction']}: FAILED - {e}")
            meta["local_path"] = None
            meta["error"] = str(e)
            results[urn] = meta

    # Save metadata
    metadata_path = os.path.join(DATA_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[download] Metadata saved to {metadata_path}")

    success = sum(1 for m in results.values() if m.get("local_path"))
    failed = total - success
    print(f"[download] Done: {success} succeeded, {failed} failed")

    return results
