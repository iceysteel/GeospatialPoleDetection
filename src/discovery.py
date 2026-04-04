import json
import math
import os
import requests
from auth import auth_headers
from ratelimit import discovery_limiter, retry_with_backoff

BASE_URL = os.environ.get("EAGLEVIEW_BASE_URL", "https://sandbox.apis.eagleview.com")

# Sandbox bounding box: Omaha, NE
BBOX = {
    "min_lon": -96.00532698173473,
    "min_lat": 41.24140396772262,
    "max_lon": -95.97589954958912,
    "max_lat": 41.25672882015283,
}

GRID_SPACING_M = 130  # meters between query points
QUERY_RADIUS_M = 75   # max radius per query


def _lat_lon_grid(bbox, spacing_m):
    """Generate a grid of (lon, lat) points across the bounding box."""
    lat_center = (bbox["min_lat"] + bbox["max_lat"]) / 2
    m_per_deg_lat = 111_320
    m_per_deg_lon = 111_320 * math.cos(math.radians(lat_center))

    lat_step = spacing_m / m_per_deg_lat
    lon_step = spacing_m / m_per_deg_lon

    points = []
    lat = bbox["min_lat"]
    while lat <= bbox["max_lat"]:
        lon = bbox["min_lon"]
        while lon <= bbox["max_lon"]:
            points.append((lon, lat))
            lon += lon_step
        lat += lat_step

    return points


def _make_request_body(lon, lat):
    """Build the rank/location request body for a single point."""
    geojson_value = json.dumps({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": None,
    })
    return {
        "center": {
            "point": {
                "geojson": {"value": geojson_value, "epsg": "EPSG:4326"}
            },
            "radius_in_meters": QUERY_RADIUS_M,
        },
        "view": {
            "obliques": {
                "cardinals": {"north": True, "east": True, "south": True, "west": True}
            },
            "orthos": {},
            "max_images_per_view": 1,
        },
        "response_props": {
            "calculated_gsd": True,
            "zoom_range": True,
            "ground_footprint": True,
            "look_at": True,
            "image_resources": {},
        },
    }


def discover_imagery(on_progress=None):
    """
    Grid the sandbox bounding box and query rank/location for each point.
    Returns a dict of unique images keyed by URN, with metadata.
    """
    grid = _lat_lon_grid(BBOX, GRID_SPACING_M)
    print(f"[discovery] Querying {len(grid)} grid points (~{len(grid) / 4.5:.0f}s at 4.5 rps)...")

    url = f"{BASE_URL}/imagery/v3/discovery/rank/location"
    images = {}  # urn -> metadata

    for i, (lon, lat) in enumerate(grid):
        if on_progress:
            on_progress(i + 1, len(grid))

        headers = auth_headers()
        headers["Content-Type"] = "application/json"
        body = _make_request_body(lon, lat)

        discovery_limiter.wait()
        resp = retry_with_backoff(
            lambda: requests.post(url, json=body, headers=headers)
        )
        data = resp.json()

        for capture_group in data.get("captures", []):
            capture_meta = capture_group.get("capture", {})

            # Process oblique images (N/E/S/W)
            obliques = capture_group.get("obliques", {})
            for direction in ("north", "east", "south", "west"):
                dir_data = obliques.get(direction, {})
                for img in dir_data.get("images", []):
                    urn = img.get("urn")
                    if urn and urn not in images:
                        images[urn] = {
                            "urn": urn,
                            "type": "oblique",
                            "direction": direction,
                            "capture": capture_meta,
                            "query_point": {"lon": lon, "lat": lat},
                            **{k: v for k, v in img.items() if k != "urn"},
                        }

            # Process ortho images
            orthos = capture_group.get("orthos", {})
            for img in orthos.get("images", []):
                urn = img.get("urn")
                if urn and urn not in images:
                    images[urn] = {
                        "urn": urn,
                        "type": "ortho",
                        "direction": "top",
                        "capture": capture_meta,
                        "query_point": {"lon": lon, "lat": lat},
                        **{k: v for k, v in img.items() if k != "urn"},
                    }

    print(f"\n[discovery] Found {len(images)} unique images")
    oblique_count = sum(1 for v in images.values() if v["type"] == "oblique")
    ortho_count = sum(1 for v in images.values() if v["type"] == "ortho")
    print(f"  Oblique: {oblique_count}  Ortho: {ortho_count}")

    return images
