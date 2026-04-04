"""
Utilities for working with oblique imagery:
- Find which downloaded images cover a GPS point
- Convert GPS to pixel coordinates within an oblique image
- Crop a region around a GPS point from an existing image
- Convert pixel coordinates to GPS (georeferencing)
"""
import json
import math
import os
import numpy as np
import cv2
from PIL import Image


def point_in_polygon(lat, lon, ring):
    """Ray casting test for point-in-polygon."""
    inside = False
    for i in range(len(ring)):
        j = (i - 1) % len(ring)
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
    return inside


def parse_footprint(meta):
    """Parse ground footprint polygon from image metadata."""
    try:
        gj = json.loads(meta['ground_footprint']['geojson']['value'])
        feat = gj['features'][0] if gj['type'] == 'FeatureCollection' else gj
        geom = feat.get('geometry', feat)
        coords = geom['coordinates'][0] if geom['type'] == 'MultiPolygon' else geom['coordinates']
        ring = coords[0] if isinstance(coords[0][0], list) else coords
        return ring  # list of [lon, lat] pairs
    except:
        return None


def build_homography(img_w, img_h, query_lat, query_lon, azimuth, crop_radius=50):
    """
    Build a homography that maps pixel coordinates to ground offset (east_m, north_m)
    from the image center (query_lat, query_lon).

    The image is a crop of an oblique view:
    - Center pixel = (query_lat, query_lon) on the ground
    - Crop covers crop_radius meters in each direction on the ground
    - 'Up' in image = forward along azimuth direction on ground
    - 'Right' in image = 90° clockwise from azimuth on ground
    """
    az_rad = math.radians(azimuth)
    r = crop_radius

    # 4 GCPs: edge midpoints → ground positions
    gcp_px = np.float32([
        [img_w / 2, 0],          # top center
        [img_w, img_h / 2],      # right center
        [img_w / 2, img_h],      # bottom center
        [0, img_h / 2],          # left center
    ])
    gcp_ground = np.float32([
        [math.sin(az_rad) * r, math.cos(az_rad) * r],      # forward
        [math.cos(az_rad) * r, -math.sin(az_rad) * r],     # right
        [-math.sin(az_rad) * r, -math.cos(az_rad) * r],    # backward
        [-math.cos(az_rad) * r, math.sin(az_rad) * r],     # left
    ])

    H_px_to_ground, _ = cv2.findHomography(gcp_px, gcp_ground)
    H_ground_to_px, _ = cv2.findHomography(gcp_ground, gcp_px)

    return H_px_to_ground, H_ground_to_px


def pixel_to_gps(px, py, img_w, img_h, query_lat, query_lon, azimuth, crop_radius=50):
    """Convert pixel coordinates to GPS using homography."""
    H, _ = build_homography(img_w, img_h, query_lat, query_lon, azimuth, crop_radius)

    det_px = np.float32([[px, py]]).reshape(-1, 1, 2)
    det_gnd = cv2.perspectiveTransform(det_px, H)
    de_m, dn_m = det_gnd[0][0]

    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * math.cos(math.radians(query_lat))
    det_lat = query_lat + dn_m / m_per_deg_lat
    det_lon = query_lon + de_m / m_per_deg_lon

    return det_lat, det_lon


def gps_to_pixel(lat, lon, img_w, img_h, query_lat, query_lon, azimuth, crop_radius=50):
    """Convert GPS to pixel coordinates within the image using inverse homography."""
    _, H_inv = build_homography(img_w, img_h, query_lat, query_lon, azimuth, crop_radius)

    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * math.cos(math.radians(query_lat))
    de_m = (lon - query_lon) * m_per_deg_lon
    dn_m = (lat - query_lat) * m_per_deg_lat

    gnd_pt = np.float32([[de_m, dn_m]]).reshape(-1, 1, 2)
    px_pt = cv2.perspectiveTransform(gnd_pt, H_inv)
    px, py = px_pt[0][0]

    return int(round(px)), int(round(py))


def find_images_at_point(lat, lon, metadata, direction=None, max_dist_m=45):
    """
    Find all downloaded oblique images whose crop center is within max_dist_m
    of the given GPS point. This ensures the point actually falls within
    the downloaded crop (not just the full image footprint).

    Returns list of (urn, metadata) sorted by distance (closest first).
    """
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * math.cos(math.radians(lat))

    results = []
    for urn, m in metadata.items():
        if m['type'] != 'oblique' or not m.get('local_path'):
            continue
        if direction and m['direction'] != direction:
            continue
        if not os.path.exists(m['local_path']):
            continue

        qp = m.get('query_point')
        if not qp:
            continue

        dlat = (lat - qp['lat']) * m_per_deg_lat
        dlon = (lon - qp['lon']) * m_per_deg_lon
        dist = math.sqrt(dlat**2 + dlon**2)

        if dist <= max_dist_m:
            gsd = m.get('calculated_gsd', {}).get('value', 999)
            results.append((urn, m, dist))

    results.sort(key=lambda x: x[2])  # closest first
    return [(urn, m) for urn, m, _ in results]


def crop_at_point(lat, lon, metadata, direction, crop_size=800):
    """
    Find a downloaded image covering (lat, lon) from the given direction,
    and return a crop centered on that point.

    Returns: (crop_image, full_image, pixel_x, pixel_y, image_meta) or None
    """
    images = find_images_at_point(lat, lon, metadata, direction=direction)
    if not images:
        return None

    urn, m = images[0]  # best GSD
    img = Image.open(m['local_path']).convert('RGB')
    img_w, img_h = img.size

    qp = m['query_point']
    azimuth = m.get('look_at', {}).get('azimuth', 0)
    download_radius = m.get('download_radius', 50)

    # Convert target GPS to pixel in this image
    px, py = gps_to_pixel(lat, lon, img_w, img_h, qp['lat'], qp['lon'], azimuth, download_radius)

    # Check if pixel is within image bounds
    if px < 0 or px >= img_w or py < 0 or py >= img_h:
        # Try next image
        if len(images) > 1:
            urn, m = images[1]
            img = Image.open(m['local_path']).convert('RGB')
            img_w, img_h = img.size
            qp = m['query_point']
            azimuth = m.get('look_at', {}).get('azimuth', 0)
            download_radius = m.get('download_radius', 50)
            px, py = gps_to_pixel(lat, lon, img_w, img_h, qp['lat'], qp['lon'], azimuth, download_radius)
            if px < 0 or px >= img_w or py < 0 or py >= img_h:
                return None
        else:
            return None

    # Crop around the pixel
    half = crop_size // 2
    x1 = max(0, px - half)
    y1 = max(0, py - half)
    x2 = min(img_w, px + half)
    y2 = min(img_h, py + half)

    crop = img.crop((x1, y1, x2, y2))

    return crop, img, px, py, m
