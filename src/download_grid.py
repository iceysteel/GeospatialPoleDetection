#!/usr/bin/env python3
"""
Download a dense grid of oblique crops for any area.
Reusable — specify center lat/lon and output directory.

Usage:
  python src/download_grid.py --lat 41.2486 --lon -95.9929 --output data/holdout_grid
  python src/download_grid.py --lat 41.248644 --lon -95.998878 --output data/testarea_grid
"""
import sys, os, json, math, argparse
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from auth import auth_headers
from ratelimit import images_limiter
import requests

BASE_URL = os.environ.get('EAGLEVIEW_BASE_URL', 'https://sandbox.apis.eagleview.com')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

GRID_SPACING = 80  # meters
CROP_RADIUS = 50   # meters
RADIUS_LAT = 0.0018
RADIUS_LON = 0.0024
DIRECTIONS = ['north', 'east', 'south', 'west']


def find_best_oblique_urn(meta, lat, lon, direction):
    best_urn, best_gsd = None, 999
    for urn, m in meta.items():
        if m['type'] != 'oblique' or m['direction'] != direction or not m.get('ground_footprint'):
            continue
        gj = json.loads(m['ground_footprint']['geojson']['value'])
        feat = gj['features'][0] if gj['type'] == 'FeatureCollection' else gj
        geom = feat.get('geometry', feat)
        coords = geom['coordinates'][0] if geom['type'] == 'MultiPolygon' else geom['coordinates']
        ring = coords[0] if isinstance(coords[0][0], list) else coords
        inside = False
        for i in range(len(ring)):
            j = (i - 1) % len(ring)
            xi, yi = ring[i]; xj, yj = ring[j]
            if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
                inside = not inside
        if not inside:
            continue
        gsd = m.get('calculated_gsd', {}).get('value', 999)
        if gsd < best_gsd:
            best_gsd = gsd
            best_urn = urn
    return best_urn, best_gsd


def download_crop(urn, meta, lat, lon, out_path, radius=CROP_RADIUS):
    if os.path.exists(out_path):
        return True
    m = meta[urn]
    max_zoom = m.get('zoom_range', {}).get('maximum_zoom_level')
    params = {
        'center.x': lon, 'center.y': lat,
        'center.radius': radius, 'epsg': 'EPSG:4326',
        'format': 'IMAGE_FORMAT_PNG',
    }
    if max_zoom:
        params['zoom'] = max_zoom
    url = f'{BASE_URL}/imagery/v3/images/{urn}/location'
    for r in [radius, 35, 25]:
        params['center.radius'] = r
        images_limiter.wait()
        resp = requests.get(url, params=params, headers=auth_headers())
        if resp.status_code != 413:
            break
    if resp.status_code == 200 and len(resp.content) > 1000:
        with open(out_path, 'wb') as f:
            f.write(resp.content)
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lat', type=float, required=True)
    parser.add_argument('--lon', type=float, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='r')
    args = parser.parse_args()

    grid_dir = args.output
    os.makedirs(grid_dir, exist_ok=True)

    with open(os.path.join(DATA_DIR, 'metadata.json')) as f:
        meta = json.load(f)

    center_lat, center_lon = args.lat, args.lon
    area = (center_lat - RADIUS_LAT, center_lon - RADIUS_LON,
            center_lat + RADIUS_LAT, center_lon + RADIUS_LON)

    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * math.cos(math.radians(center_lat))
    spacing_lat = GRID_SPACING / m_per_deg_lat
    spacing_lon = GRID_SPACING / m_per_deg_lon

    rows = int((area[2] - area[0]) / spacing_lat) + 1
    cols = int((area[3] - area[1]) / spacing_lon) + 1

    print(f"Grid: {rows}x{cols} = {rows*cols} cells, {GRID_SPACING}m spacing")
    print(f"Center: ({center_lat}, {center_lon})")
    print(f"Area: lat [{area[0]:.6f}, {area[2]:.6f}] lon [{area[1]:.6f}, {area[3]:.6f}]")

    grid = []
    downloaded = 0

    for r in range(rows):
        for c in range(cols):
            lat = area[0] + r * spacing_lat
            lon = area[1] + c * spacing_lon
            cell_name = f'{args.prefix}{r}_c{c}'
            cell_dir = os.path.join(grid_dir, cell_name)
            os.makedirs(cell_dir, exist_ok=True)

            cell = {
                'name': cell_name, 'lat': lat, 'lon': lon,
                'row': r, 'col': c, 'images': {},
            }

            for d in DIRECTIONS:
                urn, gsd = find_best_oblique_urn(meta, lat, lon, d)
                if not urn:
                    continue
                out_path = os.path.join(cell_dir, f'{d}.png')
                if download_crop(urn, meta, lat, lon, out_path):
                    cell['images'][d] = os.path.abspath(out_path)
                    m = meta[urn]
                    cell[f'{d}_meta'] = {
                        'urn': urn,
                        'azimuth': m.get('look_at', {}).get('azimuth', 0),
                        'gsd': m.get('calculated_gsd', {}).get('value', 0.04),
                        'elevation': m.get('look_at', {}).get('elevation', 45),
                        'crop_radius': CROP_RADIUS,
                    }
                    downloaded += 1

            n_views = len(cell['images'])
            if n_views >= 2:
                grid.append(cell)

            total = (r * cols + c + 1)
            if total % 10 == 0:
                print(f"  [{total}/{rows*cols}] {len(grid)} cells, {downloaded} images downloaded", flush=True)

    with open(os.path.join(grid_dir, 'index.json'), 'w') as f:
        json.dump(grid, f, indent=2)

    print(f"\nDone: {len(grid)} cells with 2+ views, {downloaded} images total")
    print(f"Saved to {grid_dir}/index.json")


if __name__ == '__main__':
    main()
