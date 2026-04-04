#!/usr/bin/env python3
"""
Download a dense grid of oblique crops covering the test area.
80m spacing, 50m radius = 20m overlap between adjacent crops.
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from auth import auth_headers
from ratelimit import images_limiter
import requests

BASE_URL = os.environ.get('EAGLEVIEW_BASE_URL', 'https://sandbox.apis.eagleview.com')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
GRID_DIR = os.path.join(DATA_DIR, 'testarea_grid')

# Focus area
FOCUS_LAT, FOCUS_LON = 41.248644, -95.998878
RADIUS_LAT, RADIUS_LON = 0.0018, 0.0024
AREA = (FOCUS_LAT - RADIUS_LAT, FOCUS_LON - RADIUS_LON,
        FOCUS_LAT + RADIUS_LAT, FOCUS_LON + RADIUS_LON)

GRID_SPACING = 80  # meters
CROP_RADIUS = 50   # meters
DIRECTIONS = ['north', 'east', 'south', 'west']


def find_best_oblique_urn(meta, lat, lon, direction):
    """Find the best (lowest GSD) oblique image URN covering a point."""
    best_urn = None
    best_gsd = 999
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
        if inside:
            gsd = m.get('calculated_gsd', {}).get('value', 999)
            if gsd < best_gsd:
                best_gsd = gsd
                best_urn = urn
    return best_urn


def main():
    with open(os.path.join(DATA_DIR, 'metadata.json')) as f:
        meta = json.load(f)

    # Generate grid
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * math.cos(math.radians(FOCUS_LAT))
    lat_step = GRID_SPACING / m_per_deg_lat
    lon_step = GRID_SPACING / m_per_deg_lon

    grid = []
    lat = AREA[0]
    row = 0
    while lat <= AREA[2]:
        lon = AREA[1]
        col = 0
        while lon <= AREA[3]:
            grid.append({'lat': round(lat, 6), 'lon': round(lon, 6), 'row': row, 'col': col,
                        'name': f'r{row}_c{col}'})
            lon += lon_step
            col += 1
        lat += lat_step
        row += 1

    print(f"Grid: {len(grid)} points at {GRID_SPACING}m spacing")
    print(f"Area: {AREA}")

    os.makedirs(GRID_DIR, exist_ok=True)

    # Download crops
    downloaded = 0
    cached = 0
    failed = 0
    index = []

    for i, gp in enumerate(grid):
        lat, lon, name = gp['lat'], gp['lon'], gp['name']
        cell_dir = os.path.join(GRID_DIR, name)
        os.makedirs(cell_dir, exist_ok=True)

        cell_info = {'name': name, 'lat': lat, 'lon': lon, 'row': gp['row'], 'col': gp['col'], 'images': {}}

        for d in DIRECTIONS:
            out_path = os.path.join(cell_dir, f'{d}.png')
            if os.path.exists(out_path):
                cell_info['images'][d] = out_path
                cached += 1
                continue

            urn = find_best_oblique_urn(meta, lat, lon, d)
            if not urn:
                continue

            m = meta[urn]
            max_zoom = m.get('zoom_range', {}).get('maximum_zoom_level')
            params = {
                'center.x': lon, 'center.y': lat,
                'center.radius': CROP_RADIUS, 'epsg': 'EPSG:4326',
                'format': 'IMAGE_FORMAT_PNG',
            }
            if max_zoom:
                params['zoom'] = max_zoom

            url = f'{BASE_URL}/imagery/v3/images/{urn}/location'
            for radius in [CROP_RADIUS, 35, 25]:
                params['center.radius'] = radius
                images_limiter.wait()
                resp = requests.get(url, params=params, headers=auth_headers())
                if resp.status_code != 413:
                    break

            if resp.status_code == 200:
                with open(out_path, 'wb') as f:
                    f.write(resp.content)
                cell_info['images'][d] = out_path
                # Save metadata for this crop
                cell_info[f'{d}_meta'] = {
                    'urn': urn, 'gsd': m.get('calculated_gsd', {}).get('value'),
                    'azimuth': m.get('look_at', {}).get('azimuth'),
                    'elevation': m.get('look_at', {}).get('elevation'),
                    'crop_radius': params['center.radius'],
                }
                downloaded += 1
            else:
                failed += 1

        index.append(cell_info)
        total = downloaded + cached + failed
        print(f"  [{i+1}/{len(grid)}] {name}: {len(cell_info['images'])} dirs | total: {downloaded} new, {cached} cached, {failed} failed")

    # Save index
    with open(os.path.join(GRID_DIR, 'index.json'), 'w') as f:
        json.dump(index, f, indent=2, default=str)

    print(f"\nDone! {downloaded} downloaded, {cached} cached, {failed} failed")
    print(f"Saved to {GRID_DIR}/")


if __name__ == '__main__':
    main()
