#!/usr/bin/env python3
"""
3D-based georeferencing using MASt3R point clouds.

Instead of a 2D homography (flat ground assumption), uses the dense 3D
reconstruction to establish a similarity transform from MASt3R's arbitrary
coordinate system to metric ground offsets (east_m, north_m).

Ground control points: each view's image center pixel corresponds to the
grid cell center GPS (lat, lon), since crops are centered on the cell.
"""
import math
import numpy as np
import torch


def fit_3d_to_ground(pts3d, orig_sizes, dir_list, cell_meta, cell_lat, cell_lon):
    """
    Fit a similarity transform from MASt3R 3D space → ground (east_m, north_m).

    Uses the image center of each view as a GCP. All views are cropped centered
    on (cell_lat, cell_lon), so each center pixel maps to (0, 0) in ground coords.

    But each view sees the ground from a different angle, so their center 3D points
    differ in MASt3R space. We also use the crop_radius to establish scale: the
    image edges are at ±crop_radius meters from center.

    Returns a transform function: f(p3d_tensor) -> (lat, lon)
    """
    gcps_3d = []
    gcps_ground = []  # (east_m, north_m) offsets from cell center

    for i, d in enumerate(dir_list):
        dm = cell_meta.get(d, {})
        azimuth = dm.get('azimuth', 0)
        crop_radius = dm.get('crop_radius', 50)

        pv = pts3d[i]
        h, w = pv.shape[:2]

        # Center pixel → ground (0, 0) — this is the strongest GCP
        center_3d = pv[h // 2, w // 2]
        if not torch.isnan(center_3d).any():
            gcps_3d.append(center_3d.detach().cpu().numpy())
            gcps_ground.append([0.0, 0.0])

        # Edge midpoints → known ground offsets based on azimuth + crop_radius
        az_rad = math.radians(azimuth)
        r = crop_radius
        edge_points = [
            # (pixel_y, pixel_x, east_m, north_m)
            (0, w // 2, math.sin(az_rad) * r, math.cos(az_rad) * r),        # top = forward
            (h // 2, w - 1, math.cos(az_rad) * r, -math.sin(az_rad) * r),   # right
            (h - 1, w // 2, -math.sin(az_rad) * r, -math.cos(az_rad) * r),  # bottom = backward
            (h // 2, 0, -math.cos(az_rad) * r, math.sin(az_rad) * r),       # left
        ]
        for py, px, east_m, north_m in edge_points:
            p3d = pv[py, px]
            if not torch.isnan(p3d).any():
                gcps_3d.append(p3d.detach().cpu().numpy())
                gcps_ground.append([east_m, north_m])

    if len(gcps_3d) < 4:
        return None  # Not enough GCPs

    gcps_3d = np.array(gcps_3d, dtype=np.float64)      # (N, 3)
    gcps_ground = np.array(gcps_ground, dtype=np.float64)  # (N, 2)

    # Fit: ground_2d = scale * R @ p3d[:2] + t
    # We project 3D→2D by dropping the vertical component, but we don't know
    # which axis is vertical in MASt3R space. Use least-squares affine fit
    # of all 3 MASt3R coords → 2 ground coords.

    # Affine model: [east, north] = A @ [x, y, z] + b
    # Solve via least squares: [x y z 1] @ [A b]^T = [east north]
    ones = np.ones((len(gcps_3d), 1))
    X = np.hstack([gcps_3d, ones])  # (N, 4)
    Y = gcps_ground                  # (N, 2)

    # Solve X @ W = Y  →  W = (X^T X)^-1 X^T Y
    W, residuals, rank, sv = np.linalg.lstsq(X, Y, rcond=None)
    # W is (4, 2): first 3 rows are the 3x2 transform matrix, last row is translation

    A = W[:3, :]  # (3, 2) — linear part
    b = W[3, :]   # (2,)   — translation

    # Compute fit quality
    predicted = X @ W
    errors_m = np.sqrt(np.sum((predicted - Y) ** 2, axis=1))
    mean_error = float(np.mean(errors_m))
    max_error = float(np.max(errors_m))

    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * math.cos(math.radians(cell_lat))

    def transform(p3d_tensor):
        """Convert MASt3R 3D point to GPS (lat, lon)."""
        p = p3d_tensor.detach().cpu().numpy().astype(np.float64)
        ground = A.T @ p + b  # [east_m, north_m]
        lat = cell_lat + ground[1] / m_per_deg_lat
        lon = cell_lon + ground[0] / m_per_deg_lon
        return float(lat), float(lon)

    return {
        'transform': transform,
        'A': A,
        'b': b,
        'n_gcps': len(gcps_3d),
        'mean_error_m': mean_error,
        'max_error_m': max_error,
    }
