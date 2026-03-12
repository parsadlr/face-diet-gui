"""
Runtime helpers for the pixel-per-degree mapping.

Typical usage
-------------
    from face_diet_gui.processing.pixel_degree_calibration.mapping_utils import (
        load_pixel_degree_mapping,
        evaluate_ppd,
        evaluate_ppd_xy,
        estimate_distance_from_face_size,
    )

    mapping = load_pixel_degree_mapping("path/to/mapping.json")
    ppd = evaluate_ppd(mapping, x=960, y=540)
    ppd_x, ppd_y = evaluate_ppd_xy(mapping, x=960, y=540)
    dist_m = estimate_distance_from_face_size(mapping, bbox_width_px=120,
                                              x_center=960, y_center=540,
                                              physical_face_width_m=0.16)
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


def load_pixel_degree_mapping(path: str) -> Dict:
    """Load a fitted pixel-per-degree mapping from JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _norm_coords(mapping: Dict, x: float, y: float) -> Tuple[float, float]:
    """Normalise pixel (x, y) to [-1, 1] using the stored image centre."""
    cx = mapping["center_x"]
    cy = mapping["center_y"]
    return (x - cx) / cx, (y - cy) / cy


def _design_row(
    x_norm: float, y_norm: float, exponents: List[Tuple[int, int]]
) -> List[float]:
    """Build a single polynomial feature vector."""
    return [(x_norm ** i) * (y_norm ** j) for i, j in exponents]


def _eval(coeffs: list, x_norm: float, y_norm: float, exponents) -> float:
    row = _design_row(x_norm, y_norm, exponents)
    return sum(c * r for c, r in zip(coeffs, row))


def evaluate_ppd(mapping: Dict, x: float, y: float) -> float:
    """
    Evaluate scalar (isotropic) pixels-per-degree at pixel (x, y).

    Derived as the geometric mean of the directional surfaces:
        ppd_scalar = sqrt(ppd_x * ppd_y)
    This is consistent with the original definition of radius_eq_px as
    sqrt(semi_major * semi_minor).
    """
    ppd_x, ppd_y = evaluate_ppd_xy(mapping, x, y)
    return float((ppd_x * ppd_y) ** 0.5) if ppd_x > 0 and ppd_y > 0 else 0.0


def evaluate_ppd_xy(mapping: Dict, x: float, y: float) -> Tuple[float, float]:
    """
    Evaluate anisotropic pixels-per-degree at pixel (x, y).

    Returns
    -------
    (ppd_x, ppd_y) : horizontal and vertical pixels per degree.
    """
    x_n, y_n = _norm_coords(mapping, x, y)
    exponents = mapping["exponents"]
    ppd_x = _eval(mapping["coefficients_x"], x_n, y_n, exponents)
    ppd_y = _eval(mapping["coefficients_y"], x_n, y_n, exponents)
    return ppd_x, ppd_y


def estimate_distance_from_face_size(
    mapping: Dict,
    bbox_width_px: float,
    x_center: float,
    y_center: float,
    physical_face_width_m: float,
) -> float:
    """
    Estimate viewing distance from the apparent bounding-box width of a face.

    Uses the horizontal PPD at (x_center, y_center) and a small-angle
    approximation:

        theta_deg = bbox_width_px / ppd_x
        distance  = physical_face_width_m / tan(theta_rad)
                  ≈ physical_face_width_m / theta_rad   (small angles)

    Parameters
    ----------
    mapping : Dict
        Loaded mapping (from load_pixel_degree_mapping).
    bbox_width_px : float
        Bounding-box width in pixels.
    x_center, y_center : float
        Centre of the bounding box in pixel coordinates.
    physical_face_width_m : float
        Assumed physical face width in metres.

    Returns
    -------
    float
        Estimated distance in metres (inf if ppd is non-positive).
    """
    ppd_x, _ = evaluate_ppd_xy(mapping, x_center, y_center)
    if ppd_x <= 0:
        return float("inf")

    theta_deg = bbox_width_px / ppd_x
    theta_rad = math.radians(theta_deg)
    if theta_rad <= 0:
        return float("inf")

    return physical_face_width_m / theta_rad
