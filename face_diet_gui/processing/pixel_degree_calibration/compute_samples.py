"""
Compute pixel-per-degree (PPD) samples from detected calibration ellipses.

For each detected ellipse the known physical geometry (target diameter +
viewing distance) gives the angular radius in degrees.  We then project the
ellipse semi-axes onto the image x and y axes to get direction-specific PPD.

Semi-axis projection onto image axes
-------------------------------------
OpenCV fitEllipse returns angle = rotation of the major axis from the x-axis
(degrees, counterclockwise).  The image-aligned projections are:

    extent_x = sqrt((semi_major * cos a)² + (semi_minor * sin a)²)
    extent_y = sqrt((semi_major * sin a)² + (semi_minor * cos a)²)

where a = radians(angle_deg).  These are the x- and y-half-widths of the
tightest bounding rectangle around the ellipse.

Three PPD values are stored per sample:
    pixels_per_degree       – scalar, from area-equivalent radius
    pixels_per_degree_x     – horizontal direction
    pixels_per_degree_y     – vertical direction
"""
import argparse
import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np


def _angular_radius_deg(diameter_m: float, distance_m: float) -> float:
    """Exact angular radius (degrees) of the physical circle at given distance."""
    r = diameter_m / 2.0
    return math.degrees(2.0 * math.atan(r / (2.0 * distance_m)))


def _ellipse_projected_extents(
    semi_major: float, semi_minor: float, angle_deg: float
) -> tuple:
    """Return (extent_x, extent_y): half-widths of the axis-aligned bounding box."""
    a = math.radians(angle_deg)
    extent_x = math.sqrt((semi_major * math.cos(a)) ** 2 + (semi_minor * math.sin(a)) ** 2)
    extent_y = math.sqrt((semi_major * math.sin(a)) ** 2 + (semi_minor * math.cos(a)) ** 2)
    return extent_x, extent_y


def compute_samples(
    masks_dir: str,
    circle_diameter_m: float,
    distance_m: float,
    output_path: str,
) -> None:
    """
    Read detections.json from masks_dir, compute PPD samples and write JSON.
    """
    masks_path = Path(masks_dir)
    meta_path = masks_path / "detections.json"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"detections.json not found in {masks_path}. "
            "Run detect_target.py first."
        )

    with meta_path.open("r", encoding="utf-8") as f:
        records: List[dict] = json.load(f)

    theta_deg = _angular_radius_deg(circle_diameter_m, distance_m)
    print(f"Angular radius of target: {theta_deg:.4f} deg")

    samples = []
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    for rec in records:
        if not rec.get("found"):
            continue

        cx = rec.get("center_x")
        cy = rec.get("center_y")
        semi_major = rec.get("semi_major_px")
        semi_minor = rec.get("semi_minor_px")
        angle_deg = rec.get("angle_deg")
        radius_eq = rec.get("radius_eq_px")

        if any(v is None for v in (cx, cy, semi_major, semi_minor, angle_deg, radius_eq)):
            continue

        if image_width is None:
            image_width = rec.get("width")
            image_height = rec.get("height")

        extent_x, extent_y = _ellipse_projected_extents(semi_major, semi_minor, angle_deg)

        samples.append(
            {
                "image": rec.get("image"),
                "center_x": float(cx),
                "center_y": float(cy),
                "semi_major_px": float(semi_major),
                "semi_minor_px": float(semi_minor),
                "angle_deg": float(angle_deg),
                "radius_eq_px": float(radius_eq),
                "pixels_per_degree": float(radius_eq / theta_deg),
                "pixels_per_degree_x": float(extent_x / theta_deg),
                "pixels_per_degree_y": float(extent_y / theta_deg),
            }
        )

    if not samples:
        print("No valid detections to compute samples from.")
        return

    output = {
        "image_width": int(image_width) if image_width is not None else None,
        "image_height": int(image_height) if image_height is not None else None,
        "circle_diameter_m": float(circle_diameter_m),
        "distance_m": float(distance_m),
        "angular_radius_deg": float(theta_deg),
        "num_samples": len(samples),
        "samples": samples,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Computed {len(samples)} PPD samples → {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute pixel-per-degree samples from detected calibration ellipses."
    )
    p.add_argument("--masks-dir", required=True, help="Directory containing detections.json.")
    p.add_argument("--diameter-m", type=float, default=0.19, help="Target diameter in metres (default 0.19).")
    p.add_argument("--distance-m", type=float, default=1.0, help="Camera-to-target distance in metres (default 1.0).")
    p.add_argument("--output", required=True, help="Output JSON file for PPD samples.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    compute_samples(
        masks_dir=args.masks_dir,
        circle_diameter_m=args.diameter_m,
        distance_m=args.distance_m,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
