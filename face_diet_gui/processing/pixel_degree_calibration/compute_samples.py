"""
Compute pixel-per-degree (PPD) samples from calibration mask metadata.

The known physical geometry (target diameter + viewing distance) gives the
angular radius in degrees.  For each detection:

* **Preferred (GUI / new detections.json):** horizontal and vertical semi-extents
  come from the mask axis-aligned bounding box — half of (right − left) and
  half of (bottom − top) in pixels — and map to ``pixels_per_degree_x`` and
  ``pixels_per_degree_y`` respectively.  Sample position uses the bbox centre.

* **Legacy:** if ``mask_extent_x_px`` / ``mask_extent_y_px`` are absent, ellipse
  semi-axes are projected onto image x and y:

    extent_x = sqrt((semi_major * cos a)² + (semi_minor * sin a)²)
    extent_y = sqrt((semi_major * sin a)² + (semi_minor * cos a)²)

Three PPD values are stored per sample:
    pixels_per_degree       – scalar, sqrt(extent_x * extent_y) / angular_radius
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
        if cx is None or cy is None:
            continue

        if image_width is None:
            image_width = rec.get("width")
            image_height = rec.get("height")

        mask_ex = rec.get("mask_extent_x_px")
        mask_ey = rec.get("mask_extent_y_px")
        if mask_ex is not None and mask_ey is not None:
            extent_x = float(mask_ex)
            extent_y = float(mask_ey)
            radius_eq = float(rec.get("radius_eq_px") or math.sqrt(extent_x * extent_y))
            semi_major = float(rec.get("semi_major_px") or extent_x)
            semi_minor = float(rec.get("semi_minor_px") or extent_y)
            angle_deg = float(rec.get("angle_deg") or 0.0)
        else:
            semi_major = rec.get("semi_major_px")
            semi_minor = rec.get("semi_minor_px")
            angle_deg = rec.get("angle_deg")
            radius_eq = rec.get("radius_eq_px")
            if any(
                v is None for v in (semi_major, semi_minor, angle_deg, radius_eq)
            ):
                continue
            semi_major = float(semi_major)
            semi_minor = float(semi_minor)
            angle_deg = float(angle_deg)
            radius_eq = float(radius_eq)
            extent_x, extent_y = _ellipse_projected_extents(
                semi_major, semi_minor, angle_deg
            )

        entry = {
            "image": rec.get("image"),
            "center_x": float(cx),
            "center_y": float(cy),
            "semi_major_px": semi_major,
            "semi_minor_px": semi_minor,
            "angle_deg": angle_deg,
            "radius_eq_px": radius_eq,
            "pixels_per_degree": float(radius_eq / theta_deg),
            "pixels_per_degree_x": float(extent_x / theta_deg),
            "pixels_per_degree_y": float(extent_y / theta_deg),
        }
        for key in (
            "mask_xmin_px",
            "mask_xmax_px",
            "mask_ymin_px",
            "mask_ymax_px",
            "mask_width_px",
            "mask_height_px",
            "mask_extent_x_px",
            "mask_extent_y_px",
        ):
            if key in rec and rec[key] is not None:
                entry[key] = rec[key]
        samples.append(entry)

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
        description="Compute pixel-per-degree samples from mask bbox or ellipse metadata."
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
