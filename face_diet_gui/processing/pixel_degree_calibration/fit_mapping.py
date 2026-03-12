"""
Fit two 2D polynomial surfaces ppd_x(x, y) and ppd_y(x, y) from calibration
samples and save to a compact JSON model.

    coefficients_x  – pixels-per-degree in the horizontal direction
    coefficients_y  – pixels-per-degree in the vertical direction

A scalar (isotropic) PPD can always be derived on-the-fly as
    ppd_scalar = sqrt(ppd_x * ppd_y)
and does not need its own fitted surface.

Pixel coordinates are normalised to [-1, 1] using the image centre before
fitting so the polynomial coefficients are numerically well-conditioned.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _design_matrix(
    x_norm: np.ndarray, y_norm: np.ndarray, order: int
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Build a 2D polynomial design matrix of total degree <= order.

    Returns (X, exponents) where column k corresponds to x^i * y^j for
    exponents[k] = (i, j).
    """
    terms: List[Tuple[int, int]] = []
    cols = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            terms.append((i, j))
            cols.append((x_norm ** i) * (y_norm ** j))
    return np.stack(cols, axis=1), terms


def _fit_and_rmse(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, float]:
    coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
    if residuals.size > 0 and len(y) > 0:
        rmse = float(np.sqrt(residuals[0] / len(y)))
    else:
        # Compute manually when lstsq doesn't return residuals (underdetermined)
        rmse = float(np.sqrt(np.mean((X @ coeffs - y) ** 2)))
    return coeffs, rmse


def fit_mapping(samples_path: str, output_path: str, order: int = 2) -> None:
    """
    Load PPD samples and fit two 2D polynomial surfaces (ppd_x, ppd_y).
    """
    samples_file = Path(samples_path)
    if not samples_file.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_file}")

    with samples_file.open("r", encoding="utf-8") as f:
        data: Dict = json.load(f)

    samples: List[Dict] = data.get("samples", [])
    if not samples:
        print(f"No samples in {samples_file}")
        return

    width = data.get("image_width")
    height = data.get("image_height")
    if width is None or height is None:
        raise ValueError("image_width / image_height missing from samples file")

    xs = np.array([s["center_x"] for s in samples], dtype=np.float64)
    ys = np.array([s["center_y"] for s in samples], dtype=np.float64)
    ppd_x = np.array([s["pixels_per_degree_x"] for s in samples], dtype=np.float64)
    ppd_y = np.array([s["pixels_per_degree_y"] for s in samples], dtype=np.float64)

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    x_norm = (xs - cx) / cx
    y_norm = (ys - cy) / cy

    X, exponents = _design_matrix(x_norm, y_norm, order=order)

    coeffs_x, rmse_x = _fit_and_rmse(X, ppd_x)
    coeffs_y, rmse_y = _fit_and_rmse(X, ppd_y)

    print(
        f"Fit (order={order}, n={len(samples)} samples):\n"
        f"  x RMSE = {rmse_x:.4f} px/deg\n"
        f"  y RMSE = {rmse_y:.4f} px/deg"
    )

    mapping = {
        "image_width": int(width),
        "image_height": int(height),
        "center_x": float(cx),
        "center_y": float(cy),
        "poly_order": int(order),
        "exponents": exponents,
        "coefficients_x": coeffs_x.tolist(),
        "coefficients_y": coeffs_y.tolist(),
        "circle_diameter_m": data.get("circle_diameter_m"),
        "distance_m": data.get("distance_m"),
        "angular_radius_deg": data.get("angular_radius_deg"),
        "num_samples": len(samples),
        "fit_rmse_x": rmse_x,
        "fit_rmse_y": rmse_y,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print(f"Mapping saved → {out_path}")

    # Auto-visualise: save heatmaps next to the mapping file
    from face_diet_gui.processing.pixel_degree_calibration.visualize_mapping import (
        visualize_mapping,
    )
    print("Generating visualizations …")
    visualize_mapping(
        mapping_path=str(out_path),
        output_dir=str(out_path.parent),
        samples_path=samples_path,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit 2D polynomial PPD mapping from calibration samples."
    )
    p.add_argument("--samples", required=True, help="PPD samples JSON (from compute_samples.py).")
    p.add_argument("--output", required=True, help="Output path for the mapping JSON.")
    p.add_argument("--order", type=int, default=2, help="Polynomial order (default 2 = quadratic).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fit_mapping(samples_path=args.samples, output_path=args.output, order=args.order)


if __name__ == "__main__":
    main()
