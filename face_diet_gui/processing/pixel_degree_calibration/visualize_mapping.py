"""
Visualise the fitted pixel-per-degree mapping as heatmaps.

Produces three PNG files saved alongside the mapping JSON:
    ppd_scalar.png  – isotropic (area-equivalent) PPD surface
    ppd_x.png       – horizontal PPD surface
    ppd_y.png       – vertical PPD surface

Each image uses the same PIL-based rendering as the GUI:
    • viridis colourmap
    • labelled colorbar strip on the right
    • hollow white circles at every calibration sample location

The saved PNGs are pixel-identical in style to the heatmaps shown in the
Fit Mapping tab, making it easy to compare on-screen with exported images.
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from face_diet_gui.processing.pixel_degree_calibration.mapping_utils import (
    evaluate_ppd,
    evaluate_ppd_xy,
    load_pixel_degree_mapping,
)

# ── colorbar layout (shared with tab_fit.py) ──────────────────────────────
CBAR_TOTAL_W = 70   # total width reserved for colorbar area
CBAR_BAR_W   = 18   # width of the actual colour gradient strip
CBAR_LEFT    = 8    # gap between heatmap edge and bar


# ── viridis colouring ─────────────────────────────────────────────────────

def viridis_rgb(arr_01: np.ndarray) -> np.ndarray:
    """
    Map a float32 array normalised to [0, 1] → uint8 RGB (viridis).
    Falls back to a hand-crafted approximation when matplotlib is absent.
    """
    try:
        import matplotlib.cm as cm  # type: ignore[import]
        rgba = (cm.viridis(arr_01) * 255).astype(np.uint8)
        return rgba[:, :, :3]
    except ImportError:
        r = np.clip((arr_01 * 2.0 - 0.6) * 255, 0, 255).astype(np.uint8)
        g = np.clip(np.sin(arr_01 * np.pi) * 280, 0, 255).astype(np.uint8)
        b = np.clip((1.0 - arr_01 * 1.4) * 255, 0, 255).astype(np.uint8)
        return np.stack([r, g, b], axis=2)


# ── heatmap rendering ─────────────────────────────────────────────────────

def make_heatmap_pil(
    grid: np.ndarray,
    vmin: float,
    vmax: float,
    canvas_w: int,
    canvas_h: int,
    samples_xy: Optional[List[Tuple[float, float]]] = None,
) -> Image.Image:
    """
    Render *grid* as a viridis heatmap PIL image of size (canvas_w × canvas_h)
    with a labelled colorbar strip on the right.

    Parameters
    ----------
    grid        : 2-D float array of PPD values (rows=y, cols=x).
    vmin, vmax  : colour scale limits.
    canvas_w/h  : total output image dimensions.
    samples_xy  : optional list of (img_x, img_y) sample centres to overlay
                  as hollow white circles.
    """
    if vmax <= vmin:
        vmax = vmin + 1.0

    plot_w = max(canvas_w - CBAR_TOTAL_W, 10)
    plot_h = max(canvas_h, 10)

    # ── heatmap body ──────────────────────────────────────────────────────
    norm     = np.clip((grid.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb      = viridis_rgb(norm)
    pil_heat = Image.fromarray(rgb).resize((plot_w, plot_h), Image.BILINEAR)

    # ── colorbar gradient ─────────────────────────────────────────────────
    cbar_vals = np.linspace(1.0, 0.0, plot_h, dtype=np.float32)
    cbar_col  = viridis_rgb(
        cbar_vals[:, np.newaxis] * np.ones((1, CBAR_BAR_W), np.float32)
    )
    pil_cbar = Image.fromarray(cbar_col)

    # ── compose canvas ────────────────────────────────────────────────────
    bg    = Image.new("RGB", (canvas_w, canvas_h), (28, 28, 28))
    bg.paste(pil_heat, (0, 0))
    bar_x = plot_w + CBAR_LEFT
    bg.paste(pil_cbar, (bar_x, 0))

    # ── colorbar tick labels ──────────────────────────────────────────────
    draw  = ImageDraw.Draw(bg)
    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except OSError:
        font = ImageFont.load_default()

    n_tick  = 6
    label_x = bar_x + CBAR_BAR_W + 3
    for i in range(n_tick):
        frac  = i / (n_tick - 1)
        val   = vmax - frac * (vmax - vmin)
        y_pos = int(frac * (plot_h - 1))
        draw.text((label_x, y_pos - 5), f"{val:.1f}", fill=(200, 200, 200), font=font)
        draw.line(
            [(bar_x - 2, y_pos), (bar_x + CBAR_BAR_W + 1, y_pos)],
            fill=(160, 160, 160), width=1,
        )

    # ── sample centre overlay ─────────────────────────────────────────────
    if samples_xy:
        grid_h, grid_w = grid.shape
        r = max(4, min(8, plot_w // 80))
        for img_x, img_y in samples_xy:
            # Map image coords → canvas coords (grid spans 0..grid_w, 0..grid_h)
            px = int(img_x / max(grid_w, 1) * plot_w)
            py = int(img_y / max(grid_h, 1) * plot_h)
            draw.ellipse(
                [(px - r, py - r), (px + r, py + r)],
                fill=None, outline=(255, 255, 255), width=2,
            )

    return bg


# ── grid evaluation ───────────────────────────────────────────────────────

def _eval_grids(mapping: dict, step: int = 8):
    """Vectorised 2-D polynomial evaluation. Returns (xs, ys, x_grid, y_grid)."""
    w  = mapping["image_width"]
    h  = mapping["image_height"]
    cx = mapping["center_x"]
    cy = mapping["center_y"]

    xs = np.arange(0, w, step, dtype=np.float64)
    ys = np.arange(0, h, step, dtype=np.float64)
    xg, yg = np.meshgrid(xs, ys)

    x_norm = (xg - cx) / cx
    y_norm = (yg - cy) / cy

    exponents = mapping["exponents"]
    cols = [(x_norm ** i) * (y_norm ** j) for i, j in exponents]
    X = np.stack(cols, axis=-1)

    cx_ = np.array(mapping["coefficients_x"], dtype=np.float64)
    cy_ = np.array(mapping["coefficients_y"], dtype=np.float64)

    return xs, ys, (X * cx_).sum(-1), (X * cy_).sum(-1)


# ── main visualise function ───────────────────────────────────────────────

_SAVE_W = 1000   # saved image width  (pixels)
_SAVE_H =  700   # saved image height (pixels)


def visualize_mapping(
    mapping_path: str,
    output_dir: Optional[str] = None,
    samples_path: Optional[str] = None,
    grid_step: int = 8,
) -> None:
    mapping = load_pixel_degree_mapping(mapping_path)
    out_dir = Path(output_dir) if output_dir else Path(mapping_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating mapping on {grid_step}px grid …")
    xs, ys, x_grid, y_grid = _eval_grids(mapping, step=grid_step)
    scalar_grid = np.sqrt(np.maximum(x_grid * y_grid, 0.0))

    # ── sample points for overlay ─────────────────────────────────────────
    samples_xy: List[Tuple[float, float]] = []
    if samples_path and Path(samples_path).exists():
        with open(samples_path, "r", encoding="utf-8") as f:
            sdata = json.load(f)
        for s in sdata.get("samples", []):
            samples_xy.append((float(s["center_x"]), float(s["center_y"])))

    # The grids are sampled at `step` intervals; convert sample image coords
    # to the grid's own coordinate space for correct overlay positioning.
    # make_heatmap_pil expects coords normalised to grid dimensions.
    # We pass raw image coords and let make_heatmap_pil scale by image size.
    img_w = mapping["image_width"]
    img_h = mapping["image_height"]

    plots = [
        (scalar_grid, "PPD scalar (area-equiv.)", "ppd_scalar.png"),
        (x_grid,      "PPD horizontal (x-axis)",  "ppd_x.png"),
        (y_grid,      "PPD vertical (y-axis)",     "ppd_y.png"),
    ]

    for grid, title, fname in plots:
        vmin = float(np.nanmin(grid))
        vmax = float(np.nanmax(grid))

        # Convert sample image coords → grid-shape-relative coords
        # make_heatmap_pil maps (img_x / grid_w * plot_w), so we need to pass
        # coords relative to the grid dimensions (grid_w cols = xs range).
        # The grid has shape (len(ys), len(xs)); pass coords scaled to that.
        grid_h, grid_w = grid.shape
        scaled_xy = [
            (sx / img_w * grid_w, sy / img_h * grid_h)
            for sx, sy in samples_xy
        ]

        pil = make_heatmap_pil(
            grid, vmin, vmax, _SAVE_W, _SAVE_H, samples_xy=scaled_xy
        )
        out_path = out_dir / fname
        pil.save(str(out_path))
        print(f"  Saved: {out_path.name}  (range {vmin:.1f}–{vmax:.1f} px/deg)")


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise the fitted PPD mapping as heatmap images."
    )
    p.add_argument("--mapping", required=True,
                   help="Path to mapping JSON (from fit_mapping.py).")
    p.add_argument("--samples", default=None,
                   help="Optional path to samples JSON for scatter overlay.")
    p.add_argument("--out-dir", default=None,
                   help="Output directory. Defaults to the mapping file's directory.")
    p.add_argument("--grid-step", type=int, default=8,
                   help="Grid step in pixels (default 8).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    visualize_mapping(
        mapping_path=args.mapping,
        output_dir=args.out_dir,
        samples_path=args.samples,
        grid_step=args.grid_step,
    )


if __name__ == "__main__":
    main()
