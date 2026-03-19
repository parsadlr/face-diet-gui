"""
estimate_distance.py — batch face distance estimation using the PPD mapping.

For every image in a folder:
  1. Detect faces with InsightFace (same model / pipeline as face_diet_gui).
  2. For each face, look up the pixel-per-degree (PPD) mapping at the face
     bounding-box centre.
  3. Convert the bounding-box width and height from pixels to degrees (angular
     size) using numerical integration of 1/ppd along the span.
  4. Estimate viewing distance independently from:
       (a) bounding-box WIDTH  + known physical face width
       (b) bounding-box HEIGHT + known physical face height
       (c) a combined (geometric-mean) estimate
  5. Optionally draw annotated results and save them.

────────────────────────────────────────────────────────────────────────────
Calculation steps in detail
────────────────────────────────────────────────────────────────────────────

Step 1 – PPD lookup
    The mapping stores two 2-D polynomial surfaces:
        ppd_x(u, v)  – pixels per degree in the horizontal direction
        ppd_y(u, v)  – pixels per degree in the vertical direction
    evaluated at image-pixel position (u, v).

Step 2 – Pixels → angular size (degrees)  [integrated]
    The correct angular span from pixel x1 to x2 (at fixed y = cy) is:

        theta_w = integral[x1 -> x2]  1/ppd_x(x, cy) dx
        theta_h = integral[y1 -> y2]  1/ppd_y(cx, y) dy

    Computed numerically with the trapezoidal rule (60 samples).
    This is more accurate than the centre-point approximation
    (theta = W_px / ppd(centre)) especially for large, close faces.

    For comparison, the centre-point approximation is also reported:
        theta_w_cp = W_px / ppd_x(cx, cy)
        theta_h_cp = H_px / ppd_y(cx, cy)

Step 3 – Angular size → viewing distance
    Exact thin-lens formula (no small-angle approximation):
        distance = (D_physical / 2) / tan(theta_rad / 2)

    Applied separately to width and height, plus a combined estimate:
        distance_combined = sqrt(distance_w * distance_h)   [geometric mean]

────────────────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────────────────
    python -m face_diet_gui.processing.pixel_degree_calibration.estimate_distance ^
        --images  D:/photos/ ^
        --mapping D:/calibration/ppd_mapping.json ^
        --output  D:/results/

    python -m face_diet_gui.processing.pixel_degree_calibration.estimate_distance ^
        --images  D:/photos/ ^
        --mapping D:/calibration/ppd_mapping.json ^
        --face-width-m  0.130 ^
        --face-height-m 0.190 ^
        --no-draw
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from face_diet_gui.processing.pixel_degree_calibration.mapping_utils import (
    evaluate_ppd_xy,
    load_pixel_degree_mapping,
)

# ── default physical face dimensions (metres) ────────────────────────────
# These correspond to what the InsightFace bounding box captures.
# Override via CLI if your subject differs.
DEFAULT_FACE_WIDTH_M  = 0.130   # horizontal bbox span (cheekbone to cheekbone)
DEFAULT_FACE_HEIGHT_M = 0.190   # vertical   bbox span (hairline to chin)

# ── image extensions to process ──────────────────────────────────────────
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── geometry helpers ──────────────────────────────────────────────────────

def pixels_to_degrees_integrated(
    x1: float, x2: float,
    y_fixed: float,
    mapping: Dict,
    axis: str,
    n_samples: int = 60,
) -> Optional[float]:
    """
    Accurate angular span via numerical integration of 1/ppd along an axis.

        theta_w = integral[x1->x2]  1/ppd_x(x, y_fixed) dx   (axis="x")
        theta_h = integral[y1->y2]  1/ppd_y(x_fixed, y) dy   (axis="y")

    Trapezoidal rule, 60 samples → < 0.01% error for a 2nd-order polynomial.
    """
    samples = np.linspace(x1, x2, n_samples)
    inv_ppd = np.empty(n_samples)
    for i, s in enumerate(samples):
        if axis == "x":
            pv, _ = evaluate_ppd_xy(mapping, s, y_fixed)
        else:
            _, pv = evaluate_ppd_xy(mapping, y_fixed, s)
        if pv <= 0:
            return None
        inv_ppd[i] = 1.0 / pv
    result = float(np.trapz(inv_ppd, samples))
    return result if result > 0 else None


def pixels_to_degrees_cp(px: float, ppd: float) -> Optional[float]:
    """Centre-point approximation: theta = px / ppd(centre)."""
    if ppd <= 0 or px <= 0:
        return None
    return px / ppd


def degrees_to_distance(angular_deg: Optional[float], physical_m: float) -> Optional[float]:
    """
    distance = (physical_m / 2) / tan(angular_rad / 2)
    Returns None if angular_deg is None or non-positive.
    """
    if angular_deg is None or angular_deg <= 0:
        return None
    theta_rad = math.radians(angular_deg)
    return (physical_m / 2.0) / math.tan(theta_rad / 2.0)


# ── per-face estimation ───────────────────────────────────────────────────

def estimate_face_distances(
    bbox: Tuple[int, int, int, int],
    mapping: Dict,
    face_width_m: float,
    face_height_m: float,
) -> Dict:
    """
    Estimate distance from a single face bounding box.
    Returns a dict with all intermediate and final values.
    """
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0

    ppd_x, ppd_y = evaluate_ppd_xy(mapping, cx, cy)

    # integrated angular spans (primary)
    theta_w_int = pixels_to_degrees_integrated(x, x + w, cy, mapping, axis="x")
    theta_h_int = pixels_to_degrees_integrated(y, y + h, cx, mapping, axis="y")

    # centre-point angular spans (for comparison)
    theta_w_cp = pixels_to_degrees_cp(w, ppd_x)
    theta_h_cp = pixels_to_degrees_cp(h, ppd_y)

    # distances — integrated
    dist_w_int = degrees_to_distance(theta_w_int, face_width_m)
    dist_h_int = degrees_to_distance(theta_h_int, face_height_m)
    if dist_w_int is not None and dist_h_int is not None:
        dist_combined_int = math.sqrt(dist_w_int * dist_h_int)
    else:
        dist_combined_int = dist_w_int or dist_h_int

    # distances — centre-point
    dist_w_cp = degrees_to_distance(theta_w_cp, face_width_m)
    dist_h_cp = degrees_to_distance(theta_h_cp, face_height_m)
    if dist_w_cp is not None and dist_h_cp is not None:
        dist_combined_cp = math.sqrt(dist_w_cp * dist_h_cp)
    else:
        dist_combined_cp = dist_w_cp or dist_h_cp

    return {
        "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
        "center_x": cx, "center_y": cy,
        "ppd_x": ppd_x, "ppd_y": ppd_y,
        # angular spans
        "angular_width_deg_int":  theta_w_int,
        "angular_height_deg_int": theta_h_int,
        "angular_width_deg_cp":   theta_w_cp,
        "angular_height_deg_cp":  theta_h_cp,
        # distances — integrated (primary recommendation)
        "dist_from_bbox_w_m":  dist_w_int,
        "dist_from_bbox_h_m":  dist_h_int,
        "dist_combined_m":     dist_combined_int,
        # distances — centre-point (for comparison)
        "dist_from_bbox_w_cp_m": dist_w_cp,
        "dist_from_bbox_h_cp_m": dist_h_cp,
        "dist_combined_cp_m":    dist_combined_cp,
    }


# ── drawing ───────────────────────────────────────────────────────────────

def _put_text_bg(
    img: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_scale: float,
    text_color: Tuple[int, int, int],
    thickness: int = 1,
) -> int:
    """Draw ASCII text with a dark background rectangle. Returns line height."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    lx, ly = origin
    cv2.rectangle(img, (lx - 3, ly - th - 4), (lx + tw + 3, ly + baseline), (0, 0, 0), -1)
    cv2.putText(img, text, (lx, ly), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return th + baseline + 4


def draw_result(image_bgr: np.ndarray, result: Dict, conf: float) -> np.ndarray:
    out = image_bgr.copy()
    x, y, w, h = result["bbox_x"], result["bbox_y"], result["bbox_w"], result["bbox_h"]

    font_scale = 0.65
    thickness  = 1
    line_gap   = 5

    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 210, 100), 2)

    def _fm(v): return f"{v:.3f} m"   if v is not None else "-"
    def _fd(v): return f"{v:.2f} deg" if v is not None else "-"

    ppd_x = result.get("ppd_x", 0.0)
    ppd_y = result.get("ppd_y", 0.0)

    lines = [
        (f"bbox: {w} x {h} px  conf: {conf:.2f}",                    (0, 210, 100)),
        (f"ppd:  x={ppd_x:.2f}  y={ppd_y:.2f} px/deg",              (180, 220, 255)),
        (f"ang (int):  W={_fd(result.get('angular_width_deg_int'))}  H={_fd(result.get('angular_height_deg_int'))}", (220, 200, 80)),
        (f"ang (cp):   W={_fd(result.get('angular_width_deg_cp'))}  H={_fd(result.get('angular_height_deg_cp'))}", (170, 155, 60)),
        (f"dist W:  {_fm(result.get('dist_from_bbox_w_m'))}  cp: {_fm(result.get('dist_from_bbox_w_cp_m'))}", (100, 235, 180)),
        (f"dist H:  {_fm(result.get('dist_from_bbox_h_m'))}  cp: {_fm(result.get('dist_from_bbox_h_cp_m'))}", (100, 235, 180)),
        (f"dist combined (int): {_fm(result.get('dist_combined_m'))}",    (80, 255, 80)),
        (f"dist combined (cp):  {_fm(result.get('dist_combined_cp_m'))}", (60, 200, 60)),
    ]

    font   = cv2.FONT_HERSHEY_SIMPLEX
    line_h = cv2.getTextSize("A", font, font_scale, thickness)[0][1] + 8
    total_h = len(lines) * (line_h + line_gap)

    cur_y = y - (len(lines) - 1) * (line_h + line_gap) - 4 if y - total_h - 4 >= 0 \
            else y + h + line_h + 2

    for text, color in lines:
        _put_text_bg(out, text, (x, cur_y), font_scale, color, thickness)
        cur_y += line_h + line_gap

    return out


# ── main pipeline ─────────────────────────────────────────────────────────

def process_folder(
    images_dir: str,
    mapping_path: str,
    output_dir: Optional[str],
    face_width_m:  float = DEFAULT_FACE_WIDTH_M,
    face_height_m: float = DEFAULT_FACE_HEIGHT_M,
    draw: bool = True,
    model_name: str = "buffalo_l",
    use_gpu: bool = False,
) -> List[Dict]:
    images_path = Path(images_dir)
    out_path    = Path(output_dir) if output_dir else images_path / "distance_results"
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading PPD mapping from {mapping_path} ...")
    mapping = load_pixel_degree_mapping(mapping_path)
    print(f"  Image size in mapping: {mapping['image_width']} x {mapping['image_height']} px")

    from face_diet_gui.processing.face_detection import initialize_detector, detect_faces_in_frame
    print(f"Loading InsightFace model '{model_name}' ...")
    detector = initialize_detector(model_name=model_name, use_gpu=use_gpu)

    img_paths = sorted(p for p in images_path.iterdir() if p.suffix.lower() in _IMG_EXTS)
    if not img_paths:
        print(f"No images found in {images_path}")
        return []

    print(f"Processing {len(img_paths)} image(s) ...\n")
    all_results: List[Dict] = []

    for img_path in img_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  [skip] could not read {img_path.name}")
            continue

        detections = detect_faces_in_frame(detector, bgr)
        annotated  = bgr.copy()
        print(f"  {img_path.name}: {len(detections)} face(s) detected")

        for i, det in enumerate(detections):
            bbox = det["bbox"]
            conf = det.get("confidence", 0.0)

            result = estimate_face_distances(
                bbox=bbox,
                mapping=mapping,
                face_width_m=face_width_m,
                face_height_m=face_height_m,
            )
            result["image"]      = img_path.name
            result["face_index"] = i
            result["confidence"] = conf
            all_results.append(result)

            def _s(v): return f"{v:.3f}" if v is not None else "-"
            def _sd(v): return f"{v:.2f} deg" if v is not None else "-"

            print(
                f"    face {i}: bbox=({bbox[0]},{bbox[1]},{bbox[2]}x{bbox[3]})  conf={conf:.2f}\n"
                f"      ppd: x={result['ppd_x']:.2f}  y={result['ppd_y']:.2f} px/deg\n"
                f"      ang (integrated):   W={_sd(result['angular_width_deg_int'])}  H={_sd(result['angular_height_deg_int'])}\n"
                f"      ang (centre-point): W={_sd(result['angular_width_deg_cp'])}  H={_sd(result['angular_height_deg_cp'])}\n"
                f"      dist integrated:   W={_s(result['dist_from_bbox_w_m'])}m  H={_s(result['dist_from_bbox_h_m'])}m  combined={_s(result['dist_combined_m'])}m\n"
                f"      dist centre-point: W={_s(result['dist_from_bbox_w_cp_m'])}m  H={_s(result['dist_from_bbox_h_cp_m'])}m  combined={_s(result['dist_combined_cp_m'])}m"
            )

            if draw:
                annotated = draw_result(annotated, result, conf)

        if draw:
            cv2.imwrite(str(out_path / f"annotated_{img_path.name}"), annotated)

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = out_path / "distance_estimates.csv"
    _csv_fields = [
        "image", "face_index", "confidence",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "center_x", "center_y",
        "ppd_x", "ppd_y",
        "angular_width_deg_int", "angular_height_deg_int",
        "angular_width_deg_cp",  "angular_height_deg_cp",
        "dist_from_bbox_w_m",    "dist_from_bbox_h_m",    "dist_combined_m",
        "dist_from_bbox_w_cp_m", "dist_from_bbox_h_cp_m", "dist_combined_cp_m",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to {out_path}/")
    print(f"  CSV:             {csv_path.name}")
    if draw:
        print(f"  Annotated images: annotated_*")
    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate face viewing distance using the PPD calibration mapping."
    )
    p.add_argument("--images",  required=True, help="Folder containing input images.")
    p.add_argument("--mapping", required=True, help="Path to ppd_mapping.json.")
    p.add_argument("--output",  default=None,
                   help="Output folder. Defaults to <images>/distance_results/.")
    p.add_argument("--face-width-m",  type=float, default=DEFAULT_FACE_WIDTH_M,
                   help=f"Physical width [m] spanned by the bbox horizontally "
                        f"(default {DEFAULT_FACE_WIDTH_M} m).")
    p.add_argument("--face-height-m", type=float, default=DEFAULT_FACE_HEIGHT_M,
                   help=f"Physical height [m] spanned by the bbox vertically "
                        f"(default {DEFAULT_FACE_HEIGHT_M} m).")
    p.add_argument("--model",   default="buffalo_l",
                   help="InsightFace model name (default: buffalo_l).")
    p.add_argument("--gpu",     action="store_true", help="Use GPU for inference.")
    p.add_argument("--no-draw", action="store_true", help="Skip annotated images.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    process_folder(
        images_dir    = args.images,
        mapping_path  = args.mapping,
        output_dir    = args.output,
        face_width_m  = args.face_width_m,
        face_height_m = args.face_height_m,
        draw          = not args.no_draw,
        model_name    = args.model,
        use_gpu       = args.gpu,
    )


if __name__ == "__main__":
    main()
