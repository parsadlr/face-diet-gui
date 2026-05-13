"""
estimate_distance.py — batch face distance estimation using the PPD mapping.

For every image in a folder:
  1. Detect faces with InsightFace (same model / pipeline as face_diet_gui).
  2. For each face, look up the pixel-per-degree (PPD) mapping at the face
     bounding-box centre.
  3. Convert bounding-box width and height from pixels to degrees using
     numerical integration of 1/ppd along each axis (horizontal at mid-box y,
     vertical at mid-box x).
  4. Estimate viewing distance from width and height separately, then take the
     geometric mean sqrt(dist_w * dist_h) as the combined distance (both use
     integrated angles), and apply ``dist + 0.3*ln(dist)`` to that combined value.
  5. Optionally draw annotated results and save them.
  6. Writes error plots (``error_histogram.png``, ``rmse_*.png``, ``mean_bias_*.png``,
     etc.) into the same ``--output`` folder via ``visualize_estimate_errors``
     (unless ``--no-error-plots``), when the CSV has rows with ground truth.

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

    Applied to width and height independently, then combined as a geometric mean:
        distance_combined = sqrt(distance_w * distance_h)

    The integrated combined distance is then corrected (final reported value):
        distance_combined := distance_combined + 0.3 * ln(distance_combined)
    using natural logarithm, distance in metres (only if combined > 0).

    The CSV stores W, H, and combined estimates each with its own error vs ground truth.

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
        --no-draw

Experiment-style stems (optional) encode metadata and ground-truth distance::

    sub-04_F_dist-200_pose-90R_TR.jpg
    -> participant sub-04, gender F, 200 cm true distance, pose 90R, FOV TR.

CSV columns include participant, gender, ground_truth_cm, pose, fov_position,
estimated_distance_w_m / _h_m / _combined_m (integrated PPD; combined is
geometric mean of W and H), and error_w_m / error_h_m / error_combined_m
(signed: estimate - ground_truth_m). Physical width/height defaults differ for
M vs F unless --face-width-m / --face-height-m override uniformly.

With ``--output DIR``, CSV, annotated images (unless ``--no-draw``), and
error-plot PNGs (histogram, RMSE, bias) under ``DIR`` when ground truth is present.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
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
# Female / male defaults differ slightly; override via CLI or use
# --face-width-m / --face-height-m to force the same values for every image.
DEFAULT_FACE_WIDTH_M_FEMALE  = 0.1364
DEFAULT_FACE_HEIGHT_M_FEMALE = 0.1758
DEFAULT_FACE_WIDTH_M_MALE    = 0.1448
DEFAULT_FACE_HEIGHT_M_MALE   = 0.1908


# ── experiment image stem: sub-04_F_dist-200_pose-90R_TR ──────────────────
# participant, gender (M|F), ground-truth distance (cm), pose, FOV quadrant.
_FILENAME_STEM_RE = re.compile(
    r"^(?P<participant>sub-[A-Za-z0-9-]+)_(?P<gender>[MF])_"
    r"dist-(?P<dist_cm>\d+)_pose-(?P<pose>90R|45R|0|45L|90L)_"
    r"(?P<fov_position>TR|TL|BR|BL|C)$",
    re.IGNORECASE,
)


def parse_experiment_stem(stem: str) -> Optional[Dict[str, str]]:
    """
    Parse a filename stem like ``sub-04_F_dist-200_pose-90R_TR``.

    Returns dict keys: participant, gender (M|F upper), dist_cm (str),
    pose, fov_position (TR|TL|BR|BL|C upper). Returns None if no match.
    """
    m = _FILENAME_STEM_RE.match(stem.strip())
    if not m:
        return None
    g = m.group("gender").upper()
    pos = m.group("fov_position").upper()
    pose = m.group("pose").upper()
    return {
        "participant": m.group("participant"),
        "gender": g,
        "dist_cm": m.group("dist_cm"),
        "pose": pose,
        "fov_position": pos,
    }

# ── image extensions to process ──────────────────────────────────────────
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def resolve_face_dimensions_m(
    gender: str,
    *,
    uniform_width_m: Optional[float],
    uniform_height_m: Optional[float],
    female_width_m: float,
    female_height_m: float,
    male_width_m: float,
    male_height_m: float,
) -> Tuple[float, float]:
    """Pick bbox physical width/height (m) from gender and optional uniform overrides."""
    if uniform_width_m is not None and uniform_height_m is not None:
        return uniform_width_m, uniform_height_m
    if uniform_width_m is not None or uniform_height_m is not None:
        # Partial uniform override: fill missing side from gender-specific defaults
        fw, fh = (
            (male_width_m, male_height_m)
            if gender.upper() == "M"
            else (female_width_m, female_height_m)
        )
        return (
            uniform_width_m if uniform_width_m is not None else fw,
            uniform_height_m if uniform_height_m is not None else fh,
        )
    if gender.upper() == "M":
        return male_width_m, male_height_m
    return female_width_m, female_height_m


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


def correct_combined_distance_m(dist_m: Optional[float]) -> Optional[float]:
    """
    Final combined distance (metres): ``dist + 0.3 * ln(dist)`` (natural log).

    Returns None if ``dist_m`` is None or not positive.
    """
    if dist_m is None or dist_m <= 0:
        return None
    return dist_m + 0.25 * math.log(dist_m)


# ── per-face estimation ───────────────────────────────────────────────────

def estimate_face_distances(
    bbox: Tuple[int, int, int, int],
    mapping: Dict,
    face_width_m: float,
    face_height_m: float,
) -> Dict:
    """
    Estimate distance from a single face bounding box.

    Primary viewing distance is ``dist_combined_m``: geometric mean of integrated
    W/H distances, then ``dist + 0.3*ln(dist)``. Per-axis distances are uncorrected.
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
        dist_combined_geom = math.sqrt(dist_w_int * dist_h_int)
    else:
        dist_combined_geom = dist_w_int or dist_h_int
    dist_combined_corrected = correct_combined_distance_m(dist_combined_geom)
    if dist_combined_corrected is not None:
        dist_combined_corrected = float(dist_combined_corrected)

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
        # distances — integrated (dist_combined_m = geom mean + 0.3*ln(geom))
        "dist_from_bbox_w_m":  dist_w_int,
        "dist_from_bbox_h_m":  dist_h_int,
        "dist_combined_geom_m": dist_combined_geom,
        "dist_combined_m":      dist_combined_corrected,
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


def _combined_distance_for_overlay(result: Dict) -> Optional[float]:
    """Same combined (corrected) distance as CSV primary column — after all fields are set."""
    for key in ("estimated_distance_combined_m", "dist_combined_m"):
        v = result.get(key)
        if v is None or v == "":
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def draw_result(image_bgr: np.ndarray, result: Dict, conf: float) -> np.ndarray:
    out = image_bgr.copy()
    x, y, w, h = result["bbox_x"], result["bbox_y"], result["bbox_w"], result["bbox_h"]

    font_scale = 0.65
    thickness  = 1
    line_gap   = 5

    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 210, 100), 2)

    def _fm(v) -> str:
        if v is None or v == "":
            return "-"
        try:
            return f"{float(v):.3f} m"
        except (TypeError, ValueError):
            return "-"

    def _fd(v): return f"{v:.2f} deg" if v is not None else "-"

    ppd_x = result.get("ppd_x", 0.0)
    ppd_y = result.get("ppd_y", 0.0)

    lines = [
        (f"bbox: {w} x {h} px  conf: {conf:.2f}", (0, 210, 100)),
        (f"ppd:  x={ppd_x:.2f}  y={ppd_y:.2f} px/deg", (180, 220, 255)),
        (
            f"ang (int):  W={_fd(result.get('angular_width_deg_int'))}  "
            f"H={_fd(result.get('angular_height_deg_int'))}",
            (220, 200, 80),
        ),
        (
            f"ang (cp):   W={_fd(result.get('angular_width_deg_cp'))}  "
            f"H={_fd(result.get('angular_height_deg_cp'))}",
            (170, 155, 60),
        ),
        (
            f"dist W:  {_fm(result.get('dist_from_bbox_w_m'))}  "
            f"cp: {_fm(result.get('dist_from_bbox_w_cp_m'))}",
            (100, 235, 180),
        ),
        (
            f"dist H:  {_fm(result.get('dist_from_bbox_h_m'))}  "
            f"cp: {_fm(result.get('dist_from_bbox_h_cp_m'))}",
            (100, 235, 180),
        ),
        (
            f"dist combined geom (int): {_fm(result.get('dist_combined_geom_m'))}",
            (120, 200, 120),
        ),
        (
            f"dist combined CORRECTED: {_fm(_combined_distance_for_overlay(result))}",
            (80, 255, 80),
        ),
        (
            f"dist combined (cp, uncorrected): {_fm(result.get('dist_combined_cp_m'))}",
            (60, 200, 60),
        ),
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
    female_face_width_m: float = DEFAULT_FACE_WIDTH_M_FEMALE,
    female_face_height_m: float = DEFAULT_FACE_HEIGHT_M_FEMALE,
    male_face_width_m: float = DEFAULT_FACE_WIDTH_M_MALE,
    male_face_height_m: float = DEFAULT_FACE_HEIGHT_M_MALE,
    uniform_face_width_m: Optional[float] = None,
    uniform_face_height_m: Optional[float] = None,
    draw: bool = True,
    save_error_plots: bool = True,
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

        meta = parse_experiment_stem(img_path.stem)
        gender_for_dims = (meta["gender"] if meta else "F")
        fw_m, fh_m = resolve_face_dimensions_m(
            gender_for_dims,
            uniform_width_m=uniform_face_width_m,
            uniform_height_m=uniform_face_height_m,
            female_width_m=female_face_width_m,
            female_height_m=female_face_height_m,
            male_width_m=male_face_width_m,
            male_height_m=male_face_height_m,
        )

        detections = detect_faces_in_frame(detector, bgr)
        annotated  = bgr.copy()
        print(f"  {img_path.name}: {len(detections)} face(s) detected")
        if meta:
            print(
                f"    meta: {meta['participant']} {meta['gender']} "
                f"gt={meta['dist_cm']}cm pose={meta['pose']} fov={meta['fov_position']}"
            )
        else:
            print("    meta: (stem does not match experiment pattern; using F dims fallback)")
        print(f"    face dims used: W={fw_m:.3f}m H={fh_m:.3f}m (gender={gender_for_dims})")

        for i, det in enumerate(detections):
            bbox = det["bbox"]
            conf = det.get("confidence", 0.0)

            result = estimate_face_distances(
                bbox=bbox,
                mapping=mapping,
                face_width_m=fw_m,
                face_height_m=fh_m,
            )
            result["image"]      = img_path.name
            result["face_index"] = i
            result["confidence"] = conf
            result["face_width_m"]  = fw_m
            result["face_height_m"] = fh_m

            if meta:
                result["participant"]    = meta["participant"]
                result["gender"]         = meta["gender"]
                result["ground_truth_cm"] = meta["dist_cm"]
                result["pose"]           = meta["pose"]
                result["fov_position"]    = meta["fov_position"]
                gt_m = float(meta["dist_cm"]) / 100.0
                result["ground_truth_m"] = gt_m
            else:
                result["participant"]     = ""
                result["gender"]          = ""
                result["ground_truth_cm"] = ""
                result["pose"]            = ""
                result["fov_position"]     = ""
                result["ground_truth_m"]  = ""

            def _est_cell(val: Optional[float]):
                return "" if val is None else val

            result["estimated_distance_w_m"] = _est_cell(result.get("dist_from_bbox_w_m"))
            result["estimated_distance_h_m"] = _est_cell(result.get("dist_from_bbox_h_m"))
            result["estimated_distance_combined_m"] = _est_cell(result.get("dist_combined_m"))

            if meta:
                gt_m = float(meta["dist_cm"]) / 100.0

                def _signed_err(est: Optional[float]):
                    if est is None:
                        return ""
                    return est - gt_m

                result["error_w_m"] = _signed_err(result.get("dist_from_bbox_w_m"))
                result["error_h_m"] = _signed_err(result.get("dist_from_bbox_h_m"))
                result["error_combined_m"] = _signed_err(result.get("dist_combined_m"))
            else:
                result["error_w_m"] = ""
                result["error_h_m"] = ""
                result["error_combined_m"] = ""

            all_results.append(result)

            def _s(v): return f"{v:.3f}" if v is not None else "-"
            def _sd(v): return f"{v:.2f} deg" if v is not None else "-"

            print(
                f"    face {i}: bbox=({bbox[0]},{bbox[1]},{bbox[2]}x{bbox[3]})  conf={conf:.2f}\n"
                f"      ppd: x={result['ppd_x']:.2f}  y={result['ppd_y']:.2f} px/deg\n"
                f"      ang (integrated):   W={_sd(result['angular_width_deg_int'])}  "
                f"H={_sd(result['angular_height_deg_int'])}\n"
                f"      ang (centre-point): W={_sd(result['angular_width_deg_cp'])}  "
                f"H={_sd(result['angular_height_deg_cp'])}\n"
                f"      dist integrated:   W={_s(result['dist_from_bbox_w_m'])}m  "
                f"H={_s(result['dist_from_bbox_h_m'])}m  "
                f"combined={_s(result['dist_combined_m'])}m (primary)\n"
                f"      dist centre-point: W={_s(result['dist_from_bbox_w_cp_m'])}m  "
                f"H={_s(result['dist_from_bbox_h_cp_m'])}m  "
                f"combined={_s(result['dist_combined_cp_m'])}m"
            )

            if draw:
                annotated = draw_result(annotated, result, conf)

        if draw:
            cv2.imwrite(str(out_path / f"annotated_{img_path.name}"), annotated)

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = out_path / "distance_estimates.csv"
    _csv_fields = [
        "participant", "gender", "ground_truth_cm", "ground_truth_m",
        "pose", "fov_position",
        "image", "face_index", "confidence",
        "face_width_m", "face_height_m",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "center_x", "center_y",
        "ppd_x", "ppd_y",
        "angular_width_deg_int", "angular_height_deg_int",
        "angular_width_deg_cp",  "angular_height_deg_cp",
        "estimated_distance_w_m", "estimated_distance_h_m", "estimated_distance_combined_m",
        "dist_from_bbox_w_cp_m", "dist_from_bbox_h_cp_m", "dist_combined_cp_m",
        "error_w_m", "error_h_m", "error_combined_m",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to {out_path}/")
    print(f"  CSV:             {csv_path.name}")
    if draw:
        print(f"  Annotated images: annotated_*")

    if save_error_plots:
        try:
            from face_diet_gui.processing.pixel_degree_calibration.visualize_estimate_errors import (
                run as run_error_plots,
            )

            run_error_plots(csv_path, out_path)
        except ImportError as exc:
            print(f"  [skip] Error plots: missing dependency ({exc}).")
        except ValueError as exc:
            print(f"  [skip] Error plots: {exc}")
            print(
                "  Tip: use experiment-style image names (e.g. sub-01_F_dist-200_pose-0_TR) "
                "so ground_truth_m is filled in the CSV; otherwise RMSE plots cannot run."
            )
        except Exception as exc:  # noqa: BLE001 — surface unexpected plot failures
            print(f"  [skip] Error plots: {exc}")

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate face viewing distance using the PPD calibration mapping "
                    "(CSV: W, H, and combined integrated estimates plus signed errors vs GT)."
    )
    p.add_argument("--images",  required=True, help="Folder containing input images.")
    p.add_argument("--mapping", required=True, help="Path to ppd_mapping.json.")
    p.add_argument(
        "--output",
        default=None,
        help="Output folder for CSV, annotated_*.jpg, and error plots from visualize_estimate_errors "
             "(error_histogram.png, rmse_*.png, mean_bias_*.png, …) "
             "(default: <images>/distance_results/).",
    )
    p.add_argument(
        "--face-width-m",
        type=float,
        default=None,
        dest="uniform_face_width_m",
        help="If set, use this bbox width [m] for every face (both genders). "
             "Combine with --face-height-m for a full uniform override. "
             "If only one dimension is set, the other follows gender-specific defaults.",
    )
    p.add_argument(
        "--face-height-m",
        type=float,
        default=None,
        dest="uniform_face_height_m",
        help="If set, use this bbox height [m] for every face (both genders).",
    )
    p.add_argument(
        "--female-face-width-m",
        type=float,
        default=DEFAULT_FACE_WIDTH_M_FEMALE,
        help=f"Female default bbox width [m] (default {DEFAULT_FACE_WIDTH_M_FEMALE}).",
    )
    p.add_argument(
        "--female-face-height-m",
        type=float,
        default=DEFAULT_FACE_HEIGHT_M_FEMALE,
        help=f"Female default bbox height [m] (default {DEFAULT_FACE_HEIGHT_M_FEMALE}).",
    )
    p.add_argument(
        "--male-face-width-m",
        type=float,
        default=DEFAULT_FACE_WIDTH_M_MALE,
        help=f"Male default bbox width [m] (default {DEFAULT_FACE_WIDTH_M_MALE}).",
    )
    p.add_argument(
        "--male-face-height-m",
        type=float,
        default=DEFAULT_FACE_HEIGHT_M_MALE,
        help=f"Male default bbox height [m] (default {DEFAULT_FACE_HEIGHT_M_MALE}).",
    )
    p.add_argument("--model",   default="buffalo_l",
                   help="InsightFace model name (default: buffalo_l).")
    p.add_argument("--gpu",     action="store_true", help="Use GPU for inference.")
    p.add_argument("--no-draw", action="store_true", help="Skip annotated images.")
    p.add_argument(
        "--no-error-plots",
        action="store_true",
        help="Do not write error PNGs from visualize_estimate_errors (histogram, rmse_*, mean_bias_*, …).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    process_folder(
        images_dir             = args.images,
        mapping_path           = args.mapping,
        output_dir             = args.output,
        female_face_width_m    = args.female_face_width_m,
        female_face_height_m   = args.female_face_height_m,
        male_face_width_m      = args.male_face_width_m,
        male_face_height_m     = args.male_face_height_m,
        uniform_face_width_m   = args.uniform_face_width_m,
        uniform_face_height_m  = args.uniform_face_height_m,
        draw                   = not args.no_draw,
        save_error_plots       = not args.no_error_plots,
        model_name             = args.model,
        use_gpu                = args.gpu,
    )


if __name__ == "__main__":
    main()
