"""
repopulate_distance.py — recompute the ``distance`` column in existing
face-detections CSV files using the PPD-based distance estimator.

Previously the column stored a unitless relative value (150 / bbox_height).
It now stores physical viewing distance in metres.

Scans all ``*_face-detections.csv`` (BIDS style) and legacy
``face_detections.csv`` files under ``{derivatives_dir}/{participant}/{session}/``
and overwrites only the ``distance`` column.

Usage
-----
  # Preview — print what would change, write nothing:
  python scripts/repopulate_distance.py --derivatives-dir /path/to/derivatives --dry-run

  # Apply to all participants/sessions:
  python scripts/repopulate_distance.py --derivatives-dir /path/to/derivatives

  # Single participant:
  python scripts/repopulate_distance.py --derivatives-dir /path/to/derivatives --participant sub-01
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Hardcoded PPD mapping — identical to the one in face_attributes.py
# ---------------------------------------------------------------------------
_PPD_MAPPING = {
    "center_x":       959.5,
    "center_y":       539.5,
    "exponents":      [[0,0],[0,1],[0,2],[1,0],[1,1],[2,0]],
    "coefficients_x": [
        14.05476210542249,
        -0.0876786050844629,
        -0.27582551113140186,
        -0.3989892392520565,
         0.8074276506823053,
        11.044886529750261,
    ],
    "coefficients_y": [
        15.210399867098543,
         0.4202796457210294,
         2.076840035068318,
        -0.35104979073345033,
         0.6885119864666166,
         5.952656667352058,
    ],
}

_FACE_WIDTH_M_FEMALE  = 0.1364
_FACE_HEIGHT_M_FEMALE = 0.1758
_FACE_WIDTH_M_MALE    = 0.1448
_FACE_HEIGHT_M_MALE   = 0.1908


# ---------------------------------------------------------------------------
# Inlined PPD math (mirrors mapping_utils + estimate_distance, no package dep)
# ---------------------------------------------------------------------------

def _evaluate_ppd_xy(x: float, y: float):
    """Return (ppd_x, ppd_y) at pixel position (x, y)."""
    cx, cy = _PPD_MAPPING["center_x"], _PPD_MAPPING["center_y"]
    xn, yn = (x - cx) / cx, (y - cy) / cy
    exps = _PPD_MAPPING["exponents"]
    row = [(xn ** i) * (yn ** j) for i, j in exps]
    ppd_x = sum(c * r for c, r in zip(_PPD_MAPPING["coefficients_x"], row))
    ppd_y = sum(c * r for c, r in zip(_PPD_MAPPING["coefficients_y"], row))
    return ppd_x, ppd_y


def _integrate_degrees(a: float, b: float, fixed: float, axis: str,
                        n: int = 60) -> Optional[float]:
    """Trapezoidal integral of 1/ppd along one axis."""
    samples = np.linspace(a, b, n)
    inv_ppd = np.empty(n)
    for i, s in enumerate(samples):
        px, py = _evaluate_ppd_xy(s, fixed) if axis == "x" else _evaluate_ppd_xy(fixed, s)
        pv = px if axis == "x" else py
        if pv <= 0:
            return None
        inv_ppd[i] = 1.0 / pv
    result = float(np.trapz(inv_ppd, samples))
    return result if result > 0 else None


def _degrees_to_dist(deg: Optional[float], physical_m: float) -> Optional[float]:
    if deg is None or deg <= 0:
        return None
    return (physical_m / 2.0) / math.tan(math.radians(deg) / 2.0)


def _compute_distance(x: float, y: float, w: float, h: float,
                      gender: Optional[str]) -> Optional[float]:
    """Return distance in metres for one detection row, or None on failure."""
    if w <= 0 or h <= 0:
        return None

    is_male = isinstance(gender, str) and gender.strip().upper() in {"M", "MAN", "MALE"}
    fw = _FACE_WIDTH_M_MALE  if is_male else _FACE_WIDTH_M_FEMALE
    fh = _FACE_HEIGHT_M_MALE if is_male else _FACE_HEIGHT_M_FEMALE

    cx, cy = x + w / 2.0, y + h / 2.0
    theta_w = _integrate_degrees(x, x + w, cy, "x")
    theta_h = _integrate_degrees(y, y + h, cx, "y")

    dist_w = _degrees_to_dist(theta_w, fw)
    dist_h = _degrees_to_dist(theta_h, fh)

    if dist_w is not None and dist_h is not None:
        combined = math.sqrt(dist_w * dist_h)
    elif dist_w is not None:
        combined = dist_w
    elif dist_h is not None:
        combined = dist_h
    else:
        return None

    # Correction: dist + 0.25 * ln(dist)
    if combined > 0:
        combined = combined + 0.25 * math.log(combined)

    return float(combined)


def repopulate_csv(csv_path: Path, dry_run: bool = False) -> bool:
    """
    Recompute the ``distance`` column for one CSV file.

    Returns True if a change was made (or would be made in dry-run mode),
    False if the file was skipped.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [WARN] could not read {csv_path}: {e}")
        return False

    required = {"x", "y", "w", "h"}
    if not required.issubset(df.columns):
        print(f"  [SKIP] missing bbox columns in {csv_path.name}")
        return False

    if "distance" not in df.columns:
        print(f"  [SKIP] no 'distance' column in {csv_path.name}")
        return False

    gender_col = df["gender"] if "gender" in df.columns else None

    new_distances = []
    for i, row in df.iterrows():
        g = gender_col.iloc[i] if gender_col is not None else None
        dist = _compute_distance(row["x"], row["y"], row["w"], row["h"], g)
        new_distances.append(dist if dist is not None else float("inf"))

    if dry_run:
        old_sample = df["distance"].iloc[0] if len(df) > 0 else "n/a"
        new_sample = new_distances[0] if new_distances else "n/a"
        print(f"  [DRY-RUN] {csv_path.name}  ({len(df)} rows)  "
              f"sample: {old_sample!r} -> {new_sample!r}")
        return True

    df["distance"] = new_distances
    df.to_csv(csv_path, index=False)
    print(f"  ✓ {csv_path.name}  ({len(df)} rows updated)")
    return True


def find_detection_csvs(derivatives_dir: Path,
                        participant: Optional[str] = None) -> list[Path]:
    """Discover all face-detections CSV files under derivatives."""
    csvs = []
    search_root = derivatives_dir / participant if participant else derivatives_dir
    for p in sorted(search_root.rglob("*")):
        if not p.is_file():
            continue
        name = p.name
        if name.endswith("_face-detections.csv") or name == "face_detections.csv":
            # Skip anything inside the annotations subtree
            if "annotations" in p.parts:
                continue
            csvs.append(p)
    return csvs


def main():
    parser = argparse.ArgumentParser(
        description="Repopulate the 'distance' column in face-detections CSVs "
                    "using the PPD-based distance estimator (metres)."
    )
    parser.add_argument(
        "--derivatives-dir", required=True,
        help="Root derivatives directory.",
    )
    parser.add_argument(
        "--participant", default=None,
        help="Restrict to a single participant folder (e.g. sub-01).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without writing anything.",
    )
    args = parser.parse_args()

    derivatives_dir = Path(args.derivatives_dir).expanduser().resolve()
    if not derivatives_dir.exists():
        print(f"ERROR: derivatives directory not found: {derivatives_dir}", file=sys.stderr)
        sys.exit(1)

    csvs = find_detection_csvs(derivatives_dir, args.participant)
    if not csvs:
        print("No face-detections CSV files found — nothing to do.")
        sys.exit(0)

    print(f"Found {len(csvs)} file(s) to process.\n")
    updated = 0
    for csv_path in csvs:
        print(f"{csv_path.relative_to(derivatives_dir)}")
        if repopulate_csv(csv_path, dry_run=args.dry_run):
            updated += 1

    action = "would update" if args.dry_run else "updated"
    print(f"\nDone. {action} {updated} / {len(csvs)} file(s).")


if __name__ == "__main__":
    main()
