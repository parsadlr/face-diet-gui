"""
Plot RMSE and mean signed error (bias) from estimate_distance CSV.

**RMSE** (per method W / H / combined): sqrt(mean((estimate - ground_truth_m)^2)).

**Mean signed error** (bias): mean(estimate - ground_truth_m). Positive = over-estimated
distance on average. Dashed line at 0 on bias figures.

RMSE PNGs (same folder as the CSV by default):
  - rmse_overall.png, rmse_by_pose.png, rmse_by_fov.png, rmse_by_ground_truth_distance.png
  - combined_rmse_by_pose.png, combined_rmse_by_fov.png,
    combined_rmse_by_ground_truth_distance.png, combined_rmse_by_gender.png

Bias PNGs (parallel naming):
  - mean_bias_overall.png, mean_bias_by_pose.png, mean_bias_by_fov.png,
    mean_bias_by_ground_truth_distance.png
  - combined_mean_bias_by_pose.png, combined_mean_bias_by_fov.png,
    combined_mean_bias_by_ground_truth_distance.png, combined_mean_bias_by_gender.png

Error histogram:
  - error_histogram.png — signed combined error (``error_combined_m``), one value per row

``estimate_distance`` calls this automatically into your ``--output`` folder
(unless ``--no-error-plots``).

Usage:
    python -m face_diet_gui.processing.pixel_degree_calibration.visualize_estimate_errors ^
        --csv  D:/results/distance_estimates.csv ^
        --out  D:/results
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

POSE_ORDER: List[str] = ["90R", "45R", "0", "45L", "90L"]
FOV_ORDER: List[str] = ["TR", "TL", "BR", "BL", "C"]
GENDER_ORDER: List[str] = ["F", "M"]
GENDER_DISPLAY: Dict[str, str] = {"F": "Female", "M": "Male"}

ERR_COLS = ["error_w_m", "error_h_m", "error_combined_m"]
METHOD_LABELS = ["Width (W)", "Height (H)", "Combined"]
COMBINED_COLOR = "#55a868"

# Fixed y-axis limits so RMSE / bias PNGs are comparable across runs.
YLIM_RMSE_M = (0.0, 0.35)
YLIM_BIAS_M = (-0.20, 0.20)


def _rmse(errors: pd.Series) -> float:
    """sqrt(mean(e^2)); NaN if no finite values."""
    mse = errors.pow(2).mean(skipna=True)
    if pd.isna(mse) or mse < 0:
        return float("nan")
    return float(math.sqrt(mse))


def _mean_signed_error(errors: pd.Series) -> float:
    """Mean signed error (bias); NaN if no finite values."""
    m = errors.mean(skipna=True)
    return float(m) if pd.notna(m) else float("nan")


def _prepare_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    missing = [c for c in ["ground_truth_m"] + ERR_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing columns {missing}. Re-run estimate_distance.py or check path."
        )
    df = df.copy()
    df["ground_truth_m"] = pd.to_numeric(df["ground_truth_m"], errors="coerce")
    if "ground_truth_cm" in df.columns:
        df["ground_truth_cm"] = pd.to_numeric(df["ground_truth_cm"], errors="coerce")
    for c in ERR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["ground_truth_m"].notna()].copy()
    if df.empty:
        raise ValueError("No rows with valid ground_truth_m after loading CSV.")
    return df


def _ordered_categories(values: Sequence[str], preferred: List[str]) -> List[str]:
    u = [str(v).strip() for v in values if pd.notna(v) and str(v).strip() != ""]
    seen = []
    for p in preferred:
        if p in u and p not in seen:
            seen.append(p)
    for v in sorted(set(u)):
        if v not in seen:
            seen.append(v)
    return seen


def _plot_overall_rmse(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    rmses = [_rmse(df[c]) for c in ERR_COLS]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(METHOD_LABELS))
    colors = ["#4c72b0", "#dd8452", COMBINED_COLOR]
    ax.bar(x, rmses, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_LABELS)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Distance RMSE (all poses & FOV)\nsqrt(mean((estimate - truth)^2))")
    ax.set_ylim(YLIM_RMSE_M)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_grouped_rmse_by_category(
    df: pd.DataFrame,
    category_col: str,
    category_order: List[str],
    title: str,
    subtitle: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    sub = df[df[category_col].notna() & (df[category_col].astype(str).str.strip() != "")].copy()
    if sub.empty:
        print(f"  [skip] {out_path.name}: no rows with column {category_col!r}")
        return
    cats = _ordered_categories(sub[category_col].unique(), category_order)
    rmse_w: List[float] = []
    rmse_h: List[float] = []
    rmse_c: List[float] = []
    counts: List[int] = []
    for cat in cats:
        block = sub[sub[category_col].astype(str) == cat]
        counts.append(len(block))
        rmse_w.append(_rmse(block["error_w_m"]))
        rmse_h.append(_rmse(block["error_h_m"]))
        rmse_c.append(_rmse(block["error_combined_m"]))

    x = np.arange(len(cats))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(7.0, len(cats) * 1.1), 4.5))
    ax.bar(x - width, rmse_w, width, label=METHOD_LABELS[0], color="#4c72b0")
    ax.bar(x, rmse_h, width, label=METHOD_LABELS[1], color="#dd8452")
    ax.bar(x + width, rmse_c, width, label=METHOD_LABELS[2], color=COMBINED_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cats, counts)])
    ax.set_ylabel("RMSE (m)")
    ax.set_title(f"{title}\n{subtitle}")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(YLIM_RMSE_M)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _tick_label_for_gt_distance(block: pd.DataFrame, d_m: float) -> str:
    """Prefer integer cm from CSV when available, else metres."""
    if "ground_truth_cm" in block.columns:
        cm = block["ground_truth_cm"].dropna()
        if len(cm):
            try:
                v = float(cm.iloc[0])
                if abs(v - round(v)) < 1e-6:
                    return f"{int(round(v))} cm"
            except (TypeError, ValueError):
                pass
    return f"{d_m:g} m"


def _plot_rmse_by_ground_truth_distance(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    sub = df[df["ground_truth_m"].notna()].copy()
    if sub.empty:
        print(f"  [skip] {out_path.name}: no ground_truth_m values")
        return

    distances = np.sort(sub["ground_truth_m"].dropna().unique())
    if len(distances) == 0:
        print(f"  [skip] {out_path.name}: no unique ground-truth distances")
        return

    rmse_w: List[float] = []
    rmse_h: List[float] = []
    rmse_c: List[float] = []
    counts: List[int] = []
    labels: List[str] = []
    for d in distances:
        block = sub[np.isclose(sub["ground_truth_m"].to_numpy(dtype=float), float(d), rtol=0, atol=1e-6)]
        counts.append(len(block))
        rmse_w.append(_rmse(block["error_w_m"]))
        rmse_h.append(_rmse(block["error_h_m"]))
        rmse_c.append(_rmse(block["error_combined_m"]))
        labels.append(_tick_label_for_gt_distance(block, float(d)))

    x = np.arange(len(distances))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(7.0, len(distances) * 1.0), 4.5))
    ax.bar(x - width, rmse_w, width, label=METHOD_LABELS[0], color="#4c72b0")
    ax.bar(x, rmse_h, width, label=METHOD_LABELS[1], color="#dd8452")
    ax.bar(x + width, rmse_c, width, label=METHOD_LABELS[2], color=COMBINED_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, counts)])
    ax.set_ylabel("RMSE (m)")
    ax.set_title("RMSE by ground-truth distance\nAll poses and FOV positions pooled per distance")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(YLIM_RMSE_M)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_combined_only_by_category(
    df: pd.DataFrame,
    category_col: str,
    category_order: List[str],
    title: str,
    subtitle: str,
    out_path: Path,
    *,
    tick_label_map: Optional[Dict[str, str]] = None,
) -> None:
    """Single bar per category: RMSE of combined estimate only."""
    import matplotlib.pyplot as plt

    sub = df[df[category_col].notna() & (df[category_col].astype(str).str.strip() != "")].copy()
    if sub.empty:
        print(f"  [skip] {out_path.name}: no rows with column {category_col!r}")
        return
    cats = _ordered_categories(sub[category_col].unique(), category_order)
    rmses: List[float] = []
    counts: List[int] = []
    for cat in cats:
        block = sub[sub[category_col].astype(str) == cat]
        counts.append(len(block))
        rmses.append(_rmse(block["error_combined_m"]))

    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(max(6.5, len(cats) * 1.0), 4.2))
    ax.bar(x, rmses, color=COMBINED_COLOR, width=0.65, label="Combined estimate")
    ax.set_xticks(x)
    tick_names = [tick_label_map.get(c, c) if tick_label_map else c for c in cats]
    ax.set_xticklabels([f"{tn}\n(n={n})" for tn, n in zip(tick_names, counts)])
    ax.set_ylabel("RMSE (m)")
    ax.set_title(f"{title}\n{subtitle}")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(YLIM_RMSE_M)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_combined_rmse_by_gender(df: pd.DataFrame, out_path: Path) -> None:
    """Female vs Male: RMSE of combined estimate (rows with gender F or M in CSV)."""
    import matplotlib.pyplot as plt

    if "gender" not in df.columns:
        print(f"  [skip] {out_path.name}: no gender column in CSV")
        return

    sub = df[df["gender"].notna() & (df["gender"].astype(str).str.strip() != "")].copy()
    sub["gender"] = sub["gender"].astype(str).str.strip().str.upper()
    sub = sub[sub["gender"].isin(GENDER_ORDER)]
    if sub.empty:
        print(f"  [skip] {out_path.name}: no rows with gender F or M")
        return

    cats = _ordered_categories(sub["gender"].unique(), GENDER_ORDER)
    rmses: List[float] = []
    counts: List[int] = []
    for cat in cats:
        block = sub[sub["gender"] == cat]
        counts.append(len(block))
        rmses.append(_rmse(block["error_combined_m"]))

    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(max(5.5, len(cats) * 1.2), 4.2))
    ax.bar(x, rmses, color=COMBINED_COLOR, width=0.55, label="Combined estimate")
    ax.set_xticks(x)
    tick_names = [GENDER_DISPLAY.get(c, c) for c in cats]
    ax.set_xticklabels([f"{tn}\n(n={n})" for tn, n in zip(tick_names, counts)])
    ax.set_ylabel("RMSE (m)")
    ax.set_title(
        "Combined estimate RMSE by gender\n"
        "All poses, FOV positions, and distances pooled"
    )
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(YLIM_RMSE_M)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_combined_only_by_ground_truth_distance(df: pd.DataFrame, out_path: Path) -> None:
    """One bar per ground-truth distance: RMSE of combined estimate only."""
    import matplotlib.pyplot as plt

    sub = df[df["ground_truth_m"].notna()].copy()
    if sub.empty:
        print(f"  [skip] {out_path.name}: no ground_truth_m values")
        return

    distances = np.sort(sub["ground_truth_m"].dropna().unique())
    if len(distances) == 0:
        print(f"  [skip] {out_path.name}: no unique ground-truth distances")
        return

    rmses: List[float] = []
    counts: List[int] = []
    labels: List[str] = []
    for d in distances:
        block = sub[np.isclose(sub["ground_truth_m"].to_numpy(dtype=float), float(d), rtol=0, atol=1e-6)]
        counts.append(len(block))
        rmses.append(_rmse(block["error_combined_m"]))
        labels.append(_tick_label_for_gt_distance(block, float(d)))

    x = np.arange(len(distances))
    fig, ax = plt.subplots(figsize=(max(6.5, len(distances) * 1.0), 4.2))
    ax.bar(x, rmses, color=COMBINED_COLOR, width=0.65, label="Combined estimate")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, counts)])
    ax.set_ylabel("RMSE (m)")
    ax.set_title(
        "Combined estimate RMSE by ground-truth distance\n"
        "All poses and FOV positions pooled per distance"
    )
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(YLIM_RMSE_M)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── mean signed error (bias) figures ───────────────────────────────────────


def _plot_overall_mean_bias(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    means = [_mean_signed_error(df[c]) for c in ERR_COLS]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(METHOD_LABELS))
    colors = ["#4c72b0", "#dd8452", COMBINED_COLOR]
    ax.bar(x, means, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_LABELS)
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Mean error (m)")
    ax.set_title("Mean signed error / bias (all poses & FOV)\nmean(estimate - truth)")
    ax.set_ylim(YLIM_BIAS_M)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_grouped_mean_bias_by_category(
    df: pd.DataFrame,
    category_col: str,
    category_order: List[str],
    title: str,
    subtitle: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    sub = df[df[category_col].notna() & (df[category_col].astype(str).str.strip() != "")].copy()
    if sub.empty:
        print(f"  [skip] {out_path.name}: no rows with column {category_col!r}")
        return
    cats = _ordered_categories(sub[category_col].unique(), category_order)
    mw: List[float] = []
    mh: List[float] = []
    mc: List[float] = []
    counts: List[int] = []
    for cat in cats:
        block = sub[sub[category_col].astype(str) == cat]
        counts.append(len(block))
        mw.append(_mean_signed_error(block["error_w_m"]))
        mh.append(_mean_signed_error(block["error_h_m"]))
        mc.append(_mean_signed_error(block["error_combined_m"]))

    x = np.arange(len(cats))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(7.0, len(cats) * 1.1), 4.5))
    ax.bar(x - width, mw, width, label=METHOD_LABELS[0], color="#4c72b0")
    ax.bar(x, mh, width, label=METHOD_LABELS[1], color="#dd8452")
    ax.bar(x + width, mc, width, label=METHOD_LABELS[2], color=COMBINED_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cats, counts)])
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Mean error (m)")
    ax.set_title(f"{title}\n{subtitle}")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(YLIM_BIAS_M)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_mean_bias_by_ground_truth_distance(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    sub = df[df["ground_truth_m"].notna()].copy()
    if sub.empty:
        print(f"  [skip] {out_path.name}: no ground_truth_m values")
        return

    distances = np.sort(sub["ground_truth_m"].dropna().unique())
    if len(distances) == 0:
        print(f"  [skip] {out_path.name}: no unique ground-truth distances")
        return

    mw: List[float] = []
    mh: List[float] = []
    mc: List[float] = []
    counts: List[int] = []
    labels: List[str] = []
    for d in distances:
        block = sub[np.isclose(sub["ground_truth_m"].to_numpy(dtype=float), float(d), rtol=0, atol=1e-6)]
        counts.append(len(block))
        mw.append(_mean_signed_error(block["error_w_m"]))
        mh.append(_mean_signed_error(block["error_h_m"]))
        mc.append(_mean_signed_error(block["error_combined_m"]))
        labels.append(_tick_label_for_gt_distance(block, float(d)))

    x = np.arange(len(distances))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(7.0, len(distances) * 1.0), 4.5))
    ax.bar(x - width, mw, width, label=METHOD_LABELS[0], color="#4c72b0")
    ax.bar(x, mh, width, label=METHOD_LABELS[1], color="#dd8452")
    ax.bar(x + width, mc, width, label=METHOD_LABELS[2], color=COMBINED_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, counts)])
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Mean error (m)")
    ax.set_title(
        "Mean signed error by ground-truth distance\n"
        "All poses and FOV positions pooled per distance"
    )
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(YLIM_BIAS_M)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_combined_mean_bias_by_category(
    df: pd.DataFrame,
    category_col: str,
    category_order: List[str],
    title: str,
    subtitle: str,
    out_path: Path,
    *,
    tick_label_map: Optional[Dict[str, str]] = None,
) -> None:
    import matplotlib.pyplot as plt

    sub = df[df[category_col].notna() & (df[category_col].astype(str).str.strip() != "")].copy()
    if sub.empty:
        print(f"  [skip] {out_path.name}: no rows with column {category_col!r}")
        return
    cats = _ordered_categories(sub[category_col].unique(), category_order)
    means: List[float] = []
    counts: List[int] = []
    for cat in cats:
        block = sub[sub[category_col].astype(str) == cat]
        counts.append(len(block))
        means.append(_mean_signed_error(block["error_combined_m"]))

    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(max(6.5, len(cats) * 1.0), 4.2))
    ax.bar(x, means, color=COMBINED_COLOR, width=0.65, label="Combined estimate")
    ax.set_xticks(x)
    tick_names = [tick_label_map.get(c, c) if tick_label_map else c for c in cats]
    ax.set_xticklabels([f"{tn}\n(n={n})" for tn, n in zip(tick_names, counts)])
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Mean error (m)")
    ax.set_title(f"{title}\n{subtitle}")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(YLIM_BIAS_M)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_combined_mean_bias_by_gender(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if "gender" not in df.columns:
        print(f"  [skip] {out_path.name}: no gender column in CSV")
        return

    sub = df[df["gender"].notna() & (df["gender"].astype(str).str.strip() != "")].copy()
    sub["gender"] = sub["gender"].astype(str).str.strip().str.upper()
    sub = sub[sub["gender"].isin(GENDER_ORDER)]
    if sub.empty:
        print(f"  [skip] {out_path.name}: no rows with gender F or M")
        return

    cats = _ordered_categories(sub["gender"].unique(), GENDER_ORDER)
    means: List[float] = []
    counts: List[int] = []
    for cat in cats:
        block = sub[sub["gender"] == cat]
        counts.append(len(block))
        means.append(_mean_signed_error(block["error_combined_m"]))

    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(max(5.5, len(cats) * 1.2), 4.2))
    ax.bar(x, means, color=COMBINED_COLOR, width=0.55, label="Combined estimate")
    ax.set_xticks(x)
    tick_names = [GENDER_DISPLAY.get(c, c) for c in cats]
    ax.set_xticklabels([f"{tn}\n(n={n})" for tn, n in zip(tick_names, counts)])
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Mean error (m)")
    ax.set_title(
        "Combined estimate mean signed error by gender\n"
        "All poses, FOV positions, and distances pooled"
    )
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(YLIM_BIAS_M)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_combined_mean_bias_by_ground_truth_distance(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    sub = df[df["ground_truth_m"].notna()].copy()
    if sub.empty:
        print(f"  [skip] {out_path.name}: no ground_truth_m values")
        return

    distances = np.sort(sub["ground_truth_m"].dropna().unique())
    if len(distances) == 0:
        print(f"  [skip] {out_path.name}: no unique ground-truth distances")
        return

    means: List[float] = []
    counts: List[int] = []
    labels: List[str] = []
    for d in distances:
        block = sub[np.isclose(sub["ground_truth_m"].to_numpy(dtype=float), float(d), rtol=0, atol=1e-6)]
        counts.append(len(block))
        means.append(_mean_signed_error(block["error_combined_m"]))
        labels.append(_tick_label_for_gt_distance(block, float(d)))

    x = np.arange(len(distances))
    fig, ax = plt.subplots(figsize=(max(6.5, len(distances) * 1.0), 4.2))
    ax.bar(x, means, color=COMBINED_COLOR, width=0.65, label="Combined estimate")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, counts)])
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Mean error (m)")
    ax.set_title(
        "Combined estimate mean signed error by ground-truth distance\n"
        "All poses and FOV positions pooled per distance"
    )
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(YLIM_BIAS_M)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_combined_error_histogram(df: pd.DataFrame, out_path: Path) -> None:
    """Histogram of signed combined distance error (m), one bin count per CSV row."""
    import matplotlib.pyplot as plt

    errs = pd.to_numeric(df["error_combined_m"], errors="coerce").to_numpy(dtype=float)
    errs = errs[np.isfinite(errs)]
    if errs.size == 0:
        print(f"  [skip] {out_path.name}: no finite error_combined_m values")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errs, bins="auto", color="#4c72b0", edgecolor="white", linewidth=0.6, alpha=0.92)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Signed error (m)  —  combined estimate − ground truth")
    ax.set_ylabel("Count")
    ax.set_title(
        "Combined distance error (all samples)\n"
        "geometric-mean integrated estimate vs ground truth"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(csv_path: str | Path, out_dir: str | Path | None = None) -> None:
    # Non-interactive backend: avoids GUI / main-thread issues when saving PNGs on some systems.
    import matplotlib

    matplotlib.use("Agg")

    csv_path = Path(csv_path)
    out_dir = Path(out_dir) if out_dir is not None else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_abs = out_dir.resolve()

    try:
        df = _prepare_frame(csv_path)
    except ValueError as exc:
        raise ValueError(
            f"{exc} "
            "Plots need at least one row with a numeric ground_truth_m "
            "(experiment-style filenames such as sub-01_F_dist-200_pose-0_TR)."
        ) from exc

    def _try_plot(filename: str, fn) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  [plot warning] {filename}: {exc}")

    _try_plot(
        "error_histogram.png",
        lambda: _plot_combined_error_histogram(df, out_dir / "error_histogram.png"),
    )
    _try_plot(
        "rmse_overall.png",
        lambda: _plot_overall_rmse(df, out_dir / "rmse_overall.png"),
    )
    _try_plot(
        "rmse_by_pose.png",
        lambda: _plot_grouped_rmse_by_category(
            df,
            "pose",
            POSE_ORDER,
            "RMSE by pose",
            "All FOV positions pooled per pose",
            out_dir / "rmse_by_pose.png",
        ),
    )
    _try_plot(
        "rmse_by_fov.png",
        lambda: _plot_grouped_rmse_by_category(
            df,
            "fov_position",
            FOV_ORDER,
            "RMSE by FOV position",
            "All poses pooled per FOV",
            out_dir / "rmse_by_fov.png",
        ),
    )
    _try_plot(
        "rmse_by_ground_truth_distance.png",
        lambda: _plot_rmse_by_ground_truth_distance(
            df, out_dir / "rmse_by_ground_truth_distance.png"
        ),
    )
    _try_plot(
        "combined_rmse_by_pose.png",
        lambda: _plot_combined_only_by_category(
            df,
            "pose",
            POSE_ORDER,
            "Combined estimate RMSE by pose",
            "All FOV positions pooled per pose",
            out_dir / "combined_rmse_by_pose.png",
        ),
    )
    _try_plot(
        "combined_rmse_by_fov.png",
        lambda: _plot_combined_only_by_category(
            df,
            "fov_position",
            FOV_ORDER,
            "Combined estimate RMSE by FOV position",
            "All poses pooled per FOV",
            out_dir / "combined_rmse_by_fov.png",
        ),
    )
    _try_plot(
        "combined_rmse_by_ground_truth_distance.png",
        lambda: _plot_combined_only_by_ground_truth_distance(
            df, out_dir / "combined_rmse_by_ground_truth_distance.png"
        ),
    )
    _try_plot(
        "combined_rmse_by_gender.png",
        lambda: _plot_combined_rmse_by_gender(df, out_dir / "combined_rmse_by_gender.png"),
    )

    # Mean signed error (bias): positive bar => mean over-estimate of distance (m).
    _try_plot(
        "mean_bias_overall.png",
        lambda: _plot_overall_mean_bias(df, out_dir / "mean_bias_overall.png"),
    )
    _try_plot(
        "mean_bias_by_pose.png",
        lambda: _plot_grouped_mean_bias_by_category(
            df,
            "pose",
            POSE_ORDER,
            "Mean signed error by pose",
            "All FOV positions pooled per pose",
            out_dir / "mean_bias_by_pose.png",
        ),
    )
    _try_plot(
        "mean_bias_by_fov.png",
        lambda: _plot_grouped_mean_bias_by_category(
            df,
            "fov_position",
            FOV_ORDER,
            "Mean signed error by FOV position",
            "All poses pooled per FOV",
            out_dir / "mean_bias_by_fov.png",
        ),
    )
    _try_plot(
        "mean_bias_by_ground_truth_distance.png",
        lambda: _plot_mean_bias_by_ground_truth_distance(
            df, out_dir / "mean_bias_by_ground_truth_distance.png"
        ),
    )
    _try_plot(
        "combined_mean_bias_by_pose.png",
        lambda: _plot_combined_mean_bias_by_category(
            df,
            "pose",
            POSE_ORDER,
            "Combined estimate mean signed error by pose",
            "All FOV positions pooled per pose",
            out_dir / "combined_mean_bias_by_pose.png",
        ),
    )
    _try_plot(
        "combined_mean_bias_by_fov.png",
        lambda: _plot_combined_mean_bias_by_category(
            df,
            "fov_position",
            FOV_ORDER,
            "Combined estimate mean signed error by FOV position",
            "All poses pooled per FOV",
            out_dir / "combined_mean_bias_by_fov.png",
        ),
    )
    _try_plot(
        "combined_mean_bias_by_ground_truth_distance.png",
        lambda: _plot_combined_mean_bias_by_ground_truth_distance(
            df, out_dir / "combined_mean_bias_by_ground_truth_distance.png"
        ),
    )
    _try_plot(
        "combined_mean_bias_by_gender.png",
        lambda: _plot_combined_mean_bias_by_gender(
            df, out_dir / "combined_mean_bias_by_gender.png"
        ),
    )

    written = sorted(out_dir.glob("*.png"))
    print(f"PNG plots ({len(written)} file(s)) -> {out_abs}")
    for p in written:
        print(f"  {p.name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot combined-error histogram, RMSE, and mean signed error (bias) from distance_estimates.csv."
    )
    p.add_argument("--csv", required=True, type=Path, help="Path to distance_estimates.csv.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output folder for PNGs (default: same folder as the CSV).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.csv, args.out)


if __name__ == "__main__":
    main()
