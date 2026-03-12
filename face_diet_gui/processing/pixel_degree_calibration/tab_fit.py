"""
Tab 2 — Fit Mapping

Layout:
  ┌─ Inputs ───────────────────────────────────────────────────────────────┐
  │  Masks dir:  [___________] Browse      Output dir: [___________] Browse │
  │  Diameter(m): [0.19]  Distance(m): [1.0]  Poly order: [2]              │
  └────────────────────────────────────────────────────────────────────────┘
  [  Compute Samples & Fit Mapping  ]
  ┌─ Results ──────────────────────────────────────────────────────────────┐
  │  n samples: —    scalar RMSE: —    x RMSE: —    y RMSE: —             │
  └────────────────────────────────────────────────────────────────────────┘
  ┌─ Heatmaps ─────────────────────────────────────────────────────────────┐
  │  [Scalar PPD]  [Horizontal PPD]  [Vertical PPD]                        │
  │  ┌──────────────────────────────────────────────────────┐ ┌── cbar ──┐ │
  │  │  tk.Canvas — viridis heatmap, hover shows values     │ │ gradient │ │
  │  └──────────────────────────────────────────────────────┘ └──────────┘ │
  │  Hover: x=432px  y=315px  |  Scalar: 85.3  X: 84.1  Y: 86.2 px/deg   │
  └────────────────────────────────────────────────────────────────────────┘
  ┌─ Log ──────────────────────────────────────────────────────────────────┐
  │  CTkTextbox (read-only, console output)                                │
  └────────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import io
import json
import sys
import threading
import tkinter as tk
from pathlib import Path
from typing import Dict, Optional, Tuple

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from face_diet_gui.processing.pixel_degree_calibration.visualize_mapping import (
    CBAR_TOTAL_W as _CBAR_TOTAL_W,
    make_heatmap_pil  as _make_heatmap_pil,
    viridis_rgb       as _viridis_rgb,
)

_GRID_STEP = 8   # evaluation grid step in pixels


# ── vectorised grid evaluation ────────────────────────────────────────────

def _eval_grids(mapping: Dict, step: int = _GRID_STEP
                ) -> Tuple[np.ndarray, np.ndarray,
                           np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised 2-D polynomial evaluation across the full image grid.
    Returns (xs, ys, scalar_grid, x_grid, y_grid).
    scalar_grid is derived as sqrt(x_grid * y_grid).
    """
    w   = mapping["image_width"]
    h   = mapping["image_height"]
    cx  = mapping["center_x"]
    cy  = mapping["center_y"]
    xs  = np.arange(0, w, step, dtype=np.float64)
    ys  = np.arange(0, h, step, dtype=np.float64)
    xg, yg = np.meshgrid(xs, ys)

    x_norm = (xg - cx) / cx
    y_norm = (yg - cy) / cy

    exponents = mapping["exponents"]
    cols = [(x_norm ** i) * (y_norm ** j) for i, j in exponents]
    X = np.stack(cols, axis=-1)

    cx_ = np.array(mapping["coefficients_x"], dtype=np.float64)
    cy_ = np.array(mapping["coefficients_y"], dtype=np.float64)

    x_grid = (X * cx_).sum(-1)
    y_grid = (X * cy_).sum(-1)
    scalar_grid = np.sqrt(np.maximum(x_grid * y_grid, 0.0))

    return xs, ys, scalar_grid, x_grid, y_grid


# ── tiny stream ───────────────────────────────────────────────────────────

class _TextboxStream(io.TextIOBase):
    def __init__(self, widget: ctk.CTkTextbox) -> None:
        super().__init__()
        self._widget = widget

    def write(self, s: str) -> int:
        self._widget.after(0, self._append, s)
        return len(s)

    def _append(self, s: str) -> None:
        self._widget.configure(state="normal")
        self._widget.insert("end", s)
        self._widget.see("end")
        self._widget.configure(state="disabled")

    def flush(self) -> None:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# FitTab
# ══════════════════════════════════════════════════════════════════════════════

class FitTab(ctk.CTkFrame):
    """Tab 2: run compute_samples → fit_mapping, display results & heatmaps."""

    _HEATMAP_KEYS = [
        ("Scalar PPD",     "coefficients_scalar"),
        ("Horizontal PPD", "coefficients_x"),
        ("Vertical PPD",   "coefficients_y"),
    ]
    # Maps label → grid key in _grids dict
    _GRID_KEY = {
        "Scalar PPD":     "scalar",
        "Horizontal PPD": "x",
        "Vertical PPD":   "y",
    }

    def __init__(self, parent: tk.Widget, seg_tab, **kwargs) -> None:
        super().__init__(parent, fg_color="transparent", **kwargs)
        self._seg_tab = seg_tab

        self._output_dir: Optional[Path] = None
        self._selected_heatmap: str = "Scalar PPD"

        # Evaluated grids (filled after fitting)
        self._mapping_data: Optional[Dict]  = None
        self._grids: Dict[str, np.ndarray]  = {}
        self._grid_xs: Optional[np.ndarray] = None
        self._grid_ys: Optional[np.ndarray] = None

        # Sample points: list of (center_x, center_y, ppd_scalar, ppd_x, ppd_y)
        self._samples: list = []

        # Canvas rendering state
        self._heat_tk_photo = None   # PhotoImage reference (must be kept alive)
        self._heat_plot_w: int = 0   # pixel width of heatmap area within canvas
        self._heat_plot_h: int = 0   # pixel height

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        # ── Inputs ──
        inp = ctk.CTkFrame(self, corner_radius=8)
        inp.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 4))

        row0 = ctk.CTkFrame(inp, fg_color="transparent")
        row0.pack(fill="x", padx=10, pady=(8, 4))

        ctk.CTkLabel(row0, text="Masks dir:", anchor="w").pack(side="left", padx=(0, 4))
        self._masks_var = tk.StringVar()
        ctk.CTkEntry(row0, textvariable=self._masks_var, width=300).pack(side="left", padx=(0, 4))
        ctk.CTkButton(row0, text="Browse", width=70, command=self._browse_masks).pack(side="left", padx=(0, 20))

        ctk.CTkLabel(row0, text="Output dir:", anchor="w").pack(side="left", padx=(0, 4))
        self._output_var = tk.StringVar()
        ctk.CTkEntry(row0, textvariable=self._output_var, width=300).pack(side="left", padx=(0, 4))
        ctk.CTkButton(row0, text="Browse", width=70, command=self._browse_output).pack(side="left")

        row1 = ctk.CTkFrame(inp, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=(0, 8))

        self._diameter_var = tk.StringVar(value="0.19")
        self._distance_var = tk.StringVar(value="1.0")
        self._order_var    = tk.StringVar(value="2")

        for label, var, width in [
            ("Target diameter (m):", self._diameter_var, 70),
            ("Distance (m):",        self._distance_var, 70),
            ("Polynomial order:",    self._order_var,    50),
        ]:
            ctk.CTkLabel(row1, text=label).pack(side="left", padx=(0, 4))
            ctk.CTkEntry(row1, textvariable=var, width=width).pack(side="left", padx=(0, 16))

        # ── Fit button ──
        self._fit_btn = ctk.CTkButton(
            self,
            text="  Compute Samples & Fit Mapping  ",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            command=self._run_fit,
        )
        self._fit_btn.grid(row=1, column=0, pady=(0, 6), padx=80)

        # ── Results ──
        res = ctk.CTkFrame(self, corner_radius=8)
        res.grid(row=2, column=0, sticky="ew", padx=4, pady=(0, 4))

        res_inner = ctk.CTkFrame(res, fg_color="transparent")
        res_inner.pack(fill="x", padx=10, pady=6)

        self._lbl_n      = ctk.CTkLabel(res_inner, text="Samples: —")
        self._lbl_rmse_x = ctk.CTkLabel(res_inner, text="X RMSE: —")
        self._lbl_rmse_y = ctk.CTkLabel(res_inner, text="Y RMSE: —")
        for lbl in (self._lbl_n, self._lbl_rmse_x, self._lbl_rmse_y):
            lbl.pack(side="left", padx=(0, 24))

        # ── Lower area: heatmaps (left) + log (right) ──
        lower = ctk.CTkFrame(self, fg_color="transparent")
        lower.grid(row=3, column=0, sticky="nsew", padx=4, pady=(0, 4))
        lower.columnconfigure(0, weight=3)
        lower.columnconfigure(1, weight=2)
        lower.rowconfigure(0, weight=1)

        # -- heatmap panel --
        heat_outer = ctk.CTkFrame(lower, corner_radius=8)
        heat_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        heat_outer.rowconfigure(1, weight=1)
        heat_outer.columnconfigure(0, weight=1)

        # title + tab buttons on same row
        heat_title_row = ctk.CTkFrame(heat_outer, fg_color="transparent")
        heat_title_row.grid(row=0, column=0, sticky="ew", padx=10, pady=(6, 2))
        heat_title_row.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            heat_title_row, text="Heatmaps", font=ctk.CTkFont(weight="bold"), anchor="w"
        ).grid(row=0, column=0, sticky="w")

        tab_bar = ctk.CTkFrame(heat_title_row, fg_color="transparent")
        tab_bar.grid(row=0, column=1, sticky="e")

        self._heat_buttons: Dict[str, ctk.CTkButton] = {}
        for label, _ in self._HEATMAP_KEYS:
            btn = ctk.CTkButton(
                tab_bar, text=label, width=120, height=28,
                command=lambda l=label: self._show_heatmap(l),
                fg_color="gray20",
            )
            btn.pack(side="left", padx=2)
            self._heat_buttons[label] = btn

        # tk.Canvas for interactive heatmap display
        self._heat_canvas = tk.Canvas(
            heat_outer, bg="#1c1c1c", highlightthickness=0, cursor="crosshair"
        )
        self._heat_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 2))
        self._heat_canvas.bind("<Configure>", self._on_canvas_resize)
        self._heat_canvas.bind("<Motion>",    self._on_heat_motion)
        self._heat_canvas.bind("<Leave>",     self._on_heat_leave)

        # placeholder text shown before fitting
        self._heat_placeholder = self._heat_canvas.create_text(
            20, 20, text="(run fitting to see heatmaps)",
            fill="#555555", anchor="nw", font=("Segoe UI", 11),
        )

        # hover info label
        self._hover_lbl = ctk.CTkLabel(
            heat_outer, text="", text_color="gray",
            font=ctk.CTkFont(family="Consolas", size=11), anchor="w",
        )
        self._hover_lbl.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 6))

        # -- log panel --
        log_outer = ctk.CTkFrame(lower, corner_radius=8)
        log_outer.grid(row=0, column=1, sticky="nsew")
        log_outer.rowconfigure(1, weight=1)
        log_outer.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            log_outer, text="Log", font=ctk.CTkFont(weight="bold"), anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(6, 2))

        self._log_box = ctk.CTkTextbox(
            log_outer, state="disabled", font=ctk.CTkFont(family="Consolas", size=11)
        )
        self._log_box.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        # Activate first button
        self._show_heatmap("Scalar PPD")

    # ── public: sync masks dir ────────────────────────────────────────────────

    def sync_masks_dir_from_seg(self) -> None:
        if self._seg_tab and self._seg_tab.masks_dir:
            self._masks_var.set(str(self._seg_tab.masks_dir))

    # ── browse ────────────────────────────────────────────────────────────────

    def _browse_masks(self) -> None:
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Select masks directory")
        if d:
            self._masks_var.set(d)

    def _browse_output(self) -> None:
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self._output_var.set(d)
            self._output_dir = Path(d)

    # ── fitting pipeline ──────────────────────────────────────────────────────

    def _run_fit(self) -> None:
        masks_dir_str  = self._masks_var.get().strip()
        output_dir_str = self._output_var.get().strip()

        if not masks_dir_str:
            self._log("ERROR: set the masks directory first.\n")
            return
        if not output_dir_str:
            output_dir_str = masks_dir_str
            self._output_var.set(output_dir_str)

        try:
            diameter = float(self._diameter_var.get())
            distance = float(self._distance_var.get())
            order    = int(self._order_var.get())
        except ValueError:
            self._log("ERROR: invalid numeric parameters.\n")
            return

        masks_dir  = Path(masks_dir_str)
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = output_dir

        samples_path = output_dir / "ppd_samples.json"
        mapping_path = output_dir / "ppd_mapping.json"

        self._fit_btn.configure(state="disabled", text="Running…")
        self._log_clear()

        def worker():
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stream = _TextboxStream(self._log_box)
            sys.stdout = stream
            sys.stderr = stream
            try:
                from face_diet_gui.processing.pixel_degree_calibration.compute_samples import (
                    compute_samples,
                )
                from face_diet_gui.processing.pixel_degree_calibration.fit_mapping import (
                    fit_mapping,
                )
                compute_samples(
                    masks_dir=str(masks_dir),
                    circle_diameter_m=diameter,
                    distance_m=distance,
                    output_path=str(samples_path),
                )
                fit_mapping(
                    samples_path=str(samples_path),
                    output_path=str(mapping_path),
                    order=order,
                )
                self.after(0, self._on_fit_done, str(mapping_path))
            except Exception as exc:
                print(f"\nERROR: {exc}")
                import traceback
                traceback.print_exc()
                self.after(0, self._on_fit_error)
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

        threading.Thread(target=worker, daemon=True).start()

    def _on_fit_done(self, mapping_path: str) -> None:
        self._fit_btn.configure(state="normal", text="  Compute Samples & Fit Mapping  ")
        self._load_results(mapping_path)
        self._load_grids(mapping_path)
        self._show_heatmap(self._selected_heatmap)

    def _on_fit_error(self) -> None:
        self._fit_btn.configure(state="normal", text="  Compute Samples & Fit Mapping  ")

    # ── results labels ────────────────────────────────────────────────────────

    def _load_results(self, mapping_path: str) -> None:
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            self._lbl_n.configure(text=f"Samples: {m.get('num_samples', '—')}")
            self._lbl_rmse_x.configure(text=f"X RMSE: {m.get('fit_rmse_x', 0):.4f} px/deg")
            self._lbl_rmse_y.configure(text=f"Y RMSE: {m.get('fit_rmse_y', 0):.4f} px/deg")
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    # ── grid computation ──────────────────────────────────────────────────────

    def _load_grids(self, mapping_path: str) -> None:
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                self._mapping_data = json.load(f)
            xs, ys, s, x, y = _eval_grids(self._mapping_data, step=_GRID_STEP)
            self._grid_xs = xs
            self._grid_ys = ys
            self._grids = {"scalar": s, "x": x, "y": y}
        except Exception as exc:
            self._log(f"[warn] Could not compute grids for hover: {exc}\n")

        # Load sample centre points from ppd_samples.json (same directory)
        self._samples = []
        try:
            samples_path = Path(mapping_path).parent / "ppd_samples.json"
            if samples_path.exists():
                with samples_path.open("r", encoding="utf-8") as f:
                    sdata = json.load(f)
                for s in sdata.get("samples", []):
                    self._samples.append((
                        float(s["center_x"]),
                        float(s["center_y"]),
                        float(s.get("pixels_per_degree",   0)),
                        float(s.get("pixels_per_degree_x", 0)),
                        float(s.get("pixels_per_degree_y", 0)),
                    ))
        except Exception as exc:
            self._log(f"[warn] Could not load sample points: {exc}\n")

    # ── heatmap display ───────────────────────────────────────────────────────

    def _show_heatmap(self, label: str) -> None:
        self._selected_heatmap = label

        for k, btn in self._heat_buttons.items():
            btn.configure(fg_color=["#3B8ED0", "#1F6AA5"] if k == label else "gray20")

        self._render_to_canvas()

    def _render_to_canvas(self) -> None:
        grid_key = self._GRID_KEY.get(self._selected_heatmap)
        grid     = self._grids.get(grid_key) if grid_key else None

        cw = max(self._heat_canvas.winfo_width(),  200)
        ch = max(self._heat_canvas.winfo_height(), 150)

        if grid is None:
            self._heat_canvas.delete("all")
            self._heat_canvas.create_text(
                cw // 2, ch // 2,
                text="(run fitting to see heatmaps)",
                fill="#555555", font=("Segoe UI", 11),
            )
            return

        vmin = float(np.nanmin(grid))
        vmax = float(np.nanmax(grid))

        plot_w = cw - _CBAR_TOTAL_W
        plot_h = ch

        # Build sample list in grid-relative coords (same convention as
        # visualize_mapping.make_heatmap_pil so both outputs look identical)
        samples_xy = None
        if self._samples and self._mapping_data is not None:
            grid_h, grid_w = grid.shape
            img_w = self._mapping_data.get("image_width", 1)
            img_h = self._mapping_data.get("image_height", 1)
            samples_xy = [
                (cx_ / img_w * grid_w, cy_ / img_h * grid_h)
                for cx_, cy_, *_ in self._samples
            ]

        pil = _make_heatmap_pil(grid, vmin, vmax, cw, ch, samples_xy=samples_xy)

        self._heat_tk_photo = ImageTk.PhotoImage(pil)
        self._heat_plot_w   = plot_w
        self._heat_plot_h   = plot_h

        self._heat_canvas.delete("all")
        self._heat_canvas.create_image(0, 0, image=self._heat_tk_photo, anchor="nw")

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self._render_to_canvas()

    # ── hover ─────────────────────────────────────────────────────────────────

    def _on_heat_motion(self, event: tk.Event) -> None:
        if self._mapping_data is None or self._grid_xs is None:
            return
        if self._heat_plot_w <= 0 or self._heat_plot_h <= 0:
            return

        # Ignore cursor over the colorbar area
        if event.x > self._heat_plot_w:
            self._hover_lbl.configure(text="")
            return

        # Map canvas pixel → image coordinate
        xs, ys = self._grid_xs, self._grid_ys
        img_x = xs[0] + (event.x / self._heat_plot_w) * (xs[-1] - xs[0])
        img_y = ys[0] + (event.y / self._heat_plot_h) * (ys[-1] - ys[0])

        # Evaluate PPD at hover point
        try:
            from face_diet_gui.processing.pixel_degree_calibration.mapping_utils import (
                evaluate_ppd, evaluate_ppd_xy,
            )
            s        = evaluate_ppd(self._mapping_data, img_x, img_y)
            px, py   = evaluate_ppd_xy(self._mapping_data, img_x, img_y)
            self._hover_lbl.configure(
                text=(
                    f"x={img_x:.0f}px  y={img_y:.0f}px  │  "
                    f"Scalar: {s:.1f}  X: {px:.1f}  Y: {py:.1f}  px/deg"
                )
            )
        except Exception:
            pass

    def _on_heat_leave(self, _event: tk.Event) -> None:
        self._hover_lbl.configure(text="")

    # ── log helpers ───────────────────────────────────────────────────────────

    def _log(self, text: str) -> None:
        self._log_box.configure(state="normal")
        self._log_box.insert("end", text)
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _log_clear(self) -> None:
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")
