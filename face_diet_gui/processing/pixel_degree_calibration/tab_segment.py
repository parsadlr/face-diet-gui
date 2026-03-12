"""
Tab 1 — Segment Targets

Layout:
  ┌─ Paths ─────────────────────────────────────────────────────────────────┐
  │  Samples dir: [________] Browse      Masks dir: [________] Browse       │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─ Image list ──────┐  ┌─ Canvas (image + mask overlay) ──────────────────┐
  │  frame_000.png  ● │  │                                                  │
  │  ...              │  │   Click on the target to segment it              │
  └───────────────────┘  └──────────────────────────────────────────────────┘
  ┌─ Actions ───────────────────────────────────────────────────────────────┐
  │  [Clear mask]  [Save mask]  [Save all]   Tolerance ±: [20]              │
  └─────────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import json
import math
import tkinter as tk
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

# ── constants ──────────────────────────────────────────────────────────────
_LIST_WIDTH = 220
_ICON_NONE = "○"
_ICON_UNSAVED = "◉"
_ICON_SAVED = "✓"
_OVERLAY_ALPHA = 0.45          # green tint opacity
_OVERLAY_COLOR = (0, 200, 80)  # BGR green


# ── segmentation helpers ────────────────────────────────────────────────────

def segment_at_click(
    image_bgr: np.ndarray,
    click_x: int,
    click_y: int,
    tolerance: int = 20,
) -> Optional[np.ndarray]:
    """
    Flood-fill "magic wand" segmentation.

    Samples the grayscale value of the seed pixel (click_x, click_y) and
    grows a region to all 4-connected neighbours whose grayscale value stays
    within ±tolerance of the seed.  Returns a binary mask (uint8, 0/255).

    Because the reference value is the seed pixel itself, the segmentation
    adapts automatically to dark targets at the image edges — no manual
    brightness adjustment is required.

    Returns None if the click is outside the image bounds.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Clamp coords to image bounds
    cy = max(0, min(click_y, h - 1))
    cx = max(0, min(click_x, w - 1))

    # floodFill needs a mask that is 2 pixels larger in each dimension
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    cv2.floodFill(
        gray.copy(), flood_mask,
        seedPoint=(cx, cy),
        newVal=0,
        loDiff=tolerance,
        upDiff=tolerance,
        flags=flags,
    )
    # Trim the 1-pixel border added by floodFill
    result = flood_mask[1:-1, 1:-1]
    if result[cy, cx] == 0:
        return None  # seed point was not filled (shouldn't normally happen)
    return result


def _ellipse_from_mask(mask: np.ndarray) -> Optional[dict]:
    """
    Fit an ellipse to the largest contour in mask and return a metadata dict
    in the same schema as detect_target.py.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None

    (cx, cy), (ma, mi), angle = cv2.fitEllipse(cnt)
    semi_major = ma / 2.0
    semi_minor = mi / 2.0
    radius_eq = math.sqrt(semi_major * semi_minor)

    return {
        "found": True,
        "center_x": float(cx),
        "center_y": float(cy),
        "semi_major_px": float(semi_major),
        "semi_minor_px": float(semi_minor),
        "angle_deg": float(angle),
        "radius_eq_px": float(radius_eq),
        "width": mask.shape[1],
        "height": mask.shape[0],
    }


# ── thumbnail + overlay helpers ────────────────────────────────────────────

def _fit_image_to_canvas(
    image_bgr: np.ndarray,
    canvas_w: int,
    canvas_h: int,
) -> Tuple[np.ndarray, int, int, float]:
    """Return (scaled_rgb, offset_x, offset_y, scale) fitting image inside canvas."""
    h, w = image_bgr.shape[:2]
    scale = min(canvas_w / w, canvas_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    scaled = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
    off_x = (canvas_w - new_w) // 2
    off_y = (canvas_h - new_h) // 2
    return rgb, off_x, off_y, scale


def _apply_overlay(rgb: np.ndarray, mask_resized: np.ndarray) -> np.ndarray:
    """Blend a green tint onto rgb where mask_resized > 0."""
    result = rgb.copy()
    m = mask_resized > 0
    color_rgb = (_OVERLAY_COLOR[2], _OVERLAY_COLOR[1], _OVERLAY_COLOR[0])
    for c, v in enumerate(color_rgb):
        result[:, :, c] = np.where(
            m,
            np.clip(
                (1 - _OVERLAY_ALPHA) * rgb[:, :, c] + _OVERLAY_ALPHA * v, 0, 255
            ).astype(np.uint8),
            rgb[:, :, c],
        )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SegmentTab
# ══════════════════════════════════════════════════════════════════════════════

class SegmentTab(ctk.CTkFrame):
    """
    Tab 1: browse calibration images, click to segment the target, save masks.
    """

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, fg_color="transparent", **kwargs)

        # ── state ────────────────────────────────────────────────────────────
        self._samples_dir: Optional[Path] = None
        self._masks_dir: Optional[Path] = None
        self._image_paths: List[Path] = []

        # Keyed by image filename stem → mask ndarray (uint8) in original size
        self._masks_memory: Dict[str, np.ndarray] = {}
        self._masks_saved: Dict[str, bool] = {}

        # Currently displayed image
        self._current_index: int = -1          # index into _image_paths
        self._current_name: Optional[str] = None
        self._current_bgr: Optional[np.ndarray] = None

        # For canvas → image coordinate mapping
        self._canvas_scale: float = 1.0
        self._canvas_off_x: int = 0
        self._canvas_off_y: int = 0
        self._tk_photo: Optional[ImageTk.PhotoImage] = None  # must hold reference

        self._build_ui()
        self._bind_keys()

    # ── keyboard navigation ───────────────────────────────────────────────────

    def _bind_keys(self) -> None:
        """Bind arrow keys on the canvas and, once the root window is available,
        on the root as well so navigation works regardless of focus."""
        for key in ("<Up>", "<Down>", "<Left>", "<Right>"):
            self._canvas.bind(key, self._on_arrow_key)
        # Bind on the root window lazily (root may not exist yet at __init__ time)
        self.after(100, self._bind_root_keys)

    def _bind_root_keys(self) -> None:
        try:
            root = self.winfo_toplevel()
            for key in ("<Up>", "<Down>", "<Left>", "<Right>"):
                root.bind(key, self._on_arrow_key)
        except Exception:
            pass

    def _on_arrow_key(self, event: tk.Event) -> None:
        if event.keysym in ("Up", "Left"):
            self._navigate(-1)
        elif event.keysym in ("Down", "Right"):
            self._navigate(1)

    def _navigate(self, delta: int) -> None:
        if not self._image_paths:
            return
        new_idx = max(0, min(len(self._image_paths) - 1, self._current_index + delta))
        if new_idx == self._current_index:
            return
        self._select_image_by_index(new_idx)

    def _select_image_by_index(self, idx: int) -> None:
        if not (0 <= idx < len(self._image_paths)):
            return
        self._current_index = idx
        self._select_image(self._image_paths[idx])
        self._scroll_list_to(idx)

    def _scroll_list_to(self, idx: int) -> None:
        """Scroll the image list so row at idx is visible."""
        try:
            total = len(self._image_paths)
            if total == 0:
                return
            fraction = idx / total
            self._list_scroll._parent_canvas.yview_moveto(fraction)
        except Exception:
            pass

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        # -- paths row --
        paths_frame = ctk.CTkFrame(self, corner_radius=8)
        paths_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(4, 4))

        ctk.CTkLabel(paths_frame, text="Samples dir:", anchor="w").pack(side="left", padx=(10, 4))
        self._samples_var = tk.StringVar()
        ctk.CTkEntry(paths_frame, textvariable=self._samples_var, width=280).pack(side="left", padx=(0, 4))
        ctk.CTkButton(paths_frame, text="Browse", width=70, command=self._browse_samples).pack(side="left", padx=(0, 16))

        ctk.CTkLabel(paths_frame, text="Masks dir:", anchor="w").pack(side="left", padx=(0, 4))
        self._masks_var = tk.StringVar()
        ctk.CTkEntry(paths_frame, textvariable=self._masks_var, width=280).pack(side="left", padx=(0, 4))
        ctk.CTkButton(paths_frame, text="Browse", width=70, command=self._browse_masks).pack(side="left", padx=(0, 10))

        # -- list panel (left) --
        list_frame = ctk.CTkFrame(self, width=_LIST_WIDTH, corner_radius=8)
        list_frame.grid(row=1, column=0, sticky="nsew", padx=(4, 2), pady=(0, 4))
        list_frame.grid_propagate(False)
        list_frame.rowconfigure(0, weight=0)
        list_frame.rowconfigure(1, weight=1)

        ctk.CTkLabel(list_frame, text="Images", font=ctk.CTkFont(weight="bold"), anchor="w").pack(
            fill="x", padx=10, pady=(8, 4)
        )

        self._list_scroll = ctk.CTkScrollableFrame(list_frame, fg_color="transparent")
        self._list_scroll.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        # -- canvas (right) --
        canvas_outer = ctk.CTkFrame(self, corner_radius=8)
        canvas_outer.grid(row=1, column=1, sticky="nsew", padx=(2, 4), pady=(0, 4))
        canvas_outer.rowconfigure(0, weight=1)
        canvas_outer.columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(
            canvas_outer,
            bg="#1e1e1e",
            highlightthickness=0,
            cursor="crosshair",
        )
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._canvas.bind("<Button-1>", self._on_canvas_click)
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        self._canvas_hint = self._canvas.create_text(
            20, 20,
            text="← Select an image from the list",
            fill="#555555",
            anchor="nw",
            font=("Segoe UI", 11),
        )

        # -- actions bar --
        act_frame = ctk.CTkFrame(self, corner_radius=8)
        act_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 4))

        ctk.CTkButton(act_frame, text="Clear mask", width=110, command=self._clear_mask).pack(
            side="left", padx=(10, 6), pady=6
        )
        ctk.CTkButton(act_frame, text="Save mask", width=110, command=self._save_current_mask).pack(
            side="left", padx=(0, 6), pady=6
        )
        ctk.CTkButton(act_frame, text="Save all", width=110, command=self._save_all_masks).pack(
            side="left", padx=(0, 20), pady=6
        )

        ctk.CTkLabel(act_frame, text="Tolerance ±:").pack(side="left", padx=(0, 4))
        self._thresh_var = tk.StringVar(value="20")
        thresh_entry = ctk.CTkEntry(act_frame, textvariable=self._thresh_var, width=50)
        thresh_entry.pack(side="left", padx=(0, 4))
        ctk.CTkButton(act_frame, text="▲", width=28, command=lambda: self._adjust_thresh(5)).pack(side="left", padx=1)
        ctk.CTkButton(act_frame, text="▼", width=28, command=lambda: self._adjust_thresh(-5)).pack(side="left", padx=1)

        self._status_label = ctk.CTkLabel(act_frame, text="", text_color="gray", anchor="w")
        self._status_label.pack(side="left", padx=(16, 0))

    # ── properties (read by FitTab) ──────────────────────────────────────────

    @property
    def masks_dir(self) -> Optional[Path]:
        return self._masks_dir

    # ── browse callbacks ─────────────────────────────────────────────────────

    def _browse_samples(self) -> None:
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Select samples directory")
        if d:
            self._samples_var.set(d)
            self._load_sample_dir(Path(d))

    def _browse_masks(self) -> None:
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Select masks directory")
        if d:
            self._masks_var.set(d)
            self._masks_dir = Path(d)
            self._masks_dir.mkdir(parents=True, exist_ok=True)
            self._reload_saved_state()
            self._refresh_list()

    # ── directory loading ────────────────────────────────────────────────────

    def _load_sample_dir(self, path: Path) -> None:
        self._samples_dir = path
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        self._image_paths = sorted(
            [p for p in path.iterdir() if p.suffix.lower() in exts],
            key=lambda p: p.name,
        )
        self._masks_memory.clear()
        self._masks_saved.clear()
        self._current_index = -1
        self._current_name = None
        self._current_bgr = None
        self._reload_saved_state()
        self._refresh_list()
        self._set_status(f"Loaded {len(self._image_paths)} images.")

    def _reload_saved_state(self) -> None:
        """Check which mask PNGs already exist on disk and mark them saved."""
        if self._masks_dir is None:
            return
        for name in list(self._masks_saved.keys()):
            self._masks_saved[name] = False
        for p in self._image_paths:
            mask_path = self._masks_dir / f"mask_{p.name}"
            if mask_path.exists():
                self._masks_saved[p.stem] = True

    # ── list rendering ───────────────────────────────────────────────────────

    def _refresh_list(self) -> None:
        for w in self._list_scroll.winfo_children():
            w.destroy()

        for idx, img_path in enumerate(self._image_paths):
            name = img_path.stem
            icon = self._icon_for(name)
            color = self._icon_color(name)
            is_selected = (idx == self._current_index)

            row = ctk.CTkFrame(
                self._list_scroll,
                fg_color="#2a4a6b" if is_selected else "transparent",
                corner_radius=4,
                cursor="hand2",
            )
            row.pack(fill="x", pady=1)

            lbl = ctk.CTkLabel(
                row,
                text=f"{icon}  {img_path.name}",
                anchor="w",
                text_color="#ffffff" if is_selected else color,
                font=ctk.CTkFont(size=12, weight="bold" if is_selected else "normal"),
            )
            lbl.pack(fill="x", padx=6, pady=2)

            # Capture path in closure
            def _on_click(event=None, p=img_path):
                self._select_image(p)

            row.bind("<Button-1>", _on_click)
            lbl.bind("<Button-1>", _on_click)

    def _icon_for(self, name: str) -> str:
        if self._masks_saved.get(name):
            return _ICON_SAVED
        if name in self._masks_memory:
            return _ICON_UNSAVED
        return _ICON_NONE

    def _icon_color(self, name: str) -> str:
        if self._masks_saved.get(name):
            return "#4CAF50"
        if name in self._masks_memory:
            return "#FF9800"
        return "#AAAAAA"

    # ── image selection & display ────────────────────────────────────────────

    def _select_image(self, img_path: Path) -> None:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            self._set_status(f"Could not open {img_path.name}")
            return
        # Keep index in sync (list-click path)
        try:
            self._current_index = self._image_paths.index(img_path)
        except ValueError:
            pass
        self._current_name = img_path.stem
        self._current_bgr = bgr

        # If a saved mask exists and is not yet in memory, load it
        if (
            self._current_name not in self._masks_memory
            and self._masks_dir is not None
        ):
            mask_path = self._masks_dir / f"mask_{img_path.name}"
            if mask_path.exists():
                loaded = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if loaded is not None:
                    self._masks_memory[self._current_name] = loaded
                    self._masks_saved[self._current_name] = True

        self._render_canvas()
        self._refresh_list()
        n = len(self._image_paths)
        self._set_status(f"[{self._current_index + 1}/{n}]  {img_path.name}  — ↑↓ or ←→ to navigate")

    def _render_canvas(self) -> None:
        if self._current_bgr is None:
            return

        cw = max(self._canvas.winfo_width(), 100)
        ch = max(self._canvas.winfo_height(), 100)

        rgb, off_x, off_y, scale = _fit_image_to_canvas(self._current_bgr, cw, ch)
        self._canvas_scale = scale
        self._canvas_off_x = off_x
        self._canvas_off_y = off_y

        # Apply mask overlay if present
        name = self._current_name
        if name and name in self._masks_memory:
            mask_orig = self._masks_memory[name]
            h, w = self._current_bgr.shape[:2]
            new_w, new_h = rgb.shape[1], rgb.shape[0]
            mask_small = cv2.resize(mask_orig, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            rgb = _apply_overlay(rgb, mask_small)

        pil_img = Image.fromarray(rgb)
        self._tk_photo = ImageTk.PhotoImage(pil_img)

        self._canvas.delete("all")
        self._canvas.create_image(off_x, off_y, image=self._tk_photo, anchor="nw")

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self._render_canvas()

    # ── click-to-segment ─────────────────────────────────────────────────────

    def _on_canvas_click(self, event: tk.Event) -> None:
        if self._current_bgr is None:
            return

        # Map canvas coords to image coords
        img_x = int((event.x - self._canvas_off_x) / self._canvas_scale)
        img_y = int((event.y - self._canvas_off_y) / self._canvas_scale)

        h, w = self._current_bgr.shape[:2]
        if not (0 <= img_x < w and 0 <= img_y < h):
            return

        thresh = self._get_threshold()
        mask = segment_at_click(self._current_bgr, img_x, img_y, tolerance=thresh)

        if mask is None:
            self._set_status("Clicked on background — try clicking directly on the target.")
            return

        self._masks_memory[self._current_name] = mask
        self._masks_saved[self._current_name] = False
        self._render_canvas()
        self._refresh_list()
        self._set_status("Target segmented — click Save mask to write to disk.")

    def _get_threshold(self) -> int:
        try:
            v = int(self._thresh_var.get())
            return max(1, min(255, v))
        except ValueError:
            return 20

    def _adjust_thresh(self, delta: int) -> None:
        self._thresh_var.set(str(self._get_threshold() + delta))

    # ── save / clear ─────────────────────────────────────────────────────────

    def _resolve_masks_dir(self) -> Optional[Path]:
        """Return the masks dir, prompting via entry if not yet set."""
        raw = self._masks_var.get().strip()
        if raw:
            self._masks_dir = Path(raw)
            self._masks_dir.mkdir(parents=True, exist_ok=True)
        if self._masks_dir is None:
            self._set_status("Set a masks directory first.")
            return None
        return self._masks_dir

    def _save_mask(self, name: str) -> bool:
        """Write mask PNG + update detections.json.  Returns True on success."""
        masks_dir = self._resolve_masks_dir()
        if masks_dir is None:
            return False
        mask = self._masks_memory.get(name)
        if mask is None:
            return False

        # Find original image path to reconstruct filename
        img_path = next(
            (p for p in self._image_paths if p.stem == name), None
        )
        if img_path is None:
            return False

        mask_path = masks_dir / f"mask_{img_path.name}"
        cv2.imwrite(str(mask_path), mask)

        # Compute ellipse metadata and update detections.json
        meta = _ellipse_from_mask(mask)
        if meta:
            meta["image"] = img_path.name
            self._update_detections_json(masks_dir, img_path.name, meta)

        self._masks_saved[name] = True
        return True

    def _update_detections_json(
        self, masks_dir: Path, image_name: str, meta: dict
    ) -> None:
        json_path = masks_dir / "detections.json"
        records: List[dict] = []
        if json_path.exists():
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    records = json.load(f)
            except (json.JSONDecodeError, OSError):
                records = []

        # Replace or append record for this image
        records = [r for r in records if r.get("image") != image_name]
        records.append(meta)
        records.sort(key=lambda r: r.get("image", ""))

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    def _remove_from_detections_json(
        self, masks_dir: Path, image_name: str
    ) -> None:
        json_path = masks_dir / "detections.json"
        if not json_path.exists():
            return
        try:
            with json_path.open("r", encoding="utf-8") as f:
                records = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        records = [r for r in records if r.get("image") != image_name]
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    def _save_current_mask(self) -> None:
        if self._current_name is None:
            return
        if self._current_name not in self._masks_memory:
            self._set_status("No mask in memory for this image.")
            return
        if self._save_mask(self._current_name):
            self._refresh_list()
            self._set_status(f"Mask saved for {self._current_name}.")

    def _save_all_masks(self) -> None:
        n = 0
        for name in list(self._masks_memory.keys()):
            if self._save_mask(name):
                n += 1
        self._refresh_list()
        self._set_status(f"Saved {n} mask(s).")

    def _clear_mask(self) -> None:
        if self._current_name is None:
            return
        name = self._current_name
        self._masks_memory.pop(name, None)
        self._masks_saved.pop(name, None)

        # Delete mask PNG + detections.json entry if they exist
        if self._masks_dir is not None:
            img_path = next(
                (p for p in self._image_paths if p.stem == name), None
            )
            if img_path:
                mask_file = self._masks_dir / f"mask_{img_path.name}"
                if mask_file.exists():
                    mask_file.unlink()
                self._remove_from_detections_json(self._masks_dir, img_path.name)

        self._render_canvas()
        self._refresh_list()
        self._set_status(f"Mask cleared for {name}.")

    # ── status bar ───────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        self._status_label.configure(text=msg)
