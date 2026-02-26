"""Tab 3: Resolve Mismatches — consensus face/non-face across reviewers."""

import json
import re
import threading
import queue
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from collections import defaultdict

import customtkinter as ctk
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
import tkinter

from face_diet_gui.core.settings_manager import SettingsManager, ReviewerRegistry
from face_diet_gui.gui.widgets.directory_tree_widget import DirectoryTreeWidget
from face_diet_gui.core.pipeline_helpers import (
    ProcessingStopped,
    _discard_annotations_for_session,
    _load_review_status_for_session,
    _load_mismatches_resolved_flag,
    _get_sessions_with_review_status,
    _format_time,
    _run_stage1_via_subprocess,
    _run_stage2_via_subprocess,
    _run_stage3_via_subprocess,
)
from face_diet_gui.gui.common import BTN_DISABLED_FG, _show_full_frame_toplevel, ProgressReporter


class MismatchResolutionTab(ctk.CTkFrame):
    """Tab: Resolve face/non-face mismatches (consensus). Left = session list (click to select), right = gallery + Save."""

    def __init__(self, master, settings_manager: SettingsManager, project_dir: Path, reviewer_id: str):
        super().__init__(master)
        self.settings = settings_manager
        self.project_dir = Path(project_dir) if project_dir else None
        self.reviewer_id = reviewer_id or ""
        self.registry = ReviewerRegistry(project_dir) if project_dir else None
        self.selected_participant: Optional[str] = None
        self.selected_session: Optional[str] = None
        self.session_dir: Optional[Path] = None
        self.df: Optional[pd.DataFrame] = None
        self.mismatch_indices: List[int] = []
        self.reviewer_labels: Dict[int, Dict[str, bool]] = {}
        self.reviewer_ids: List[str] = []
        self.annotations: Dict[int, bool] = {}
        self.image_cache: Dict[int, Image.Image] = {}
        self.items_per_page = 24
        self.current_page = 0
        self.current_df_filtered: Optional[pd.DataFrame] = None
        self.current_total_pages = 1
        self._session_rows: List[Tuple[ctk.CTkFrame, str, str]] = []
        self._setup_ui()
        # Session list is loaded when tab is first shown (via <Map>), not at startup, to avoid blocking main window

    def _setup_ui(self):
        ctk.CTkLabel(
            self,
            text="Resolve Mismatches",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(10, 15))
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        content.grid_columnconfigure(0, weight=0, minsize=360)
        content.grid_columnconfigure(1, weight=1, uniform="mismatch")
        content.grid_rowconfigure(0, weight=1)
        # Left: session list (narrower)
        left = ctk.CTkFrame(content)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(left, text="Participants & Sessions", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        ctk.CTkLabel(
            left,
            text="Click a session (2+ reviewers) to resolve mismatches.",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(pady=(0, 5))
        self.session_list_frame = ctk.CTkScrollableFrame(left, height=450)
        self.session_list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        # Right: gallery panel
        right = ctk.CTkFrame(content)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)
        self.right_placeholder = ctk.CTkLabel(
            right,
            text="Select a session with 2+ reviewers to resolve mismatches.",
            font=ctk.CTkFont(size=13),
            text_color="gray"
        )
        self.right_placeholder.grid(row=0, column=0, rowspan=4, sticky="nsew", padx=20, pady=40)
        self.gallery_panel = ctk.CTkFrame(right, fg_color="transparent")
        self.gallery_panel.grid(row=0, column=0, rowspan=6, sticky="nsew")
        self.gallery_panel.grid_remove()
        self.gallery_panel.grid_columnconfigure(0, weight=1)
        self.gallery_panel.grid_rowconfigure(4, weight=1)
        ctk.CTkLabel(
            self.gallery_panel,
            text="Mismatch instances (default: face; uncheck for non-face)",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        stats_frame = ctk.CTkFrame(self.gallery_panel, fg_color="transparent")
        stats_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        self.mismatch_stats_label = ctk.CTkLabel(stats_frame, text="", font=ctk.CTkFont(size=12))
        self.mismatch_stats_label.pack(side="left")
        ctrl = ctk.CTkFrame(self.gallery_panel, fg_color="transparent")
        ctrl.grid(row=2, column=0, sticky="ew", padx=10, pady=4)
        ctk.CTkLabel(ctrl, text="Items per page:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 5))
        self.items_per_page_var = ctk.IntVar(value=self.items_per_page)
        ctk.CTkEntry(ctrl, textvariable=self.items_per_page_var, width=60).pack(side="left", padx=(0, 5))
        ctk.CTkButton(ctrl, text="Apply", command=self._on_items_per_page_apply, width=60).pack(side="left", padx=(0, 15))
        ctk.CTkButton(ctrl, text="Check all (page)", command=self._check_all_page, width=110).pack(side="left", padx=(10, 5))
        ctk.CTkButton(ctrl, text="Uncheck all (page)", command=self._uncheck_all_page, width=120).pack(side="left", padx=(0, 5))
        # Pagination row: page info, Go to page, page numbers (same layout as Tab 2)
        pagination_row = ctk.CTkFrame(self.gallery_panel, fg_color="transparent")
        pagination_row.grid(row=3, column=0, sticky="ew", padx=10, pady=4)
        self.gallery_panel.grid_columnconfigure(0, weight=1)
        self.page_info_label = ctk.CTkLabel(pagination_row, text="", font=ctk.CTkFont(size=11), text_color="gray")
        self.page_info_label.pack(side="left", padx=(0, 15))
        ctk.CTkLabel(pagination_row, text="Go to page:", font=ctk.CTkFont(size=11), text_color="gray").pack(side="left", padx=(0, 4))
        self.page_go_entry_tab3 = ctk.CTkEntry(pagination_row, width=50, font=ctk.CTkFont(size=11))
        self.page_go_entry_tab3.pack(side="left", padx=(0, 4))
        def _on_page_go_tab3():
            try:
                n = int(self.page_go_entry_tab3.get().strip())
                if hasattr(self, "current_total_pages") and 1 <= n <= self.current_total_pages:
                    self.current_page = n - 1
                    self._display_gallery()
            except (ValueError, tkinter.TclError):
                pass
        ctk.CTkButton(pagination_row, text="Go", width=40, height=24, font=ctk.CTkFont(size=11), command=_on_page_go_tab3).pack(side="left", padx=(0, 12))
        self.page_go_entry_tab3.bind("<Return>", lambda e: _on_page_go_tab3())
        self.page_numbers_frame = ctk.CTkFrame(pagination_row, fg_color="transparent")
        self.page_numbers_frame.pack(side="left", padx=0, pady=0)
        self.gallery_frame = ctk.CTkScrollableFrame(self.gallery_panel)
        self.gallery_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=5)
        self.gallery_panel.grid_rowconfigure(4, weight=1)
        btn_row = ctk.CTkFrame(self.gallery_panel, fg_color="transparent")
        btn_row.grid(row=5, column=0, sticky="ew", pady=10)
        ctk.CTkButton(
            btn_row,
            text="Save consensus",
            command=self._save_consensus,
            width=180,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#28a745",
            hover_color="#218838",
            text_color="white"
        ).pack(side="left")

    def _load_session_list(self):
        """Load session list in background so tab opens immediately; paint when ready."""
        for w in self.session_list_frame.winfo_children():
            w.destroy()
        self._session_rows.clear()
        if not self.project_dir or not self.project_dir.exists():
            return
        ctk.CTkLabel(
            self.session_list_frame,
            text="Loading sessions…",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(pady=20)
        project_dir = self.project_dir

        def _fetch():
            items = _get_sessions_with_review_status(project_dir)
            self.after(0, lambda: self._paint_session_list(items))

        threading.Thread(target=_fetch, daemon=True).start()

    def _paint_session_list(self, items: List[Dict]):
        """Paint session list (must run on main thread). Called after background fetch."""
        for w in self.session_list_frame.winfo_children():
            w.destroy()
        self._session_rows.clear()
        if not items:
            ctk.CTkLabel(self.session_list_frame, text="No sessions found.", font=ctk.CTkFont(size=12), text_color="gray").pack(pady=20)
            return
        by_participant: Dict[str, List[Dict]] = defaultdict(list)
        for item in items:
            by_participant[item["participant"]].append(item)
        for participant_name in sorted(by_participant.keys()):
            ctk.CTkLabel(
                self.session_list_frame,
                text=participant_name,
                font=ctk.CTkFont(size=12, weight="bold")
            ).pack(anchor="w", padx=8, pady=(10, 2))
            for item in sorted(by_participant[participant_name], key=lambda x: x["session"]):
                session = item["session"]
                n = item.get("reviewers_with_tab2_count", 0)  # number of reviewers who have submitted (tab2)
                mismatch_count = item["mismatch_count"]
                resolved = item.get("resolved", False)
                if n == 0:
                    status_text = "Not reviewed"
                    status_color = "#dc3545"  # red
                elif n == 1:
                    status_text = "1 reviewer"
                    status_color = "#ffc107"  # yellow
                elif resolved:
                    status_text = f"{n} reviewers (Resolved)"
                    status_color = "#28a745"  # green
                row = ctk.CTkFrame(self.session_list_frame, fg_color=("gray92", "gray18"))
                row.pack(fill="x", padx=5, pady=2)
                ctk.CTkLabel(row, text="   ", width=20).pack(side="left")
                session_lbl = ctk.CTkLabel(row, text=session, font=ctk.CTkFont(size=12), anchor="w")
                session_lbl.pack(side="left", padx=(0, 8), pady=3)
                if n == 0 or n == 1 or resolved:
                    status_lbl = ctk.CTkLabel(row, text=status_text, font=ctk.CTkFont(size=12), text_color=status_color)
                    status_lbl.pack(side="left", padx=(0, 8), pady=3)
                    status_widgets = [status_lbl]
                else:
                    status_lbl_1 = ctk.CTkLabel(row, text=f"{n} reviewers (", font=ctk.CTkFont(size=12), text_color="#007bff")
                    status_lbl_1.pack(side="left", padx=(0, 0), pady=3)
                    status_lbl_2 = ctk.CTkLabel(row, text=f"{mismatch_count} mismatches)", font=ctk.CTkFont(size=12), text_color="#dc3545")
                    status_lbl_2.pack(side="left", padx=(0, 8), pady=3)
                    status_widgets = [status_lbl_1, status_lbl_2]
                can_select = n >= 2
                if can_select:
                    self._session_rows.append((row, participant_name, session))
                    for w in [row, session_lbl] + status_widgets:
                        w.bind("<Button-1>", lambda e, p=participant_name, s=session: self._on_session_click(p, s))
                        w.configure(cursor="hand2")
                    row.bind("<Button-1>", lambda e, p=participant_name, s=session: self._on_session_click(p, s))
        # Re-apply selection highlight so the current session row shows updated status (e.g. "Resolved" after save)
        if getattr(self, "selected_participant", None) and getattr(self, "selected_session", None):
            for (row, p, s) in self._session_rows:
                if p == self.selected_participant and s == self.selected_session:
                    row.configure(fg_color=("gray75", "gray25"))
                    break

    def _on_session_click(self, participant: str, session: str):
        if self.selected_participant == participant and self.selected_session == session:
            return
        self.selected_participant = participant
        self.selected_session = session
        self.session_dir = self.project_dir / participant / session
        for (row, p, s) in self._session_rows:
            if p == participant and s == session:
                row.configure(fg_color=("gray75", "gray25"))
            else:
                row.configure(fg_color=("gray92", "gray18"))
        self.right_placeholder.grid_remove()
        self.gallery_panel.grid(row=0, column=0, rowspan=6, sticky="nsew")
        self._load_mismatch_data()

    def _load_mismatch_data(self):
        self.mismatch_stats_label.configure(text="Loading…")
        for w in self.gallery_frame.winfo_children():
            w.destroy()
        thread = threading.Thread(target=self._load_mismatch_data_thread, daemon=True)
        thread.start()

    def _load_mismatch_data_thread(self):
        try:
            df = pd.read_csv(self.session_dir / "face_detections.csv")
            # Match Tab 2: instance_index in tab2 files is in sorted-by-confidence order (ascending)
            if "confidence" in df.columns:
                df = df.sort_values("confidence", ascending=True).reset_index(drop=True)
            reviewer_ids_all = self.registry.get_reviewer_ids()
            with_tab2 = [r for r in reviewer_ids_all if self.registry.get_is_face_annotation_path(r, self.selected_participant, self.selected_session).exists()]
            reviewers_with_tab2 = [r for r in with_tab2 if _load_review_status_for_session(self.registry, r, self.selected_participant, self.selected_session).get("reviewed", False)]
            if len(reviewers_with_tab2) < 2:
                self.after(0, lambda: self.mismatch_stats_label.configure(text="Need 2+ reviewers (fully reviewed) for this session."))
                return
            per_reviewer = {}
            for rid in reviewers_with_tab2:
                ann_df = pd.read_csv(self.registry.get_is_face_annotation_path(rid, self.selected_participant, self.selected_session))
                per_reviewer[rid] = dict(zip(ann_df["instance_index"].astype(int), ann_df["is_face"].astype(bool)))
            mismatch_indices = []
            reviewer_labels = {}
            consensus_path = self.registry.get_consensus_annotation_path(self.selected_participant, self.selected_session)
            if consensus_path.exists():
                # Only show mismatches introduced by reviewers who submitted AFTER the consensus was saved.
                try:
                    consensus_mtime = consensus_path.stat().st_mtime
                    cons_df = pd.read_csv(consensus_path)
                    consensus = dict(zip(cons_df["instance_index"].astype(int), cons_df["is_face"].astype(bool)))
                    post_consensus = [r for r in reviewers_with_tab2
                                      if self.registry.get_is_face_annotation_path(r, self.selected_participant, self.selected_session).stat().st_mtime > consensus_mtime]
                    for idx in df.index:
                        idx_int = int(idx)
                        cons_val = consensus.get(idx_int, True)
                        for r in post_consensus:
                            if per_reviewer[r].get(idx_int, True) != cons_val:
                                mismatch_indices.append(idx_int)
                                reviewer_labels[idx_int] = {r: per_reviewer[r].get(idx_int, True) for r in reviewers_with_tab2}
                                break
                except Exception:
                    pass  # consensus unreadable – show no mismatches (treat as resolved)
            else:
                # No consensus: pairwise disagreement across all reviewers
                for idx in df.index:
                    idx_int = int(idx)
                    vals = [per_reviewer[r].get(idx_int, True) for r in reviewers_with_tab2]
                    if len(set(vals)) > 1:
                        mismatch_indices.append(idx_int)
                        reviewer_labels[idx_int] = {r: per_reviewer[r].get(idx_int, True) for r in reviewers_with_tab2}
            self.df = df
            self.mismatch_indices = mismatch_indices
            self.reviewer_labels = reviewer_labels
            self.reviewer_ids = reviewers_with_tab2
            # Tab 3 gallery default: all images checked (face). If consensus file exists for this session, use those labels instead.
            self.annotations = {idx: True for idx in mismatch_indices}
            # Load saved consensus from _annotations/consensus/ if it exists (overwrites defaults for this session)
            consensus_path = self.registry.get_consensus_annotation_path(self.selected_participant, self.selected_session)
            if consensus_path.exists():
                try:
                    ann_df = pd.read_csv(consensus_path)
                    for _, row in ann_df.iterrows():
                        idx = int(row["instance_index"])
                        self.annotations[idx] = bool(row["is_face"])
                except Exception:
                    pass
            self.image_cache.clear()
            self.current_page = 0
            self.after(0, self._on_mismatch_data_loaded)
        except Exception as e:
            self.after(0, lambda: self.mismatch_stats_label.configure(text=f"Error: {str(e)}"))

    def _on_mismatch_data_loaded(self):
        n = len(self.mismatch_indices)
        self.mismatch_stats_label.configure(
            text=f"{n} instance(s) with mismatches."
        )
        if n == 0:
            self._clear_gallery()
            self.page_info_label.configure(text="No mismatches in this session.")
            return
        try:
            self.items_per_page = int(self.items_per_page_var.get())
            if self.items_per_page < 1:
                self.items_per_page = 24
                self.items_per_page_var.set(24)
        except (ValueError, tkinter.TclError):
            self.items_per_page = 24
        self._display_gallery()

    def _display_gallery(self):
        for w in self.gallery_frame.winfo_children():
            w.destroy()
        if not self.mismatch_indices:
            return
        total = len(self.mismatch_indices)
        total_pages = max(1, (total + self.items_per_page - 1) // self.items_per_page)
        if self.current_page >= total_pages:
            self.current_page = total_pages - 1
        start = self.current_page * self.items_per_page
        end = min(start + self.items_per_page, total)
        page_indices = self.mismatch_indices[start:end]
        self.current_df_filtered = self.df.loc[page_indices] if hasattr(self.df, 'loc') else None
        self.current_total_pages = total_pages
        self.page_info_label.configure(text=f"Showing {start + 1}-{end} of {total} | Page {self.current_page + 1} of {total_pages}")
        self._update_page_numbers()
        threading.Thread(target=self._load_gallery_page_images, args=(page_indices,), daemon=True).start()

    def _load_gallery_page_images(self, indices: List[int]):
        video_files = list(self.session_dir.glob("scenevideo.*"))
        video_path = video_files[0] if video_files else None
        images_data = []
        for idx in indices:
            row = self.df.loc[idx].to_dict()
            crop = self._extract_face_crop(row, video_path)
            if crop:
                self.image_cache[idx] = crop
                images_data.append((idx, row, crop))
            else:
                from PIL import Image as PILImage
                import numpy as np
                ph = np.zeros((80, 80, 3), dtype=np.uint8)
                ph[:] = (60, 60, 60)
                images_data.append((idx, row, Image.fromarray(ph)))
        self.after(0, lambda: self._paint_gallery_grid(images_data))

    def _extract_face_crop(self, face_info: dict, video_path: Optional[Path]) -> Optional[Image.Image]:
        if not video_path:
            return None
        try:
            frame_number = int(face_info["frame_number"])
            x, y, w, h = int(face_info["x"]), int(face_info["y"]), int(face_info["w"]), int(face_info["h"])
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            h_img, w_img = frame.shape[:2]
            pad = int(max(w, h) * 0.1)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
            if x2 <= x1 or y2 <= y1:
                return None
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                return None
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            return Image.fromarray(face_crop_rgb)
        except Exception:
            return None

    def _paint_gallery_grid(self, images_data: List):
        for w in self.gallery_frame.winfo_children():
            w.destroy()
        self._current_page_images_data = images_data
        self._card_nonface_labels = {}  # idx -> label widget (show/hide only, no repaint on toggle)
        self._card_checkbox_vars = {}   # idx -> BooleanVar (for check all / uncheck all without repaint)
        try:
            self.gallery_frame.update_idletasks()
            frame_width = max(self.gallery_frame.winfo_width(), 140)
            per_row = max(1, frame_width // 140)
        except Exception:
            per_row = 6
        for i, (idx, row_data, image) in enumerate(images_data):
            r, c = i // per_row, i % per_row
            container = ctk.CTkFrame(self.gallery_frame, width=130, height=155)
            container.grid(row=r, column=c, padx=5, pady=5)
            container.grid_propagate(False)
            img_resized = image.resize((120, 120), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_resized)
            img_label = ctk.CTkLabel(container, image=img_tk, text="")
            img_label.image = img_tk
            img_label.place(x=5, y=5, relwidth=0.92, relheight=0.77)
            is_face = self.annotations.get(idx, True)
            var = ctk.BooleanVar(value=is_face)
            self._card_checkbox_vars[idx] = var
            cb = ctk.CTkCheckBox(
                container, text="", variable=var, width=20,
                command=lambda idx=idx, v=var: self._on_mismatch_check(idx, v),
                checkbox_width=18, checkbox_height=18
            )
            cb.place(x=10, y=10)
            ctk.CTkLabel(container, text=f"#{idx}", font=ctk.CTkFont(size=10), text_color="gray").place(x=5, y=128)
            nonface_label = ctk.CTkLabel(
                container, text="NON-FACE", font=ctk.CTkFont(size=9, weight="bold"),
                text_color="red", fg_color="black"
            )
            self._card_nonface_labels[idx] = nonface_label
            if not is_face:
                nonface_label.place(relx=0.5, rely=0.45, anchor="center")
            def toggle(e, idx=idx, v=var):
                v.set(not v.get())
                self._on_mismatch_check(idx, v)
            img_label.bind("<Button-1>", toggle)
            def on_double(e, rd=row_data):
                if self.session_dir:
                    _show_full_frame_toplevel(self, self.session_dir, rd)
            img_label.bind("<Double-Button-1>", on_double)
        for col in range(per_row):
            self.gallery_frame.grid_columnconfigure(col, weight=1, minsize=130)
        total_rows = (len(images_data) + per_row - 1) // per_row
        for row in range(total_rows):
            self.gallery_frame.grid_rowconfigure(row, weight=0, minsize=160)

    def _on_mismatch_check(self, idx: int, var: ctk.BooleanVar):
        """Update annotation to match checkbox; show/hide NON-FACE label only (no repaint, like Tab 2)."""
        is_face = var.get()
        self.annotations[idx] = is_face
        if getattr(self, "_card_nonface_labels", None) and idx in self._card_nonface_labels:
            lbl = self._card_nonface_labels[idx]
            if is_face:
                lbl.place_forget()
            else:
                lbl.place(relx=0.5, rely=0.45, anchor="center")

    def _check_all_page(self):
        """Mark all instances on the current page as face."""
        data = getattr(self, "_current_page_images_data", None)
        if not data:
            return
        for idx, _, _ in data:
            self.annotations[idx] = True
            if getattr(self, "_card_checkbox_vars", None) and idx in self._card_checkbox_vars:
                self._card_checkbox_vars[idx].set(True)
            if getattr(self, "_card_nonface_labels", None) and idx in self._card_nonface_labels:
                self._card_nonface_labels[idx].place_forget()

    def _uncheck_all_page(self):
        """Mark all instances on the current page as non-face."""
        data = getattr(self, "_current_page_images_data", None)
        if not data:
            return
        for idx, _, _ in data:
            self.annotations[idx] = False
            if getattr(self, "_card_checkbox_vars", None) and idx in self._card_checkbox_vars:
                self._card_checkbox_vars[idx].set(False)
            if getattr(self, "_card_nonface_labels", None) and idx in self._card_nonface_labels:
                self._card_nonface_labels[idx].place(relx=0.5, rely=0.45, anchor="center")

    def _clear_gallery(self):
        for w in self.gallery_frame.winfo_children():
            w.destroy()

    def _update_page_numbers(self):
        for w in self.page_numbers_frame.winfo_children():
            w.destroy()
        total_pages = getattr(self, 'current_total_pages', 1)
        cur = self.current_page
        if total_pages <= 0:
            return
        def go(p):
            if 0 <= p < total_pages:
                self.current_page = p
                self._display_gallery()
        first = ctk.CTkLabel(self.page_numbers_frame, text="<<", font=ctk.CTkFont(size=12), cursor="hand2")
        first.pack(side="left", padx=3)
        if cur > 0:
            first.bind("<Button-1>", lambda e: go(0))
            first.configure(text_color="#3b8ed0")
        else:
            first.configure(text_color="gray")
        prev = ctk.CTkLabel(self.page_numbers_frame, text="<", font=ctk.CTkFont(size=12), cursor="hand2")
        prev.pack(side="left", padx=3)
        if cur > 0:
            prev.bind("<Button-1>", lambda e: go(cur - 1))
            prev.configure(text_color="#3b8ed0")
        else:
            prev.configure(text_color="gray")
        for p in range(total_pages):
            lbl = ctk.CTkLabel(self.page_numbers_frame, text=str(p + 1), font=ctk.CTkFont(size=12, weight="bold" if p == cur else "normal"), cursor="hand2")
            lbl.pack(side="left", padx=3)
            if p == cur:
                lbl.configure(text_color="#3b8ed0")
            else:
                lbl.bind("<Button-1>", lambda e, p=p: go(p))
                lbl.configure(text_color="#3b8ed0")
        nxt = ctk.CTkLabel(self.page_numbers_frame, text=">", font=ctk.CTkFont(size=12), cursor="hand2")
        nxt.pack(side="left", padx=3)
        if cur < total_pages - 1:
            nxt.bind("<Button-1>", lambda e: go(cur + 1))
            nxt.configure(text_color="#3b8ed0")
        else:
            nxt.configure(text_color="gray")
        last = ctk.CTkLabel(self.page_numbers_frame, text=">>", font=ctk.CTkFont(size=12), cursor="hand2")
        last.pack(side="left", padx=3)
        if cur < total_pages - 1:
            last.bind("<Button-1>", lambda e: go(total_pages - 1))
            last.configure(text_color="#3b8ed0")
        else:
            last.configure(text_color="gray")

    def _on_items_per_page_apply(self):
        try:
            v = int(self.items_per_page_var.get())
            if v < 1:
                v = 24
                self.items_per_page_var.set(24)
            self.items_per_page = v
            self.current_page = 0
            if self.mismatch_indices:
                self._display_gallery()
        except (ValueError, tkinter.TclError):
            pass

    def _save_consensus(self):
        if self.df is None or not self.selected_participant or not self.selected_session:
            return
        try:
            # Global consensus and resolved flag under _annotations/consensus/{participant}/{session}/
            consensus_path = self.registry.get_consensus_annotation_path(self.selected_participant, self.selected_session)
            consensus_path.parent.mkdir(parents=True, exist_ok=True)
            records = []
            for idx in self.df.index:
                is_face = self.annotations.get(idx, True)
                records.append({"instance_index": idx, "is_face": is_face})
            df_out = pd.DataFrame(records)
            with open(consensus_path, "w", newline="", encoding="utf-8") as f:
                df_out.to_csv(f, index=False)
                f.flush()
                os.fsync(f.fileno())
            messagebox.showinfo("Saved", f"Consensus saved for {self.selected_participant} / {self.selected_session}.")
            self._load_session_list()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{str(e)}")

    def set_project_dir(self, project_dir: Path):
        self.project_dir = Path(project_dir) if project_dir else None
        self.registry = ReviewerRegistry(project_dir) if project_dir else None
        if self.project_dir and self.project_dir.exists():
            self._load_session_list()

    def update_project_and_reviewer(self, project_dir: Path, reviewer_id: str):
        """Called when user changes project or reviewer via Back to setup."""
        self.project_dir = Path(project_dir) if project_dir else None
        self.reviewer_id = reviewer_id or ""
        self.registry = ReviewerRegistry(project_dir) if project_dir else None
        if self.project_dir and self.project_dir.exists():
            self._load_session_list()


