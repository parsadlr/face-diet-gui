"""Tab 2: Face Instance Review — manual review of detected faces before clustering."""

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


class FaceInstanceReviewTab(ctk.CTkFrame):
    """Tab 2: Face Instance Review - Manual review of detected faces before clustering."""

    def __init__(self, master, settings_manager: SettingsManager,
                 project_dir: Path, reviewer_id: str):
        super().__init__(master)
        self.settings = settings_manager
        self.project_dir: Path = project_dir
        self.reviewer_id: str = reviewer_id

        # Selected participant / session
        self.selected_participant: Optional[str] = None
        self.selected_session: Optional[str] = None
        self.session_dir: Optional[Path] = None

        # Data storage
        self.df: Optional[pd.DataFrame] = None
        self.annotations: Dict = {}  # instance_index -> {'is_face': bool, 'reviewed_at': str}
        self.current_page = 0
        self.items_per_page = 100
        self.selected_instances = set()

        # Image cache
        self.image_cache: Dict[int, Image.Image] = {}
        self._session_display_to_raw: Dict[str, str] = {}

        self._setup_ui()
        self._load_settings()
    
    def _setup_ui(self):
        """Setup UI: compact top form, gallery expands to use all space, action buttons fixed at bottom."""
        # Title
        ctk.CTkLabel(
            self,
            text="Face Instance Review",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(10, 15), fill="x")

        # Top form: participant/session, stats, controls, pagination (compact, no scroll)
        self.form_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.form_frame.pack(fill="x", padx=20, pady=(0, 6))

        # Participant / session selector — dropdowns
        sel_frame = ctk.CTkFrame(self.form_frame)
        sel_frame.pack(fill="x", pady=(0, 6))

        row = ctk.CTkFrame(sel_frame, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(
            row, text="Participant:", font=ctk.CTkFont(size=13, weight="bold"), width=90
        ).pack(side="left", padx=(0, 6))
        self.participant_var = ctk.StringVar(value="— select —")
        self.participant_dropdown = ctk.CTkOptionMenu(
            row,
            variable=self.participant_var,
            values=["— select —"],
            width=220,
            command=self._on_participant_selected_dropdown
        )
        self.participant_dropdown.pack(side="left", padx=(0, 20))

        ctk.CTkLabel(
            row, text="Session:", font=ctk.CTkFont(size=13, weight="bold"), width=70
        ).pack(side="left", padx=(0, 6))
        self.session_var = ctk.StringVar(value="— select —")
        self.session_dropdown = ctk.CTkOptionMenu(
            row,
            variable=self.session_var,
            values=["— select —"],
            width=220,
            command=self._on_session_selected_dropdown
        )
        self.session_dropdown.pack(side="left", padx=(0, 15))

        self.load_btn = ctk.CTkButton(
            row,
            text="Load Session",
            command=self._load_session,
            width=120,
            height=32,
            state="disabled"
        )
        self.load_btn.pack(side="left", padx=(0, 15))

        self.reviewed_var = ctk.BooleanVar(value=False)
        self.reviewed_checkbox = ctk.CTkCheckBox(
            row,
            text="I have fully reviewed this session",
            variable=self.reviewed_var,
            command=self._on_reviewed_checkbox_changed,
            font=ctk.CTkFont(size=12),
            state="disabled"
        )
        self.reviewed_checkbox.pack(side="left", padx=(0, 5), pady=0)

        self._populate_participants_dropdown()

        # Control panel (items per page, Apply, Check all, Uncheck all)
        self._create_control_panel()

        # Statistics (total instances, valid faces, non-face) — below controls, above pages
        self.stats_frame = ctk.CTkFrame(self.form_frame)
        self.stats_frame.pack(fill="x", pady=(0, 6))
        self.stats_label = ctk.CTkLabel(
            self.stats_frame,
            text="No session loaded",
            font=ctk.CTkFont(size=12)
        )
        self.stats_label.pack(pady=(10, 10))

        # Pagination
        self._create_pagination_controls()

        # Gallery: takes all remaining space (expand=True) so more images visible
        self.gallery_container = ctk.CTkFrame(self)
        self.gallery_container.pack(fill="both", expand=True, padx=20, pady=(0, 6))
        self.gallery_frame = ctk.CTkScrollableFrame(self.gallery_container)
        self.gallery_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Action buttons fixed at bottom
        self._create_action_buttons()
    
    def _create_control_panel(self):
        """Create control panel with filters and settings."""
        panel = ctk.CTkFrame(self.form_frame)
        panel.pack(fill="x", pady=(0, 6))
        
        # Items per page
        ctk.CTkLabel(
            panel,
            text="Items per page:",
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=(10, 5))
        
        self.items_per_page_var = ctk.IntVar(value=100)
        ctk.CTkEntry(
            panel,
            width=80,
            textvariable=self.items_per_page_var
        ).pack(side="left", padx=(0, 15))
        
        # Min confidence
        ctk.CTkLabel(
            panel,
            text="Min Confidence:",
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=(10, 5))
        
        self.min_confidence_var = ctk.DoubleVar(value=0.0)
        ctk.CTkEntry(
            panel,
            width=80,
            textvariable=self.min_confidence_var
        ).pack(side="left", padx=(0, 5))
        
        # Apply (items per page + min confidence)
        self.apply_btn = ctk.CTkButton(
            panel,
            text="Apply",
            command=self._on_apply_filters,
            width=80,
            height=30,
            state="disabled"
        )
        self.apply_btn.pack(side="left", padx=(10, 10))
        
        # Check all / Uncheck all (page)
        self.check_all_page_btn = ctk.CTkButton(
            panel, text="Check all (page)", command=self._check_all_page, width=110, height=30, state="disabled"
        )
        self.check_all_page_btn.pack(side="left", padx=(5, 5))
        self.uncheck_all_page_btn = ctk.CTkButton(
            panel, text="Uncheck all (page)", command=self._uncheck_all_page, width=120, height=30, state="disabled"
        )
        self.uncheck_all_page_btn.pack(side="left", padx=(0, 5))
    
    def _create_pagination_controls(self):
        """Create pagination controls (single horizontal row when empty or filled)."""
        self.pagination_frame = ctk.CTkFrame(self.form_frame, fg_color="transparent")
        self.pagination_frame.pack(fill="x", pady=(0, 6))
        row = ctk.CTkFrame(self.pagination_frame, fg_color="transparent")
        row.pack(fill="x", padx=0, pady=4)
        # Page info (left); go-to-page entry; page numbers (right)
        self.page_info_label = ctk.CTkLabel(
            row,
            text="Load a session to see instances and pages",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.page_info_label.pack(side="left", padx=(0, 15))
        ctk.CTkLabel(row, text="Go to page:", font=ctk.CTkFont(size=11), text_color="gray").pack(side="left", padx=(10, 4))
        self.page_go_entry = ctk.CTkEntry(row, width=50, font=ctk.CTkFont(size=11), state="disabled")
        self.page_go_entry.pack(side="left", padx=(0, 4))
        def _on_page_go_tab2():
            try:
                n = int(self.page_go_entry.get().strip())
                if hasattr(self, "current_total_pages") and 1 <= n <= self.current_total_pages:
                    self._go_to_page(n - 1)
            except (ValueError, tkinter.TclError):
                pass
        self.page_go_btn = ctk.CTkButton(row, text="Go", width=40, height=24, font=ctk.CTkFont(size=11), command=_on_page_go_tab2, state="disabled")
        self.page_go_btn.pack(side="left", padx=(0, 12))
        self.page_go_entry.bind("<Return>", lambda e: _on_page_go_tab2())
        self.page_numbers_frame = ctk.CTkFrame(row, fg_color="transparent")
        self.page_numbers_frame.pack(side="left", padx=0, pady=0)
    
    def _create_action_buttons(self):
        """Create action buttons fixed at bottom of tab."""
        action_frame = ctk.CTkFrame(self)
        action_frame.pack(fill="x", padx=20, pady=(6, 10))
        # Save annotations button (centered)
        self.save_btn = ctk.CTkButton(
            action_frame,
            text="Save Annotations",
            command=self._save_annotations,
            width=180,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#28a745",
            hover_color="#218838",
            text_color="white",
            text_color_disabled="white",
            state="disabled"
        )
        self.save_btn.configure(fg_color=BTN_DISABLED_FG)  # initial disabled look
        self.save_btn.pack(anchor="center", pady=5)
    
    def _get_review_status_path(self, participant: str, session: str) -> Path:
        """Path to review_status.json for this reviewer/participant/session."""
        registry = ReviewerRegistry(self.project_dir)
        ann_path = registry.get_is_face_annotation_path(self.reviewer_id, participant, session)
        return ann_path.parent / "review_status.json"

    def _load_review_status(self, participant: str, session: str) -> Dict:
        """Load {reviewed: bool, last_save: str|None} for this session."""
        path = self._get_review_status_path(participant, session)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {
                    "reviewed": bool(data.get("reviewed", False)),
                    "last_save": data.get("last_save"),
                }
            except Exception:
                pass
        # If no status file, use tab2 CSV mtime as last_save for display
        registry = ReviewerRegistry(self.project_dir)
        ann_path = registry.get_is_face_annotation_path(self.reviewer_id, participant, session)
        last_save = None
        if ann_path.exists():
            try:
                from datetime import datetime
                mtime = ann_path.stat().st_mtime
                last_save = datetime.fromtimestamp(mtime).isoformat()
            except Exception:
                pass
        return {"reviewed": False, "last_save": last_save}

    def _save_review_status(self, participant: str, session: str, reviewed: bool, last_save: Optional[str] = None):
        """Save review status. If last_save is None, keep existing value."""
        path = self._get_review_status_path(participant, session)
        path.parent.mkdir(parents=True, exist_ok=True)
        current = self._load_review_status(participant, session)
        if last_save is not None:
            current["last_save"] = last_save
        current["reviewed"] = reviewed
        with open(path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)

    def _format_last_save(self, last_save_iso: Optional[str]) -> str:
        """Format last_save ISO string or None for display in session list."""
        if not last_save_iso:
            return "not saved"
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(last_save_iso.replace("Z", "+00:00"))
            return dt.strftime("%d %b %Y, %H:%M")
        except Exception:
            return last_save_iso[:16] if last_save_iso else "not saved"

    def _refresh_session_dropdown(self):
        """Rebuild session list for current participant and keep current session selected."""
        if not self.selected_participant or not self.selected_session:
            return
        choice = self.selected_participant
        p_path = self.project_dir / choice
        sessions = sorted([
            d.name for d in p_path.iterdir()
            if d.is_dir()
            and not d.name.startswith('_')
            and not d.name.startswith('.')
            and (d / "face_detections.csv").exists()
        ])
        display_values = []
        for sname in sessions:
            status = self._load_review_status(choice, sname)
            last_save_str = self._format_last_save(status["last_save"])
            label = f"{sname} — last save: {last_save_str}"
            if status["reviewed"]:
                label += " ✓ Reviewed"
            display_values.append(label)
        self._session_display_to_raw = {display_values[i]: sessions[i] for i in range(len(sessions))}
        self.session_dropdown.configure(values=["— select —"] + display_values)
        # Keep current session selected (find display string for selected_session)
        for disp, raw in self._session_display_to_raw.items():
            if raw == self.selected_session:
                self.session_var.set(disp)
                break

    def update_project_and_reviewer(self, project_dir: Path, reviewer_id: str):
        """Called when user changes project or reviewer via Back to setup."""
        self.project_dir = project_dir
        self.reviewer_id = reviewer_id

    def _on_reviewed_checkbox_changed(self):
        """User toggled 'I have fully reviewed this session'."""
        if not self.selected_participant or not self.selected_session:
            return
        reviewed = self.reviewed_var.get()
        self._save_review_status(
            self.selected_participant,
            self.selected_session,
            reviewed=reviewed,
            last_save=None,
        )
        # Refresh session dropdown without clearing selection
        self._refresh_session_dropdown()
        # Confirm to user
        session_label = f"{self.selected_participant} / {self.selected_session}"
        if reviewed:
            messagebox.showinfo("Session reviewed", f"Marked as fully reviewed:\n{session_label}")
        else:
            messagebox.showinfo("Session not reviewed", f"Marked as not fully reviewed:\n{session_label}")

    def _populate_participants_dropdown(self):
        """Populate the participant dropdown from the project directory."""
        if not self.project_dir or not self.project_dir.exists():
            self.participant_dropdown.configure(values=["— select —"])
            self.participant_var.set("— select —")
            self.session_dropdown.configure(values=["— select —"])
            self.session_var.set("— select —")
            return

        participants = sorted([
            d.name for d in self.project_dir.iterdir()
            if d.is_dir() and not d.name.startswith('_') and not d.name.startswith('.')
        ])
        values = ["— select —"] + participants
        self.participant_dropdown.configure(values=values)
        self.participant_var.set("— select —")
        self.session_dropdown.configure(values=["— select —"])
        self.session_var.set("— select —")
        self.selected_participant = None
        self.selected_session = None
        self.load_btn.configure(state="disabled")

    def _set_session_loaded_controls(self, enabled: bool):
        """Enable or disable all controls that require a loaded session (checkbox, save, filter controls)."""
        state = "normal" if enabled else "disabled"
        self.reviewed_checkbox.configure(state=state)
        self.save_btn.configure(
            state=state,
            fg_color="#28a745" if enabled else BTN_DISABLED_FG
        )
        self.check_all_page_btn.configure(state=state)
        self.uncheck_all_page_btn.configure(state=state)
        self.apply_btn.configure(state=state)
        if not enabled:
            self.page_info_label.configure(text="Load a session to see instances and pages")
            for w in self.page_numbers_frame.winfo_children():
                w.destroy()
            self.page_go_entry.configure(state="disabled")
            self.page_go_btn.configure(state="disabled")
        else:
            self.page_go_entry.configure(state="normal")
            self.page_go_btn.configure(state="normal")

    def _on_participant_selected_dropdown(self, choice: str):
        """Called when participant dropdown selection changes."""
        if choice == "— select —":
            self.selected_participant = None
            self.session_dropdown.configure(values=["— select —"])
            self.session_var.set("— select —")
            self.selected_session = None
            self.load_btn.configure(state="disabled")
            self._set_session_loaded_controls(False)
            return
        self.selected_participant = choice
        self.selected_session = None
        self.load_btn.configure(state="disabled")
        self._set_session_loaded_controls(False)

        p_path = self.project_dir / choice
        sessions = sorted([
            d.name for d in p_path.iterdir()
            if d.is_dir()
            and not d.name.startswith('_')
            and not d.name.startswith('.')
            and (d / "face_detections.csv").exists()
        ])
        display_values = []
        for sname in sessions:
            status = self._load_review_status(choice, sname)
            last_save_str = self._format_last_save(status["last_save"])
            label = f"{sname} — last save: {last_save_str}"
            if status["reviewed"]:
                label += " ✓ Reviewed"
            display_values.append(label)
        self._session_display_to_raw = {display_values[i]: sessions[i] for i in range(len(sessions))}
        self.session_dropdown.configure(values=["— select —"] + display_values)
        self.session_var.set("— select —")

    def _on_session_selected_dropdown(self, choice: str):
        """Called when session dropdown selection changes."""
        if choice == "— select —":
            self.selected_session = None
            self.load_btn.configure(state="disabled")
            self._set_session_loaded_controls(False)
            return
        raw = getattr(self, "_session_display_to_raw", {}).get(choice, choice)
        self.selected_session = raw
        self.load_btn.configure(state="normal")

    def _load_session(self):
        """Load face detections from the selected participant/session."""
        if not self.selected_participant or not self.selected_session:
            messagebox.showerror("Error", "Please select a participant and session.")
            return

        self.session_dir = self.project_dir / self.selected_participant / self.selected_session

        # Load data in background thread
        thread = threading.Thread(target=self._load_data_thread, daemon=True)
        thread.start()
    
    def _load_data_thread(self):
        """Load face detections in background thread."""
        try:
            csv_path = self.session_dir / "face_detections.csv"
            
            if not csv_path.exists():
                self.after(0, lambda: messagebox.showerror(
                    "Error",
                    f"Could not find face_detections.csv in:\n{self.session_dir}"
                ))
                return
            
            # Clear annotations so we don't carry over the previous session's flags (index-based)
            self.annotations.clear()
            
            # Load CSV
            self.df = pd.read_csv(csv_path)
            
            # Load existing annotations if they exist (for this session only)
            self._load_existing_annotations()
            
            # Sort by confidence (lowest first)
            if 'confidence' in self.df.columns:
                self.df = self.df.sort_values('confidence', ascending=True).reset_index(drop=True)
            
            # Initialize annotations for all instances (default: is_face=True)
            for idx in self.df.index:
                if idx not in self.annotations:
                    self.annotations[idx] = {'is_face': True, 'reviewed_at': None}
            
            # Update UI and enable session-dependent controls
            self.after(0, self._update_stats)
            self.after(0, self._display_gallery)
            self.after(0, lambda: self._set_session_loaded_controls(True))
            # Show user-controlled "session reviewed" checkbox state
            status = self._load_review_status(self.selected_participant, self.selected_session)
            self.after(0, lambda: self.reviewed_var.set(status["reviewed"]))
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\n[ERROR] Error loading session data:")
            print(error_details)
            
            self.after(0, lambda: messagebox.showerror(
                "Error",
                f"Failed to load session data:\n{str(e)}\n\nCheck terminal for details."
            ))
    
    def _load_existing_annotations(self):
        """Load existing annotation file if it exists."""
        registry = ReviewerRegistry(self.project_dir)
        annotation_file = registry.get_is_face_annotation_path(
            self.reviewer_id, self.selected_participant, self.selected_session
        )

        if annotation_file.exists():
            try:
                ann_df = pd.read_csv(annotation_file)
                for _, row in ann_df.iterrows():
                    idx = int(row['instance_index'])
                    self.annotations[idx] = {
                        'is_face': bool(row['is_face']),
                        'reviewed_at': str(row['reviewed_at'])
                    }
                print(f"Loaded existing annotations from {annotation_file}")
            except Exception as e:
                print(f"Warning: Could not load annotations: {e}")

    def _load_settings(self):
        """Load settings into UI (no per-tab reviewer or session dir to restore)."""
        pass
    
    def _update_stats(self):
        """Update statistics display."""
        if self.df is None:
            return
        
        total = len(self.df)
        marked_nonface = sum(1 for ann in self.annotations.values() if not ann['is_face'])
        valid_faces = total - marked_nonface
        
        stats_text = (
            f"Total Instances: {total} | "
            f"Valid Faces: {valid_faces} | "
            f"Non-Face: {marked_nonface}"
        )
        
        self.stats_label.configure(text=stats_text)
        self._update_face_nonface_label()
    
    def _on_apply_filters(self):
        """Apply items per page and min confidence, then refresh gallery."""
        if self.df is None:
            return
        try:
            new_value = int(self.items_per_page_var.get())
            if new_value < 1:
                new_value = 100
                self.items_per_page_var.set(100)
            self.items_per_page = new_value
        except (ValueError, tkinter.TclError):
            self.items_per_page_var.set(self.items_per_page)
            messagebox.showerror("Invalid Input", "Please enter a valid number for items per page.")
            return
        self.current_page = 0
        self._display_gallery()
    
    def _display_gallery(self):
        """Display current page of face instances."""
        if self.df is None:
            return
        
        # Apply confidence filter
        min_conf = float(self.min_confidence_var.get())
        if 'confidence' in self.df.columns and min_conf > 0.0:
            df_filtered = self.df[self.df['confidence'] >= min_conf].copy()
        else:
            df_filtered = self.df.copy()
        
        # Calculate pagination
        total_items = len(df_filtered)
        total_pages = max(1, (total_items + self.items_per_page - 1) // self.items_per_page)
        
        # Ensure current page is valid
        if self.current_page >= total_pages:
            self.current_page = max(0, total_pages - 1)
        
        # Get page slice
        start_idx = self.current_page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, total_items)
        page_df = df_filtered.iloc[start_idx:end_idx]
        
        # Store for page navigation
        self.current_df_filtered = df_filtered
        self.current_total_pages = total_pages
        
        # Update page info label
        self.page_info_label.configure(
            text=f"Showing {start_idx + 1}-{end_idx} of {total_items} instances | Page {self.current_page + 1} of {total_pages}"
        )
        
        # Update page numbers
        self._update_page_numbers(total_pages)
        
        # Load images in background thread
        thread = threading.Thread(
            target=self._load_gallery_images_thread,
            args=(page_df,),
            daemon=True
        )
        thread.start()
    
    def _load_gallery_images_thread(self, page_df: pd.DataFrame):
        """Load and display gallery images in background thread."""
        # Show loading message in stats label (not in gallery frame)
        self.after(0, lambda: self.stats_label.configure(text="Loading images..."))
        
        try:
            # Load all images first
            images_data = []
            for idx, row in page_df.iterrows():
                crop = self._extract_face_crop(row.to_dict())
                if crop:
                    self.image_cache[idx] = crop
                    images_data.append((idx, row, crop))
            
            # Create gallery UI in main thread (will clear first)
            self.after(0, lambda: self._create_gallery_grid(images_data))
            
            # Restore stats
            self.after(0, lambda: self._update_stats())
            
        except Exception as e:
            print(f"Error loading gallery images: {e}")
            import traceback
            traceback.print_exc()
            self.after(0, lambda: messagebox.showerror(
                "Error",
                f"Failed to load gallery images:\n{str(e)}"
            ))
    
    def _clear_gallery(self):
        """Clear all gallery items."""
        for widget in self.gallery_frame.winfo_children():
            widget.destroy()
    
    def _calculate_images_per_row(self):
        """Calculate number of images per row based on gallery frame width."""
        try:
            # Get gallery frame width
            self.gallery_frame.update_idletasks()
            frame_width = self.gallery_frame.winfo_width()
            
            # If width is 1 (not yet rendered), use default
            if frame_width <= 1:
                frame_width = 850  # Default width
            
            # Image frame width: 130px + padding (5px each side) = 140px per image
            image_width = 140
            images_per_row = max(1, frame_width // image_width)
            
            return images_per_row
        except:
            return 6  # Default fallback
    
    def _create_gallery_grid(self, images_data: List):
        """Create gallery grid layout matching popup gallery style."""
        # Ensure gallery is completely clear (no pack widgets left)
        self._clear_gallery()
        self._card_nonface_labels = {}

        images_per_row = self._calculate_images_per_row()

        # Store images for resize handling
        self.gallery_frame.images_data = images_data
        self.gallery_frame.images_per_row = images_per_row

        for i, (instance_idx, row_data, image) in enumerate(images_data):
            row_idx = i // images_per_row
            col_idx = i % images_per_row
            
            # Create container frame (same size as image)
            img_container = ctk.CTkFrame(self.gallery_frame, width=130, height=155)
            img_container.grid(row=row_idx, column=col_idx, padx=5, pady=5)
            img_container.grid_propagate(False)
            
            # Resize image
            img_resized = image.resize((120, 120), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_resized)
            
            # Image label as background
            img_label = ctk.CTkLabel(img_container, image=img_tk, text="")
            img_label.image = img_tk  # Keep reference
            img_label.place(x=5, y=5, relwidth=0.92, relheight=0.77)
            
            # Checkbox: checked = face, unchecked = non-face (red text shown below)
            is_face = self.annotations.get(instance_idx, {}).get('is_face', True)
            checkbox_var = ctk.BooleanVar(value=is_face)
            checkbox = ctk.CTkCheckBox(
                img_container,
                text="",
                variable=checkbox_var,
                width=20,
                command=lambda idx=instance_idx, var=checkbox_var: self._on_face_checkbox_toggle(idx, var),
                checkbox_width=18,
                checkbox_height=18
            )
            checkbox.place(x=10, y=10)
            
            # Make entire image clickable to toggle face/non-face
            def toggle_on_click(event, idx=instance_idx, var=checkbox_var):
                var.set(not var.get())
                self._on_face_checkbox_toggle(idx, var)
            
            img_label.bind("<Button-1>", toggle_on_click)
            # Double-click: show full frame (not just face crop)
            def on_double(event, rd=row_data):
                if self.session_dir:
                    _show_full_frame_toplevel(self, self.session_dir, rd)
            img_label.bind("<Double-Button-1>", on_double)
            img_container.configure(cursor="hand2")
            img_label.configure(cursor="hand2")
            
            # Info frame at bottom (confidence + index) - use pack instead of place for labels
            info_frame = ctk.CTkFrame(img_container, fg_color="transparent")
            info_frame.place(x=5, y=128, relwidth=0.92)
            
            # Confidence label
            confidence = row_data.get('confidence', 0.0)
            conf_text = f"{confidence:.3f}"
            conf_label = ctk.CTkLabel(
                info_frame,
                text=conf_text,
                font=ctk.CTkFont(size=10),
                text_color="#ffa500" if confidence < 0.9 else "#90ee90"
            )
            conf_label.pack(side="left")
            
            # Instance index label
            idx_label = ctk.CTkLabel(
                info_frame,
                text=f"#{instance_idx}",
                font=ctk.CTkFont(size=9),
                text_color="gray"
            )
            idx_label.pack(side="right")

            # NON-FACE label (always create; show/hide on toggle so unchecked shows red flag)
            nonface_label = ctk.CTkLabel(
                img_container,
                text="NON-FACE",
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color="red",
                fg_color="black"
            )
            self._card_nonface_labels[instance_idx] = nonface_label
            if not is_face:
                nonface_label.place(relx=0.5, rely=0.4, anchor="center")

        # Configure grid to enable proper scrolling
        for col in range(images_per_row):
            self.gallery_frame.grid_columnconfigure(col, weight=1, minsize=140)
        
        # Calculate total rows needed
        total_rows = (len(images_data) + images_per_row - 1) // images_per_row
        for row in range(total_rows):
            self.gallery_frame.grid_rowconfigure(row, weight=0, minsize=160)
        
        # Force update of the scrollable frame's internal canvas
        self.gallery_frame.update_idletasks()
        
        # Bind resize event to regenerate layout (only once)
        if not hasattr(self, '_resize_bound'):
            self.gallery_frame.bind('<Configure>', self._on_gallery_resize)
            self._resize_bound = True
    
    def _on_gallery_resize(self, event=None):
        """Handle gallery frame resize to adjust layout."""
        if not hasattr(self.gallery_frame, 'images_data') or not self.gallery_frame.images_data:
            return
        
        new_images_per_row = self._calculate_images_per_row()
        
        # Only regenerate if images per row changed
        if new_images_per_row != self.gallery_frame.images_per_row:
            self.gallery_frame.images_per_row = new_images_per_row

            # Regenerate grid layout with new column count
            self._clear_gallery()
            self._card_nonface_labels = {}

            for i, (instance_idx, row_data, image) in enumerate(self.gallery_frame.images_data):
                row_idx = i // new_images_per_row
                col_idx = i % new_images_per_row
                
                # Create container frame
                img_container = ctk.CTkFrame(self.gallery_frame, width=130, height=155)
                img_container.grid(row=row_idx, column=col_idx, padx=5, pady=5)
                img_container.grid_propagate(False)
                
                # Resize image
                img_resized = image.resize((120, 120), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img_resized)
                
                # Image label
                img_label = ctk.CTkLabel(img_container, image=img_tk, text="")
                img_label.image = img_tk
                img_label.place(x=5, y=5, relwidth=0.92, relheight=0.77)
                
                # Checkbox: checked = face, unchecked = non-face (red text shown below)
                is_face = self.annotations.get(instance_idx, {}).get('is_face', True)
                checkbox_var = ctk.BooleanVar(value=is_face)
                checkbox = ctk.CTkCheckBox(
                    img_container,
                    text="",
                    variable=checkbox_var,
                    width=20,
                    command=lambda idx=instance_idx, var=checkbox_var: self._on_face_checkbox_toggle(idx, var),
                    checkbox_width=18,
                    checkbox_height=18
                )
                checkbox.place(x=10, y=10)
                
                # Make clickable
                def toggle_on_click(event, idx=instance_idx, var=checkbox_var):
                    var.set(not var.get())
                    self._on_face_checkbox_toggle(idx, var)
                
                img_label.bind("<Button-1>", toggle_on_click)
                def on_double(event, rd=row_data):
                    if self.session_dir:
                        _show_full_frame_toplevel(self, self.session_dir, rd)
                img_label.bind("<Double-Button-1>", on_double)
                img_container.configure(cursor="hand2")
                img_label.configure(cursor="hand2")
                
                # Info at bottom
                info_frame = ctk.CTkFrame(img_container, fg_color="transparent")
                info_frame.place(x=5, y=128, relwidth=0.92)
                
                confidence = row_data.get('confidence', 0.0)
                ctk.CTkLabel(
                    info_frame,
                    text=f"{confidence:.3f}",
                    font=ctk.CTkFont(size=10),
                    text_color="#ffa500" if confidence < 0.9 else "#90ee90"
                ).pack(side="left")
                
                ctk.CTkLabel(
                    info_frame,
                    text=f"#{instance_idx}",
                    font=ctk.CTkFont(size=9),
                    text_color="gray"
                ).pack(side="right")

                # NON-FACE label (always create; show/hide on toggle)
                nonface_label = ctk.CTkLabel(
                    img_container,
                    text="NON-FACE",
                    font=ctk.CTkFont(size=10, weight="bold"),
                    text_color="red",
                    fg_color="black"
                )
                self._card_nonface_labels[instance_idx] = nonface_label
                if not is_face:
                    nonface_label.place(relx=0.5, rely=0.4, anchor="center")

            # Configure grid columns and rows
            for col in range(new_images_per_row):
                self.gallery_frame.grid_columnconfigure(col, weight=1, minsize=140)
            
            total_rows = (len(self.gallery_frame.images_data) + new_images_per_row - 1) // new_images_per_row
            for row in range(total_rows):
                self.gallery_frame.grid_rowconfigure(row, weight=0, minsize=160)
            
            # Force update scrollable region
            self.gallery_frame.update_idletasks()
    
    def _extract_face_crop(self, face_info: dict) -> Optional[Image.Image]:
        """Extract face crop from video frame."""
        try:
            frame_number = int(face_info['frame_number'])
            x, y, w, h = int(face_info['x']), int(face_info['y']), int(face_info['w']), int(face_info['h'])
            
            # Find video file
            video_files = list(self.session_dir.glob("scenevideo.*"))
            if not video_files:
                return self._create_placeholder_image()
            
            video_path = video_files[0]
            
            # Open video and extract frame
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return self._create_placeholder_image()
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return self._create_placeholder_image()
            
            # Get frame dimensions
            h_img, w_img = frame.shape[:2]
            
            # Add padding
            pad = int(max(w, h) * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)
            
            if x2 <= x1 or y2 <= y1:
                return self._create_placeholder_image()
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return self._create_placeholder_image()
            
            # Convert to RGB
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            return Image.fromarray(face_crop_rgb)
            
        except Exception as e:
            print(f"Error extracting face crop: {e}")
            return self._create_placeholder_image()
    
    def _create_placeholder_image(self) -> Image.Image:
        """Create a placeholder image."""
        img = Image.new('RGB', (120, 120), color=(100, 100, 100))
        return img
    
    def _on_face_checkbox_toggle(self, instance_idx: int, checkbox_var: ctk.BooleanVar):
        """Toggle face/non-face for this instance; annotations persist across pages."""
        from datetime import datetime
        is_face = checkbox_var.get()
        self.annotations[instance_idx] = {
            'is_face': is_face,
            'reviewed_at': datetime.now().isoformat()
        }
        # Update NON-FACE label visibility on current page
        if getattr(self, "_card_nonface_labels", None) and instance_idx in self._card_nonface_labels:
            lbl = self._card_nonface_labels[instance_idx]
            if is_face:
                lbl.place_forget()
            else:
                lbl.place(relx=0.5, rely=0.4, anchor="center")
        self._update_stats()
        self._update_face_nonface_label()

    def _check_all_page(self):
        """Mark all instances on the current page as face."""
        if not getattr(self.gallery_frame, "images_data", None):
            return
        from datetime import datetime
        now = datetime.now().isoformat()
        for instance_idx, _, _ in self.gallery_frame.images_data:
            self.annotations[instance_idx] = {"is_face": True, "reviewed_at": now}
        self._create_gallery_grid(self.gallery_frame.images_data)
        self._update_stats()
        self._update_face_nonface_label()

    def _uncheck_all_page(self):
        """Mark all instances on the current page as non-face."""
        if not getattr(self.gallery_frame, "images_data", None):
            return
        from datetime import datetime
        now = datetime.now().isoformat()
        for instance_idx, _, _ in self.gallery_frame.images_data:
            self.annotations[instance_idx] = {"is_face": False, "reviewed_at": now}
        self._create_gallery_grid(self.gallery_frame.images_data)
        self._update_stats()
        self._update_face_nonface_label()
    
    def _update_face_nonface_label(self):
        """No-op: face/non-face count was removed from bottom bar."""
        pass
    
    def _save_annotations(self):
        """Save annotations to reviewer overlay CSV."""
        if self.df is None:
            return

        try:
            registry = ReviewerRegistry(self.project_dir)
            annotation_file = registry.get_is_face_annotation_path(
                self.reviewer_id, self.selected_participant, self.selected_session
            )
            annotation_file.parent.mkdir(parents=True, exist_ok=True)

            annotation_records = []
            for idx in self.df.index:
                ann = self.annotations.get(idx, {'is_face': True, 'reviewed_at': None})
                annotation_records.append({
                    'instance_index': idx,
                    'is_face': ann['is_face'],
                    'reviewed_at': ann['reviewed_at'] if ann['reviewed_at'] else '',
                })

            ann_df = pd.DataFrame(annotation_records)
            ann_df.to_csv(annotation_file, index=False)

            total = len(ann_df)
            marked_nonface = sum(1 for r in annotation_records if not r['is_face'])
            valid_faces = total - marked_nonface

            # Update review status with last_save = now (keep user's "reviewed" choice)
            from datetime import datetime
            now_iso = datetime.now().isoformat()
            self._save_review_status(
                self.selected_participant,
                self.selected_session,
                reviewed=self.reviewed_var.get(),
                last_save=now_iso,
            )

            # Refresh session dropdown to show last save and reviewed state (keep selection)
            if self.selected_participant and self.selected_session:
                self._refresh_session_dropdown()

            messagebox.showinfo(
                "Saved",
                f"Annotations saved!\n\n"
                f"File: {annotation_file}\n"
                f"Total instances: {total}\n"
                f"Valid faces: {valid_faces}\n"
                f"Non-face: {marked_nonface}"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to save annotations:\n{str(e)}")
    
    def _update_page_numbers(self, total_pages: int = None):
        """Update the clickable page number labels (matching popup gallery style)."""
        # Clear existing page elements
        for widget in self.page_numbers_frame.winfo_children():
            widget.destroy()
        
        if total_pages is None:
            total_pages = self.current_total_pages if hasattr(self, 'current_total_pages') else 1
        
        current_page = self.current_page
        
        if total_pages == 0:
            return
        
        # First page button (<<)
        first_label = ctk.CTkLabel(
            self.page_numbers_frame,
            text="<<",
            font=ctk.CTkFont(size=12),
            cursor="hand2"
        )
        first_label.pack(side="left", padx=3)
        if current_page > 0:
            first_label.bind("<Button-1>", lambda e: self._go_to_page(0))
            first_label.configure(text_color="#3b8ed0")
        else:
            first_label.configure(text_color="gray")
        
        # Previous page label (<)
        prev_label = ctk.CTkLabel(
            self.page_numbers_frame,
            text="<",
            font=ctk.CTkFont(size=12),
            cursor="hand2"
        )
        prev_label.pack(side="left", padx=3)
        if current_page > 0:
            prev_label.bind("<Button-1>", lambda e: self._go_to_page(current_page - 1))
            prev_label.configure(text_color="#3b8ed0")
        else:
            prev_label.configure(text_color="gray")
        
        # Show page numbers (limit to reasonable number to avoid UI overflow)
        max_visible_pages = 15
        if total_pages <= max_visible_pages:
            pages_to_show = list(range(total_pages))
        else:
            pages_to_show = []
            pages_to_show.append(0)
            
            start = max(1, current_page - 2)
            end = min(total_pages - 1, current_page + 2)
            for p in range(start, end + 1):
                if p not in pages_to_show:
                    pages_to_show.append(p)
            
            if total_pages - 1 not in pages_to_show:
                pages_to_show.append(total_pages - 1)
            
            if pages_to_show[1] > 1:
                pages_to_show.insert(1, None)
            if pages_to_show[-2] < total_pages - 2:
                pages_to_show.insert(-1, None)
        
        # Create page number labels
        for page_num in pages_to_show:
            if page_num is None:
                ctk.CTkLabel(
                    self.page_numbers_frame,
                    text="...",
                    font=ctk.CTkFont(size=12)
                ).pack(side="left", padx=2)
            else:
                is_current = (page_num == current_page)
                page_label = ctk.CTkLabel(
                    self.page_numbers_frame,
                    text=str(page_num + 1),
                    font=ctk.CTkFont(size=12, weight="bold" if is_current else "normal"),
                    cursor="hand2"
                )
                page_label.pack(side="left", padx=3)
                
                if is_current:
                    page_label.configure(text_color="#3b8ed0")
                else:
                    page_label.configure(text_color="#3b8ed0")
                    page_label.bind("<Button-1>", lambda e, p=page_num: self._go_to_page(p))
        
        # Next page label (>)
        next_label = ctk.CTkLabel(
            self.page_numbers_frame,
            text=">",
            font=ctk.CTkFont(size=12),
            cursor="hand2"
        )
        next_label.pack(side="left", padx=3)
        if current_page < total_pages - 1:
            next_label.bind("<Button-1>", lambda e: self._go_to_page(current_page + 1))
            next_label.configure(text_color="#3b8ed0")
        else:
            next_label.configure(text_color="gray")
        
        # Last page button (>>)
        last_label = ctk.CTkLabel(
            self.page_numbers_frame,
            text=">>",
            font=ctk.CTkFont(size=12),
            cursor="hand2"
        )
        last_label.pack(side="left", padx=3)
        if current_page < total_pages - 1:
            last_label.bind("<Button-1>", lambda e: self._go_to_page(total_pages - 1))
            last_label.configure(text_color="#3b8ed0")
        else:
            last_label.configure(text_color="gray")
    
    def _go_to_page(self, page_num: int):
        """Go to specific page."""
        if hasattr(self, 'current_total_pages'):
            if 0 <= page_num < self.current_total_pages:
                self.current_page = page_num
                self._display_gallery()
        else:
            self.current_page = page_num
            self._display_gallery()


