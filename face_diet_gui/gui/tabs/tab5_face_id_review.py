"""Tab 5: Face ID Review — manual ID merging."""

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


class ManualReviewTab(ctk.CTkFrame):
    """Tab 4: Manual Review & Merging (refactored from original GUI)."""

    def __init__(self, master, settings_manager: SettingsManager,
                 project_dir: Path, reviewer_id: str):
        super().__init__(master)
        self.settings = settings_manager
        self.project_dir: Path = project_dir
        self.reviewer_id: str = reviewer_id

        # Selected participant
        self.selected_participant: Optional[str] = None
        self.participant_dir: Optional[Path] = None

        # Data storage
        self.df: Optional[pd.DataFrame] = None
        self.df_full: Optional[pd.DataFrame] = None
        self.face_groups: Dict = {}
        self.face_groups_all: Dict = {}
        self.merge_groups: Dict[str, List[str]] = {}
        self.face_id_to_merged: Dict[str, str] = {}
        self.merged_id_to_media: set = set()  # canonical (merged) face IDs marked as Media (TV, computer, flyer, etc.)
        self.session_bbox_stats: Dict[str, Dict[str, float]] = {}
        self.representative_sample_size = 30

        # Session filtering
        self.available_sessions: List[str] = []
        self.session_checkboxes: Dict[str, ctk.BooleanVar] = {}

        # UI components
        self.row_widgets: Dict[str, Dict] = {}
        self.selected_ids: set = set()

        self._setup_ui()
        self._load_settings()
    
    def _setup_ui(self):
        """Setup UI components."""
        # Title
        ctk.CTkLabel(
            self,
            text="Face ID Review",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(10, 15))

        # Participant selector (dropdown like Tab 2)
        self.sel_frame_tab4 = ctk.CTkFrame(self)
        self.sel_frame_tab4.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(
            self.sel_frame_tab4,
            text="Participant:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=(10, 4))

        self.participant_var_tab4 = ctk.StringVar(value="— select —")
        self.participant_dropdown_tab4 = ctk.CTkOptionMenu(
            self.sel_frame_tab4,
            variable=self.participant_var_tab4,
            values=["— select —"],
            width=280,
            command=self._on_participant_selected_tab4_dropdown
        )
        self.participant_dropdown_tab4.pack(side="left", padx=5, pady=8)

        self.load_participant_btn = ctk.CTkButton(
            self.sel_frame_tab4,
            text="Load Participant",
            command=self._load_participant_face_ids,
            width=140,
            height=35,
            state="disabled",
            fg_color="#007ACC",
            hover_color="#0066AA"
        )
        self.load_participant_btn.pack(side="left", padx=10)
        self.load_status_label = ctk.CTkLabel(
            self.sel_frame_tab4,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.load_status_label.pack(side="left", padx=(5, 10), pady=8)

        self._populate_participants_tab4()
        
        # Session filter panel
        self._create_session_filter_panel()

        # Merge controls
        self._create_merge_controls()
        
        # Face ID list
        self._create_face_list()
        
        # Save button
        self._create_save_button()
    
    def _create_session_filter_panel(self):
        """Create session filter panel."""
        self.session_filter_frame = ctk.CTkFrame(self)
        # Will be packed after loading data
        
        ctk.CTkLabel(
            self.session_filter_frame,
            text="Filter by Sessions:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        # Controls
        controls = ctk.CTkFrame(self.session_filter_frame)
        controls.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(
            controls,
            text="Select All",
            command=self._select_all_sessions,
            width=90,
            height=28
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            controls,
            text="Deselect All",
            command=self._deselect_all_sessions,
            width=90,
            height=28
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            controls,
            text="Apply Filter",
            command=self._apply_session_filter,
            width=90,
            height=28,
            fg_color="#28a745",
            hover_color="#218838"
        ).pack(side="left", padx=15)
        
        # Session checkboxes container
        self.session_checkboxes_frame = ctk.CTkFrame(self.session_filter_frame)
        self.session_checkboxes_frame.pack(fill="x", padx=15, pady=5)
    
    def _create_filter_controls(self):
        """Removed: min instances and min confidence are now in Tab 4 settings."""
    
    def _create_merge_controls(self):
        """Create merge/unmerge control buttons."""
        controls = ctk.CTkFrame(self)
        controls.pack(fill="x", padx=20, pady=(10, 10))
        
        # Merge button
        self.merge_btn = ctk.CTkButton(
            controls,
            text="Merge Selected",
            command=self._merge_selected,
            width=180,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#28a745",
            hover_color="#218838",
            text_color="white",
            text_color_disabled="white",
            state="disabled"
        )
        self.merge_btn.configure(fg_color=BTN_DISABLED_FG)  # initial disabled look
        self.merge_btn.pack(side="left", padx=(10, 10))
        
        # Set as Media (face from TV, computer, flyer, etc.)
        self.set_media_btn = ctk.CTkButton(
            controls,
            text="Set as Media",
            command=self._set_selected_as_media,
            width=120,
            height=35,
            font=ctk.CTkFont(size=12),
            state="disabled"
        )
        self.set_media_btn.pack(side="left", padx=(5, 5))

        # Clear Media — remove media flag from selected IDs
        self.unset_media_btn = ctk.CTkButton(
            controls,
            text="Clear Media",
            command=self._set_selected_as_non_media,
            width=120,
            height=35,
            font=ctk.CTkFont(size=12),
            state="disabled"
        )
        self.unset_media_btn.pack(side="left", padx=(0, 10))

        # Selection info
        self.selection_label = ctk.CTkLabel(
            controls,
            text="No faces selected",
            font=ctk.CTkFont(size=13)
        )
        self.selection_label.pack(side="left", padx=(10, 0))
    
    def _create_face_list(self):
        """Create the scrollable face ID list."""
        list_container = ctk.CTkFrame(self)
        list_container.pack(fill="both", expand=True, padx=20, pady=(0, 15))
        
        # Header — column widths must match _create_face_row
        header = ctk.CTkFrame(list_container, height=40)
        header.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(header, text="Select",       width=50,  font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=2)
        ctk.CTkLabel(header, text="Face Preview", width=560, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=2)
        ctk.CTkLabel(header, text="Face ID(s)",   width=520, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=2)
        ctk.CTkLabel(header, text="Media",        width=80,  font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=2)
        ctk.CTkLabel(header, text="Instances",    width=100, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=2)
        ctk.CTkLabel(header, text="Actions",      width=150, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=2)
        
        # Scrollable frame for face list
        self.face_list_frame = ctk.CTkScrollableFrame(list_container, height=400)
        self.face_list_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))
    
    def _create_save_button(self):
        """Create the save and export buttons."""
        save_frame = ctk.CTkFrame(self, fg_color="transparent")
        save_frame.pack(fill="x", padx=20)
        
        # Button container for horizontal layout
        button_container = ctk.CTkFrame(save_frame, fg_color="transparent")
        button_container.pack(pady=10)
        
        self.save_btn = ctk.CTkButton(
            button_container,
            text="Save Annotations",
            command=self._save_results,
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
        self.save_btn.pack(side="left", padx=5)
    
    def _populate_participants_tab4(self):
        """Populate the participant dropdown from the project directory."""
        if not self.project_dir or not self.project_dir.exists():
            self.participant_dropdown_tab4.configure(values=["— select —"])
            return

        registry = ReviewerRegistry(self.project_dir)
        participants = []
        for d in sorted(self.project_dir.iterdir()):
            if not d.is_dir() or d.name.startswith(("_", ".")):
                continue
            # Tab 4 output is one CSV in participant folder
            if registry.get_face_ids_path(d.name).exists():
                participants.append(d.name)

        values = ["— select —"] + participants
        self.participant_dropdown_tab4.configure(values=values)
        if not participants:
            self.participant_var_tab4.set("— select —")

    def _on_participant_selected_tab4_dropdown(self, choice: str):
        """Called when participant dropdown selection changes."""
        if choice == "— select —" or not choice:
            self.selected_participant = None
            self.participant_dir = None
            self.load_participant_btn.configure(state="disabled")
            return
        self.selected_participant = choice
        self.participant_dir = self.project_dir / choice
        self.load_participant_btn.configure(state="normal")

    def _load_participant_face_ids(self):
        """Load all face IDs for the selected participant (no min instances/confidence filter)."""
        if not self.selected_participant:
            messagebox.showerror("Error", "Please select a participant first.")
            return
        self._validate_and_load()

    def update_project_and_reviewer(self, project_dir: Path, reviewer_id: str):
        """Called when user changes project or reviewer via Back to setup."""
        self.project_dir = project_dir
        self.reviewer_id = reviewer_id
        self._populate_participants_tab4()

    def _select_all_sessions(self):
        """Select all session checkboxes."""
        for var in self.session_checkboxes.values():
            var.set(True)
    
    def _deselect_all_sessions(self):
        """Deselect all session checkboxes."""
        for var in self.session_checkboxes.values():
            var.set(False)
    
    def _apply_session_filter(self):
        """Apply session filter and refresh display."""
        if self.df_full is None:
            return
        
        # Get selected sessions
        selected_sessions = [
            session for session, var in self.session_checkboxes.items()
            if var.get()
        ]
        
        if not selected_sessions:
            messagebox.showwarning("Warning", "Please select at least one session!")
            return
        
        # Filter dataframe
        self.df = self.df_full[self.df_full['session_name'].isin(selected_sessions)].copy()
        
        # Recalculate face groups for filtered data
        self._recalculate_face_groups()
        
        # Refresh display
        self._display_face_list()
    
    def _recalculate_face_groups(self):
        """Recalculate face groups based on current filtered data."""
        min_instances = int(self.settings.get("stage3.min_instances", 1))

        df_filtered = self.df.copy()
        face_counts = df_filtered['face_id'].value_counts()
        eligible_ids = face_counts[face_counts >= min_instances].index.tolist()
        
        # Rebuild face_groups
        self.face_groups = {}
        for face_id in eligible_ids:
            face_instances = df_filtered[df_filtered['face_id'] == face_id]
            count = len(face_instances)
            
            representative = self._find_representative_face(face_instances)
            thumbnail = self._extract_face_crop(representative) if representative else None
            
            # Check if this face_id is part of a merged group
            original_ids = [face_id]
            if face_id in self.face_groups_all:
                original_ids = self.face_groups_all[face_id].get('original_ids', [face_id])
            
            self.face_groups[face_id] = {
                'count': count,
                'representative': representative,
                'thumbnail': thumbnail,
                'original_ids': original_ids
            }
        
        # Sort by count
        self.face_groups = dict(
            sorted(self.face_groups.items(), key=lambda x: x[1]['count'], reverse=True)
        )
    
    def _apply_min_filters(self):
        """Recalculate groups using current settings and refresh the face list."""
        if self.df_full is None:
            return
        self._recalculate_face_groups()
        if not self.face_groups:
            messagebox.showwarning("No Results", "No face IDs meet the current Min Instances filter.")
        self._display_face_list()

    def _validate_and_load(self):
        """Validate that clustering has been run for this participant, then load."""
        registry = ReviewerRegistry(self.project_dir)
        face_ids_path = registry.get_face_ids_path(self.selected_participant)
        if not face_ids_path.exists():
            messagebox.showerror(
                "Error",
                f"No face ID assignments found for participant '{self.selected_participant}'.\n\n"
                f"Please run Tab 4 (Face ID Clustering) first."
            )
            return

        self.load_status_label.configure(text="Loading…")
        self.load_participant_btn.configure(state="disabled")
        thread = threading.Thread(target=self._load_data_thread, daemon=True)
        thread.start()

    def _load_data_thread(self):
        """Load and process data in background thread."""
        try:
            registry = ReviewerRegistry(self.project_dir)
            participant = self.selected_participant
            participant_path = self.project_dir / participant

            # Load face_ids overlay from participant folder (Tab 4 output, one CSV per participant, shared)
            face_ids_path = registry.get_face_ids_path(participant)
            face_ids_df = pd.read_csv(face_ids_path)

            # Load all per-session face_detections.csv
            session_dfs = []
            for session_dir in sorted(participant_path.iterdir()):
                if not session_dir.is_dir() or session_dir.name.startswith(("_", ".")):
                    continue
                csv_path = session_dir / "face_detections.csv"
                if not csv_path.exists():
                    continue
                sdf = pd.read_csv(csv_path)
                sdf['session_name'] = session_dir.name
                sdf['instance_index'] = sdf.index
                session_dfs.append(sdf)

            if not session_dfs:
                self.after(0, lambda: self.load_status_label.configure(text=""))
                self.after(0, lambda: self.load_participant_btn.configure(state="normal"))
                self.after(0, lambda: messagebox.showerror(
                    "Error", "No face_detections.csv files found for this participant."
                ))
                return

            base_df = pd.concat(session_dfs, ignore_index=True)

            # Merge face_ids onto base data
            base_df = base_df.merge(
                face_ids_df[['session_name', 'instance_index', 'face_id']],
                on=['session_name', 'instance_index'],
                how='inner'
            )

            self.df_full = base_df
            self.df = self.df_full.copy()

            # Get available sessions
            self.available_sessions = sorted(self.df['session_name'].unique())
            
            # Initialize merge tracking for ALL IDs
            all_face_ids = self.df['face_id'].unique()
            self.face_id_to_merged = {fid: fid for fid in all_face_ids}
            self.merged_id_to_media = set()
            # Load saved merges and media if present
            merges_path = registry.get_merges_path(self.reviewer_id, participant)
            if merges_path.exists():
                try:
                    merges_df = pd.read_csv(merges_path)
                    for _, r in merges_df.iterrows():
                        self.face_id_to_merged[str(r['face_id'])] = str(r['merged_face_id'])
                    if 'is_media' in merges_df.columns:
                        for _, r in merges_df.iterrows():
                            if r.get('is_media') in (True, 'True', 1, '1'):
                                self.merged_id_to_media.add(str(r['merged_face_id']))
                except Exception:
                    pass
            
            # Create session filter UI
            self.after(0, self._create_session_checkboxes)
            
            # Apply min_instances from Tab 4 settings
            min_instances_load = int(self.settings.get("stage3.min_instances", 1))
            df_for_build = self.df.copy()
            face_counts = df_for_build['face_id'].value_counts()
            eligible_ids = face_counts[face_counts >= min_instances_load].index.tolist()
            
            # Compute per-session bbox extents (use full df)
            self.session_bbox_stats = {}
            for session_name, session_df in self.df.groupby('session_name'):
                max_x2 = float((session_df['x'] + session_df['w']).max())
                max_y2 = float((session_df['y'] + session_df['h']).max())
                self.session_bbox_stats[session_name] = {
                    'max_x2': max_x2,
                    'max_y2': max_y2,
                }
            
            # Build face groups (all face IDs)
            self.face_groups = {}
            for face_id in eligible_ids:
                face_instances = df_for_build[df_for_build['face_id'] == face_id]
                count = len(face_instances)
                representative = self._find_representative_face(face_instances)
                thumbnail = self._extract_face_crop(representative) if representative else None
                self.face_groups[face_id] = {
                    'count': count,
                    'representative': representative,
                    'thumbnail': thumbnail,
                    'original_ids': [face_id]
                }
                self.face_id_to_merged[face_id] = face_id
            
            self.face_groups = dict(
                sorted(self.face_groups.items(), key=lambda x: x[1]['count'], reverse=True)
            )
            self.face_groups_all = self.face_groups.copy()

            # Restore previously saved merge state so Tab 5 reflects the reviewer's last save
            self._reconstruct_saved_merges()

            # Display results
            self.after(0, self._display_face_list)
            self.after(0, lambda: self.load_status_label.configure(text=""))
            self.after(0, lambda: self.save_btn.configure(state="normal", fg_color="#28a745"))
            self.after(0, lambda: self.load_participant_btn.configure(state="normal"))
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\n[ERROR] Error during data loading:")
            print(error_details)
            
            self.after(0, lambda: self.load_status_label.configure(text=""))
            self.after(0, lambda: self.load_participant_btn.configure(state="normal"))
            self.after(0, lambda: messagebox.showerror(
                "Error",
                f"Failed to load data:\n{str(e)}\n\nCheck terminal for details."
            ))
    
    def _reconstruct_saved_merges(self):
        """Rebuild merged face_groups from the saved face_id_to_merged mapping."""
        from collections import defaultdict
        groups = defaultdict(list)
        for orig_id, merged_id in self.face_id_to_merged.items():
            groups[merged_id].append(orig_id)

        for merged_id, original_ids in groups.items():
            if len(original_ids) <= 1:
                continue  # not merged
            # Only reconstruct if the canonical merged_id exists in face_groups
            if merged_id not in self.face_groups:
                continue
            all_thumbs, total_count = [], 0
            for orig_id in original_ids:
                if orig_id not in self.face_groups:
                    continue
                g = self.face_groups[orig_id]
                total_count += g['count']
                t = g.get('thumbnail')
                if t:
                    all_thumbs.append(t)
                all_thumbs.extend(g.get('thumbnails', []))
            # Deduplicate thumbnails by id()
            seen, unique_thumbs = set(), []
            for t in all_thumbs:
                if id(t) not in seen:
                    seen.add(id(t))
                    unique_thumbs.append(t)
            self.face_groups[merged_id] = {
                'count': total_count,
                'representative': self.face_groups[merged_id]['representative'],
                'thumbnail': self.face_groups[merged_id].get('thumbnail'),
                'thumbnails': unique_thumbs[:6],
                'original_ids': original_ids,
            }
            for orig_id in original_ids:
                if orig_id != merged_id and orig_id in self.face_groups:
                    del self.face_groups[orig_id]

        self.face_groups = dict(
            sorted(self.face_groups.items(), key=lambda x: x[1]['count'], reverse=True)
        )

    def _create_session_checkboxes(self):
        """Create checkboxes for session filtering."""
        # Clear existing
        for widget in self.session_checkboxes_frame.winfo_children():
            widget.destroy()
        self.session_checkboxes.clear()
        
        # Create checkboxes
        for session_name in self.available_sessions:
            var = ctk.BooleanVar(value=True)
            cb = ctk.CTkCheckBox(
                self.session_checkboxes_frame,
                text=session_name,
                variable=var,
                font=ctk.CTkFont(size=11),
                checkbox_width=16,
                checkbox_height=16
            )
            cb.pack(side="left", padx=5, pady=2)
            self.session_checkboxes[session_name] = var
        
        # Pack the filter frame
        if not self.session_filter_frame.winfo_ismapped():
            self.session_filter_frame.pack(fill="x", padx=20, pady=(0, 10), after=self.sel_frame_tab4)
    
    def _find_representative_face(self, face_instances: pd.DataFrame) -> Optional[dict]:
        """Find the face closest to centroid (simplified from original)."""
        if face_instances.empty:
            return None
        
        # Use confidence + sharpness if available
        if 'confidence' in face_instances.columns and 'sharpness' in face_instances.columns:
            conf = face_instances['confidence'].astype(float)
            sharp = face_instances['sharpness'].astype(float)
            conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
            sharp_norm = (sharp - sharp.min()) / (sharp.max() - sharp.min() + 1e-8)
            score = conf_norm * sharp_norm
            best_idx = score.idxmax()
            return face_instances.loc[best_idx].to_dict()
        elif 'confidence' in face_instances.columns:
            return face_instances.sort_values('confidence', ascending=False).iloc[0].to_dict()
        else:
            return face_instances.iloc[0].to_dict()
    
    def _extract_face_crop(self, face_info: dict) -> Optional[Image.Image]:
        """Extract face crop from video frame."""
        try:
            session_name = face_info['session_name']
            frame_number = int(face_info['frame_number'])
            x, y, w, h = int(face_info['x']), int(face_info['y']), int(face_info['w']), int(face_info['h'])

            # Sanity-check: if w or h are suspiciously small (< 5px) the CSV likely
            # has normalised or otherwise incorrect coordinates.
            if w < 5 or h < 5:
                print(f"[Tab5 crop warning] Tiny bbox for {session_name} frame {frame_number}: "
                      f"x={x} y={y} w={w} h={h}. Check face_detections.csv coordinate scale.")
                return self._create_placeholder_image()

            session_dir = self.participant_dir / session_name
            if not session_dir.exists():
                return self._create_placeholder_image()

            video_files = list(session_dir.glob("scenevideo.*"))
            if not video_files:
                return self._create_placeholder_image()

            video_path = video_files[0]
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return self._create_placeholder_image()

            # Use millisecond seek for better accuracy with compressed codecs
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.set(cv2.CAP_PROP_POS_MSEC, (frame_number / fps) * 1000.0)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                return self._create_placeholder_image()

            h_img, w_img = frame.shape[:2]

            # Sanity-check bbox against frame size
            if x >= w_img or y >= h_img:
                print(f"[Tab5 crop warning] bbox origin ({x},{y}) outside frame "
                      f"({w_img}×{h_img}) for {session_name} frame {frame_number}.")
                return self._create_placeholder_image()

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

            face_crop = cv2.resize(face_crop, (80, 80), interpolation=cv2.INTER_AREA)
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            return Image.fromarray(face_crop_rgb)

        except Exception as e:
            print(f"[Tab5 crop error] {e}")
            return self._create_placeholder_image()
    
    def _create_placeholder_image(self) -> Image.Image:
        """Create a placeholder image."""
        img = Image.new('RGB', (80, 80), color=(100, 100, 100))
        return img
    
    def _display_face_list(self):
        """Display all face IDs in the scrollable list."""
        # Clear existing widgets
        for widget in self.face_list_frame.winfo_children():
            widget.destroy()
        
        self.row_widgets.clear()
        self.selected_ids.clear()
        
        # Create rows
        for face_id, info in self.face_groups.items():
            self._create_face_row(face_id, info)
        
        self._update_selection_label()
    
    def _create_face_row(self, face_id: str, info: dict):
        """Create a row for a face ID. Column widths must match header in _create_face_list."""
        row_frame = ctk.CTkFrame(self.face_list_frame)
        row_frame.pack(fill="x", padx=5, pady=2)
        row_frame.bind("<Double-Button-1>", lambda e: self._open_gallery_popup(face_id))

        # — Select (50px) —
        checkbox_var = ctk.BooleanVar()
        checkbox = ctk.CTkCheckBox(
            row_frame, text="", variable=checkbox_var, width=50,
            command=lambda: self._on_checkbox_toggle(face_id, checkbox_var.get()),
            checkbox_width=18, checkbox_height=18
        )
        checkbox.pack(side="left", padx=2)

        # — Face Preview (560px fixed) —
        thumbnail_container = ctk.CTkFrame(row_frame, width=560, height=70)
        thumbnail_container.pack(side="left", padx=2)
        thumbnail_container.pack_propagate(False)

        thumbnails = info.get('thumbnails', [])
        if not thumbnails:
            thumb = info.get('thumbnail')
            if thumb:
                thumbnails = [thumb]
        for thumb in thumbnails[:4]:  # max 4 thumbnails in preview
            if thumb:
                thumb_tk = ImageTk.PhotoImage(thumb)
                thumb_label = ctk.CTkLabel(thumbnail_container, image=thumb_tk, text="")
                thumb_label.image = thumb_tk
                thumb_label.pack(side="left", padx=2, pady=2)
                thumb_label.bind("<Double-Button-1>", lambda e, fid=face_id: self._open_gallery_popup(fid))

        # — Face ID(s) (520px) —
        original_ids = info.get('original_ids', [face_id])
        if len(original_ids) > 1:
            id_text = f"{face_id}\n← {', '.join(original_ids)}"
        else:
            id_text = face_id

        id_label = ctk.CTkLabel(
            row_frame, text=id_text, width=520, anchor="w",
            font=ctk.CTkFont(size=12), wraplength=515
        )
        id_label.pack(side="left", padx=2)
        id_label.bind("<Double-Button-1>", lambda e: self._open_gallery_popup(face_id))

        # — Media (80px) — gray "No" / red "Yes"
        is_media = face_id in self.merged_id_to_media
        media_label = ctk.CTkLabel(
            row_frame,
            text="Yes" if is_media else "No",
            width=80,
            font=ctk.CTkFont(size=12),
            text_color="#dc3545" if is_media else "gray"
        )
        media_label.pack(side="left", padx=2)

        # — Instances (100px) —
        ctk.CTkLabel(
            row_frame, text=str(info['count']), width=100,
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=2)

        # — Actions (150px) —
        if len(original_ids) > 1:
            ctk.CTkButton(
                row_frame, text="Unmerge", width=140, height=28,
                command=lambda: self._unmerge_group(face_id),
                fg_color="#dc3545", hover_color="#c82333"
            ).pack(side="left", padx=2)

        # Store references
        self.row_widgets[face_id] = {
            'frame': row_frame,
            'checkbox': checkbox,
            'checkbox_var': checkbox_var
        }
    
    def _on_checkbox_toggle(self, face_id: str, is_selected: bool):
        """Handle checkbox toggle."""
        if is_selected:
            self.selected_ids.add(face_id)
        else:
            self.selected_ids.discard(face_id)
        self._update_selection_label()
    
    def _set_selected_as_media(self):
        """Mark selected face IDs as Media (from TV, computer, flyer, etc.)."""
        if not self.selected_ids:
            return
        for fid in self.selected_ids:
            self.merged_id_to_media.add(fid)
        messagebox.showinfo("Set as Media", f"Marked {len(self.selected_ids)} face ID(s) as Media.")
        self._display_face_list()

    def _set_selected_as_non_media(self):
        """Remove the Media flag from selected face IDs."""
        if not self.selected_ids:
            return
        removed = [fid for fid in self.selected_ids if fid in self.merged_id_to_media]
        for fid in removed:
            self.merged_id_to_media.discard(fid)
        if removed:
            messagebox.showinfo("Unset Media", f"Removed Media flag from {len(removed)} face ID(s).")
            self._display_face_list()
    
    def _update_selection_label(self):
        """Update the selection info label and merge/Set as Media button states."""
        count = len(self.selected_ids)
        if count == 0:
            text = "No faces selected"
        elif count == 1:
            text = "1 face selected"
        else:
            text = f"{count} faces selected"
        
        self.selection_label.configure(text=text)
        if count >= 2:
            self.merge_btn.configure(state="normal", fg_color="#28a745")
        else:
            self.merge_btn.configure(state="disabled", fg_color=BTN_DISABLED_FG)
        if count >= 1:
            self.set_media_btn.configure(state="normal")
            self.unset_media_btn.configure(state="normal")
        else:
            self.set_media_btn.configure(state="disabled")
            self.unset_media_btn.configure(state="disabled")
    
    def _merge_selected(self):
        """Merge selected face IDs into a single group."""
        if len(self.selected_ids) < 2:
            return
        
        selected_list = list(self.selected_ids)
        
        # Find the ID with most instances
        counts = {fid: self.face_groups[fid]['count'] for fid in selected_list}
        merged_id = max(counts, key=counts.get)
        
        # Collect all original IDs and thumbnails
        all_original_ids = []
        total_count = 0
        all_thumbnails = []
        
        for fid in selected_list:
            original_ids = self.face_groups[fid].get('original_ids', [fid])
            all_original_ids.extend(original_ids)
            total_count += self.face_groups[fid]['count']
            
            # Collect thumbnails from all selected IDs
            thumb = self.face_groups[fid].get('thumbnail')
            if thumb:
                all_thumbnails.append(thumb)
            # Also check for multiple thumbnails if they exist
            thumbs = self.face_groups[fid].get('thumbnails', [])
            if thumbs:
                all_thumbnails.extend(thumbs)
        
        # Remove duplicates while preserving order (keep first occurrence)
        seen = set()
        unique_thumbnails = []
        for thumb in all_thumbnails:
            # Use id() as a simple way to check uniqueness
            thumb_id = id(thumb)
            if thumb_id not in seen:
                seen.add(thumb_id)
                unique_thumbnails.append(thumb)
        
        # Limit to max 6 thumbnails to avoid UI overflow
        unique_thumbnails = unique_thumbnails[:6]
        
        # Create merged group
        merged_info = {
            'count': total_count,
            'representative': self.face_groups[merged_id]['representative'],
            'thumbnail': self.face_groups[merged_id]['thumbnail'],  # Keep for backward compatibility
            'thumbnails': unique_thumbnails,  # List of all thumbnails
            'original_ids': all_original_ids
        }
        
        # Propagate media label: all media → media; none → not media; mixed → False + warn
        any_media  = any(fid in self.merged_id_to_media for fid in selected_list)
        all_media  = all(fid in self.merged_id_to_media for fid in selected_list)
        mixed_media = any_media and not all_media
        for fid in selected_list:
            self.merged_id_to_media.discard(fid)
        if all_media:
            self.merged_id_to_media.add(merged_id)
        # mixed → merged group is NOT media; warn below

        # Update merge tracking
        for orig_id in all_original_ids:
            self.face_id_to_merged[orig_id] = merged_id

        # Remove selected IDs and add merged
        for fid in selected_list:
            if fid in self.face_groups:
                del self.face_groups[fid]

        self.face_groups[merged_id] = merged_info

        # Re-sort by count
        self.face_groups = dict(
            sorted(self.face_groups.items(), key=lambda x: x[1]['count'], reverse=True)
        )

        # Clear selection
        self.selected_ids.clear()

        # Refresh display
        self._display_face_list()

        if mixed_media:
            messagebox.showwarning(
                "Merged (media conflict)",
                f"Merged {len(selected_list)} face IDs into {merged_id}.\n\n"
                "Warning: some were marked as Media and some were not.\n"
                "The merged group has been set to Non-media. "
                "Use 'Set as Media' if needed."
            )
        else:
            messagebox.showinfo("Merged", f"Merged {len(selected_list)} face IDs into {merged_id}")
    
    def _unmerge_group(self, merged_id: str):
        """Unmerge a merged group back into separate IDs."""
        if merged_id not in self.face_groups:
            return
        
        merged_info = self.face_groups[merged_id]
        original_ids = merged_info.get('original_ids', [])
        
        if len(original_ids) <= 1:
            return
        
        # Capture media status before removing the merged group
        was_media = merged_id in self.merged_id_to_media

        if merged_id in self.face_groups:
            del self.face_groups[merged_id]
        self.merged_id_to_media.discard(merged_id)

        # Restore individual groups and propagate the merged group's media status
        for orig_id in original_ids:
            if orig_id in self.face_groups_all:
                self.face_groups[orig_id] = self.face_groups_all[orig_id].copy()
            self.face_id_to_merged[orig_id] = orig_id
            if was_media:
                self.merged_id_to_media.add(orig_id)
            else:
                self.merged_id_to_media.discard(orig_id)
        
        # Re-sort
        self.face_groups = dict(
            sorted(self.face_groups.items(), key=lambda x: x[1]['count'], reverse=True)
        )
        
        # Refresh display
        self._display_face_list()
        
        messagebox.showinfo(
            "Unmerged",
            f"Unmerged {merged_id} into {len(original_ids)} separate IDs"
        )
    
    def _open_gallery_popup(self, face_id: str):
        """Open gallery popup showing all instances of a face ID."""
        if face_id not in self.face_groups:
            return
        
        # Check if this is a merged group
        info = self.face_groups[face_id]
        original_ids = info.get('original_ids', [face_id])
        
        # Get all instances - if merged, get from all original IDs
        if len(original_ids) > 1:
            face_instances = self.df[self.df['face_id'].isin(original_ids)]
            is_merged = True
        else:
            face_instances = self.df[self.df['face_id'] == face_id]
            is_merged = False
        
        num_instances = len(face_instances)
        
        # Create popup window
        popup = ctk.CTkToplevel(self)
        popup.title(f"Face ID: {face_id}")
        popup.geometry("900x700")
        popup.grab_set()  # Modal
        
        # Title - show merged status if applicable
        if is_merged:
            title_text = f"Face ID: {face_id} ({num_instances} instances, merged from {len(original_ids)} IDs)"
        else:
            title_text = f"Face ID: {face_id} ({num_instances} instances)"
        
        ctk.CTkLabel(
            popup,
            text=title_text,
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=10)
        
        # Pagination parameters
        batch_size = 100
        total_pages = (num_instances + batch_size - 1) // batch_size if num_instances > 0 else 1
        
        # Store pagination state in popup
        popup.current_page = 0
        popup.total_pages = total_pages
        popup.batch_size = batch_size
        popup.face_instances = face_instances
        popup.face_id = face_id
        popup.num_instances = num_instances
        
        # Store selected instances (by their dataframe index)
        popup.selected_instances = set()
        
        # Pagination controls frame
        pagination_controls_frame = ctk.CTkFrame(popup)
        pagination_controls_frame.pack(pady=5)
        
        # Batch size input
        batch_size_frame = ctk.CTkFrame(pagination_controls_frame)
        batch_size_frame.pack(side="left", padx=5)
        
        ctk.CTkLabel(
            batch_size_frame,
            text="Instances per page:",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=5)
        
        batch_size_entry = ctk.CTkEntry(
            batch_size_frame,
            width=80,
            font=ctk.CTkFont(size=11)
        )
        batch_size_entry.insert(0, str(batch_size))
        batch_size_entry.pack(side="left", padx=5)
        
        # Load button
        load_btn = ctk.CTkButton(
            batch_size_frame,
            text="Load",
            width=80,
            height=30,
            font=ctk.CTkFont(size=12),
            command=lambda: self._gallery_load_page(popup, gallery_frame, loading_label, 
                                                   batch_size_entry, load_btn, page_numbers_frame)
        )
        load_btn.pack(side="left", padx=5)
        
        # Go to page
        ctk.CTkLabel(pagination_controls_frame, text="Go to page:", font=ctk.CTkFont(size=11)).pack(side="left", padx=(15, 4))
        popup.page_go_entry = ctk.CTkEntry(pagination_controls_frame, width=50, font=ctk.CTkFont(size=11))
        popup.page_go_entry.pack(side="left", padx=(0, 4))
        def _on_popup_page_go():
            try:
                n = int(popup.page_go_entry.get().strip())
                if 1 <= n <= popup.total_pages:
                    self._gallery_go_to_page(popup, n - 1, gallery_frame, loading_label, batch_size_entry, load_btn, page_numbers_frame)
            except (ValueError, tkinter.TclError):
                pass
        popup.page_go_btn = ctk.CTkButton(pagination_controls_frame, text="Go", width=40, height=26, font=ctk.CTkFont(size=11), command=_on_popup_page_go)
        popup.page_go_btn.pack(side="left", padx=(0, 10))
        popup.page_go_entry.bind("<Return>", lambda e: _on_popup_page_go())
        
        # Page numbers frame (will be populated dynamically)
        page_numbers_frame = ctk.CTkFrame(pagination_controls_frame)
        page_numbers_frame.pack(side="left", padx=10)
        
        # Store references in popup for later access
        popup.batch_size_entry = batch_size_entry
        popup.load_btn = load_btn
        popup.page_numbers_frame = page_numbers_frame
        
        # Loading message
        loading_label = ctk.CTkLabel(
            popup,
            text="Loading images...",
            font=ctk.CTkFont(size=14)
        )
        loading_label.pack(pady=20)
        
        # Scrollable gallery (responsive width)
        gallery_frame = ctk.CTkScrollableFrame(popup, height=450)
        gallery_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Action buttons frame
        action_frame = ctk.CTkFrame(popup)
        action_frame.pack(pady=10)
        
        # Selection label
        selection_label = ctk.CTkLabel(
            action_frame,
            text="0 instances selected",
            font=ctk.CTkFont(size=12)
        )
        selection_label.pack(side="left", padx=10)
        popup.selection_label = selection_label
        
        # Create New ID button
        ctk.CTkButton(
            action_frame,
            text="Create New ID",
            command=lambda: self._gallery_create_new_id(popup),
            width=150,
            height=35,
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=5)
        
        # Assign to Existing ID button
        ctk.CTkButton(
            action_frame,
            text="Assign to Existing ID",
            command=lambda: self._gallery_assign_to_existing(popup),
            width=180,
            height=35,
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=5)
        
        # Clear Selection button
        ctk.CTkButton(
            action_frame,
            text="Clear Selection",
            command=lambda: self._gallery_clear_selection(popup, gallery_frame, loading_label),
            width=150,
            height=35,
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=5)
        
        # Close button
        ctk.CTkButton(
            action_frame,
            text="Close",
            command=popup.destroy,
            width=100,
            height=35,
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=5)
        
        # Initialize page numbers and load first page
        self._update_gallery_page_numbers(popup, page_numbers_frame, gallery_frame, loading_label, load_btn)
        self._gallery_load_page(popup, gallery_frame, loading_label, batch_size_entry, load_btn, page_numbers_frame)
    
    def _calculate_images_per_row(self, gallery_frame):
        """Calculate number of images per row based on gallery frame width."""
        try:
            # Get gallery frame width
            gallery_frame.update_idletasks()
            frame_width = gallery_frame.winfo_width()
            
            # If width is 1 (not yet rendered), use default
            if frame_width <= 1:
                frame_width = 850  # Default width
            
            # Image frame width: 130px + padding (5px each side) = 140px per image
            image_width = 140
            images_per_row = max(1, frame_width // image_width)
            
            return images_per_row
        except:
            return 6  # Default fallback
    
    def _load_gallery_images(self, face_instances: pd.DataFrame, gallery_frame, loading_label, 
                            current_page: int = 0, batch_size: int = 100, total_pages: int = 1,
                            load_btn=None):
        """Load face crops for gallery page (in background thread)."""
        try:
            # Calculate slice for current page
            start_idx = current_page * batch_size
            end_idx = min(start_idx + batch_size, len(face_instances))
            page_instances = face_instances.iloc[start_idx:end_idx]
            
            images = []
            
            # Show loading message and update button
            self.after(0, lambda: loading_label.configure(text=f"Loading page {current_page + 1} of {total_pages}..."))
            self.after(0, lambda: loading_label.pack(pady=20))
            if load_btn:
                self.after(0, lambda: load_btn.configure(text="Loading...", state="disabled"))
            
            # Clear existing images
            self.after(0, lambda: self._clear_gallery_frame(gallery_frame))
            
            for idx, row in page_instances.iterrows():
                crop = self._extract_face_crop(row.to_dict())
                if crop:
                    # Resize to larger size for gallery
                    crop = crop.resize((120, 120), Image.LANCZOS)
                    images.append(crop)
            
            # Calculate images per row based on frame width
            images_per_row = self._calculate_images_per_row(gallery_frame)
            
            # Update UI with images
            self.after(0, lambda: loading_label.pack_forget())
            if load_btn:
                self.after(0, lambda: load_btn.configure(text="Load", state="normal"))
            
            # Create grid with responsive layout
            popup_window = gallery_frame.winfo_toplevel()
            for idx, img in enumerate(images):
                row_idx = idx // images_per_row
                col_idx = idx % images_per_row
                
                # Get the actual dataframe index for this image
                df_idx = page_instances.iloc[idx].name
                
                # Create container frame (same size as image)
                img_container = ctk.CTkFrame(gallery_frame, width=130, height=130)
                img_container.grid(row=row_idx, column=col_idx, padx=5, pady=5)
                img_container.grid_propagate(False)
                
                # Add image as background
                img_tk = ImageTk.PhotoImage(img)
                img_label = ctk.CTkLabel(img_container, image=img_tk, text="")
                img_label.image = img_tk  # Keep reference
                img_label.place(x=0, y=0, relwidth=1, relheight=1)
                
                # Checkbox overlay (top-right corner)
                checkbox_var = ctk.BooleanVar(value=df_idx in popup_window.selected_instances)
                checkbox = ctk.CTkCheckBox(
                    img_container,
                    text="",
                    variable=checkbox_var,
                    width=20,
                    command=lambda idx=df_idx, var=checkbox_var: self._gallery_toggle_selection(popup_window, idx, var),
                    checkbox_width=18,
                    checkbox_height=18,
                    fg_color="#3b8ed0",
                    hover_color="#2a6fa5"
                )
                checkbox.place(x=5, y=5)
                
                # Make entire image clickable to toggle selection
                def toggle_on_click(event, idx=df_idx, var=checkbox_var, cb=checkbox):
                    var.set(not var.get())
                    self._gallery_toggle_selection(popup_window, idx, var)
                
                img_label.bind("<Button-1>", toggle_on_click)
                row_data = page_instances.iloc[idx].to_dict()
                session_dir = self.participant_dir / row_data.get("session_name", "")
                def on_double(event, rd=row_data, sd=session_dir):
                    if sd.exists():
                        _show_full_frame_toplevel(self, sd, rd)
                img_label.bind("<Double-Button-1>", on_double)
                img_container.configure(cursor="hand2")
                img_label.configure(cursor="hand2")
            
            # Store images_per_row and images in gallery_frame for resize handling
            gallery_frame.images_per_row = images_per_row
            gallery_frame.images = images  # Keep reference to images
            gallery_frame.image_tk_objects = []  # Store PhotoImage objects to prevent garbage collection
            
            # Store PhotoImage references
            for img in images:
                img_tk = ImageTk.PhotoImage(img)
                gallery_frame.image_tk_objects.append(img_tk)
            
            # Bind resize event to regenerate layout
            def on_resize(event=None):
                if not hasattr(gallery_frame, 'images') or not gallery_frame.images:
                    return
                
                new_images_per_row = self._calculate_images_per_row(gallery_frame)
                if new_images_per_row != gallery_frame.images_per_row:
                    # Regenerate grid layout
                    self._clear_gallery_frame(gallery_frame)
                    for idx, img_tk in enumerate(gallery_frame.image_tk_objects):
                        row_idx = idx // new_images_per_row
                        col_idx = idx % new_images_per_row
                        
                        img_frame = ctk.CTkFrame(gallery_frame, width=130, height=130)
                        img_frame.grid(row=row_idx, column=col_idx, padx=5, pady=5)
                        
                        img_label = ctk.CTkLabel(img_frame, image=img_tk, text="")
                        img_label.image = img_tk  # Keep reference
                        img_label.pack(padx=5, pady=5)
                    
                    gallery_frame.images_per_row = new_images_per_row
            
            # Bind resize event to popup window (only once)
            popup_window = gallery_frame.winfo_toplevel()
            if not hasattr(popup_window, '_gallery_resize_bound'):
                popup_window.bind('<Configure>', on_resize)
                popup_window._gallery_resize_bound = True
        
        except Exception as e:
            self.after(0, lambda: loading_label.configure(text=f"Error loading images: {e}"))
            if load_btn:
                self.after(0, lambda: load_btn.configure(text="Load", state="normal"))
    
    def _clear_gallery_frame(self, gallery_frame):
        """Clear all widgets from gallery frame."""
        for widget in gallery_frame.winfo_children():
            widget.destroy()
    
    def _gallery_load_page(self, popup, gallery_frame, loading_label, batch_size_entry, load_btn, page_numbers_frame):
        """Load current page with potentially updated batch size."""
        try:
            # Get batch size from entry
            new_batch_size = int(batch_size_entry.get())
            if new_batch_size < 1:
                new_batch_size = 100
                batch_size_entry.delete(0, "end")
                batch_size_entry.insert(0, "100")
            
            # Recalculate total pages
            num_instances = len(popup.face_instances)
            total_pages = (num_instances + new_batch_size - 1) // new_batch_size if num_instances > 0 else 1
            
            # Check if batch size changed
            batch_size_changed = (new_batch_size != popup.batch_size)
            
            # Update popup state
            popup.batch_size = new_batch_size
            popup.total_pages = total_pages
            
            # If batch size changed, reset to page 0; otherwise ensure current page is valid
            if batch_size_changed:
                popup.current_page = 0
            elif popup.current_page >= total_pages:
                popup.current_page = max(0, total_pages - 1)
            
            # Update page numbers
            self._update_gallery_page_numbers(popup, page_numbers_frame, gallery_frame, loading_label, load_btn)
            
            # Load images for current page
            thread = threading.Thread(
                target=self._load_gallery_images,
                args=(popup.face_instances, gallery_frame, loading_label, 
                      popup.current_page, popup.batch_size, popup.total_pages, load_btn),
                daemon=True
            )
            thread.start()
            
        except ValueError:
            # Invalid batch size, reset to default
            batch_size_entry.delete(0, "end")
            batch_size_entry.insert(0, str(popup.batch_size))
            messagebox.showerror("Invalid Input", "Please enter a valid number for instances per page.")
    
    def _gallery_go_to_page(self, popup, page_num, gallery_frame, loading_label, batch_size_entry, load_btn, page_numbers_frame):
        """Go to a specific page number."""
        if 0 <= page_num < popup.total_pages:
            popup.current_page = page_num
            self._update_gallery_page_numbers(popup, page_numbers_frame, gallery_frame, loading_label, load_btn)
            self._gallery_load_page(popup, gallery_frame, loading_label, batch_size_entry, load_btn, page_numbers_frame)
    
    def _update_gallery_page_numbers(self, popup, page_numbers_frame, gallery_frame, loading_label, load_btn):
        """Update the clickable page number labels."""
        # Clear existing page elements
        for widget in page_numbers_frame.winfo_children():
            widget.destroy()
        
        total_pages = popup.total_pages
        current_page = popup.current_page
        
        if total_pages == 0:
            return
        
        # First page button (<<)
        first_label = ctk.CTkLabel(
            page_numbers_frame,
            text="<<",
            font=ctk.CTkFont(size=12),
            cursor="hand2"
        )
        first_label.pack(side="left", padx=3)
        if current_page > 0:
            first_label.bind("<Button-1>", lambda e: self._gallery_go_to_page(
                popup, 0, gallery_frame, loading_label, 
                popup.batch_size_entry, popup.load_btn, page_numbers_frame
            ))
            first_label.configure(text_color="#3b8ed0")
        else:
            first_label.configure(text_color="gray")
        
        # Previous page label (<)
        prev_label = ctk.CTkLabel(
            page_numbers_frame,
            text="<",
            font=ctk.CTkFont(size=12),
            cursor="hand2"
        )
        prev_label.pack(side="left", padx=3)
        if current_page > 0:
            prev_label.bind("<Button-1>", lambda e: self._gallery_go_to_page(
                popup, current_page - 1, gallery_frame, loading_label, 
                popup.batch_size_entry, popup.load_btn, page_numbers_frame
            ))
            prev_label.configure(text_color="#3b8ed0")
        else:
            prev_label.configure(text_color="gray")
        
        # Show page numbers (limit to reasonable number to avoid UI overflow)
        max_visible_pages = 15
        if total_pages <= max_visible_pages:
            # Show all pages
            pages_to_show = list(range(total_pages))
        else:
            # Show first few, current area, and last few
            pages_to_show = []
            # Always show first page
            pages_to_show.append(0)
            
            # Show pages around current
            start = max(1, current_page - 2)
            end = min(total_pages - 1, current_page + 2)
            for p in range(start, end + 1):
                if p not in pages_to_show:
                    pages_to_show.append(p)
            
            # Always show last page
            if total_pages - 1 not in pages_to_show:
                pages_to_show.append(total_pages - 1)
            
            # Add ellipsis if needed
            if pages_to_show[1] > 1:
                pages_to_show.insert(1, None)  # None indicates ellipsis
            if pages_to_show[-2] < total_pages - 2:
                pages_to_show.insert(-1, None)
        
        # Create page number labels
        for page_num in pages_to_show:
            if page_num is None:
                # Ellipsis
                ctk.CTkLabel(
                    page_numbers_frame,
                    text="...",
                    font=ctk.CTkFont(size=12)
                ).pack(side="left", padx=2)
            else:
                # Page number label (clickable)
                is_current = (page_num == current_page)
                page_label = ctk.CTkLabel(
                    page_numbers_frame,
                    text=str(page_num + 1),
                    font=ctk.CTkFont(size=12, weight="bold" if is_current else "normal"),
                    cursor="hand2"
                )
                page_label.pack(side="left", padx=3)
                
                if is_current:
                    page_label.configure(text_color="#3b8ed0")
                else:
                    page_label.configure(text_color="#3b8ed0")
                    page_label.bind("<Button-1>", lambda e, p=page_num: self._gallery_go_to_page(
                        popup, p, gallery_frame, loading_label, 
                        popup.batch_size_entry, popup.load_btn, page_numbers_frame
                    ))
        
        # Next page label (>)
        next_label = ctk.CTkLabel(
            page_numbers_frame,
            text=">",
            font=ctk.CTkFont(size=12),
            cursor="hand2"
        )
        next_label.pack(side="left", padx=3)
        if current_page < total_pages - 1:
            next_label.bind("<Button-1>", lambda e: self._gallery_go_to_page(
                popup, current_page + 1, gallery_frame, loading_label, 
                popup.batch_size_entry, popup.load_btn, page_numbers_frame
            ))
            next_label.configure(text_color="#3b8ed0")
        else:
            next_label.configure(text_color="gray")
        
        # Last page label (>>)
        last_label = ctk.CTkLabel(
            page_numbers_frame,
            text=">>",
            font=ctk.CTkFont(size=12),
            cursor="hand2"
        )
        last_label.pack(side="left", padx=3)
        if current_page < total_pages - 1:
            last_label.bind("<Button-1>", lambda e: self._gallery_go_to_page(
                popup, total_pages - 1, gallery_frame, loading_label, 
                popup.batch_size_entry, popup.load_btn, page_numbers_frame
            ))
            last_label.configure(text_color="#3b8ed0")
        else:
            last_label.configure(text_color="gray")
    
    def _gallery_toggle_selection(self, popup, df_idx, checkbox_var):
        """Toggle selection of an instance in gallery."""
        if checkbox_var.get():
            popup.selected_instances.add(df_idx)
        else:
            popup.selected_instances.discard(df_idx)
        
        # Update selection label
        count = len(popup.selected_instances)
        popup.selection_label.configure(text=f"{count} instance{'s' if count != 1 else ''} selected")
    
    def _gallery_clear_selection(self, popup, gallery_frame, loading_label):
        """Clear all selections and reload current page."""
        popup.selected_instances.clear()
        popup.selection_label.configure(text="0 instances selected")
        
        # Reload current page to uncheck all checkboxes
        self._gallery_load_page(popup, gallery_frame, loading_label, 
                              popup.batch_size_entry, popup.load_btn, popup.page_numbers_frame)
    
    def _gallery_create_new_id(self, popup):
        """Create a new face ID from selected instances."""
        if not popup.selected_instances:
            messagebox.showwarning("No Selection", "Please select at least one instance.")
            return
        
        # Confirm action
        count = len(popup.selected_instances)
        response = messagebox.askyesno(
            "Create New ID",
            f"Create a new face ID from {count} selected instance{'s' if count != 1 else ''}?\n\n"
            f"These instances will be removed from their current ID(s)."
        )
        
        if not response:
            return
        
        # Generate new face ID
        existing_ids = list(self.face_groups.keys())
        max_id_num = 0
        for fid in existing_ids:
            if fid.startswith("FACE_"):
                try:
                    num = int(fid.split("_")[1])
                    max_id_num = max(max_id_num, num)
                except:
                    pass
        new_face_id = f"FACE_{max_id_num + 1:05d}"
        
        # Update dataframe: set new face_id for selected instances
        for df_idx in popup.selected_instances:
            self.df.at[df_idx, 'face_id'] = new_face_id
            if self.df_full is not None:
                self.df_full.at[df_idx, 'face_id'] = new_face_id
        
        # Find representative for new ID
        selected_rows = self.df.loc[list(popup.selected_instances)]
        representative = self._find_representative_face(selected_rows)
        thumbnail = self._extract_face_crop(representative) if representative else None
        
        # Create new face group
        self.face_groups[new_face_id] = {
            'count': count,
            'representative': representative,
            'thumbnail': thumbnail,
            'original_ids': [new_face_id]
        }
        
        # Update existing face groups (decrease counts)
        for df_idx in popup.selected_instances:
            old_face_id = popup.face_instances.loc[df_idx, 'face_id']
            if old_face_id in self.face_groups:
                self.face_groups[old_face_id]['count'] -= 1
                if self.face_groups[old_face_id]['count'] <= 0:
                    del self.face_groups[old_face_id]
        
        # Re-sort face groups
        self.face_groups = dict(
            sorted(self.face_groups.items(), key=lambda x: x[1]['count'], reverse=True)
        )
        
        # Refresh main face list
        self.after(0, self._display_face_list)
        
        # Clear selection and update gallery
        popup.selected_instances.clear()
        popup.selection_label.configure(text="0 instances selected")
        
        # Show success message (popup stays open)
        messagebox.showinfo(
            "Success",
            f"Created new face ID: {new_face_id}\n"
            f"Assigned {count} instance{'s' if count != 1 else ''} to it."
        )
    
    def _gallery_assign_to_existing(self, popup):
        """Assign selected instances to an existing face ID."""
        if not popup.selected_instances:
            messagebox.showwarning("No Selection", "Please select at least one instance.")
            return
        
        # Create selection dialog
        self._show_id_selection_dialog(popup)
    
    def _show_id_selection_dialog(self, gallery_popup):
        """Show dialog to select an existing face ID."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Select Face ID")
        dialog.geometry("600x700")
        dialog.grab_set()  # Modal
        
        # Title
        count = len(gallery_popup.selected_instances)
        ctk.CTkLabel(
            dialog,
            text=f"Assign {count} instance{'s' if count != 1 else ''} to which Face ID?",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Scrollable list of face IDs
        list_frame = ctk.CTkScrollableFrame(dialog, width=550, height=550)
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create clickable rows for each face ID
        for face_id, info in self.face_groups.items():
            # Skip the current face ID
            if face_id == gallery_popup.face_id:
                continue
            
            row_frame = ctk.CTkFrame(list_frame)
            row_frame.pack(fill="x", padx=5, pady=3)
            
            # Make row clickable
            def select_id(fid=face_id):
                dialog.destroy()
                self._gallery_assign_to_id(gallery_popup, fid)
            
            row_frame.bind("<Button-1>", lambda e, fid=face_id: select_id(fid))
            row_frame.configure(cursor="hand2")
            
            # Thumbnail
            thumb = info.get('thumbnail')
            if thumb:
                thumb_tk = ImageTk.PhotoImage(thumb)
                thumb_label = ctk.CTkLabel(row_frame, image=thumb_tk, text="", cursor="hand2")
                thumb_label.image = thumb_tk
                thumb_label.pack(side="left", padx=5)
                thumb_label.bind("<Button-1>", lambda e, fid=face_id: select_id(fid))
            
            # Face ID and count
            id_label = ctk.CTkLabel(
                row_frame,
                text=f"{face_id} ({info['count']} instances)",
                font=ctk.CTkFont(size=12),
                anchor="w",
                cursor="hand2"
            )
            id_label.pack(side="left", fill="x", expand=True, padx=5)
            id_label.bind("<Button-1>", lambda e, fid=face_id: select_id(fid))
        
        # Cancel button
        ctk.CTkButton(
            dialog,
            text="Cancel",
            command=dialog.destroy,
            width=150,
            height=35,
            font=ctk.CTkFont(size=12)
        ).pack(pady=10)
    
    def _gallery_assign_to_id(self, gallery_popup, target_face_id):
        """Assign selected instances to a specific face ID."""
        count = len(gallery_popup.selected_instances)
        
        # Confirm action
        response = messagebox.askyesno(
            "Confirm Assignment",
            f"Assign {count} selected instance{'s' if count != 1 else ''} to {target_face_id}?"
        )
        
        if not response:
            return
        
        # Update dataframe: set target face_id for selected instances
        for df_idx in gallery_popup.selected_instances:
            old_face_id = self.df.at[df_idx, 'face_id']
            self.df.at[df_idx, 'face_id'] = target_face_id
            if self.df_full is not None:
                self.df_full.at[df_idx, 'face_id'] = target_face_id
            
            # Update face group counts
            if old_face_id in self.face_groups:
                self.face_groups[old_face_id]['count'] -= 1
                if self.face_groups[old_face_id]['count'] <= 0:
                    del self.face_groups[old_face_id]
        
        # Increase count for target face ID
        if target_face_id in self.face_groups:
            self.face_groups[target_face_id]['count'] += count
        
        # Re-sort face groups
        self.face_groups = dict(
            sorted(self.face_groups.items(), key=lambda x: x[1]['count'], reverse=True)
        )
        
        # Refresh main face list
        self.after(0, self._display_face_list)
        
        # Clear selection and update gallery
        gallery_popup.selected_instances.clear()
        gallery_popup.selection_label.configure(text="0 instances selected")
        
        # Show success message (popup stays open)
        messagebox.showinfo(
            "Success",
            f"Assigned {count} instance{'s' if count != 1 else ''} to {target_face_id}"
        )
    
    def _save_results(self):
        """Save merged results to reviewer overlay annotation file."""
        if self.df_full is None:
            return

        from datetime import datetime

        response = messagebox.askyesno(
            "Confirm Save",
            f"Save face ID merge annotations for reviewer '{self.reviewer_id}'?\n\n"
            f"The base data files will NOT be modified.\n"
            f"Continue?"
        )
        if not response:
            return

        try:
            registry = ReviewerRegistry(self.project_dir)
            annotation_file = registry.get_merges_path(
                self.reviewer_id, self.selected_participant
            )
            annotation_file.parent.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().isoformat()
            annotation_records = [
                {
                    'face_id': face_id,
                    'merged_face_id': merged_id,
                    'is_media': merged_id in self.merged_id_to_media,
                    'reviewed_at': timestamp,
                }
                for face_id, merged_id in self.face_id_to_merged.items()
            ]

            ann_df = pd.DataFrame(annotation_records)
            ann_df.to_csv(annotation_file, index=False)

            unique_original_ids = len(set(self.face_id_to_merged.keys()))
            unique_merged_ids = len(set(self.face_id_to_merged.values()))

            messagebox.showinfo(
                "Saved",
                f"Merges and media flags saved.\n\n"
                f"File: {annotation_file}\n"
                f"Original unique IDs: {unique_original_ids}\n"
                f"Merged unique IDs: {unique_merged_ids}\n"
                f"Reduction: {unique_original_ids - unique_merged_ids} IDs"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to save annotations:\n{str(e)}")
    
    def _load_settings(self):
        """No per-tab settings to load; min_instances is in Tab 4 settings."""


