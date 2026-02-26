"""Tab 4: Face ID Clustering (Stage 3) — select participants for clustering."""

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


class FaceIDAssignmentTab(ctk.CTkFrame):
    """Tab 4: Face ID Assignment (Stage 3) — select participants for clustering."""

    def __init__(self, master, settings_manager: SettingsManager,
                 project_dir: Path, reviewer_id: str):
        super().__init__(master)
        self.settings = settings_manager
        self.project_dir: Path = project_dir
        self.reviewer_id: str = reviewer_id
        self.participant_widgets: Dict = {}
        self.processing_thread: Optional[threading.Thread] = None
        self.is_processing = False
        self._current_process_holder: List = [None]
        self._stop_requested = False

        self._setup_ui()
        self._load_settings()
        # Participant/session list is loaded when tab is first shown (via <Map>), not at startup

    def _setup_ui(self):
        """Setup UI components."""
        # Title
        ctk.CTkLabel(
            self,
            text="Face ID Clustering",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(10, 15))

        # Main content area: three equal columns — Participants & Sessions | Settings | Progress
        # Use uniform so all three columns get the same width regardless of content
        content_frame = ctk.CTkFrame(self)
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)
        content_frame.grid_columnconfigure(0, weight=1, uniform="tab3cols")
        content_frame.grid_columnconfigure(1, weight=1, uniform="tab3cols")
        content_frame.grid_columnconfigure(2, weight=1, uniform="tab3cols")
        content_frame.grid_rowconfigure(0, weight=1)
        
        # Column 1: Participants & Sessions
        col_list = ctk.CTkFrame(content_frame)
        col_list.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        col_list.grid_columnconfigure(0, weight=1)
        col_list.grid_rowconfigure(2, weight=1)
        
        # Column 2: Settings
        col_settings = ctk.CTkFrame(content_frame)
        col_settings.grid(row=0, column=1, sticky="nsew", padx=6)
        col_settings.grid_columnconfigure(0, weight=1)
        col_settings.grid_rowconfigure(0, weight=1)
        self._create_settings_panel(col_settings)
        
        # Column 3: Progress
        col_progress = ctk.CTkFrame(content_frame)
        col_progress.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        col_progress.grid_columnconfigure(0, weight=1)
        col_progress.grid_rowconfigure(0, weight=1)
        self._create_progress_panel(col_progress)
        
        ctk.CTkLabel(
            col_list,
            text="Participants & Sessions",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        ctk.CTkLabel(
            col_list,
            text="Check participants to include in clustering.",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(pady=(0, 5))
        
        select_frame = ctk.CTkFrame(col_list, fg_color="transparent")
        select_frame.pack(fill="x", padx=5, pady=2)
        
        ctk.CTkButton(
            select_frame,
            text="Select All",
            command=self._select_all_participants,
            width=90,
            height=26
        ).pack(side="left", padx=3)
        
        ctk.CTkButton(
            select_frame,
            text="Deselect All",
            command=self._deselect_all_participants,
            width=90,
            height=26
        ).pack(side="left", padx=3)
        
        self.participant_list_frame = ctk.CTkScrollableFrame(col_list, height=400)
        self.participant_list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Process buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=20)
        self.process_btn = ctk.CTkButton(
            btn_frame,
            text="Assign Face IDs",
            command=self._start_processing,
            width=180,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#28a745",
            hover_color="#218838",
            text_color="white",
            text_color_disabled="white"
        )
        self.process_btn.pack(side="left", padx=(0, 12))
        self.stop_btn = ctk.CTkButton(
            btn_frame,
            text="Stop",
            command=self._stop_processing,
            width=180,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#dc3545",
            hover_color="#c82333",
            text_color="white",
            text_color_disabled="white",
            state="disabled"
        )
        self.stop_btn.configure(fg_color=BTN_DISABLED_FG)  # initial disabled look
        self.stop_btn.pack(side="left")
    
    def _create_settings_panel(self, parent):
        """Create settings panel."""
        settings_frame = ctk.CTkScrollableFrame(parent)
        settings_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            settings_frame,
            text="Clustering Settings",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 15))
        
        # Algorithm
        algo_frame = ctk.CTkFrame(settings_frame)
        algo_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(algo_frame, text="Algorithm:", width=150, anchor="w").pack(side="left")
        self.algorithm_var = ctk.StringVar(value="leiden")
        ctk.CTkOptionMenu(
            algo_frame,
            values=["leiden", "louvain"],
            variable=self.algorithm_var,
            width=120
        ).pack(side="left", padx=5)
        
        # Similarity threshold
        sim_frame = ctk.CTkFrame(settings_frame)
        sim_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(sim_frame, text="Similarity Threshold:", width=150, anchor="w").pack(side="left")
        self.sim_threshold_var = ctk.DoubleVar(value=0.6)
        ctk.CTkEntry(sim_frame, textvariable=self.sim_threshold_var, width=80).pack(side="left", padx=5)
        
        # k-neighbors
        k_frame = ctk.CTkFrame(settings_frame)
        k_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(k_frame, text="k-Neighbors:", width=150, anchor="w").pack(side="left")
        self.k_neighbors_var = ctk.IntVar(value=50)
        ctk.CTkEntry(k_frame, textvariable=self.k_neighbors_var, width=80).pack(side="left", padx=5)

        # Min confidence
        conf_frame = ctk.CTkFrame(settings_frame)
        conf_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(conf_frame, text="Min Confidence:", width=150, anchor="w").pack(side="left")
        self.min_confidence_var = ctk.DoubleVar(value=0.0)
        ctk.CTkEntry(conf_frame, textvariable=self.min_confidence_var, width=80).pack(side="left", padx=5)

        # Min instances — filter out face IDs with fewer than N instances in Tab 5
        mi_frame = ctk.CTkFrame(settings_frame)
        mi_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(mi_frame, text="Min Instances:", width=150, anchor="w").pack(side="left")
        self.min_instances_var = ctk.IntVar(value=5)
        ctk.CTkEntry(mi_frame, textvariable=self.min_instances_var, width=80).pack(side="left", padx=5)

        # Enable refinement
        self.enable_refinement_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            settings_frame,
            text="Enable small cluster refinement",
            variable=self.enable_refinement_var,
            command=self._toggle_refinement_frame,
            checkbox_width=18,
            checkbox_height=18
        ).pack(anchor="w", padx=5, pady=10)

        # Refinement settings (in a sub-frame); visually inactive when checkbox is off
        self.refine_frame = ctk.CTkFrame(settings_frame)
        self.refine_frame.pack(fill="x", padx=15, pady=5)
        self._refine_labels = []  # track labels for dimming

        # k-voting
        kv_frame = ctk.CTkFrame(self.refine_frame)
        kv_frame.pack(fill="x", padx=5, pady=2)
        kv_lbl = ctk.CTkLabel(kv_frame, text="k-Voting:", width=130, anchor="w")
        kv_lbl.pack(side="left")
        self._refine_labels.append(kv_lbl)
        self.k_voting_var = ctk.IntVar(value=10)
        self.kv_entry = ctk.CTkEntry(kv_frame, textvariable=self.k_voting_var, width=70)
        self.kv_entry.pack(side="left", padx=5)

        # Min votes
        mv_frame = ctk.CTkFrame(self.refine_frame)
        mv_frame.pack(fill="x", padx=5, pady=2)
        mv_lbl = ctk.CTkLabel(mv_frame, text="Min Votes:", width=130, anchor="w")
        mv_lbl.pack(side="left")
        self._refine_labels.append(mv_lbl)
        self.min_votes_var = ctk.IntVar(value=5)
        self.mv_entry = ctk.CTkEntry(mv_frame, textvariable=self.min_votes_var, width=70)
        self.mv_entry.pack(side="left", padx=5)

    def _toggle_refinement_frame(self):
        """Enable/disable the refinement sub-frame with full inactive visual style."""
        enabled = self.enable_refinement_var.get()
        state = "normal" if enabled else "disabled"
        label_color = ("gray10", "gray90") if enabled else ("gray60", "gray50")
        frame_color = ("gray86", "gray17") if enabled else ("gray78", "gray26")
        for entry in (self.kv_entry, self.mv_entry):
            entry.configure(state=state)
        for lbl in self._refine_labels:
            lbl.configure(text_color=label_color)
        self.refine_frame.configure(fg_color=frame_color)
        for child in self.refine_frame.winfo_children():
            if isinstance(child, ctk.CTkFrame):
                child.configure(fg_color=frame_color)

    def _create_progress_panel(self, parent):
        """Create progress panel."""
        progress_frame = ctk.CTkFrame(parent)
        progress_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Current step info (centered, gray, above progress bar)
        self.current_step_label = ctk.CTkLabel(
            progress_frame,
            text="Ready to process",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.current_step_label.pack(pady=(10, 8))
        
        # Progress bar with percentage
        progress_container = ctk.CTkFrame(progress_frame, fg_color="transparent")
        progress_container.pack(fill="x", padx=20, pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(progress_container, width=450, height=12)
        self.progress_bar.pack(side="left", fill="x", expand=True)
        self.progress_bar.set(0)
        self.progress_bar.configure(progress_color="#3b8ed0")
        
        self.progress_percentage_label = ctk.CTkLabel(
            progress_container,
            text="0%",
            font=ctk.CTkFont(size=11),
            width=50
        )
        self.progress_percentage_label.pack(side="left", padx=(5, 0))
        
        # Time estimate
        self.time_estimate_label = ctk.CTkLabel(
            progress_frame,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.time_estimate_label.pack(pady=(2, 5))
        
        # Steps list
        ctk.CTkLabel(
            progress_frame,
            text="Processing Steps",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=10, pady=(5, 5))
        
        # Scrollable steps frame
        self.steps_frame = ctk.CTkScrollableFrame(progress_frame, height=120)
        self.steps_frame.pack(fill="both", expand=True, padx=10, pady=(0, 2))
        
        # Detailed log (collapsible)
        self.show_log_var = ctk.BooleanVar(value=False)
        self.log_toggle_btn = ctk.CTkCheckBox(
            progress_frame,
            text="Show detailed log",
            variable=self.show_log_var,
            command=self._toggle_detailed_log,
            font=ctk.CTkFont(size=10),
            checkbox_width=16,
            checkbox_height=16
        )
        self.log_toggle_btn.pack(anchor="w", padx=10, pady=(2, 0))
        
        self.log_textbox = ctk.CTkTextbox(progress_frame, height=100)
        # Initially hidden
        
        # Status label (compatibility)
        self.status_label = self.current_step_label
    
    def _toggle_detailed_log(self):
        """Toggle detailed log visibility."""
        if self.show_log_var.get():
            self.log_textbox.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        else:
            self.log_textbox.pack_forget()
    
    def set_project_dir(self, project_dir: Path):
        """Called by app when project directory changes."""
        self.project_dir = project_dir
        self._load_participants_and_sessions()

    def update_project_and_reviewer(self, project_dir: Path, reviewer_id: str):
        """Called when user changes project or reviewer via Back to setup."""
        self.project_dir = project_dir
        self.reviewer_id = reviewer_id
        self._load_participants_and_sessions()

    def _load_participants_and_sessions(self):
        """Load participant/session list in background so tab opens immediately; paint when ready."""
        for w in self.participant_list_frame.winfo_children():
            w.destroy()
        self.participant_widgets.clear()
        if not self.project_dir or not self.project_dir.exists():
            ctk.CTkLabel(
                self.participant_list_frame,
                text="No project directory",
                font=ctk.CTkFont(size=12),
                text_color="gray"
            ).pack(pady=20)
            return
        ctk.CTkLabel(
            self.participant_list_frame,
            text="Loading participants & sessions…",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(pady=20)
        project_dir = self.project_dir

        def _fetch():
            items = _get_sessions_with_review_status(project_dir)
            self.after(0, lambda: self._paint_participants_and_sessions(items))

        threading.Thread(target=_fetch, daemon=True).start()

    def _paint_participants_and_sessions(self, items: List[Dict]):
        """Paint participant/session list (must run on main thread). Called after background fetch."""
        for w in self.participant_list_frame.winfo_children():
            w.destroy()
        self.participant_widgets.clear()
        if not items:
            ctk.CTkLabel(
                self.participant_list_frame,
                text="No sessions with face_detections.csv found.",
                font=ctk.CTkFont(size=12),
                text_color="gray"
            ).pack(pady=20)
            return
        by_participant: Dict[str, List[Dict]] = defaultdict(list)
        for item in items:
            by_participant[item["participant"]].append(item)
        for participant_name in sorted(by_participant.keys()):
            session_items = by_participant[participant_name]
            participant_dir = self.project_dir / participant_name
            part_frame = ctk.CTkFrame(self.participant_list_frame)
            part_frame.pack(fill="x", padx=5, pady=(8, 2))
            var = ctk.BooleanVar(value=True)
            cb = ctk.CTkCheckBox(
                part_frame,
                text=f"{participant_name} ({len(session_items)} sessions)",
                variable=var,
                font=ctk.CTkFont(size=12, weight="bold"),
                checkbox_width=18,
                checkbox_height=18
            )
            cb.pack(side="left", padx=10, pady=5)
            self.participant_widgets[participant_name] = {
                "frame": part_frame,
                "checkbox_var": var,
                "path": participant_dir,
                "session_count": len(session_items),
            }
            for item in sorted(session_items, key=lambda x: x["session"]):
                session = item["session"]
                n = item.get("reviewers_with_tab2_count", 0)
                mismatch_count = item.get("mismatch_count", 0)
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
                row = ctk.CTkFrame(self.participant_list_frame, fg_color=("gray92", "gray18"))
                row.pack(fill="x", padx=5, pady=1)
                ctk.CTkLabel(row, text="   ", width=20).pack(side="left")
                session_lbl = ctk.CTkLabel(row, text=session, font=ctk.CTkFont(size=12), anchor="w")
                session_lbl.pack(side="left", padx=(0, 8), pady=3)
                if n == 0 or n == 1 or resolved:
                    status_lbl = ctk.CTkLabel(row, text=status_text, font=ctk.CTkFont(size=12), text_color=status_color)
                    status_lbl.pack(side="left", padx=(0, 8), pady=3)
                else:
                    ctk.CTkLabel(row, text=f"{n} reviewers (", font=ctk.CTkFont(size=12), text_color="#007bff").pack(side="left", padx=(0, 0), pady=3)
                    ctk.CTkLabel(row, text=f"{mismatch_count} mismatches)", font=ctk.CTkFont(size=12), text_color="#dc3545").pack(side="left", padx=(0, 8), pady=3)

    def _select_all_participants(self):
        """Select all participants."""
        for data in self.participant_widgets.values():
            data['checkbox_var'].set(True)
    
    def _deselect_all_participants(self):
        """Deselect all participants."""
        for data in self.participant_widgets.values():
            data['checkbox_var'].set(False)
    
    def _load_settings(self):
        """Load settings into UI."""
        self.algorithm_var.set(self.settings.get("stage3.algorithm", "leiden"))
        self.sim_threshold_var.set(self.settings.get("stage3.similarity_threshold", 0.6))
        self.k_neighbors_var.set(self.settings.get("stage3.k_neighbors", 50))
        self.min_confidence_var.set(self.settings.get("stage3.min_confidence", 0.0))
        self.min_instances_var.set(self.settings.get("stage3.min_instances", 5))
        self.enable_refinement_var.set(self.settings.get("stage3.enable_refinement", True))
        self.k_voting_var.set(self.settings.get("stage3.k_voting", 10))
        self.min_votes_var.set(self.settings.get("stage3.min_votes", 5))
        self._toggle_refinement_frame()

    def _save_settings(self):
        """Save current settings."""
        self.settings.set("stage3.algorithm", self.algorithm_var.get())
        self.settings.set("stage3.similarity_threshold", self.sim_threshold_var.get())
        self.settings.set("stage3.k_neighbors", self.k_neighbors_var.get())
        self.settings.set("stage3.min_confidence", self.min_confidence_var.get())
        self.settings.set("stage3.min_instances", self.min_instances_var.get())
        self.settings.set("stage3.enable_refinement", self.enable_refinement_var.get())
        self.settings.set("stage3.k_voting", self.k_voting_var.get())
        self.settings.set("stage3.min_votes", self.min_votes_var.get())
        self.settings.save_settings()
    
    def _stop_processing(self):
        """Request stop and terminate the current subprocess."""
        self._stop_requested = True
        proc = self._current_process_holder[0] if self._current_process_holder else None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
    
    def _start_processing(self):
        """Start processing in background thread."""
        if self.is_processing:
            messagebox.showwarning("Processing", "Processing is already running!")
            return
        
        if not self.project_dir:
            messagebox.showerror("Error", "Please select a project directory first!")
            return
        
        # Get selected participants
        selected_participants = [
            (name, data['path'])
            for name, data in self.participant_widgets.items()
            if data['checkbox_var'].get()
        ]
        
        if not selected_participants:
            messagebox.showerror("Error", "Please select at least one participant!")
            return
        
        # Warn if any selected participant already has face ID clustering (re-run will overwrite participant CSV)
        participants_already_done = [
            name for (name, path) in selected_participants
            if (path / "face_ids.csv").exists()
        ]
        if participants_already_done:
            n = len(participants_already_done)
            participant_list = "\n".join(f"  • {p}" for p in participants_already_done[:10])
            if n > 10:
                participant_list += f"\n  ... and {n - 10} more"
            ok = messagebox.askyesno(
                "Overwrite face ID clustering",
                "The following participant(s) already have face ID clustering results in their folder:\n\n"
                + participant_list
                + "\n\nRe-running will overwrite face_ids.csv in each participant folder. "
                "This cannot be undone.\n\n"
                "Are you sure you want to continue?",
                icon=messagebox.WARNING,
                default="no"
            )
            if not ok:
                return
        
        # Save settings
        self._save_settings()
        
        # Clear log
        self.log_textbox.delete("1.0", "end")
        
        self._stop_requested = False
        self._current_process_holder[0] = None
        
        # Disable Start, enable Stop
        self.process_btn.configure(
            state="disabled",
            text="Processing...",
            fg_color=BTN_DISABLED_FG
        )
        self.stop_btn.configure(state="normal", fg_color="#dc3545")
        self.is_processing = True

        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_worker,
            args=(selected_participants,),
            daemon=True
        )
        self.processing_thread.start()
    
    def _processing_worker(self, selected_participants: List[Tuple[str, Path]]):
        """Worker thread for processing."""
        import time
        
        try:
            reporter = ProgressReporter(self)
            reporter.start_time = time.time()
            
            total_participants = len(selected_participants)
            
            # Clear steps frame
            self.after(0, lambda: [w.destroy() for w in self.steps_frame.winfo_children()])
            
            # Create all steps upfront
            for idx, (participant_name, _) in enumerate(selected_participants, 1):
                step_id = f"participant_{idx}"
                self.after(0, lambda sid=step_id, pn=participant_name: 
                    reporter.add_step(sid, f"{pn} - Face ID Assignment", "pending"))
            
            time.sleep(0.2)  # Give UI time to render steps
            
            for idx, (participant_name, participant_path) in enumerate(selected_participants, 1):
                session_start = time.time()
                
                # Update current step
                step_id = f"participant_{idx}"
                self.after(0, lambda p=participant_name: 
                    reporter.set_current_step("Face ID Assignment", p, None))
                self.after(0, lambda sid=step_id: reporter.update_step_status(sid, "in_progress"))
                self.after(0, lambda: reporter.update_progress(0, "0%"))  # Reset progress bar
                self.after(0, lambda: reporter.update_time_estimate("0s", None))  # Reset time
                
                try:
                    registry = ReviewerRegistry(self.project_dir)
                    annotations_dir = str(
                        registry.get_reviewer_dir(self.reviewer_id) / participant_name
                    )
                    # Face ID clustering output goes in participant folder (shared): participant_path/face_ids.csv
                    output_dir = str(participant_path)
                    # Consensus dir: {project}/_annotations/consensus/{participant}/
                    consensus_dir = str(self.project_dir / "_annotations" / "consensus" / participant_name)

                    _run_stage3_via_subprocess(
                        participant_dir=str(participant_path),
                        annotations_dir=annotations_dir,
                        output_dir=output_dir,
                        similarity_threshold=self.sim_threshold_var.get(),
                        k_neighbors=self.k_neighbors_var.get(),
                        min_confidence=self.min_confidence_var.get(),
                        algorithm=self.algorithm_var.get(),
                        enable_refinement=self.enable_refinement_var.get(),
                        min_cluster_size=self.min_instances_var.get(),
                        k_voting=self.k_voting_var.get(),
                        min_votes=self.min_votes_var.get(),
                        reporter=reporter,
                        process_holder=self._current_process_holder,
                        stop_check=lambda: self._stop_requested,
                        consensus_dir=consensus_dir,
                    )
                    self.after(0, lambda sid=step_id: reporter.update_step_status(sid, "completed"))
                
                except ProcessingStopped:
                    self.after(0, lambda sid=step_id: reporter.update_step_status(sid, "error"))
                    raise
                except Exception as e:
                    self.after(0, lambda sid=step_id: reporter.update_step_status(sid, "error"))
                    raise
            
            # Done
            self.after(0, lambda: reporter.update_status(f"✓ All Complete!"))
            self.after(0, lambda: reporter.update_status(f"[OK] Processed {total_participants} participant(s)"))
            self.after(0, lambda: reporter.update_progress(1.0, "100%"))
            elapsed_str = _format_time(time.time() - reporter.start_time)
            self.after(0, lambda e=elapsed_str: reporter.update_time_estimate(e, None))
            self.after(0, lambda: messagebox.showinfo(
                "Success",
                f"Successfully assigned face IDs for {total_participants} participant(s)!"
            ))
        
        except ProcessingStopped:
            self.after(0, lambda: reporter.log("\n[STOPPED] Processing stopped by user."))
            self.after(0, lambda: reporter.update_status("[Stopped] Processing stopped by user."))
            self.after(0, lambda: messagebox.showinfo("Stopped", "Processing was stopped."))
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = str(e)
            self.after(0, lambda ed=error_details: reporter.log(f"\n[ERROR]\n{ed}"))
            self.after(0, lambda: reporter.update_status("[ERROR] Error during processing"))
            self.after(0, lambda em=error_msg: messagebox.showerror(
                "Error",
                f"Processing failed:\n{em}\n\nCheck detailed log for details."
            ))
        
        finally:
            self.after(0, lambda: self.process_btn.configure(
                state="normal",
                text="Assign Face IDs",
                fg_color="#28a745"
            ))
            self.after(0, lambda: self.stop_btn.configure(state="disabled", fg_color=BTN_DISABLED_FG))
            self.is_processing = False

