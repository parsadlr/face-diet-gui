"""
Tab 1: Video Processing (Stages 1 & 2) — select sessions and run face detection + attribute extraction.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import customtkinter as ctk
import threading
from tkinter import filedialog, messagebox
import tkinter

from face_diet_gui.core.settings_manager import SettingsManager
from face_diet_gui.gui.widgets.directory_tree_widget import DirectoryTreeWidget
from face_diet_gui.core.pipeline_helpers import (
    ProcessingStopped,
    _discard_annotations_for_session,
    _format_time,
    _run_stage1_via_subprocess,
    _run_stage2_via_subprocess,
)
from face_diet_gui.gui.common import BTN_DISABLED_FG, ProgressReporter


class VideoProcessingTab(ctk.CTkFrame):
    """Tab 1: Video Processing (Stages 1 & 2)."""

    def __init__(self, master, settings_manager: SettingsManager,
                 project_dir: Path, reviewer_id: str):
        super().__init__(master)
        self.settings = settings_manager
        self.project_dir: Path = project_dir
        self.reviewer_id: str = reviewer_id
        self.processing_thread: Optional[threading.Thread] = None
        self.is_processing = False
        self._current_process_holder: List = [None]
        self._stop_requested = False

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Setup UI components."""
        # Title
        ctk.CTkLabel(
            self,
            text="Video Processing: Face Detection & Attributes",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(10, 15))

        # Main content area: three equal columns — Participants & Sessions | Settings | Progress
        content_frame = ctk.CTkFrame(self)
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Column 1: Participants & sessions (directory tree)
        col_tree = ctk.CTkFrame(content_frame)
        col_tree.pack(side="left", fill="both", expand=True, padx=(0, 6))
        ctk.CTkLabel(
            col_tree,
            text="Participants & Sessions",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        self.tree_widget = DirectoryTreeWidget(col_tree)
        self.tree_widget.pack(fill="both", expand=True, pady=5)

        # Column 2: Settings (same expand so it doesn't look squeezed)
        col_settings = ctk.CTkFrame(content_frame)
        col_settings.pack(side="left", fill="both", expand=True, padx=6)
        self._create_settings_panel(col_settings)

        # Column 3: Progress
        col_progress = ctk.CTkFrame(content_frame)
        col_progress.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self._create_progress_panel(col_progress)

        # Process buttons at bottom
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=20)
        self.process_btn = ctk.CTkButton(
            btn_frame,
            text="Start Processing",
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
        settings_frame = ctk.CTkFrame(parent)
        settings_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            settings_frame,
            text="Processing Settings",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 15))

        # Stage 1 settings
        stage1_frame = ctk.CTkFrame(settings_frame)
        stage1_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            stage1_frame,
            text="Stage 1: Face Detection",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=5, pady=5)

        # Sampling rate
        self.use_original_fps_var = ctk.BooleanVar(value=True)
        self.original_fps_checkbox = ctk.CTkCheckBox(
            stage1_frame,
            text="Use original frame rate",
            variable=self.use_original_fps_var,
            command=self._on_original_fps_toggle,
            checkbox_width=18,
            checkbox_height=18
        )
        self.original_fps_checkbox.pack(anchor="w", padx=5, pady=2)

        sr_frame = ctk.CTkFrame(stage1_frame)
        sr_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(sr_frame, text="Sampling Rate:", width=120, anchor="w").pack(side="left")
        self.sampling_rate_var = ctk.IntVar(value=30)
        self.sampling_rate_entry = ctk.CTkEntry(sr_frame, textvariable=self.sampling_rate_var, width=80, state="disabled")
        self.sampling_rate_entry.pack(side="left", padx=5)
        ctk.CTkLabel(sr_frame, text="frames", text_color="gray").pack(side="left")

        # Min confidence threshold
        conf_frame = ctk.CTkFrame(stage1_frame)
        conf_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(conf_frame, text="Min Confidence:", width=120, anchor="w").pack(side="left")
        self.min_confidence_stage1_var = ctk.DoubleVar(value=0.0)
        ctk.CTkEntry(conf_frame, textvariable=self.min_confidence_stage1_var, width=80).pack(side="left", padx=5)
        ctk.CTkLabel(conf_frame, text="(0.0-1.0)", text_color="gray", font=ctk.CTkFont(size=12)).pack(side="left")

        # GPU checkbox
        self.use_gpu_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            stage1_frame,
            text="Use GPU (if available)",
            variable=self.use_gpu_var,
            checkbox_width=18,
            checkbox_height=18
        ).pack(anchor="w", padx=5, pady=2)

        # Debug mode: Process only 5% of video (TEMPORARY - for debugging)
        self.debug_mode_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            stage1_frame,
            text="DEBUG: Process only 5% of video",
            variable=self.debug_mode_var,
            checkbox_width=18,
            checkbox_height=18,
            text_color="orange"
        ).pack(anchor="w", padx=5, pady=5)

        # Stage 2 settings
        stage2_frame = ctk.CTkFrame(settings_frame)
        stage2_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            stage2_frame,
            text="Stage 2: Attribute Extraction",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=5, pady=5)

        # Batch size
        bs_frame = ctk.CTkFrame(stage2_frame)
        bs_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(bs_frame, text="Batch Size:", width=120, anchor="w").pack(side="left")
        self.batch_size_var = ctk.IntVar(value=32)
        ctk.CTkEntry(bs_frame, textvariable=self.batch_size_var, width=80).pack(side="left", padx=5)

    def _on_original_fps_toggle(self):
        """Handle original FPS checkbox toggle."""
        if self.use_original_fps_var.get():
            self.sampling_rate_entry.configure(state="disabled")
        else:
            self.sampling_rate_entry.configure(state="normal")

    def _toggle_detailed_log(self):
        """Toggle detailed log visibility."""
        if self.show_log_var.get():
            self.log_textbox.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        else:
            self.log_textbox.pack_forget()

    def _create_progress_panel(self, parent):
        """Create progress panel."""
        progress_frame = ctk.CTkFrame(parent)
        progress_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.current_step_label = ctk.CTkLabel(
            progress_frame,
            text="Ready to process",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.current_step_label.pack(pady=(10, 8))

        progress_container = ctk.CTkFrame(progress_frame, fg_color="transparent")
        progress_container.pack(fill="x", padx=20, pady=5)

        self.progress_bar = ctk.CTkProgressBar(progress_container, width=350, height=12)
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

        self.time_estimate_label = ctk.CTkLabel(
            progress_frame,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.time_estimate_label.pack(pady=(2, 5))

        ctk.CTkLabel(
            progress_frame,
            text="Processing Steps",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=10, pady=(5, 5))

        self.steps_frame = ctk.CTkScrollableFrame(progress_frame, height=180)
        self.steps_frame.pack(fill="both", expand=True, padx=10, pady=(0, 2))

        self.show_log_var = ctk.BooleanVar(value=False)
        self.log_toggle_btn = ctk.CTkCheckBox(
            progress_frame,
            text="Show detailed log",
            variable=self.show_log_var,
            command=self._toggle_detailed_log,
            font=ctk.CTkFont(size=10)
        )
        self.log_toggle_btn.pack(anchor="w", padx=10, pady=(2, 0))

        self.log_textbox = ctk.CTkTextbox(progress_frame, height=100)

        self.status_label = self.current_step_label

    def _browse_directory(self):
        """Browse for project directory."""
        folder = filedialog.askdirectory(title="Select Project Root Directory")
        if not folder:
            return

        self.project_dir = Path(folder)
        self.dir_entry.configure(state="normal")
        self.dir_entry.delete(0, "end")
        self.dir_entry.insert(0, str(self.project_dir))
        self.dir_entry.configure(state="readonly")

        self.tree_widget.build_tree(str(self.project_dir))

        self.settings.set("last_project_dir", str(self.project_dir))
        self.settings.save_settings()

    def _load_settings(self):
        """Load settings into UI."""
        if self.project_dir and self.project_dir.exists():
            self.tree_widget.build_tree(str(self.project_dir))

        self.use_original_fps_var.set(self.settings.get("stage1.use_original_fps", True))
        self.sampling_rate_var.set(self.settings.get("stage1.sampling_rate", 30))
        self.min_confidence_stage1_var.set(self.settings.get("stage1.min_confidence", 0.0))
        self.use_gpu_var.set(self.settings.get("stage1.use_gpu", False))
        self.batch_size_var.set(self.settings.get("stage2.batch_size", 32))

        self._on_original_fps_toggle()

    def set_project_dir(self, project_dir: Path):
        """Called by app when project directory changes."""
        self.project_dir = project_dir
        self.tree_widget.build_tree(str(project_dir))

    def update_project_and_reviewer(self, project_dir: Path, reviewer_id: str):
        """Called when user changes project or reviewer via Back to setup."""
        self.project_dir = project_dir
        self.reviewer_id = reviewer_id
        self.tree_widget.build_tree(str(project_dir))

    def _get_min_confidence(self):
        """Get min confidence value with validation."""
        try:
            val = self.min_confidence_stage1_var.get()
            if val is None or val == "":
                return 0.0
            return float(val)
        except (ValueError, tkinter.TclError):
            return 0.0

    def _get_batch_size(self):
        """Get batch size value with validation."""
        try:
            val = self.batch_size_var.get()
            if val is None or val == "":
                return 32
            return int(val)
        except (ValueError, tkinter.TclError):
            return 32

    def _save_settings(self):
        """Save current settings."""
        self.settings.set("stage1.use_original_fps", self.use_original_fps_var.get())
        self.settings.set("stage1.sampling_rate", self.sampling_rate_var.get())
        self.settings.set("stage1.min_confidence", self._get_min_confidence())
        self.settings.set("stage1.use_gpu", self.use_gpu_var.get())
        self.settings.set("stage2.batch_size", self._get_batch_size())
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

        selected_sessions = self.tree_widget.get_selected_sessions()

        if not selected_sessions:
            messagebox.showerror("Error", "Please select at least one session to process!")
            return

        sessions_already_done = [
            (p, s, path) for (p, s, path) in selected_sessions
            if (path / "face_detections.csv").exists()
        ]
        if sessions_already_done:
            n = len(sessions_already_done)
            session_list = "\n".join(f"  • {p} / {s}" for (p, s, _) in sessions_already_done[:10])
            if n > 10:
                session_list += f"\n  ... and {n - 10} more"
            ok = messagebox.askyesno(
                "Overwrite face detection — annotations will be lost",
                "The following session(s) already have face detection results:\n\n"
                + session_list
                + "\n\nRe-running will:\n"
                "  • Overwrite face_detections.csv for these sessions\n"
                "  • Permanently delete is_face and merges annotations that depend on these sessions\n\n"
                "This cannot be undone. Are you sure you want to continue?",
                icon=messagebox.WARNING,
                default="no"
            )
            if not ok:
                return

        self._save_settings()

        self.log_textbox.delete("1.0", "end")

        self._stop_requested = False
        self._current_process_holder[0] = None

        self.process_btn.configure(
            state="disabled",
            text="Processing...",
            fg_color=BTN_DISABLED_FG
        )
        self.stop_btn.configure(state="normal", fg_color="#dc3545")
        self.is_processing = True

        self.processing_thread = threading.Thread(
            target=self._processing_worker,
            args=(selected_sessions,),
            daemon=True
        )
        self.processing_thread.start()

    def _processing_worker(self, selected_sessions: List[Tuple[str, str, Path]]):
        """Worker thread for processing."""
        import time

        try:
            reporter = ProgressReporter(self)
            reporter.start_time = time.time()

            total_sessions = len(selected_sessions)

            self.after(0, lambda: [w.destroy() for w in self.steps_frame.winfo_children()])

            for idx, (participant_name, session_name, _) in enumerate(selected_sessions, 1):
                step_id_s1 = f"session_{idx}_stage1"
                step_id_s2 = f"session_{idx}_stage2"
                self.after(0, lambda sid=step_id_s1, pn=participant_name, sn=session_name:
                    reporter.add_step(sid, f"{pn}/{sn} - Stage 1: Face Detection", "pending"))
                self.after(0, lambda sid=step_id_s2, pn=participant_name, sn=session_name:
                    reporter.add_step(sid, f"{pn}/{sn} - Stage 2: Attribute Extraction", "pending"))

            time.sleep(0.2)

            for idx, (participant_name, session_name, session_path) in enumerate(selected_sessions, 1):
                # Determine sampling rate
                if self.use_original_fps_var.get():
                    sampling_rate = 1
                else:
                    try:
                        sampling_rate = self.sampling_rate_var.get()
                        if not sampling_rate or sampling_rate <= 0:
                            sampling_rate = 30
                    except (ValueError, tkinter.TclError):
                        sampling_rate = 30

                step_id_s1 = f"session_{idx}_stage1"
                self.after(0, lambda p=participant_name, s=session_name:
                    reporter.set_current_step("Face Detection", p, s))
                self.after(0, lambda sid=step_id_s1: reporter.update_step_status(sid, "in_progress"))
                self.after(0, lambda: reporter.update_progress(0, "0%"))
                self.after(0, lambda: reporter.update_time_estimate("0s", None))
                self.after(0, lambda: reporter.log(f"Running Stage 1 for {session_name}..."))

                try:
                    _run_stage1_via_subprocess(
                        session_dir=str(session_path),
                        sampling_rate=sampling_rate,
                        use_gpu=self.use_gpu_var.get(),
                        min_confidence=self._get_min_confidence(),
                        reporter=reporter,
                        debug_mode=self.debug_mode_var.get(),
                        settings=self.settings,
                        process_holder=self._current_process_holder,
                        stop_check=lambda: self._stop_requested,
                    )
                    self.after(0, lambda sid=step_id_s1: reporter.update_step_status(sid, "completed"))
                    _discard_annotations_for_session(
                        self.project_dir, participant_name, session_name
                    )
                    self.after(0, lambda p=participant_name, s=session_name: reporter.log(
                        f"Discarded previous annotations for {p}/{s} (is_face, merges)."
                    ))
                except ProcessingStopped:
                    self.after(0, lambda sid=step_id_s1: reporter.update_step_status(sid, "error"))
                    raise
                except Exception:
                    self.after(0, lambda sid=step_id_s1: reporter.update_step_status(sid, "error"))
                    raise

                step_id_s2 = f"session_{idx}_stage2"
                self.after(0, lambda p=participant_name, s=session_name:
                    reporter.set_current_step("Attribute Extraction", p, s))
                self.after(0, lambda sid=step_id_s2: reporter.update_step_status(sid, "in_progress"))
                self.after(0, lambda: reporter.update_progress(0, "0%"))
                self.after(0, lambda: reporter.update_time_estimate("0s", None))
                self.after(0, lambda: reporter.log(f"Running Stage 2 for {session_name}..."))

                try:
                    _run_stage2_via_subprocess(
                        session_dir=str(session_path),
                        batch_size=self._get_batch_size(),
                        reporter=reporter,
                        debug_mode=self.debug_mode_var.get(),
                        settings=self.settings,
                        process_holder=self._current_process_holder,
                        stop_check=lambda: self._stop_requested,
                    )
                    self.after(0, lambda sid=step_id_s2: reporter.update_step_status(sid, "completed"))
                except ProcessingStopped:
                    self.after(0, lambda sid=step_id_s2: reporter.update_step_status(sid, "error"))
                    raise
                except Exception:
                    self.after(0, lambda sid=step_id_s2: reporter.update_step_status(sid, "error"))
                    raise

            self.after(0, lambda: reporter.update_status(f"[OK] All Complete!"))
            self.after(0, lambda: reporter.update_status(f"[OK] Processed {total_sessions} session(s)"))
            self.after(0, lambda: reporter.update_progress(1.0, "100%"))
            elapsed_str = _format_time(time.time() - reporter.start_time)
            self.after(0, lambda e=elapsed_str: reporter.update_time_estimate(e, None))
            self.after(0, lambda: messagebox.showinfo(
                "Success",
                f"Successfully processed {total_sessions} session(s)!"
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
                text="Start Processing",
                fg_color="#28a745"
            ))
            self.after(0, lambda: self.stop_btn.configure(state="disabled", fg_color=BTN_DISABLED_FG))
            self.is_processing = False
