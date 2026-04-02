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
                 data_dir: Path, derivatives_dir: Path, reviewer_id: str):
        super().__init__(master)
        self.settings = settings_manager
        self.data_dir: Path = Path(data_dir)
        self.derivatives_dir: Path = Path(derivatives_dir)
        self.reviewer_id: str = reviewer_id
        self.processing_thread: Optional[threading.Thread] = None
        self.is_processing = False
        self._current_process_holder: List = [None]
        self._stop_requested = False

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Setup UI components."""
        ctk.CTkLabel(
            self,
            text="Video Processing: Face Detection & Attributes",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(10, 15))

        content_frame = ctk.CTkFrame(self)
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Column 1: Participants & sessions
        col_tree = ctk.CTkFrame(content_frame)
        col_tree.pack(side="left", fill="both", expand=True, padx=(0, 6))
        ctk.CTkLabel(
            col_tree,
            text="Participants & Sessions",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        self.tree_widget = DirectoryTreeWidget(col_tree)
        self.tree_widget.pack(fill="both", expand=True, pady=5)

        # Column 2: Settings
        col_settings = ctk.CTkFrame(content_frame)
        col_settings.pack(side="left", fill="both", expand=True, padx=6)
        self._create_settings_panel(col_settings)

        # Column 3: Progress
        col_progress = ctk.CTkFrame(content_frame)
        col_progress.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self._create_progress_panel(col_progress)

        # Buttons at bottom
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
        self.stop_btn.configure(fg_color=BTN_DISABLED_FG)
        self.stop_btn.pack(side="left")

    def _create_settings_panel(self, parent):
        """Create unified settings panel."""
        settings_frame = ctk.CTkFrame(parent)
        settings_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            settings_frame,
            text="Processing Settings",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 10))

        inner = ctk.CTkFrame(settings_frame)
        inner.pack(fill="x", padx=10, pady=5)

        # --- Exclude edges ---
        self.trim_edges_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            inner,
            text="Exclude edges",
            variable=self.trim_edges_var,
            command=self._on_trim_toggle,
            checkbox_width=18,
            checkbox_height=18
        ).pack(anchor="w", padx=5, pady=(8, 2))

        self._trim_detail_frame = ctk.CTkFrame(inner, fg_color="transparent")
        self._trim_detail_frame.pack(fill="x", padx=20, pady=(0, 4))

        trim_row1 = ctk.CTkFrame(self._trim_detail_frame, fg_color="transparent")
        trim_row1.pack(fill="x", pady=1)
        ctk.CTkLabel(trim_row1, text="Skip from start:", width=130, anchor="w").pack(side="left")
        self.trim_start_var = ctk.StringVar(value="")
        self.trim_start_entry = ctk.CTkEntry(
            trim_row1, textvariable=self.trim_start_var, width=70, state="disabled",
            placeholder_text="e.g. 30"
        )
        self.trim_start_entry.pack(side="left", padx=5)
        ctk.CTkLabel(trim_row1, text="s", text_color="gray").pack(side="left")

        trim_row2 = ctk.CTkFrame(self._trim_detail_frame, fg_color="transparent")
        trim_row2.pack(fill="x", pady=1)
        ctk.CTkLabel(trim_row2, text="Skip from end:", width=130, anchor="w").pack(side="left")
        self.trim_end_var = ctk.StringVar(value="")
        self.trim_end_entry = ctk.CTkEntry(
            trim_row2, textvariable=self.trim_end_var, width=70, state="disabled",
            placeholder_text="e.g. 30"
        )
        self.trim_end_entry.pack(side="left", padx=5)
        ctk.CTkLabel(trim_row2, text="s", text_color="gray").pack(side="left")

        # --- Downsampling ---
        self.downsampling_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            inner,
            text="Downsampling",
            variable=self.downsampling_var,
            command=self._on_downsampling_toggle,
            checkbox_width=18,
            checkbox_height=18
        ).pack(anchor="w", padx=5, pady=(6, 2))

        self._ds_detail_frame = ctk.CTkFrame(inner, fg_color="transparent")
        self._ds_detail_frame.pack(fill="x", padx=20, pady=(0, 4))
        ctk.CTkLabel(self._ds_detail_frame, text="Factor:", width=100, anchor="w").pack(side="left")
        self.downsampling_factor_var = ctk.StringVar(value="")
        self.downsampling_factor_entry = ctk.CTkEntry(
            self._ds_detail_frame, textvariable=self.downsampling_factor_var, width=70,
            state="disabled", placeholder_text="e.g. 3"
        )
        self.downsampling_factor_entry.pack(side="left", padx=5)
        ctk.CTkLabel(self._ds_detail_frame, text="frames", text_color="gray").pack(side="left")

        # --- Interval Sampling ---
        self.interval_sampling_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            inner,
            text="Interval Sampling",
            variable=self.interval_sampling_var,
            command=self._on_interval_sampling_toggle,
            checkbox_width=18,
            checkbox_height=18
        ).pack(anchor="w", padx=5, pady=(6, 2))

        self._is_detail_frame = ctk.CTkFrame(inner, fg_color="transparent")
        self._is_detail_frame.pack(fill="x", padx=20, pady=(0, 4))

        row1 = ctk.CTkFrame(self._is_detail_frame, fg_color="transparent")
        row1.pack(fill="x", pady=1)
        ctk.CTkLabel(row1, text="Interval length:", width=130, anchor="w").pack(side="left")
        self.interval_length_var = ctk.StringVar(value="")
        self.interval_length_entry = ctk.CTkEntry(
            row1, textvariable=self.interval_length_var, width=70, state="disabled",
            placeholder_text="e.g. 30"
        )
        self.interval_length_entry.pack(side="left", padx=5)
        ctk.CTkLabel(row1, text="s", text_color="gray").pack(side="left")

        row2 = ctk.CTkFrame(self._is_detail_frame, fg_color="transparent")
        row2.pack(fill="x", pady=1)
        ctk.CTkLabel(row2, text="Num. of intervals:", width=130, anchor="w").pack(side="left")
        self.num_intervals_var = ctk.StringVar(value="")
        self.num_intervals_entry = ctk.CTkEntry(
            row2, textvariable=self.num_intervals_var, width=70, state="disabled",
            placeholder_text="e.g. 5"
        )
        self.num_intervals_entry.pack(side="left", padx=5)

        row3 = ctk.CTkFrame(self._is_detail_frame, fg_color="transparent")
        row3.pack(fill="x", pady=1)
        ctk.CTkLabel(row3, text="Min face fraction:", width=130, anchor="w").pack(side="left")
        self.min_face_fraction_var = ctk.StringVar(value="")
        self.min_face_fraction_entry = ctk.CTkEntry(
            row3, textvariable=self.min_face_fraction_var, width=70, state="disabled",
            placeholder_text="e.g. 0.1"
        )
        self.min_face_fraction_entry.pack(side="left", padx=5)
        ctk.CTkLabel(row3, text="(0.0-1.0)", text_color="gray", font=ctk.CTkFont(size=13)).pack(side="left")

        # --- Min Confidence ---
        conf_frame = ctk.CTkFrame(inner, fg_color="transparent")
        conf_frame.pack(fill="x", padx=5, pady=(8, 2))
        ctk.CTkLabel(conf_frame, text="Min Confidence:", width=130, anchor="w").pack(side="left")
        self.min_confidence_stage1_var = ctk.StringVar(value="")
        ctk.CTkEntry(
            conf_frame, textvariable=self.min_confidence_stage1_var, width=70,
            placeholder_text="e.g. 0.5"
        ).pack(side="left", padx=5)
        ctk.CTkLabel(conf_frame, text="(0.0-1.0)", text_color="gray", font=ctk.CTkFont(size=13)).pack(side="left")

        # --- GPU ---
        self.use_gpu_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            inner,
            text="Use GPU (if available)",
            variable=self.use_gpu_var,
            checkbox_width=18,
            checkbox_height=18
        ).pack(anchor="w", padx=5, pady=(6, 2))

        # --- Batch Size ---
        bs_frame = ctk.CTkFrame(inner, fg_color="transparent")
        bs_frame.pack(fill="x", padx=5, pady=(8, 6))
        ctk.CTkLabel(bs_frame, text="Batch Size:", width=130, anchor="w").pack(side="left")
        self.batch_size_var = ctk.StringVar(value="")
        ctk.CTkEntry(
            bs_frame, textvariable=self.batch_size_var, width=70,
            placeholder_text="e.g. 32"
        ).pack(side="left", padx=5)

    def _on_trim_toggle(self):
        state = "normal" if self.trim_edges_var.get() else "disabled"
        self.trim_start_entry.configure(state=state)
        self.trim_end_entry.configure(state=state)

    def _on_downsampling_toggle(self):
        state = "normal" if self.downsampling_var.get() else "disabled"
        self.downsampling_factor_entry.configure(state=state)

    def _on_interval_sampling_toggle(self):
        state = "normal" if self.interval_sampling_var.get() else "disabled"
        self.interval_length_entry.configure(state=state)
        self.num_intervals_entry.configure(state=state)
        self.min_face_fraction_entry.configure(state=state)

    def _toggle_detailed_log(self):
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
            font=ctk.CTkFont(size=13),
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
            font=ctk.CTkFont(size=13),
            width=50
        )
        self.progress_percentage_label.pack(side="left", padx=(5, 0))

        self.time_estimate_label = ctk.CTkLabel(
            progress_frame,
            text="",
            font=ctk.CTkFont(size=12),
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
            font=ctk.CTkFont(size=12)
        )
        self.log_toggle_btn.pack(anchor="w", padx=10, pady=(2, 0))

        self.log_textbox = ctk.CTkTextbox(progress_frame, height=100)

        self.status_label = self.current_step_label

    def _load_settings(self):
        """Load settings into UI."""
        if self.data_dir and self.data_dir.exists():
            self.tree_widget.build_tree(
                str(self.data_dir),
                derivatives_dir=str(self.derivatives_dir) if self.derivatives_dir else None
            )

        self.trim_edges_var.set(self.settings.get("trim_edges.enabled", False))
        self.trim_start_var.set(self.settings.get("trim_edges.trim_start", ""))
        self.trim_end_var.set(self.settings.get("trim_edges.trim_end", ""))
        self.downsampling_var.set(self.settings.get("downsampling.enabled", False))
        self.downsampling_factor_var.set(self.settings.get("downsampling.factor", ""))
        self.interval_sampling_var.set(self.settings.get("interval_sampling.enabled", False))
        self.interval_length_var.set(self.settings.get("interval_sampling.interval_length", ""))
        self.num_intervals_var.set(self.settings.get("interval_sampling.num_intervals", ""))
        self.min_face_fraction_var.set(self.settings.get("interval_sampling.min_face_fraction", ""))
        self.min_confidence_stage1_var.set(self.settings.get("stage1.min_confidence", ""))
        self.use_gpu_var.set(self.settings.get("stage1.use_gpu", False))
        self.batch_size_var.set(self.settings.get("stage2.batch_size", ""))

        self._on_trim_toggle()
        self._on_downsampling_toggle()
        self._on_interval_sampling_toggle()

    def update_dirs_and_reviewer(self, data_dir: Path, derivatives_dir: Path, reviewer_id: str):
        """Called when user changes dirs or reviewer via Back to setup."""
        self.data_dir = Path(data_dir)
        self.derivatives_dir = Path(derivatives_dir)
        self.reviewer_id = reviewer_id
        self.tree_widget.build_tree(
            str(self.data_dir),
            derivatives_dir=str(self.derivatives_dir)
        )

    def _get_min_confidence(self):
        try:
            val = self.min_confidence_stage1_var.get().strip()
            return float(val) if val else 0.0
        except (ValueError, tkinter.TclError):
            return 0.0

    def _get_batch_size(self):
        try:
            val = self.batch_size_var.get().strip()
            return max(1, int(val)) if val else 32
        except (ValueError, tkinter.TclError):
            return 32

    def _get_downsampling_factor(self):
        try:
            val = self.downsampling_factor_var.get().strip()
            return max(1, int(val)) if val else 3
        except (ValueError, tkinter.TclError):
            return 3

    def _get_interval_length(self):
        try:
            val = self.interval_length_var.get().strip()
            return float(val) if val else 30.0
        except (ValueError, tkinter.TclError):
            return 30.0

    def _get_num_intervals(self):
        try:
            val = self.num_intervals_var.get().strip()
            return max(1, int(val)) if val else 5
        except (ValueError, tkinter.TclError):
            return 5

    def _get_min_face_fraction(self):
        try:
            val = self.min_face_fraction_var.get().strip()
            return float(val) if val else 0.1
        except (ValueError, tkinter.TclError):
            return 0.1

    def _get_trim_start(self):
        """Return trim-from-start in seconds, or None if not set/not enabled."""
        if not self.trim_edges_var.get():
            return None
        try:
            val = self.trim_start_var.get().strip()
            return float(val) if val else None
        except (ValueError, tkinter.TclError):
            return None

    def _get_trim_end(self):
        """Return trim-from-end in seconds, or None if not set/not enabled."""
        if not self.trim_edges_var.get():
            return None
        try:
            val = self.trim_end_var.get().strip()
            return float(val) if val else None
        except (ValueError, tkinter.TclError):
            return None

    def _save_settings(self):
        self.settings.set("trim_edges.enabled", self.trim_edges_var.get())
        self.settings.set("trim_edges.trim_start", self.trim_start_var.get().strip())
        self.settings.set("trim_edges.trim_end", self.trim_end_var.get().strip())
        self.settings.set("downsampling.enabled", self.downsampling_var.get())
        self.settings.set("downsampling.factor", self.downsampling_factor_var.get().strip())
        self.settings.set("interval_sampling.enabled", self.interval_sampling_var.get())
        self.settings.set("interval_sampling.interval_length", self.interval_length_var.get().strip())
        self.settings.set("interval_sampling.num_intervals", self.num_intervals_var.get().strip())
        self.settings.set("interval_sampling.min_face_fraction", self.min_face_fraction_var.get().strip())
        self.settings.set("stage1.min_confidence", self.min_confidence_stage1_var.get().strip())
        self.settings.set("stage1.use_gpu", self.use_gpu_var.get())
        self.settings.set("stage2.batch_size", self.batch_size_var.get().strip())
        self.settings.save_settings()

    def _stop_processing(self):
        self._stop_requested = True
        proc = self._current_process_holder[0] if self._current_process_holder else None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def _bids_output_csv(self, participant: str, session: str) -> Path:
        """Return the BIDS-compliant face-detections CSV path in derivatives_dir."""
        return self.derivatives_dir / participant / session / f"{participant}_{session}_face-detections.csv"

    def _start_processing(self):
        if self.is_processing:
            messagebox.showwarning("Processing", "Processing is already running!")
            return

        if not self.data_dir:
            messagebox.showerror("Error", "Please select a data directory first!")
            return

        selected_sessions = self.tree_widget.get_selected_sessions()

        if not selected_sessions:
            messagebox.showerror("Error", "Please select at least one session to process!")
            return

        sessions_already_done = [
            (p, s, path) for (p, s, path) in selected_sessions
            if self._bids_output_csv(p, s).exists()
        ]
        if sessions_already_done:
            n = len(sessions_already_done)
            session_list = "\n".join(f"  - {p} / {s}" for (p, s, _) in sessions_already_done[:10])
            if n > 10:
                session_list += f"\n  ... and {n - 10} more"
            ok = messagebox.askyesno(
                "Overwrite face detection - annotations will be lost",
                "The following session(s) already have face detection results:\n\n"
                + session_list
                + "\n\nRe-running will:\n"
                "  - Overwrite face-detections CSV for these sessions\n"
                "  - Permanently delete is_face and merges annotations that depend on these sessions\n\n"
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

            # Resolve sampling rate from downsampling checkbox
            if self.downsampling_var.get():
                sampling_rate = self._get_downsampling_factor()
            else:
                sampling_rate = 1

            use_interval_sampling = self.interval_sampling_var.get()
            interval_length = self._get_interval_length()
            num_intervals = self._get_num_intervals()
            min_face_fraction = self._get_min_face_fraction()
            trim_start = self._get_trim_start()
            trim_end = self._get_trim_end()

            for idx, (participant_name, session_name, data_session_path) in enumerate(selected_sessions, 1):
                # data_session_path is in data_dir (for the video)
                # output CSV goes to derivatives_dir with BIDS naming
                output_csv = self._bids_output_csv(participant_name, session_name)
                output_csv.parent.mkdir(parents=True, exist_ok=True)

                # Find video path in data_dir for stage 2
                video_files = list(data_session_path.glob("scenevideo.*"))
                video_path_str = str(video_files[0]) if video_files else None

                step_id_s1 = f"session_{idx}_stage1"
                self.after(0, lambda p=participant_name, s=session_name:
                    reporter.set_current_step("Face Detection", p, s))
                self.after(0, lambda sid=step_id_s1: reporter.update_step_status(sid, "in_progress"))
                self.after(0, lambda: reporter.update_progress(0, "0%"))
                self.after(0, lambda: reporter.update_time_estimate("0s", None))
                self.after(0, lambda: reporter.log(f"Running Stage 1 for {session_name}..."))

                try:
                    _run_stage1_via_subprocess(
                        session_dir=str(data_session_path),
                        sampling_rate=sampling_rate,
                        use_gpu=self.use_gpu_var.get(),
                        min_confidence=self._get_min_confidence(),
                        reporter=reporter,
                        output_csv=str(output_csv),
                        use_interval_sampling=use_interval_sampling,
                        interval_length=interval_length,
                        num_intervals=num_intervals,
                        min_face_fraction=min_face_fraction,
                        trim_start=trim_start,
                        trim_end=trim_end,
                        settings=self.settings,
                        process_holder=self._current_process_holder,
                        stop_check=lambda: self._stop_requested,
                    )
                    self.after(0, lambda sid=step_id_s1: reporter.update_step_status(sid, "completed"))
                    _discard_annotations_for_session(
                        self.derivatives_dir, participant_name, session_name
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
                        session_dir=str(data_session_path),
                        batch_size=self._get_batch_size(),
                        reporter=reporter,
                        input_csv=str(output_csv),
                        video_path=video_path_str,
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
