"""
Face-Diet Multi-Tab GUI

A comprehensive GUI for the entire face processing pipeline:
- Tab 1: Face Detection (Stages 1 & 2)
- Tab 2: Face Instance Review - Manual detection verification
- Tab 3: Face ID Clustering (Stage 3)
- Tab 4: Face ID Review - Manual ID merging
"""

import os
import sys
import json
import re
import threading
import queue
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import customtkinter as ctk
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
import tkinter

from settings_manager import SettingsManager, ReviewerRegistry
from directory_tree_widget import DirectoryTreeWidget

# Path to venv_tf210 Python interpreter for processing
VENV_TF210_PYTHON = Path(__file__).parent / "venv_tf210" / "Scripts" / "python.exe"


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def _run_stage1_via_subprocess(session_dir: str, sampling_rate: int, use_gpu: bool, min_confidence: float, reporter, debug_mode: bool = False):
    """Run stage1_detect_faces via subprocess using venv_tf210, streaming output in real-time."""
    import time
    import threading
    import re
    import cv2
    
    if not VENV_TF210_PYTHON.exists():
        raise FileNotFoundError(
            f"venv_tf210 Python interpreter not found at: {VENV_TF210_PYTHON}\n"
            f"Please ensure venv_tf210 is set up correctly."
        )
    
    # Calculate 5% duration if debug mode is enabled
    end_time = None
    if debug_mode:
        session_path = Path(session_dir)
        video_files = list(session_path.glob("scenevideo.*"))
        if video_files:
            try:
                cap = cv2.VideoCapture(str(video_files[0]))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_duration = total_frames / fps if fps > 0 else 0.0
                    end_time = video_duration * 0.05  # 5% of video
                    cap.release()
                    reporter.log(f"DEBUG MODE: Processing only first 5% of video ({end_time:.1f}s)")
            except Exception as e:
                reporter.log(f"Warning: Could not calculate video duration for debug mode: {e}")
    
    script_path = Path(__file__).parent / "stage1_detect_faces.py"
    cmd = [
        str(VENV_TF210_PYTHON),
        "-u",  # Unbuffered output
        str(script_path),
        session_dir,
        "--sampling-rate", str(sampling_rate),
        "--min-confidence", str(min_confidence),
    ]
    if use_gpu:
        cmd.append("--gpu")
    if end_time is not None:
        cmd.extend(["--end-time", str(end_time)])
    
    # Use Popen to stream output in real-time
    # Set encoding to UTF-8 to handle Unicode characters properly on Windows
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # Capture stderr separately
        text=True,
        encoding='utf-8',
        errors='replace',  # Replace any encoding errors with placeholder
        bufsize=1,
        cwd=str(Path(__file__).parent)
    )
    
    # Pattern to match progress lines: [  X%] Frame Y ... Z/W frames
    progress_pattern = re.compile(r'\[\s*(\d+)%\].*?(\d+)/(\d+)\s+frames')
    
    # Variables for tracking
    last_percent = 0
    last_processed = 0
    last_total = 0
    step_start_time = time.time()
    error_lines = []
    
    # Stream stdout line by line
    for line in process.stdout:
        reporter.log(line.rstrip())
        
        # Try to parse progress
        match = progress_pattern.search(line)
        if match:
            percent = int(match.group(1))
            processed = int(match.group(2))
            total = int(match.group(3))
            
            # Update progress bar
            reporter.update_progress(percent / 100.0, f"{percent}%")
            
            # Calculate time estimates
            if processed > 0 and total > 0:
                elapsed = time.time() - step_start_time
                frames_remaining = total - processed
                if processed > 0:
                    avg_time_per_frame = elapsed / processed
                    estimated_remaining = avg_time_per_frame * frames_remaining
                    
                    elapsed_str = _format_time(elapsed)
                    remaining_str = _format_time(estimated_remaining)
                    reporter.update_time_estimate(elapsed_str, remaining_str)
            
            last_percent = percent
            last_processed = processed
            last_total = total
    
    # Read stderr in a separate thread to avoid blocking
    def read_stderr():
        for line in process.stderr:
            error_lines.append(line.rstrip())
            reporter.log(f"ERROR: {line.rstrip()}")
    
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()
    
    # Wait for process to complete
    return_code = process.wait()
    stderr_thread.join(timeout=1.0)  # Wait for stderr to finish reading
    
    if return_code != 0:
        error_msg = f"Stage 1 failed with return code {return_code}"
        if error_lines:
            error_msg += f"\n\nError output:\n" + "\n".join(error_lines[-10:])  # Last 10 error lines
        raise RuntimeError(error_msg)
    
    return True


def _run_stage2_via_subprocess(session_dir: str, batch_size: int, reporter, debug_mode: bool = False):
    """Run stage2_extract_attributes via subprocess using venv_tf210, streaming output in real-time."""
    import time
    import threading
    import re
    import pandas as pd
    
    if not VENV_TF210_PYTHON.exists():
        raise FileNotFoundError(
            f"venv_tf210 Python interpreter not found at: {VENV_TF210_PYTHON}\n"
            f"Please ensure venv_tf210 is set up correctly."
        )
    
    # Calculate 5% limit if debug mode is enabled
    limit = None
    if debug_mode:
        session_path = Path(session_dir)
        csv_path = session_path / "face_detections.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                total_faces = len(df)
                limit = max(1, int(total_faces * 0.05))  # 5% of faces, at least 1
                reporter.log(f"DEBUG MODE: Processing only first {limit} faces (5% of {total_faces} total)")
            except Exception as e:
                reporter.log(f"Warning: Could not calculate face limit for debug mode: {e}")
    
    script_path = Path(__file__).parent / "stage2_extract_attributes.py"
    cmd = [
        str(VENV_TF210_PYTHON),
        "-u",  # Unbuffered output
        str(script_path),
        session_dir,
        "--batch-size", str(batch_size),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    
    # Use Popen to stream output in real-time
    # Set encoding to UTF-8 to handle Unicode characters properly on Windows
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # Capture stderr separately
        text=True,
        encoding='utf-8',
        errors='replace',  # Replace any encoding errors with placeholder
        bufsize=1,
        cwd=str(Path(__file__).parent)
    )
    
    # Pattern to match progress lines: [  X%] ... Y/Z faces
    progress_pattern = re.compile(r'\[\s*(\d+)%\].*?(\d+)/(\d+)\s+faces')
    
    # Variables for tracking
    step_start_time = time.time()
    error_lines = []
    
    # Stream stdout line by line
    for line in process.stdout:
        reporter.log(line.rstrip())
        
        # Try to parse progress
        match = progress_pattern.search(line)
        if match:
            percent = int(match.group(1))
            processed = int(match.group(2))
            total = int(match.group(3))
            
            # Update progress bar
            reporter.update_progress(percent / 100.0, f"{percent}%")
            
            # Calculate time estimates
            if processed > 0 and total > 0:
                elapsed = time.time() - step_start_time
                faces_remaining = total - processed
                if processed > 0:
                    avg_time_per_face = elapsed / processed
                    estimated_remaining = avg_time_per_face * faces_remaining
                    
                    elapsed_str = _format_time(elapsed)
                    remaining_str = _format_time(estimated_remaining)
                    reporter.update_time_estimate(elapsed_str, remaining_str)
    
    # Read stderr in a separate thread to avoid blocking
    def read_stderr():
        for line in process.stderr:
            error_lines.append(line.rstrip())
            reporter.log(f"ERROR: {line.rstrip()}")
    
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()
    
    # Wait for process to complete
    return_code = process.wait()
    stderr_thread.join(timeout=1.0)  # Wait for stderr to finish reading
    
    if return_code != 0:
        error_msg = f"Stage 2 failed with return code {return_code}"
        if error_lines:
            error_msg += f"\n\nError output:\n" + "\n".join(error_lines[-10:])  # Last 10 error lines
        raise RuntimeError(error_msg)
    
    return True


def _run_stage3_via_subprocess(
    participant_dir: str,
    annotations_dir: str,
    output_dir: str,
    similarity_threshold: float,
    k_neighbors: int,
    min_confidence: float,
    algorithm: str,
    enable_refinement: bool,
    min_cluster_size: int,
    k_voting: int,
    min_votes: int,
    reporter,
):
    """Run stage3_graph_clustering via subprocess using venv_tf210, streaming output in real-time."""
    import time
    import threading

    if not VENV_TF210_PYTHON.exists():
        raise FileNotFoundError(
            f"venv_tf210 Python interpreter not found at: {VENV_TF210_PYTHON}\n"
            f"Please ensure venv_tf210 is set up correctly."
        )

    script_path = Path(__file__).parent / "stage3_graph_clustering.py"
    cmd = [
        str(VENV_TF210_PYTHON),
        "-u",
        str(script_path),
        participant_dir,
        "--threshold", str(similarity_threshold),
        "--k-neighbors", str(k_neighbors),
        "--min-confidence", str(min_confidence),
        "--algorithm", algorithm,
        "--annotations_dir", annotations_dir,
        "--output_dir", output_dir,
    ]

    if not enable_refinement:
        cmd.append("--no-refine")
    else:
        cmd.extend(["--min-cluster-size", str(min_cluster_size)])
        cmd.extend(["--k-voting", str(k_voting)])
        cmd.extend(["--min-votes", str(min_votes)])
    
    # Use Popen to stream output in real-time
    # Set encoding to UTF-8 to handle Unicode characters properly on Windows
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',  # Replace any encoding errors with placeholder
        bufsize=1,
        cwd=str(Path(__file__).parent)
    )
    
    # Pattern to match progress lines: [  X%] Processed Y/Z faces
    progress_pattern = re.compile(r'\[\s*(\d+)%\].*?Processed\s+[\d,]+/?([\d,]+)\s+.*?faces')
    
    # Variables for tracking
    step_start_time = time.time()
    
    # Stream output line by line
    for line in process.stdout:
        reporter.log(line.rstrip())
        
        # Try to parse progress
        match = progress_pattern.search(line)
        if match:
            percent = int(match.group(1))
            
            # Update progress bar
            reporter.update_progress(percent / 100.0, f"{percent}%")
            
            # Calculate time estimates
            elapsed = time.time() - step_start_time
            if percent > 0:
                estimated_total = elapsed / (percent / 100.0)
                estimated_remaining = estimated_total - elapsed
                
                elapsed_str = _format_time(elapsed)
                remaining_str = _format_time(estimated_remaining)
                reporter.update_time_estimate(elapsed_str, remaining_str)
    
    # Wait for process to complete
    return_code = process.wait()
    
    if return_code != 0:
        raise RuntimeError(f"Stage 3 failed with return code {return_code}")
    
    # Parse result from output (basic parsing)
    # The actual result dict isn't available, but we can extract info from stdout
    return {
        'total_faces': 0,  # Would need to parse from output
        'unique_global_ids': 0,  # Would need to parse from output
    }


class ProgressReporter:
    """Helper class for reporting progress to GUI with enhanced UI."""
    
    def __init__(self, tab_instance):
        self.tab = tab_instance
        self.steps = {}  # {step_id: {'frame': frame, 'status': status, 'label': label}}
        self.start_time = None
        self.current_step_start = None
        self.current_step_name = ""
    
    def update_status(self, text: str):
        """Update status label (thread-safe)."""
        if hasattr(self.tab, 'current_step_label'):
            self.tab.current_step_label.configure(text=text)
    
    
    def set_current_step(self, stage_name: str, participant: str, session: str = None):
        """Set current step information with proper formatting - all on one line."""
        import time
        self.current_step_start = time.time()
        self.current_step_name = stage_name
        
        # Format as single line: Participant: X | Session: Y | Stage
        if session:
            step_text = f"Participant: {participant}  |  Session: {session}  |  {stage_name}"
        else:
            step_text = f"Participant: {participant}  |  {stage_name}"
        
        if hasattr(self.tab, 'current_step_label'):
            self.tab.current_step_label.configure(text=step_text)
    
    def update_progress(self, value: float, percentage_text: str = None):
        """Update progress bar (0.0 to 1.0, thread-safe)."""
        if hasattr(self.tab, 'progress_bar'):
            self.tab.progress_bar.set(value)
        if hasattr(self.tab, 'progress_percentage_label') and percentage_text:
            self.tab.progress_percentage_label.configure(text=percentage_text)
    
    def update_step_time_estimate(self):
        """Update time estimate for current step."""
        import time
        if self.current_step_start and hasattr(self.tab, 'time_estimate_label'):
            elapsed = time.time() - self.current_step_start
            elapsed_str = _format_time(elapsed)
            text = f"Elapsed: {elapsed_str}"
            self.tab.time_estimate_label.configure(text=text)
    
    def update_time_estimate(self, elapsed: str, remaining: str = None):
        """Update time estimate label."""
        if hasattr(self.tab, 'time_estimate_label'):
            if remaining:
                text = f"Elapsed: {elapsed}  |  Remaining: ~{remaining}"
            else:
                text = f"Elapsed: {elapsed}"
            self.tab.time_estimate_label.configure(text=text)
    
    def add_step(self, step_id: str, step_name: str, status: str = "pending"):
        """Add a step to the steps list. Status: pending, in_progress, completed, error."""
        if not hasattr(self.tab, 'steps_frame'):
            return
        
        step_frame = ctk.CTkFrame(self.tab.steps_frame)
        step_frame.pack(fill="x", padx=5, pady=3)
        
        # Status icon - all use Unicode text with different colors
        icons = {
            "pending": "○",
            "in_progress": "◉",  # Blue circle with dot
            "completed": "●",  # Filled circle (green)
            "error": "●"  # Filled circle (red)
        }
        colors = {
            "pending": "gray",
            "in_progress": "#3b8ed0",  # Blue
            "completed": "#28a745",  # Green
            "error": "#dc3545"  # Red
        }
        
        # Create icon container
        icon_container = ctk.CTkFrame(step_frame, width=30, height=30, fg_color="transparent")
        icon_container.pack(side="left", padx=(5, 10))
        icon_container.pack_propagate(False)
        
        # Use Unicode text icon for all statuses (same format, different colors)
        icon_label = ctk.CTkLabel(
            icon_container,
            text=icons.get(status, "○"),
            font=ctk.CTkFont(size=18),
            text_color=colors.get(status, "gray"),
            width=30,
            height=30
        )
        icon_label.pack(expand=True, fill="both")
        
        text_label = ctk.CTkLabel(
            step_frame,
            text=step_name,
            font=ctk.CTkFont(size=11),
            anchor="w"
        )
        text_label.pack(side="left", fill="x", expand=True, padx=5)
        
        self.steps[step_id] = {
            'frame': step_frame,
            'icon_container': icon_container,
            'icon_label': icon_label,
            'text_label': text_label,
            'status': status
        }
    
    def update_step_status(self, step_id: str, status: str, detail: str = None):
        """Update a step's status with better icons."""
        if step_id not in self.steps:
            return
        
        # All icons use Unicode text with different colors
        icons = {
            "pending": "○",
            "in_progress": "◉",  # Blue circle with dot
            "completed": "●",  # Filled circle (green)
            "error": "●"  # Filled circle (red)
        }
        colors = {
            "pending": "gray",
            "in_progress": "#3b8ed0",  # Blue
            "completed": "#28a745",  # Green
            "error": "#dc3545"  # Red
        }
        
        step = self.steps[step_id]
        step['status'] = status
        
        # Update Unicode text icon (same format for all statuses)
        if 'icon_label' in step and step['icon_label']:
            # Update existing label
            step['icon_label'].configure(
                text=icons.get(status, "○"),
                text_color=colors.get(status, "gray")
            )
        else:
            # Create label if it doesn't exist
            icon_label = ctk.CTkLabel(
                step['icon_container'],
                text=icons.get(status, "○"),
                font=ctk.CTkFont(size=18),
                text_color=colors.get(status, "gray"),
                width=30,
                height=30
            )
            icon_label.pack(expand=True, fill="both")
            step['icon_label'] = icon_label
        
        if detail:
            step['text_label'].configure(text=detail)
    
    def log(self, message: str):
        """Add message to detailed log textbox (thread-safe)."""
        if hasattr(self.tab, 'log_textbox'):
            self.tab.log_textbox.insert("end", message + "\n")
            self.tab.log_textbox.see("end")


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

        self._setup_ui()
        self._load_settings()
    
    def _setup_ui(self):
        """Setup UI components."""
        # Title
        ctk.CTkLabel(
            self,
            text="Video Processing: Face Detection & Attributes",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(10, 20))

        # Read-only project + reviewer info bar
        info_frame = ctk.CTkFrame(self)
        info_frame.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(
            info_frame,
            text="Project:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=(10, 4))

        self.dir_label = ctk.CTkLabel(
            info_frame,
            text=str(self.project_dir) if self.project_dir else "—",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.dir_label.pack(side="left", padx=(0, 20))

        ctk.CTkLabel(
            info_frame,
            text="Reviewer:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=(10, 4))

        ctk.CTkLabel(
            info_frame,
            text=self.reviewer_id or "—",
            font=ctk.CTkFont(size=12),
            text_color="#3b8ed0"
        ).pack(side="left", padx=(0, 10))
        
        # Main content area (side-by-side)
        content_frame = ctk.CTkFrame(self)
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Left panel: Directory tree
        left_panel = ctk.CTkFrame(content_frame, width=600)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        ctk.CTkLabel(
            left_panel,
            text="Select Participants & Sessions",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        self.tree_widget = DirectoryTreeWidget(left_panel)
        self.tree_widget.pack(fill="both", expand=True, pady=5)
        
        # Right panel: Settings and progress
        right_panel = ctk.CTkFrame(content_frame, width=520)
        right_panel.pack(side="right", fill="both", padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Settings
        self._create_settings_panel(right_panel)
        
        # Progress section
        self._create_progress_panel(right_panel)
        
        # Process button at bottom
        self.process_btn = ctk.CTkButton(
            self,
            text="Start Processing",
            command=self._start_processing,
            width=200,
            height=45,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#28a745",
            hover_color="#218838",
            text_color="white",
            text_color_disabled="white"
        )
        self.process_btn.pack(pady=20)
    
    def _create_settings_panel(self, parent):
        """Create settings panel."""
        settings_frame = ctk.CTkFrame(parent)
        settings_frame.pack(fill="x", padx=10, pady=10)
        
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
        ctk.CTkLabel(conf_frame, text="(0.0-1.0)", text_color="gray", font=ctk.CTkFont(size=9)).pack(side="left")
        
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
            # Use original FPS - disable entry
            self.sampling_rate_entry.configure(state="disabled")
        else:
            # Use custom sampling rate - enable entry
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
        self.steps_frame = ctk.CTkScrollableFrame(progress_frame, height=180)
        self.steps_frame.pack(fill="both", expand=True, padx=10, pady=(0, 2))
        
        # Detailed log (collapsible)
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
        # Initially hidden
        
        # Status label (compatibility)
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
        
        # Build tree
        self.tree_widget.build_tree(str(self.project_dir))
        
        # Save to settings
        self.settings.set("last_project_dir", str(self.project_dir))
        self.settings.save_settings()
    
    def _load_settings(self):
        """Load settings into UI."""
        # Build tree from the app-level project_dir
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
        self.dir_label.configure(text=str(project_dir))
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
    
    def _start_processing(self):
        """Start processing in background thread."""
        if self.is_processing:
            messagebox.showwarning("Processing", "Processing is already running!")
            return
        
        if not self.project_dir:
            messagebox.showerror("Error", "Please select a project directory first!")
            return
        
        # Get selected sessions
        selected_sessions = self.tree_widget.get_selected_sessions()
        
        if not selected_sessions:
            messagebox.showerror("Error", "Please select at least one session to process!")
            return
        
        # Save settings
        self._save_settings()
        
        # Clear log
        self.log_textbox.delete("1.0", "end")
        
        # Disable button and change appearance
        self.process_btn.configure(
            state="disabled",
            text="Processing...",
            fg_color="#6c757d"  # Gray color for disabled state
        )
        self.is_processing = True
        
        # Start processing thread
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
            
            # Clear steps frame
            self.after(0, lambda: [w.destroy() for w in self.steps_frame.winfo_children()])
            
            # Create all steps upfront
            for idx, (participant_name, session_name, _) in enumerate(selected_sessions, 1):
                step_id_s1 = f"session_{idx}_stage1"
                step_id_s2 = f"session_{idx}_stage2"
                self.after(0, lambda sid=step_id_s1, pn=participant_name, sn=session_name: 
                    reporter.add_step(sid, f"{pn}/{sn} - Stage 1: Face Detection", "pending"))
                self.after(0, lambda sid=step_id_s2, pn=participant_name, sn=session_name: 
                    reporter.add_step(sid, f"{pn}/{sn} - Stage 2: Attribute Extraction", "pending"))
            
            time.sleep(0.2)  # Give UI time to render steps
            
            for idx, (participant_name, session_name, session_path) in enumerate(selected_sessions, 1):
                session_start = time.time()
                
                # Determine sampling rate
                if self.use_original_fps_var.get():
                    sampling_rate = 1
                else:
                    try:
                        sampling_rate = self.sampling_rate_var.get()
                        if not sampling_rate or sampling_rate <= 0:
                            sampling_rate = 30  # Default
                    except (ValueError, tkinter.TclError):
                        sampling_rate = 30  # Default
                
                # Stage 1: Face Detection
                step_id_s1 = f"session_{idx}_stage1"
                self.after(0, lambda p=participant_name, s=session_name: 
                    reporter.set_current_step("Face Detection", p, s))
                self.after(0, lambda sid=step_id_s1: reporter.update_step_status(sid, "in_progress"))
                self.after(0, lambda: reporter.update_progress(0, "0%"))  # Reset progress bar
                self.after(0, lambda: reporter.update_time_estimate("0s", None))  # Reset time
                self.after(0, lambda: reporter.log(f"Running Stage 1 for {session_name}..."))
                
                try:
                    _run_stage1_via_subprocess(
                        session_dir=str(session_path),
                        sampling_rate=sampling_rate,
                        use_gpu=self.use_gpu_var.get(),
                        min_confidence=self._get_min_confidence(),
                        reporter=reporter,
                        debug_mode=self.debug_mode_var.get()
                    )
                    self.after(0, lambda sid=step_id_s1: reporter.update_step_status(sid, "completed"))
                except Exception as e:
                    self.after(0, lambda sid=step_id_s1: reporter.update_step_status(sid, "error"))
                    raise
                
                # Stage 2: Attribute Extraction
                step_id_s2 = f"session_{idx}_stage2"
                self.after(0, lambda p=participant_name, s=session_name: 
                    reporter.set_current_step("Attribute Extraction", p, s))
                self.after(0, lambda sid=step_id_s2: reporter.update_step_status(sid, "in_progress"))
                self.after(0, lambda: reporter.update_progress(0, "0%"))  # Reset progress bar
                self.after(0, lambda: reporter.update_time_estimate("0s", None))  # Reset time
                self.after(0, lambda: reporter.log(f"Running Stage 2 for {session_name}..."))
                
                try:
                    _run_stage2_via_subprocess(
                        session_dir=str(session_path),
                        batch_size=self._get_batch_size(),
                        reporter=reporter,
                        debug_mode=self.debug_mode_var.get()
                    )
                    self.after(0, lambda sid=step_id_s2: reporter.update_step_status(sid, "completed"))
                except Exception as e:
                    self.after(0, lambda sid=step_id_s2: reporter.update_step_status(sid, "error"))
                    raise
            
            # Done
            self.after(0, lambda: reporter.update_status(f"[OK] All Complete!"))
            self.after(0, lambda: reporter.update_status(f"[OK] Processed {total_sessions} session(s)"))
            self.after(0, lambda: reporter.update_progress(1.0, "100%"))
            elapsed_str = _format_time(time.time() - reporter.start_time)
            self.after(0, lambda e=elapsed_str: reporter.update_time_estimate(e, None))
            self.after(0, lambda: messagebox.showinfo(
                "Success",
                f"Successfully processed {total_sessions} session(s)!"
            ))
        
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
            # Re-enable button and restore color
            self.after(0, lambda: self.process_btn.configure(
                state="normal",
                text="Start Processing",
                fg_color="#28a745"
            ))
            self.is_processing = False


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

        self._setup_ui()
        self._load_settings()
    
    def _setup_ui(self):
        """Setup UI components."""
        # Title
        ctk.CTkLabel(
            self,
            text="Face Instance Review - Manual Detection Verification",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(10, 20))

        # Reviewer info bar
        info_frame = ctk.CTkFrame(self)
        info_frame.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(
            info_frame,
            text="Reviewing as:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=(10, 4))
        ctk.CTkLabel(
            info_frame,
            text=self.reviewer_id or "—",
            font=ctk.CTkFont(size=13),
            text_color="#3b8ed0"
        ).pack(side="left", padx=(0, 20))

        # Participant / session selector
        sel_frame = ctk.CTkFrame(self)
        sel_frame.pack(fill="x", padx=20, pady=(0, 10))

        # --- Left: participants ---
        p_col = ctk.CTkFrame(sel_frame)
        p_col.pack(side="left", fill="both", expand=True, padx=(0, 5))

        ctk.CTkLabel(
            p_col, text="Participant", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=5, pady=(4, 2))

        self.participant_listbox = ctk.CTkScrollableFrame(p_col, height=90)
        self.participant_listbox.pack(fill="both", expand=True, padx=5, pady=(0, 4))

        # --- Right: sessions ---
        s_col = ctk.CTkFrame(sel_frame)
        s_col.pack(side="left", fill="both", expand=True, padx=(5, 0))

        ctk.CTkLabel(
            s_col, text="Session", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=5, pady=(4, 2))

        self.session_listbox = ctk.CTkScrollableFrame(s_col, height=90)
        self.session_listbox.pack(fill="both", expand=True, padx=5, pady=(0, 4))

        # Load button
        self.load_btn = ctk.CTkButton(
            sel_frame,
            text="Load Session",
            command=self._load_session,
            width=120,
            height=35,
            state="disabled"
        )
        self.load_btn.pack(side="left", padx=10, pady=5, anchor="s")

        # Populate participant list immediately
        self._populate_participants()
        
        # Statistics panel
        self.stats_frame = ctk.CTkFrame(self)
        self.stats_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        self.stats_label = ctk.CTkLabel(
            self.stats_frame,
            text="No session loaded",
            font=ctk.CTkFont(size=12)
        )
        self.stats_label.pack(pady=10)
        
        # Control panel
        self._create_control_panel()
        
        # Pagination controls
        self._create_pagination_controls()
        
        # Gallery frame
        self.gallery_container = ctk.CTkFrame(self)
        self.gallery_container.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        
        self.gallery_frame = ctk.CTkScrollableFrame(self.gallery_container, height=400)
        self.gallery_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Action buttons
        self._create_action_buttons()
    
    def _create_control_panel(self):
        """Create control panel with filters and settings."""
        panel = ctk.CTkFrame(self)
        panel.pack(fill="x", padx=20, pady=(0, 10))
        
        # Items per page
        ctk.CTkLabel(
            panel,
            text="Items per page:",
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=(10, 5))
        
        self.items_per_page_var = ctk.IntVar(value=100)
        self.items_per_page_entry = ctk.CTkEntry(
            panel,
            width=80,
            textvariable=self.items_per_page_var
        )
        self.items_per_page_entry.pack(side="left", padx=(0, 5))
        
        ctk.CTkButton(
            panel,
            text="Load",
            command=self._on_items_per_page_changed,
            width=80,
            height=30
        ).pack(side="left", padx=(0, 15))
        
        # Confidence filter
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
        
        ctk.CTkButton(
            panel,
            text="Apply Filter",
            command=self._apply_filter,
            width=100,
            height=30
        ).pack(side="left", padx=(10, 5))
    
    def _create_pagination_controls(self):
        """Create pagination controls (matching popup gallery style)."""
        self.pagination_frame = ctk.CTkFrame(self)
        self.pagination_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        # Page info label (showing current range)
        self.page_info_label = ctk.CTkLabel(
            self.pagination_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.page_info_label.pack(pady=(5, 5))
        
        # Page numbers frame (will be populated dynamically)
        self.page_numbers_frame = ctk.CTkFrame(self.pagination_frame)
        self.page_numbers_frame.pack(padx=10, pady=5)
    
    def _create_action_buttons(self):
        """Create action buttons at bottom."""
        action_frame = ctk.CTkFrame(self)
        action_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        # Selection info
        self.selection_label = ctk.CTkLabel(
            action_frame,
            text="0 instances selected",
            font=ctk.CTkFont(size=12)
        )
        self.selection_label.pack(side="left", padx=10)
        
        # Mark as non-face button
        self.mark_nonface_btn = ctk.CTkButton(
            action_frame,
            text="Mark as Non-Face",
            command=self._mark_as_nonface,
            width=160,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#dc3545",
            hover_color="#c82333",
            state="disabled"
        )
        self.mark_nonface_btn.pack(side="left", padx=5)
        
        # Mark as face button (restore mistakenly marked non-faces)
        self.mark_face_btn = ctk.CTkButton(
            action_frame,
            text="Mark as Face",
            command=self._mark_as_face,
            width=140,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#28a745",
            hover_color="#218838",
            state="disabled"
        )
        self.mark_face_btn.pack(side="left", padx=5)
        
        # Clear selection button
        self.clear_btn = ctk.CTkButton(
            action_frame,
            text="Clear Selection",
            command=self._clear_selection,
            width=140,
            height=40,
            font=ctk.CTkFont(size=13),
            state="disabled"
        )
        self.clear_btn.pack(side="left", padx=5)
        
        # Save annotations button
        self.save_btn = ctk.CTkButton(
            action_frame,
            text="Save Annotations",
            command=self._save_annotations,
            width=160,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#007bff",
            hover_color="#0056b3",
            state="disabled"
        )
        self.save_btn.pack(side="left", padx=5)
    
    def _populate_participants(self):
        """Populate the participant list from the project directory."""
        for w in self.participant_listbox.winfo_children():
            w.destroy()
        self._participant_btn_vars: Dict[str, ctk.StringVar] = {}

        if not self.project_dir or not self.project_dir.exists():
            ctk.CTkLabel(
                self.participant_listbox, text="No project loaded", text_color="gray"
            ).pack()
            return

        participants = sorted([
            d.name for d in self.project_dir.iterdir()
            if d.is_dir() and not d.name.startswith('_') and not d.name.startswith('.')
        ])

        for pname in participants:
            btn = ctk.CTkButton(
                self.participant_listbox,
                text=pname,
                height=28,
                anchor="w",
                fg_color="transparent",
                hover_color=("gray85", "gray30"),
                command=lambda p=pname: self._on_participant_selected(p)
            )
            btn.pack(fill="x", padx=2, pady=1)
            self._participant_btn_vars[pname] = btn

    def _on_participant_selected(self, participant: str):
        """Called when a participant is clicked."""
        self.selected_participant = participant
        self.selected_session = None
        self.load_btn.configure(state="disabled")

        # Highlight selected participant
        for pname, btn in self._participant_btn_vars.items():
            btn.configure(
                fg_color=("#3b8ed0" if pname == participant else "transparent")
            )

        # Populate sessions
        for w in self.session_listbox.winfo_children():
            w.destroy()

        p_path = self.project_dir / participant
        sessions = sorted([
            d.name for d in p_path.iterdir()
            if d.is_dir()
            and not d.name.startswith('_')
            and not d.name.startswith('.')
            and (d / "face_detections.csv").exists()
        ])

        self._session_btn_vars: Dict[str, ctk.CTkButton] = {}
        for sname in sessions:
            # Check if reviewer already has annotations for this session
            registry = ReviewerRegistry(self.project_dir)
            ann_path = registry.get_tab2_annotation_path(
                self.reviewer_id, participant, sname
            )
            done_marker = " ✓" if ann_path.exists() else ""
            btn = ctk.CTkButton(
                self.session_listbox,
                text=sname + done_marker,
                height=28,
                anchor="w",
                fg_color="transparent",
                hover_color=("gray85", "gray30"),
                command=lambda s=sname: self._on_session_selected(s)
            )
            btn.pack(fill="x", padx=2, pady=1)
            self._session_btn_vars[sname] = btn

    def _on_session_selected(self, session: str):
        """Called when a session is clicked."""
        self.selected_session = session
        self.load_btn.configure(state="normal")

        for sname, btn in self._session_btn_vars.items():
            btn.configure(
                fg_color=("#3b8ed0" if sname == session else "transparent")
            )

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
            
            # Load CSV
            self.df = pd.read_csv(csv_path)
            
            # Load existing annotations if they exist
            self._load_existing_annotations()
            
            # Sort by confidence (lowest first)
            if 'confidence' in self.df.columns:
                self.df = self.df.sort_values('confidence', ascending=True).reset_index(drop=True)
            
            # Initialize annotations for all instances (default: is_face=True)
            for idx in self.df.index:
                if idx not in self.annotations:
                    self.annotations[idx] = {'is_face': True, 'reviewed_at': None}
            
            # Update UI
            self.after(0, self._update_stats)
            self.after(0, self._display_gallery)
            self.after(0, lambda: self.save_btn.configure(state="normal"))
            
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
        annotation_file = registry.get_tab2_annotation_path(
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
    
    def _on_items_per_page_changed(self):
        """Handle items per page change."""
        try:
            new_value = int(self.items_per_page_var.get())
            if new_value < 1:
                new_value = 100
                self.items_per_page_var.set(100)
            
            batch_changed = (new_value != self.items_per_page)
            self.items_per_page = new_value
            
            # Reset to page 0 if batch size changed
            if batch_changed:
                self.current_page = 0
            
            self._display_gallery()
        except ValueError:
            self.items_per_page_var.set(self.items_per_page)
            messagebox.showerror("Invalid Input", "Please enter a valid number for items per page.")
    
    def _apply_filter(self):
        """Apply confidence filter and reload gallery."""
        if self.df is None:
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
            
            # Checkbox overlay (top-left corner)
            checkbox_var = ctk.BooleanVar(value=instance_idx in self.selected_instances)
            checkbox = ctk.CTkCheckBox(
                img_container,
                text="",
                variable=checkbox_var,
                width=20,
                command=lambda idx=instance_idx, var=checkbox_var: self._on_checkbox_toggle(idx, var),
                checkbox_width=18,
                checkbox_height=18,
                fg_color="#3b8ed0",
                hover_color="#2a6fa5"
            )
            checkbox.place(x=10, y=10)
            
            # Make entire image clickable to toggle selection
            def toggle_on_click(event, idx=instance_idx, var=checkbox_var):
                var.set(not var.get())
                self._on_checkbox_toggle(idx, var)
            
            img_label.bind("<Button-1>", toggle_on_click)
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
            
            # Show non-face marker if marked (overlay in center)
            if not self.annotations.get(instance_idx, {}).get('is_face', True):
                nonface_label = ctk.CTkLabel(
                    img_container,
                    text="NON-FACE",
                    font=ctk.CTkFont(size=10, weight="bold"),
                    text_color="red",
                    fg_color="black"
                )
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
                
                # Checkbox
                checkbox_var = ctk.BooleanVar(value=instance_idx in self.selected_instances)
                checkbox = ctk.CTkCheckBox(
                    img_container,
                    text="",
                    variable=checkbox_var,
                    width=20,
                    command=lambda idx=instance_idx, var=checkbox_var: self._on_checkbox_toggle(idx, var),
                    checkbox_width=18,
                    checkbox_height=18,
                    fg_color="#3b8ed0",
                    hover_color="#2a6fa5"
                )
                checkbox.place(x=10, y=10)
                
                # Make clickable
                def toggle_on_click(event, idx=instance_idx, var=checkbox_var):
                    var.set(not var.get())
                    self._on_checkbox_toggle(idx, var)
                
                img_label.bind("<Button-1>", toggle_on_click)
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
                
                # Non-face marker
                if not self.annotations.get(instance_idx, {}).get('is_face', True):
                    ctk.CTkLabel(
                        img_container,
                        text="NON-FACE",
                        font=ctk.CTkFont(size=10, weight="bold"),
                        text_color="red",
                        fg_color="black"
                    ).place(relx=0.5, rely=0.4, anchor="center")
            
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
    
    def _on_checkbox_toggle(self, instance_idx: int, checkbox_var: ctk.BooleanVar):
        """Handle checkbox toggle."""
        if checkbox_var.get():
            self.selected_instances.add(instance_idx)
        else:
            self.selected_instances.discard(instance_idx)
        
        self._update_selection_label()
    
    def _update_selection_label(self):
        """Update selection count label."""
        count = len(self.selected_instances)
        if count == 0:
            text = "0 instances selected"
        elif count == 1:
            text = "1 instance selected"
        else:
            text = f"{count} instances selected"
        
        self.selection_label.configure(text=text)
        
        # Enable/disable buttons based on selection
        if count > 0:
            self.mark_nonface_btn.configure(state="normal")
            self.mark_face_btn.configure(state="normal")
            self.clear_btn.configure(state="normal")
        else:
            self.mark_nonface_btn.configure(state="disabled")
            self.mark_face_btn.configure(state="disabled")
            self.clear_btn.configure(state="disabled")
    
    def _mark_as_nonface(self):
        """Mark selected instances as non-face."""
        if not self.selected_instances:
            return
        
        # Confirm action
        count = len(self.selected_instances)
        response = messagebox.askyesno(
            "Confirm",
            f"Mark {count} instance(s) as non-face?\n\n"
            "These instances will be excluded from clustering."
        )
        
        if not response:
            return
        
        # Mark as non-face
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        for idx in self.selected_instances:
            self.annotations[idx] = {
                'is_face': False,
                'reviewed_at': timestamp
            }
        
        # Clear selection
        self.selected_instances.clear()
        
        # Refresh display
        self._update_stats()
        self._display_gallery()
        
        messagebox.showinfo("Success", f"Marked {count} instance(s) as non-face.")
    
    def _mark_as_face(self):
        """Restore selected instances as valid faces."""
        if not self.selected_instances:
            return
        
        # Confirm action
        count = len(self.selected_instances)
        response = messagebox.askyesno(
            "Confirm",
            f"Mark {count} instance(s) as valid face(s)?\n\n"
            "These instances will be included in clustering."
        )
        
        if not response:
            return
        
        # Mark as face
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        for idx in self.selected_instances:
            self.annotations[idx] = {
                'is_face': True,
                'reviewed_at': timestamp
            }
        
        # Clear selection
        self.selected_instances.clear()
        
        # Refresh display
        self._update_stats()
        self._display_gallery()
        
        messagebox.showinfo("Success", f"Marked {count} instance(s) as valid face(s).")
    
    def _clear_selection(self):
        """Clear all selections."""
        self.selected_instances.clear()
        self._update_selection_label()
        self._display_gallery()
    
    def _save_annotations(self):
        """Save annotations to reviewer overlay CSV."""
        if self.df is None:
            return

        try:
            registry = ReviewerRegistry(self.project_dir)
            annotation_file = registry.get_tab2_annotation_path(
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

            # Refresh session list to show ✓ marker
            if self.selected_participant:
                self._on_participant_selected(self.selected_participant)

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


class FaceIDAssignmentTab(ctk.CTkFrame):
    """Tab 3: Face ID Assignment (Stage 3)."""

    def __init__(self, master, settings_manager: SettingsManager,
                 project_dir: Path, reviewer_id: str):
        super().__init__(master)
        self.settings = settings_manager
        self.project_dir: Path = project_dir
        self.reviewer_id: str = reviewer_id
        self.participant_widgets: Dict = {}
        self.processing_thread: Optional[threading.Thread] = None
        self.is_processing = False

        self._setup_ui()
        self._load_settings()
        self._load_participant_list()
    
    def _setup_ui(self):
        """Setup UI components."""
        # Title
        ctk.CTkLabel(
            self,
            text="Face ID Assignment: Global Clustering",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(10, 20))

        # Read-only info bar
        info_frame = ctk.CTkFrame(self)
        info_frame.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(
            info_frame,
            text="Project:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=(10, 4))
        self.dir_label = ctk.CTkLabel(
            info_frame,
            text=str(self.project_dir) if self.project_dir else "—",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.dir_label.pack(side="left", padx=(0, 20))

        ctk.CTkLabel(
            info_frame,
            text="Reviewer:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=(10, 4))
        ctk.CTkLabel(
            info_frame,
            text=self.reviewer_id or "—",
            font=ctk.CTkFont(size=13),
            text_color="#3b8ed0"
        ).pack(side="left", padx=(0, 10))
        
        # Main content area
        content_frame = ctk.CTkFrame(self)
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Left panel: Participant list
        left_panel = ctk.CTkFrame(content_frame, width=600)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        ctk.CTkLabel(
            left_panel,
            text="Select Participants",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        # Select all/deselect all
        select_frame = ctk.CTkFrame(left_panel)
        select_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkButton(
            select_frame,
            text="Select All",
            command=self._select_all_participants,
            width=100,
            height=28
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            select_frame,
            text="Deselect All",
            command=self._deselect_all_participants,
            width=100,
            height=28
        ).pack(side="left", padx=5)
        
        # Participant list
        self.participant_list_frame = ctk.CTkScrollableFrame(left_panel, height=400)
        self.participant_list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Right panel: Settings and progress
        right_panel = ctk.CTkFrame(content_frame, width=520)
        right_panel.pack(side="right", fill="both", padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Settings
        self._create_settings_panel(right_panel)
        
        # Progress
        self._create_progress_panel(right_panel)
        
        # Process button
        self.process_btn = ctk.CTkButton(
            self,
            text="Assign Face IDs",
            command=self._start_processing,
            width=200,
            height=45,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#007ACC",
            hover_color="#0066AA",
            text_color="white",
            text_color_disabled="white"
        )
        self.process_btn.pack(pady=20)
    
    def _create_settings_panel(self, parent):
        """Create settings panel."""
        settings_frame = ctk.CTkScrollableFrame(parent, height=300)
        settings_frame.pack(fill="x", padx=10, pady=10)
        
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
        
        # Enable refinement
        self.enable_refinement_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            settings_frame,
            text="Enable small cluster refinement",
            variable=self.enable_refinement_var,
            checkbox_width=18,
            checkbox_height=18
        ).pack(anchor="w", padx=5, pady=10)
        
        # Refinement settings (in a sub-frame)
        refine_frame = ctk.CTkFrame(settings_frame)
        refine_frame.pack(fill="x", padx=15, pady=5)
        
        # Min cluster size
        mcs_frame = ctk.CTkFrame(refine_frame)
        mcs_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(mcs_frame, text="Min Cluster Size:", width=130, anchor="w").pack(side="left")
        self.min_cluster_size_var = ctk.IntVar(value=5)
        ctk.CTkEntry(mcs_frame, textvariable=self.min_cluster_size_var, width=70).pack(side="left", padx=5)
        
        # k-voting
        kv_frame = ctk.CTkFrame(refine_frame)
        kv_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(kv_frame, text="k-Voting:", width=130, anchor="w").pack(side="left")
        self.k_voting_var = ctk.IntVar(value=10)
        ctk.CTkEntry(kv_frame, textvariable=self.k_voting_var, width=70).pack(side="left", padx=5)
        
        # Min votes
        mv_frame = ctk.CTkFrame(refine_frame)
        mv_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(mv_frame, text="Min Votes:", width=130, anchor="w").pack(side="left")
        self.min_votes_var = ctk.IntVar(value=5)
        ctk.CTkEntry(mv_frame, textvariable=self.min_votes_var, width=70).pack(side="left", padx=5)
        
        # Info: reviewer annotations will be applied automatically
        rev_info_frame = ctk.CTkFrame(settings_frame)
        rev_info_frame.pack(fill="x", padx=5, pady=10)
        ctk.CTkLabel(
            rev_info_frame,
            text="Face instance filters from Tab 2 will be applied\nautomatically using the current reviewer's annotations.",
            font=ctk.CTkFont(size=10),
            text_color="gray",
            justify="left"
        ).pack(side="left", padx=5)
    
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
    
    def _toggle_detailed_log(self):
        """Toggle detailed log visibility."""
        if self.show_log_var.get():
            self.log_textbox.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        else:
            self.log_textbox.pack_forget()
    
    def set_project_dir(self, project_dir: Path):
        """Called by app when project directory changes."""
        self.project_dir = project_dir
        self.dir_label.configure(text=str(project_dir))
        self._load_participant_list()

    def _load_participant_list(self):
        """Load participants from project directory."""
        # Clear existing
        for widget in self.participant_list_frame.winfo_children():
            widget.destroy()
        self.participant_widgets.clear()
        
        if not self.project_dir or not self.project_dir.exists():
            # Show instruction message
            msg_frame = ctk.CTkFrame(self.participant_list_frame)
            msg_frame.pack(fill="both", expand=True, padx=20, pady=40)
            
            ctk.CTkLabel(
                msg_frame,
                text="No project directory selected",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color="gray"
            ).pack(pady=(10, 5))
            
            ctk.CTkLabel(
                msg_frame,
                text="Click 'Browse' above to select your project directory",
                font=ctk.CTkFont(size=12),
                text_color="gray"
            ).pack(pady=5)
            return
        
        # Find participants (skip hidden and reserved dirs)
        participants = sorted([
            d for d in self.project_dir.iterdir()
            if d.is_dir()
            and not d.name.startswith('.')
            and not d.name.startswith('_')
        ])
        
        if not participants:
            ctk.CTkLabel(
                self.participant_list_frame,
                text="No participant directories found",
                text_color="orange"
            ).pack(pady=20)
            return
        
        # Create participant checkboxes
        participants_with_data = []
        for participant_dir in participants:
            participant_name = participant_dir.name
            
            # Count sessions with face_detections.csv
            sessions = [
                d for d in participant_dir.iterdir()
                if d.is_dir() and (d / "face_detections.csv").exists()
            ]
            
            if not sessions:
                continue  # Skip participants with no processed sessions
            
            participants_with_data.append((participant_name, participant_dir, sessions))
            
            # Create frame
            frame = ctk.CTkFrame(self.participant_list_frame)
            frame.pack(fill="x", padx=5, pady=3)
            
            # Checkbox
            var = ctk.BooleanVar(value=True)
            cb = ctk.CTkCheckBox(
                frame,
                text=f"{participant_name} ({len(sessions)} sessions)",
                variable=var,
                font=ctk.CTkFont(size=12),
                checkbox_width=18,
                checkbox_height=18
            )
            cb.pack(side="left", padx=10, pady=5)
            
            self.participant_widgets[participant_name] = {
                'frame': frame,
                'checkbox_var': var,
                'path': participant_dir,
                'session_count': len(sessions)
            }
        
        # Show message if no participants have processed data
        if not participants_with_data:
            msg_frame = ctk.CTkFrame(self.participant_list_frame)
            msg_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            ctk.CTkLabel(
                msg_frame,
                text="No processed participants found",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color="orange"
            ).pack(pady=(10, 5))
            
            ctk.CTkLabel(
                msg_frame,
                text=f"Found {len(participants)} participant(s), but none have processed sessions.\n\n"
                     f"Please process sessions in Tab 1 first.\n"
                     f"Tab 2 requires face_detections.csv files from Tab 1.",
                font=ctk.CTkFont(size=12),
                text_color="gray",
                justify="center"
            ).pack(pady=5)
    
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
        self.enable_refinement_var.set(self.settings.get("stage3.enable_refinement", True))
        self.min_cluster_size_var.set(self.settings.get("stage3.min_cluster_size", 5))
        self.k_voting_var.set(self.settings.get("stage3.k_voting", 10))
        self.min_votes_var.set(self.settings.get("stage3.min_votes", 5))
    
    def _save_settings(self):
        """Save current settings."""
        self.settings.set("stage3.algorithm", self.algorithm_var.get())
        self.settings.set("stage3.similarity_threshold", self.sim_threshold_var.get())
        self.settings.set("stage3.k_neighbors", self.k_neighbors_var.get())
        self.settings.set("stage3.min_confidence", self.min_confidence_var.get())
        self.settings.set("stage3.enable_refinement", self.enable_refinement_var.get())
        self.settings.set("stage3.min_cluster_size", self.min_cluster_size_var.get())
        self.settings.set("stage3.k_voting", self.k_voting_var.get())
        self.settings.set("stage3.min_votes", self.min_votes_var.get())
        self.settings.save_settings()
    
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
        
        # Save settings
        self._save_settings()
        
        # Clear log
        self.log_textbox.delete("1.0", "end")
        
        # Disable button and change appearance
        self.process_btn.configure(
            state="disabled",
            text="Processing...",
            fg_color="#6c757d"  # Gray color for disabled state
        )
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
                    output_dir = annotations_dir  # thin face_ids file goes into same dir

                    _run_stage3_via_subprocess(
                        participant_dir=str(participant_path),
                        annotations_dir=annotations_dir,
                        output_dir=output_dir,
                        similarity_threshold=self.sim_threshold_var.get(),
                        k_neighbors=self.k_neighbors_var.get(),
                        min_confidence=self.min_confidence_var.get(),
                        algorithm=self.algorithm_var.get(),
                        enable_refinement=self.enable_refinement_var.get(),
                        min_cluster_size=self.min_cluster_size_var.get(),
                        k_voting=self.k_voting_var.get(),
                        min_votes=self.min_votes_var.get(),
                        reporter=reporter,
                    )
                    self.after(0, lambda sid=step_id: reporter.update_step_status(sid, "completed"))
                
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
            # Re-enable button and restore color
            self.after(0, lambda: self.process_btn.configure(
                state="normal",
                text="Assign Face IDs",
                fg_color="#007ACC"
            ))
            self.is_processing = False


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
            text="Manual Review & Face ID Merging",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(10, 20))

        # Reviewer info bar
        info_frame = ctk.CTkFrame(self)
        info_frame.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(
            info_frame,
            text="Reviewing as:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=(10, 4))
        ctk.CTkLabel(
            info_frame,
            text=self.reviewer_id or "—",
            font=ctk.CTkFont(size=13),
            text_color="#3b8ed0"
        ).pack(side="left", padx=(0, 20))

        # Participant selector
        sel_frame = ctk.CTkFrame(self)
        sel_frame.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(
            sel_frame,
            text="Participant:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=(10, 4))

        self.participant_listbox_tab4 = ctk.CTkScrollableFrame(
            sel_frame, height=80, width=300
        )
        self.participant_listbox_tab4.pack(side="left", fill="x", padx=5, pady=4)

        self.review_btn = ctk.CTkButton(
            sel_frame,
            text="Load Participant",
            command=self._review_with_filters,
            width=140,
            height=35,
            state="disabled",
            fg_color="#007ACC",
            hover_color="#0066AA"
        )
        self.review_btn.pack(side="left", padx=10)

        self._populate_participants_tab4()
        
        # Session filter panel
        self._create_session_filter_panel()
        
        # Filter controls
        self._create_filter_controls()
        
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
        """Create filter controls panel."""
        panel = ctk.CTkFrame(self)
        panel.pack(fill="x", padx=20, pady=(0, 10))
        
        # Min instances filter
        ctk.CTkLabel(
            panel,
            text="Min Instances:",
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=(10, 5))
        
        self.min_instances_var = ctk.IntVar(value=1)
        ctk.CTkEntry(
            panel,
            width=60,
            textvariable=self.min_instances_var
        ).pack(side="left", padx=(0, 5))
        
        # Min confidence filter
        ctk.CTkLabel(
            panel,
            text="Min Confidence:",
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=(15, 5))
        
        self.min_confidence_var = ctk.DoubleVar(value=0.0)
        ctk.CTkEntry(
            panel,
            width=70,
            textvariable=self.min_confidence_var
        ).pack(side="left", padx=(0, 5))
        
        # Review button
        self.review_btn = ctk.CTkButton(
            panel,
            text="Review",
            command=self._review_with_filters,
            width=80,
            height=35,
            font=ctk.CTkFont(size=13),
            state="disabled"
        )
        self.review_btn.pack(side="left", padx=(10, 10))
    
    def _create_merge_controls(self):
        """Create merge/unmerge control buttons."""
        controls = ctk.CTkFrame(self)
        controls.pack(fill="x", padx=20, pady=(10, 10))
        
        # Merge button
        self.merge_btn = ctk.CTkButton(
            controls,
            text="Merge Selected",
            command=self._merge_selected,
            width=150,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#28a745",
            hover_color="#218838",
            state="disabled"
        )
        self.merge_btn.pack(side="left", padx=(10, 10))
        
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
        
        # Header
        header = ctk.CTkFrame(list_container, height=40)
        header.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(header, text="Select", width=60, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)
        ctk.CTkLabel(header, text="Face Preview", width=100, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)
        ctk.CTkLabel(header, text="Face ID(s)", width=300, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)
        ctk.CTkLabel(header, text="Instances", width=100, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)
        ctk.CTkLabel(header, text="Actions", width=120, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)
        
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
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#007ACC",
            hover_color="#0066AA",
            state="disabled"
        )
        self.save_btn.pack(side="left", padx=5)
        
        self.export_btn = ctk.CTkButton(
            button_container,
            text="Export Final CSV",
            command=self._export_final_csv,
            width=180,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#28a745",
            hover_color="#218838",
            state="disabled"
        )
        self.export_btn.pack(side="left", padx=5)
    
    def _populate_participants_tab4(self):
        """Populate the participant list from the project directory."""
        for w in self.participant_listbox_tab4.winfo_children():
            w.destroy()
        self._participant_btn_tab4: Dict[str, ctk.CTkButton] = {}

        if not self.project_dir or not self.project_dir.exists():
            ctk.CTkLabel(
                self.participant_listbox_tab4, text="No project loaded", text_color="gray"
            ).pack()
            return

        participants = sorted([
            d.name for d in self.project_dir.iterdir()
            if d.is_dir()
            and not d.name.startswith('_')
            and not d.name.startswith('.')
        ])

        for pname in participants:
            # Only show participants that have tab3_face_ids.csv
            registry = ReviewerRegistry(self.project_dir)
            face_ids_path = registry.get_tab3_face_ids_path(self.reviewer_id, pname)
            if not face_ids_path.exists():
                continue
            btn = ctk.CTkButton(
                self.participant_listbox_tab4,
                text=pname,
                height=28,
                anchor="w",
                fg_color="transparent",
                hover_color=("gray85", "gray30"),
                command=lambda p=pname: self._on_participant_selected_tab4(p)
            )
            btn.pack(fill="x", padx=2, pady=1)
            self._participant_btn_tab4[pname] = btn

    def _on_participant_selected_tab4(self, participant: str):
        """Called when a participant is clicked in Tab 4."""
        self.selected_participant = participant
        self.participant_dir = self.project_dir / participant
        self.review_btn.configure(state="normal")

        for pname, btn in self._participant_btn_tab4.items():
            btn.configure(
                fg_color=("#3b8ed0" if pname == participant else "transparent")
            )
    
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
        min_instances = int(self.min_instances_var.get())
        min_conf = float(self.min_confidence_var.get())
        
        # Apply filters
        df_filtered = self.df.copy()
        if min_conf > 0.0 and 'confidence' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['confidence'] >= min_conf]
        
        # Count instances per face ID
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
    
    def _review_with_filters(self):
        """Load and filter data using current settings."""
        if not self.selected_participant:
            messagebox.showerror("Error", "Please select a participant first.")
            return
        self._validate_and_load()

    def _validate_and_load(self):
        """Validate that clustering has been run for this reviewer/participant, then load."""
        registry = ReviewerRegistry(self.project_dir)
        face_ids_path = registry.get_tab3_face_ids_path(
            self.reviewer_id, self.selected_participant
        )

        if not face_ids_path.exists():
            messagebox.showerror(
                "Error",
                f"No face ID assignments found for reviewer '{self.reviewer_id}' "
                f"and participant '{self.selected_participant}'.\n\n"
                f"Please run Tab 3 (Face ID Clustering) first."
            )
            return

        thread = threading.Thread(target=self._load_data_thread, daemon=True)
        thread.start()

    def _load_data_thread(self):
        """Load and process data in background thread."""
        try:
            registry = ReviewerRegistry(self.project_dir)
            participant = self.selected_participant
            participant_path = self.project_dir / participant

            # Load face_ids overlay
            face_ids_path = registry.get_tab3_face_ids_path(self.reviewer_id, participant)
            face_ids_df = pd.read_csv(face_ids_path)

            # Load all per-session face_detections.csv and merge with face_ids
            session_dfs = []
            for session_dir in sorted(participant_path.iterdir()):
                if not session_dir.is_dir() or session_dir.name.startswith('_'):
                    continue
                csv_path = session_dir / "face_detections.csv"
                if not csv_path.exists():
                    continue
                sdf = pd.read_csv(csv_path)
                sdf['session_name'] = session_dir.name
                sdf['instance_index'] = sdf.index
                session_dfs.append(sdf)

            if not session_dfs:
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
            
            # Create session filter UI
            self.after(0, self._create_session_checkboxes)
            
            # Apply confidence filter
            min_conf = float(self.min_confidence_var.get())
            if 'confidence' in self.df.columns and min_conf > 0.0:
                self.df = self.df[self.df['confidence'] >= min_conf].reset_index(drop=True)
            
            # Compute counts and apply min instances
            min_instances = int(self.min_instances_var.get())
            face_counts = self.df['face_id'].value_counts()
            eligible_ids = face_counts[face_counts >= min_instances].index.tolist()
            
            if not eligible_ids:
                self.after(0, lambda: messagebox.showwarning(
                    "No Results",
                    "No face IDs meet the selected filters."
                ))
                return
            
            # Compute per-session bbox extents
            self.session_bbox_stats = {}
            for session_name, session_df in self.df.groupby('session_name'):
                max_x2 = float((session_df['x'] + session_df['w']).max())
                max_y2 = float((session_df['y'] + session_df['h']).max())
                self.session_bbox_stats[session_name] = {
                    'max_x2': max_x2,
                    'max_y2': max_y2,
                }
            
            # Build face groups
            self.face_groups = {}
            for face_id in eligible_ids:
                face_instances = self.df[self.df['face_id'] == face_id]
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
            
            # Sort by count
            self.face_groups = dict(
                sorted(self.face_groups.items(), key=lambda x: x[1]['count'], reverse=True)
            )
            
            # Store all groups
            self.face_groups_all = self.face_groups.copy()
            
            # Display results
            self.after(0, self._display_face_list)
            self.after(0, lambda: self.save_btn.configure(state="normal"))
            self.after(0, lambda: self.export_btn.configure(state="normal"))
            self.after(0, lambda: self.review_btn.configure(state="normal"))
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\n[ERROR] Error during data loading:")
            print(error_details)
            
            self.after(0, lambda: messagebox.showerror(
                "Error",
                f"Failed to load data:\n{str(e)}\n\nCheck terminal for details."
            ))
    
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
            self.session_filter_frame.pack(fill="x", padx=20, pady=(0, 10), after=self.path_entry.master)
    
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
        """Extract face crop from video frame (simplified from original)."""
        try:
            session_name = face_info['session_name']
            frame_number = int(face_info['frame_number'])
            
            x, y, w, h = int(face_info['x']), int(face_info['y']), int(face_info['w']), int(face_info['h'])
            
            # Find session directory
            session_dir = self.participant_dir / session_name
            if not session_dir.exists():
                return self._create_placeholder_image()
            
            # Find video file
            video_files = list(session_dir.glob("scenevideo.*"))
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
            
            # Resize to thumbnail
            face_crop = cv2.resize(face_crop, (80, 80), interpolation=cv2.INTER_AREA)
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            return Image.fromarray(face_crop_rgb)
            
        except Exception as e:
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
        """Create a row for a face ID."""
        row_frame = ctk.CTkFrame(self.face_list_frame)
        row_frame.pack(fill="x", padx=5, pady=3)
        
        # Make row clickable for gallery
        row_frame.bind("<Double-Button-1>", lambda e: self._open_gallery_popup(face_id))
        
        # Checkbox
        checkbox_var = ctk.BooleanVar()
        checkbox = ctk.CTkCheckBox(
            row_frame,
            text="",
            variable=checkbox_var,
            width=60,
            command=lambda: self._on_checkbox_toggle(face_id, checkbox_var.get()),
            checkbox_width=18,
            checkbox_height=18
        )
        checkbox.pack(side="left", padx=5)
        
        # Thumbnail container - adjust width based on number of thumbnails
        thumbnails = info.get('thumbnails', [])
        if not thumbnails:
            # Fallback to single thumbnail for backward compatibility
            thumb = info.get('thumbnail')
            if thumb:
                thumbnails = [thumb]
        
        # Calculate container width (80px per thumbnail + padding)
        num_thumbs = len(thumbnails)
        container_width = max(100, min(500, num_thumbs * 90))
        thumbnail_container = ctk.CTkFrame(row_frame, width=container_width)
        thumbnail_container.pack(side="left", padx=5)
        
        # Display thumbnails horizontally
        for thumb in thumbnails:
            if thumb:
                thumb_tk = ImageTk.PhotoImage(thumb)
                thumb_label = ctk.CTkLabel(thumbnail_container, image=thumb_tk, text="")
                thumb_label.image = thumb_tk  # Keep reference
                thumb_label.pack(side="left", padx=2)
                thumb_label.bind("<Double-Button-1>", lambda e: self._open_gallery_popup(face_id))
        
        # Face ID label
        original_ids = info.get('original_ids', [face_id])
        if len(original_ids) > 1:
            id_text = f"{face_id} ← merged from: {', '.join(original_ids)}"
        else:
            id_text = face_id
        
        id_label = ctk.CTkLabel(
            row_frame,
            text=id_text,
            width=300,
            anchor="w",
            font=ctk.CTkFont(size=12)
        )
        id_label.pack(side="left", padx=5)
        id_label.bind("<Double-Button-1>", lambda e: self._open_gallery_popup(face_id))
        
        # Instance count
        count_label = ctk.CTkLabel(
            row_frame,
            text=str(info['count']),
            width=100,
            font=ctk.CTkFont(size=12)
        )
        count_label.pack(side="left", padx=5)
        
        # Unmerge button (only for merged groups)
        if len(original_ids) > 1:
            unmerge_btn = ctk.CTkButton(
                row_frame,
                text="Unmerge",
                width=100,
                height=30,
                command=lambda: self._unmerge_group(face_id),
                fg_color="#dc3545",
                hover_color="#c82333"
            )
            unmerge_btn.pack(side="left", padx=5)
        
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
        
        # Enable/disable merge button
        if len(self.selected_ids) >= 2:
            self.merge_btn.configure(state="normal")
        else:
            self.merge_btn.configure(state="disabled")
    
    def _update_selection_label(self):
        """Update the selection info label."""
        count = len(self.selected_ids)
        if count == 0:
            text = "No faces selected"
        elif count == 1:
            text = "1 face selected"
        else:
            text = f"{count} faces selected"
        
        self.selection_label.configure(text=text)
    
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
        
        messagebox.showinfo(
            "Merged",
            f"Merged {len(selected_list)} face IDs into {merged_id}"
        )
    
    def _unmerge_group(self, merged_id: str):
        """Unmerge a merged group back into separate IDs."""
        if merged_id not in self.face_groups:
            return
        
        merged_info = self.face_groups[merged_id]
        original_ids = merged_info.get('original_ids', [])
        
        if len(original_ids) <= 1:
            return
        
        # Remove merged group
        if merged_id in self.face_groups:
            del self.face_groups[merged_id]
        
        # Restore individual groups
        for orig_id in original_ids:
            if orig_id in self.face_groups_all:
                self.face_groups[orig_id] = self.face_groups_all[orig_id].copy()
            self.face_id_to_merged[orig_id] = orig_id
        
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
            annotation_file = registry.get_tab4_merges_path(
                self.reviewer_id, self.selected_participant
            )
            annotation_file.parent.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().isoformat()
            annotation_records = [
                {
                    'face_id': face_id,
                    'merged_face_id': merged_id,
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
                f"Merge annotations saved!\n\n"
                f"File: {annotation_file}\n"
                f"Original unique IDs: {unique_original_ids}\n"
                f"Merged unique IDs: {unique_merged_ids}\n"
                f"Reduction: {unique_original_ids - unique_merged_ids} IDs"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to save annotations:\n{str(e)}")
    
    def _export_final_csv(self):
        """
        Export on-demand: join base data + tab3_face_ids + tab4_merges into a
        single wide CSV.  The large shared files are never duplicated on disk;
        this export is produced once per reviewer when explicitly requested.
        """
        if self.df_full is None:
            return

        try:
            registry = ReviewerRegistry(self.project_dir)
            participant = self.selected_participant

            # Reload the full base + reviewer face_ids (df_full already has face_id)
            df_out = self.df_full.copy()

            # Apply merge mapping → final_face_id
            df_out['final_face_id'] = df_out['face_id'].map(self.face_id_to_merged)
            df_out['final_face_id'].fillna(df_out['face_id'], inplace=True)

            # Save to reviewer-specific export
            export_dir = registry.get_reviewer_dir(self.reviewer_id) / participant
            export_dir.mkdir(parents=True, exist_ok=True)
            output_path = export_dir / f"final_faces_{self.reviewer_id}.csv"
            df_out.to_csv(output_path, index=False)

            unique_original_ids = len(set(self.face_id_to_merged.keys()))
            unique_merged_ids = len(set(self.face_id_to_merged.values()))

            messagebox.showinfo(
                "Exported",
                f"Final CSV exported!\n\n"
                f"File: {output_path}\n"
                f"Reviewer: {self.reviewer_id}\n\n"
                f"Total faces: {len(df_out)}\n"
                f"Original unique IDs: {unique_original_ids}\n"
                f"Final unique IDs: {unique_merged_ids}\n"
                f"Reduction: {unique_original_ids - unique_merged_ids} IDs"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to export CSV:\n{str(e)}")
    
    def _load_settings(self):
        """Load settings into UI."""
        self.min_instances_var.set(self.settings.get("tab3.min_instances", 1))
        self.min_confidence_var.set(self.settings.get("tab3.min_confidence", 0.0))


class StartupDialog(ctk.CTkToplevel):
    """
    Modal dialog shown at startup to select / create a reviewer and project directory.
    Blocks the main window until confirmed.
    """

    def __init__(self, master, settings: SettingsManager):
        super().__init__(master)
        self.settings = settings
        self.result_project_dir: Optional[Path] = None
        self.result_reviewer_id: Optional[str] = None

        self.title("Face-Diet — Setup")
        self.geometry("600x420")
        self.resizable(False, False)
        self.grab_set()   # make modal
        self.focus_force()

        self._setup_ui()
        self._load_last_values()

    # ------------------------------------------------------------------ #
    # UI                                                                   #
    # ------------------------------------------------------------------ #

    def _setup_ui(self):
        ctk.CTkLabel(
            self,
            text="Face-Diet Setup",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(pady=(20, 4))

        ctk.CTkLabel(
            self,
            text="Select your project directory and reviewer identity to begin.",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(pady=(0, 20))

        # Project directory
        proj_frame = ctk.CTkFrame(self)
        proj_frame.pack(fill="x", padx=30, pady=6)

        ctk.CTkLabel(
            proj_frame, text="Project directory:", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=6, pady=(6, 2))

        dir_row = ctk.CTkFrame(proj_frame, fg_color="transparent")
        dir_row.pack(fill="x", padx=6, pady=(0, 6))

        self.dir_entry = ctk.CTkEntry(dir_row, placeholder_text="Path to project root…", width=380)
        self.dir_entry.pack(side="left", fill="x", expand=True)

        ctk.CTkButton(
            dir_row, text="Browse", width=80, command=self._browse_dir
        ).pack(side="left", padx=(6, 0))

        # Reviewer selection
        rev_frame = ctk.CTkFrame(self)
        rev_frame.pack(fill="x", padx=30, pady=6)

        ctk.CTkLabel(
            rev_frame, text="Reviewer:", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=6, pady=(6, 2))

        self.reviewer_var = ctk.StringVar(value="")
        self.reviewer_option = ctk.CTkOptionMenu(
            rev_frame,
            variable=self.reviewer_var,
            values=["— select —"],
            width=280,
            command=self._on_reviewer_selected
        )
        self.reviewer_option.pack(side="left", padx=6, pady=(0, 6))

        ctk.CTkButton(
            rev_frame, text="+ New", width=80, command=self._show_new_reviewer_panel
        ).pack(side="left", padx=(6, 0))

        # New reviewer panel (hidden initially)
        self.new_rev_frame = ctk.CTkFrame(self)

        ctk.CTkLabel(
            self.new_rev_frame, text="New reviewer ID (no spaces):",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=6)

        self.new_id_entry = ctk.CTkEntry(self.new_rev_frame, width=120,
                                          placeholder_text="e.g. alice")
        self.new_id_entry.pack(side="left", padx=4)

        ctk.CTkLabel(
            self.new_rev_frame, text="Display name:", font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=6)

        self.new_name_entry = ctk.CTkEntry(self.new_rev_frame, width=140,
                                            placeholder_text="e.g. Alice Smith")
        self.new_name_entry.pack(side="left", padx=4)

        ctk.CTkButton(
            self.new_rev_frame, text="Create", width=70, command=self._create_reviewer
        ).pack(side="left", padx=6)

        # Status label
        self.status_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=11),
                                          text_color="orange")
        self.status_label.pack(pady=4)

        # Confirm button
        ctk.CTkButton(
            self,
            text="Continue →",
            width=180,
            height=42,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._confirm
        ).pack(pady=16)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _browse_dir(self):
        folder = filedialog.askdirectory(title="Select Project Root Directory")
        if not folder:
            return
        self.dir_entry.delete(0, "end")
        self.dir_entry.insert(0, folder)
        self._refresh_reviewer_list(Path(folder))

    def _refresh_reviewer_list(self, project_dir: Path):
        """Reload the reviewer dropdown from the project's registry."""
        try:
            registry = ReviewerRegistry(project_dir)
            ids = registry.get_reviewer_ids()
        except Exception:
            ids = []

        options = ids if ids else ["— select —"]
        self.reviewer_option.configure(values=options)
        if ids:
            self.reviewer_var.set(ids[0])
        else:
            self.reviewer_var.set("— select —")

    def _on_reviewer_selected(self, value: str):
        if value == "— select —":
            self.new_rev_frame.pack_forget()

    def _show_new_reviewer_panel(self):
        self.new_rev_frame.pack(fill="x", padx=30, pady=4)
        self.new_id_entry.focus()

    def _create_reviewer(self):
        project_dir_str = self.dir_entry.get().strip()
        if not project_dir_str or not Path(project_dir_str).exists():
            self.status_label.configure(text="Please select a valid project directory first.")
            return

        raw_id = self.new_id_entry.get().strip()
        display_name = self.new_name_entry.get().strip()
        if not raw_id:
            self.status_label.configure(text="Reviewer ID cannot be empty.")
            return

        reviewer_id = ReviewerRegistry.sanitize_id(raw_id)
        if not display_name:
            display_name = reviewer_id

        registry = ReviewerRegistry(Path(project_dir_str))
        if registry.reviewer_exists(reviewer_id):
            self.status_label.configure(text=f"Reviewer '{reviewer_id}' already exists.")
            return

        registry.add_reviewer(reviewer_id, display_name)
        self.status_label.configure(
            text=f"Reviewer '{reviewer_id}' created.", text_color="#28a745"
        )
        self._refresh_reviewer_list(Path(project_dir_str))
        self.reviewer_var.set(reviewer_id)
        self.new_rev_frame.pack_forget()

    def _load_last_values(self):
        """Pre-fill fields from last session."""
        last_dir = self.settings.get("last_project_dir", "")
        if last_dir and Path(last_dir).exists():
            self.dir_entry.insert(0, last_dir)
            self._refresh_reviewer_list(Path(last_dir))

        last_reviewer = self.settings.get("reviewer_id", "")
        if last_reviewer:
            ids = [self.reviewer_option.cget("values")[i]
                   for i in range(len(self.reviewer_option.cget("values")))]
            if last_reviewer in ids:
                self.reviewer_var.set(last_reviewer)

    def _confirm(self):
        project_dir_str = self.dir_entry.get().strip()
        if not project_dir_str or not Path(project_dir_str).exists():
            self.status_label.configure(
                text="Please select a valid project directory.", text_color="orange"
            )
            return

        reviewer_id = self.reviewer_var.get()
        if not reviewer_id or reviewer_id == "— select —":
            self.status_label.configure(
                text="Please select or create a reviewer.", text_color="orange"
            )
            return

        self.result_project_dir = Path(project_dir_str)
        self.result_reviewer_id = reviewer_id

        # Persist
        self.settings.set("last_project_dir", str(self.result_project_dir))
        self.settings.set("reviewer_id", self.result_reviewer_id)
        self.settings.save_settings()

        self.grab_release()
        self.destroy()


class FaceDietApp(ctk.CTk):
    """Main application with tabbed interface."""

    def __init__(self):
        super().__init__()

        self.title("Face-Diet: Comprehensive Face Processing Pipeline")
        self.geometry("1600x1000")

        # Initialize settings
        self.settings = SettingsManager()

        # Show startup dialog (modal — blocks until dismissed)
        self.withdraw()  # hide main window until setup is complete
        dialog = StartupDialog(self, self.settings)
        self.wait_window(dialog)

        if dialog.result_project_dir is None:
            # User closed the dialog without confirming
            self.destroy()
            return

        self.project_dir: Path = dialog.result_project_dir
        self.reviewer_id: str = dialog.result_reviewer_id

        self.deiconify()  # show main window now

        # Create tabview
        self.tabview = ctk.CTkTabview(self, width=1580, height=980)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # Add tabs
        self.tabview.add("Face Detection")
        self.tabview.add("Face Instance Review")
        self.tabview.add("Face ID Clustering")
        self.tabview.add("Face ID Review")

        # Create tab content — pass project_dir and reviewer_id to each tab
        self.tab1 = VideoProcessingTab(
            self.tabview.tab("Face Detection"),
            self.settings, self.project_dir, self.reviewer_id
        )
        self.tab1.pack(fill="both", expand=True)

        self.tab2 = FaceInstanceReviewTab(
            self.tabview.tab("Face Instance Review"),
            self.settings, self.project_dir, self.reviewer_id
        )
        self.tab2.pack(fill="both", expand=True)

        self.tab3 = FaceIDAssignmentTab(
            self.tabview.tab("Face ID Clustering"),
            self.settings, self.project_dir, self.reviewer_id
        )
        self.tab3.pack(fill="both", expand=True)

        self.tab4 = ManualReviewTab(
            self.tabview.tab("Face ID Review"),
            self.settings, self.project_dir, self.reviewer_id
        )
        self.tab4.pack(fill="both", expand=True)


def main():
    """Main entry point."""
    # Verify venv_tf210 exists for processing
    if not VENV_TF210_PYTHON.exists():
        import tkinter.messagebox as msgbox
        msgbox.showerror(
            "Missing venv_tf210",
            f"venv_tf210 Python interpreter not found at:\n{VENV_TF210_PYTHON}\n\n"
            f"Processing features (Tabs 1 & 3) will not work.\n"
            f"Tabs 2 & 4 (Review features) should still work."
        )
    
    # Set appearance
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    # Create and run app
    app = FaceDietApp()
    app.mainloop()


if __name__ == "__main__":
    main()
