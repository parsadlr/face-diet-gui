"""
Face-Diet Multi-Tab GUI

A comprehensive GUI for the entire face processing pipeline:
- Tab 1: Video Processing (Stages 1 & 2)
- Tab 2: Face ID Assignment (Stage 3)
- Tab 3: Manual Review & Merging
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

from settings_manager import SettingsManager
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
    similarity_threshold: float,
    k_neighbors: int,
    min_confidence: float,
    algorithm: str,
    enable_refinement: bool,
    min_cluster_size: int,
    k_voting: int,
    min_votes: int,
    reporter
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
        "-u",  # Unbuffered output
        str(script_path),
        participant_dir,
        "--threshold", str(similarity_threshold),
        "--k-neighbors", str(k_neighbors),
        "--min-confidence", str(min_confidence),
        "--algorithm", algorithm,
        "--output", "faces_combined.csv",
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
    
    def __init__(self, master, settings_manager: SettingsManager):
        super().__init__(master)
        self.settings = settings_manager
        self.project_dir: Optional[Path] = None
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
        
        # Directory selection
        dir_frame = ctk.CTkFrame(self)
        dir_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        ctk.CTkLabel(
            dir_frame,
            text="Project Directory:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=10)
        
        self.dir_entry = ctk.CTkEntry(
            dir_frame,
            placeholder_text="Select project root directory...",
            width=500,
            state="readonly"
        )
        self.dir_entry.pack(side="left", padx=10)
        
        ctk.CTkButton(
            dir_frame,
            text="Browse",
            command=self._browse_directory,
            width=100,
            height=35
        ).pack(side="left", padx=5)
        
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
        last_dir = self.settings.get("last_project_dir", "")
        if last_dir and Path(last_dir).exists():
            self.project_dir = Path(last_dir)
            self.dir_entry.configure(state="normal")
            self.dir_entry.delete(0, "end")
            self.dir_entry.insert(0, last_dir)
            self.dir_entry.configure(state="readonly")
            self.tree_widget.build_tree(last_dir)
        
        self.use_original_fps_var.set(self.settings.get("stage1.use_original_fps", True))
        self.sampling_rate_var.set(self.settings.get("stage1.sampling_rate", 30))
        self.min_confidence_stage1_var.set(self.settings.get("stage1.min_confidence", 0.0))
        self.use_gpu_var.set(self.settings.get("stage1.use_gpu", False))
        self.batch_size_var.set(self.settings.get("stage2.batch_size", 32))
        
        # Update UI state based on checkbox
        self._on_original_fps_toggle()
    
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


class FaceIDAssignmentTab(ctk.CTkFrame):
    """Tab 2: Face ID Assignment (Stage 3)."""
    
    def __init__(self, master, settings_manager: SettingsManager):
        super().__init__(master)
        self.settings = settings_manager
        self.project_dir: Optional[Path] = None
        self.participant_widgets: Dict = {}
        self.processing_thread: Optional[threading.Thread] = None
        self.is_processing = False
        
        self._setup_ui()
        self._load_settings()
        
        # Show initial message if no directory loaded
        if not self.project_dir:
            self._load_participant_list()
    
    def _setup_ui(self):
        """Setup UI components."""
        # Title
        ctk.CTkLabel(
            self,
            text="Face ID Assignment: Global Clustering",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(10, 20))
        
        # Directory selection
        dir_frame = ctk.CTkFrame(self)
        dir_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        ctk.CTkLabel(
            dir_frame,
            text="Project Directory:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=10)
        
        self.dir_entry = ctk.CTkEntry(
            dir_frame,
            placeholder_text="Select project root directory...",
            width=500,
            state="readonly"
        )
        self.dir_entry.pack(side="left", padx=10)
        
        ctk.CTkButton(
            dir_frame,
            text="Browse",
            command=self._browse_directory,
            width=100,
            height=35
        ).pack(side="left", padx=5)
        
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
        
        # Load participants
        self._load_participant_list()
        
        # Save to settings
        self.settings.set("last_project_dir", str(self.project_dir))
        self.settings.save_settings()
    
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
        
        # Find participants
        participants = sorted([
            d for d in self.project_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
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
        last_dir = self.settings.get("last_project_dir", "")
        if last_dir and Path(last_dir).exists():
            self.project_dir = Path(last_dir)
            self.dir_entry.configure(state="normal")
            self.dir_entry.delete(0, "end")
            self.dir_entry.insert(0, last_dir)
            self.dir_entry.configure(state="readonly")
            self._load_participant_list()
        
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
                    _run_stage3_via_subprocess(
                        participant_dir=str(participant_path),
                        similarity_threshold=self.sim_threshold_var.get(),
                        k_neighbors=self.k_neighbors_var.get(),
                        min_confidence=self.min_confidence_var.get(),
                        algorithm=self.algorithm_var.get(),
                        enable_refinement=self.enable_refinement_var.get(),
                        min_cluster_size=self.min_cluster_size_var.get(),
                        k_voting=self.k_voting_var.get(),
                        min_votes=self.min_votes_var.get(),
                        reporter=reporter
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
    """Tab 3: Manual Review & Merging (refactored from original GUI)."""
    
    def __init__(self, master, settings_manager: SettingsManager):
        super().__init__(master)
        self.settings = settings_manager
        
        # Data storage
        self.participant_dir: Optional[Path] = None
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
        
        # Participant selection
        dir_frame = ctk.CTkFrame(self)
        dir_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        ctk.CTkLabel(
            dir_frame,
            text="Participant Folder:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=10)
        
        self.path_entry = ctk.CTkEntry(
            dir_frame,
            placeholder_text="Select participant directory...",
            width=500,
            state="readonly"
        )
        self.path_entry.pack(side="left", padx=10)
        
        ctk.CTkButton(
            dir_frame,
            text="Browse",
            command=self._browse_folder,
            width=100,
            height=35
        ).pack(side="left", padx=5)
        
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
        """Create the save button."""
        save_frame = ctk.CTkFrame(self, fg_color="transparent")
        save_frame.pack(fill="x", padx=20)
        
        self.save_btn = ctk.CTkButton(
            save_frame,
            text="Save Merged Results",
            command=self._save_results,
            width=200,
            height=45,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#007ACC",
            hover_color="#0066AA",
            state="disabled"
        )
        self.save_btn.pack(pady=10)
    
    def _browse_folder(self):
        """Open folder browser and load data."""
        folder = filedialog.askdirectory(title="Select Participant Directory")
        if not folder:
            return
        
        self.participant_dir = Path(folder)
        self.path_entry.configure(state="normal")
        self.path_entry.delete(0, "end")
        self.path_entry.insert(0, str(self.participant_dir))
        self.path_entry.configure(state="readonly")
        self.review_btn.configure(state="normal")
    
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
        if not self.participant_dir:
            messagebox.showerror("Error", "Please select a participant folder first.")
            return
        self._validate_and_load()
    
    def _validate_and_load(self):
        """Validate folder structure and load data."""
        csv_path = self.participant_dir / "faces_combined.csv"
        
        if not csv_path.exists():
            messagebox.showerror(
                "Error",
                f"Could not find faces_combined.csv in:\n{self.participant_dir}"
            )
            return
        
        # Load data in background thread
        thread = threading.Thread(target=self._load_data_thread, args=(csv_path,))
        thread.daemon = True
        thread.start()
    
    def _load_data_thread(self, csv_path: Path):
        """Load and process data in background thread (simplified version from original)."""
        try:
            # Load CSV
            self.df_full = pd.read_csv(csv_path)
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
        gallery_frame = ctk.CTkScrollableFrame(popup, height=550)
        gallery_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Close button
        ctk.CTkButton(
            popup,
            text="Close",
            command=popup.destroy,
            width=150,
            height=40,
            font=ctk.CTkFont(size=14)
        ).pack(pady=10)
        
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
            for idx, img in enumerate(images):
                row_idx = idx // images_per_row
                col_idx = idx % images_per_row
                
                # Create frame for image
                img_frame = ctk.CTkFrame(gallery_frame, width=130, height=130)
                img_frame.grid(row=row_idx, column=col_idx, padx=5, pady=5)
                
                # Add image
                img_tk = ImageTk.PhotoImage(img)
                img_label = ctk.CTkLabel(img_frame, image=img_tk, text="")
                img_label.image = img_tk  # Keep reference
                img_label.pack(padx=5, pady=5)
            
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
    
    def _save_results(self):
        """Save merged results to CSV with merged_face_id column."""
        if self.df_full is None:
            return
        
        # Confirm save
        response = messagebox.askyesno(
            "Confirm Save",
            "This will add a 'merged_face_id' column to faces_combined.csv.\n"
            "The original file will be backed up. Continue?"
        )
        
        if not response:
            return
        
        try:
            # Create backup
            csv_path = self.participant_dir / "faces_combined.csv"
            backup_path = self.participant_dir / "faces_combined.backup.csv"
            
            import shutil
            shutil.copy2(csv_path, backup_path)
            
            # Add merged_face_id column
            df_out = self.df_full.copy()
            df_out['merged_face_id'] = df_out['face_id'].map(self.face_id_to_merged)
            
            # Fill any missing values (use original face_id)
            df_out['merged_face_id'].fillna(df_out['face_id'], inplace=True)
            
            # Count merges
            num_merged = len([k for k, v in self.face_id_to_merged.items() if k != v])
            unique_merged_ids = df_out['merged_face_id'].nunique()
            original_unique_ids = df_out['face_id'].nunique()
            
            # Save
            df_out.to_csv(csv_path, index=False)
            
            messagebox.showinfo(
                "Saved",
                f"Successfully saved merged results!\n\n"
                f"Output: {csv_path}\n"
                f"Backup: {backup_path}\n\n"
                f"Original unique IDs: {original_unique_ids}\n"
                f"Merged unique IDs: {unique_merged_ids}\n"
                f"Reduction: {original_unique_ids - unique_merged_ids} IDs"
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror(
                "Error",
                f"Failed to save results:\n{str(e)}"
            )
    
    def _load_settings(self):
        """Load settings into UI."""
        self.min_instances_var.set(self.settings.get("tab3.min_instances", 1))
        self.min_confidence_var.set(self.settings.get("tab3.min_confidence", 0.0))


class FaceDietApp(ctk.CTk):
    """Main application with tabbed interface."""
    
    def __init__(self):
        super().__init__()
        
        self.title("Face-Diet: Comprehensive Face Processing Pipeline")
        self.geometry("1600x1000")
        
        # Initialize settings manager
        self.settings = SettingsManager()
        
        # Create tabview
        self.tabview = ctk.CTkTabview(self, width=1580, height=980)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add tabs
        self.tabview.add("Video Processing")
        self.tabview.add("Face ID Assignment")
        self.tabview.add("Manual Review")
        
        # Create tab content
        self.tab1 = VideoProcessingTab(self.tabview.tab("Video Processing"), self.settings)
        self.tab1.pack(fill="both", expand=True)
        
        self.tab2 = FaceIDAssignmentTab(self.tabview.tab("Face ID Assignment"), self.settings)
        self.tab2.pack(fill="both", expand=True)
        
        self.tab3 = ManualReviewTab(self.tabview.tab("Manual Review"), self.settings)
        self.tab3.pack(fill="both", expand=True)


def main():
    """Main entry point."""
    # Verify venv_tf210 exists for processing
    if not VENV_TF210_PYTHON.exists():
        import tkinter.messagebox as msgbox
        msgbox.showerror(
            "Missing venv_tf210",
            f"venv_tf210 Python interpreter not found at:\n{VENV_TF210_PYTHON}\n\n"
            f"Processing features (Tabs 1 & 2) will not work.\n"
            f"Tab 3 (Manual Review) should still work."
        )
    
    # Set appearance
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    # Create and run app
    app = FaceDietApp()
    app.mainloop()


if __name__ == "__main__":
    main()
