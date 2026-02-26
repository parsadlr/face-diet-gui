"""
Shared GUI helpers: full-frame toplevel, progress reporter, constants.
"""

from pathlib import Path

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from tkinter import messagebox

from face_diet_gui.core.pipeline_helpers import _format_time


# Inactive (disabled) main action buttons: white text, desaturated background (towards gray)
BTN_DISABLED_FG = "#5a6268"


def _show_full_frame_toplevel(parent, session_dir: Path, face_info: dict):
    """Show full video frame (not just face crop) in a toplevel. Double-click from gallery."""
    try:
        frame_number = int(face_info.get("frame_number", 0))
        x, y, w, h = int(face_info.get("x", 0)), int(face_info.get("y", 0)), int(face_info.get("w", 0)), int(face_info.get("h", 0))
        video_files = list(session_dir.glob("scenevideo.*"))
        if not video_files:
            messagebox.showinfo("Full frame", "No video found for this session.", parent=parent)
            return
        cap = cv2.VideoCapture(str(video_files[0]))
        if not cap.isOpened():
            messagebox.showinfo("Full frame", "Could not open video.", parent=parent)
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showinfo("Full frame", "Could not read frame.", parent=parent)
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img = Image.fromarray(frame_rgb)
        root = parent.winfo_toplevel() if hasattr(parent, "winfo_toplevel") else parent
        tw = ctk.CTkToplevel(root)
        tw.title("Full frame")
        tw.geometry(f"{min(1000, img.width)}x{min(700, img.height)}")
        try:
            tw.transient(root)
        except Exception:
            pass
        img.thumbnail((1000, 700), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        lbl = ctk.CTkLabel(tw, image=img_tk, text="")
        lbl.image = img_tk
        lbl.pack(padx=5, pady=5)
        ctk.CTkLabel(tw, text="(Face region outlined in green. Close to return.)", font=ctk.CTkFont(size=11), text_color="gray").pack(pady=(0, 5))
    except Exception as e:
        messagebox.showerror("Error", f"Could not show full frame:\n{e}", parent=parent)


class ProgressReporter:
    """Helper class for reporting progress to GUI with enhanced UI."""

    def __init__(self, tab_instance):
        self.tab = tab_instance
        self.steps = {}
        self.start_time = None
        self.current_step_start = None
        self.current_step_name = ""

    def update_status(self, text: str):
        if hasattr(self.tab, 'current_step_label'):
            self.tab.current_step_label.configure(text=text)

    def set_current_step(self, stage_name: str, participant: str, session: str = None):
        import time
        self.current_step_start = time.time()
        self.current_step_name = stage_name
        if session:
            step_text = f"Participant: {participant}  |  Session: {session}  |  {stage_name}"
        else:
            step_text = f"Participant: {participant}  |  {stage_name}"
        if hasattr(self.tab, 'current_step_label'):
            self.tab.current_step_label.configure(text=step_text)

    def update_progress(self, value: float, percentage_text: str = None):
        if hasattr(self.tab, 'progress_bar'):
            self.tab.progress_bar.set(value)
        if hasattr(self.tab, 'progress_percentage_label') and percentage_text:
            self.tab.progress_percentage_label.configure(text=percentage_text)

    def update_step_time_estimate(self):
        import time
        if self.current_step_start and hasattr(self.tab, 'time_estimate_label'):
            elapsed = time.time() - self.current_step_start
            elapsed_str = _format_time(elapsed)
            text = f"Elapsed: {elapsed_str}"
            self.tab.time_estimate_label.configure(text=text)

    def update_time_estimate(self, elapsed: str, remaining: str = None):
        if hasattr(self.tab, 'time_estimate_label'):
            if remaining:
                text = f"Elapsed: {elapsed}  |  Remaining: ~{remaining}"
            else:
                text = f"Elapsed: {elapsed}"
            self.tab.time_estimate_label.configure(text=text)

    def add_step(self, step_id: str, step_name: str, status: str = "pending"):
        if not hasattr(self.tab, 'steps_frame'):
            return
        step_frame = ctk.CTkFrame(self.tab.steps_frame)
        step_frame.pack(fill="x", padx=5, pady=3)
        icons = {"pending": "○", "in_progress": "◉", "completed": "●", "error": "●"}
        colors = {"pending": "gray", "in_progress": "#3b8ed0", "completed": "#28a745", "error": "#dc3545"}
        icon_container = ctk.CTkFrame(step_frame, width=30, height=30, fg_color="transparent")
        icon_container.pack(side="left", padx=(5, 10))
        icon_container.pack_propagate(False)
        icon_label = ctk.CTkLabel(
            icon_container,
            text=icons.get(status, "○"),
            font=ctk.CTkFont(size=18),
            text_color=colors.get(status, "gray"),
            width=30,
            height=30
        )
        icon_label.pack(expand=True, fill="both")
        text_label = ctk.CTkLabel(step_frame, text=step_name, font=ctk.CTkFont(size=11), anchor="w")
        text_label.pack(side="left", fill="x", expand=True, padx=5)
        self.steps[step_id] = {
            'frame': step_frame,
            'icon_container': icon_container,
            'icon_label': icon_label,
            'text_label': text_label,
            'status': status
        }

    def update_step_status(self, step_id: str, status: str, detail: str = None):
        if step_id not in self.steps:
            return
        icons = {"pending": "○", "in_progress": "◉", "completed": "●", "error": "●"}
        colors = {"pending": "gray", "in_progress": "#3b8ed0", "completed": "#28a745", "error": "#dc3545"}
        step = self.steps[step_id]
        step['status'] = status
        if 'icon_label' in step and step['icon_label']:
            step['icon_label'].configure(
                text=icons.get(status, "○"),
                text_color=colors.get(status, "gray")
            )
        else:
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
        if hasattr(self.tab, 'log_textbox'):
            self.tab.log_textbox.insert("end", message + "\n")
            self.tab.log_textbox.see("end")
