"""
Face-Diet GUI application: startup dialog, main window, and entry point.

Tab classes live in face_diet_gui.gui.tabs. Run via: python -m face_diet_gui or from face_diet_gui.gui.app import main.
"""

import sys
from pathlib import Path
from typing import Optional

import customtkinter as ctk
from tkinter import filedialog
import tkinter

from face_diet_gui.core.settings_manager import SettingsManager, ReviewerRegistry
from face_diet_gui.gui.tabs import (
    VideoProcessingTab,
    FaceInstanceReviewTab,
    MismatchResolutionTab,
    FaceIDAssignmentTab,
    ManualReviewTab,
)


class StartupDialog(ctk.CTkToplevel):
    """
    Modal dialog shown at startup to select data dir, derivatives dir, and reviewer.
    Blocks the main window until confirmed.
    """

    def __init__(self, master, settings: SettingsManager):
        super().__init__(master)
        self.settings = settings
        self.result_data_dir: Optional[Path] = None
        self.result_derivatives_dir: Optional[Path] = None
        self.result_reviewer_id: Optional[str] = None

        self.title("Face-Diet — Setup")
        self.geometry("760x620")
        self.minsize(680, 560)
        self.resizable(True, True)
        self.grab_set()
        self.focus_force()

        self._setup_ui()
        self._load_last_values()

    def _setup_ui(self):
        ctk.CTkLabel(
            self,
            text="Face-Diet Setup",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(pady=(20, 4))

        ctk.CTkLabel(
            self,
            text="Configure your data directory, derivatives directory, and reviewer identity.",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(pady=(0, 16))

        # --- Data directory ---
        data_frame = ctk.CTkFrame(self)
        data_frame.pack(fill="x", padx=30, pady=6)

        ctk.CTkLabel(
            data_frame, text="Data directory (videos & eye tracking):",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=6, pady=(6, 2))

        data_row = ctk.CTkFrame(data_frame, fg_color="transparent")
        data_row.pack(fill="x", padx=6, pady=(0, 6))

        self.data_dir_entry = ctk.CTkEntry(
            data_row, placeholder_text="Path to data root (sub-XX/ses-XX/scenevideo…)", width=400
        )
        self.data_dir_entry.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            data_row, text="Browse", width=80, command=self._browse_data_dir
        ).pack(side="left", padx=(6, 0))

        # --- Derivatives directory ---
        deriv_frame = ctk.CTkFrame(self)
        deriv_frame.pack(fill="x", padx=30, pady=6)

        ctk.CTkLabel(
            deriv_frame, text="Derivatives directory (detections & annotations):",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=6, pady=(6, 2))

        deriv_row = ctk.CTkFrame(deriv_frame, fg_color="transparent")
        deriv_row.pack(fill="x", padx=6, pady=(0, 6))

        self.deriv_dir_entry = ctk.CTkEntry(
            deriv_row, placeholder_text="Path to derivatives root (sub-XX/ses-XX/…outputs)", width=400
        )
        self.deriv_dir_entry.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            deriv_row, text="Browse", width=80, command=self._browse_deriv_dir
        ).pack(side="left", padx=(6, 0))

        # --- Reviewer ---
        rev_frame = ctk.CTkFrame(self)
        rev_frame.pack(fill="x", padx=30, pady=6)

        ctk.CTkLabel(
            rev_frame, text="Reviewer:", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=6, pady=(6, 2))

        rev_row = ctk.CTkFrame(rev_frame, fg_color="transparent")
        rev_row.pack(fill="x", padx=6, pady=(0, 6))

        self.reviewer_var = ctk.StringVar(value="")
        self.reviewer_option = ctk.CTkOptionMenu(
            rev_row,
            variable=self.reviewer_var,
            values=["— select —"],
            width=280,
            command=self._on_reviewer_selected
        )
        self.reviewer_option.pack(side="left")

        ctk.CTkButton(
            rev_row, text="+ New", width=80, command=self._show_new_reviewer_panel
        ).pack(side="left", padx=(10, 0))

        self.new_rev_frame = ctk.CTkFrame(self)
        ctk.CTkLabel(
            self.new_rev_frame, text="New reviewer ID (no spaces):",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(10, 6), pady=10)
        self.new_id_entry = ctk.CTkEntry(self.new_rev_frame, width=200,
                                          placeholder_text="e.g. alice")
        self.new_id_entry.pack(side="left", padx=4, pady=10)
        ctk.CTkButton(
            self.new_rev_frame, text="Create", width=100, height=32,
            command=self._create_reviewer
        ).pack(side="left", padx=(10, 12), pady=10)

        self.status_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=13),
                                          text_color="orange")
        self.status_label.pack(pady=4)

        ctk.CTkButton(
            self,
            text="Continue",
            width=180,
            height=42,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._confirm
        ).pack(pady=16)

    # ------------------------------------------------------------------ #

    def _browse_data_dir(self):
        folder = filedialog.askdirectory(title="Select Data Root Directory")
        if not folder:
            return
        self.data_dir_entry.delete(0, "end")
        self.data_dir_entry.insert(0, folder)

    def _browse_deriv_dir(self):
        folder = filedialog.askdirectory(title="Select Derivatives Root Directory")
        if not folder:
            return
        self.deriv_dir_entry.delete(0, "end")
        self.deriv_dir_entry.insert(0, folder)
        self._refresh_reviewer_list(Path(folder))

    def _refresh_reviewer_list(self, derivatives_dir: Path):
        """Reload the reviewer dropdown from the derivatives registry."""
        try:
            registry = ReviewerRegistry(derivatives_dir)
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
        self.new_rev_frame.pack(fill="x", padx=30, pady=8)
        self.new_id_entry.focus()

    def _create_reviewer(self):
        deriv_dir_str = self.deriv_dir_entry.get().strip()
        if not deriv_dir_str:
            self.status_label.configure(text="Please select a derivatives directory first.")
            return

        deriv_dir = Path(deriv_dir_str)
        deriv_dir.mkdir(parents=True, exist_ok=True)

        raw_id = self.new_id_entry.get().strip()
        if not raw_id:
            self.status_label.configure(text="Reviewer ID cannot be empty.")
            return

        reviewer_id = ReviewerRegistry.sanitize_id(raw_id)
        registry = ReviewerRegistry(deriv_dir)
        if registry.reviewer_exists(reviewer_id):
            self.status_label.configure(text=f"Reviewer '{reviewer_id}' already exists.")
            return

        registry.add_reviewer(reviewer_id, reviewer_id)
        self.status_label.configure(
            text=f"Reviewer '{reviewer_id}' created.", text_color="#28a745"
        )
        self._refresh_reviewer_list(deriv_dir)
        self.reviewer_var.set(reviewer_id)
        self.new_rev_frame.pack_forget()

    def _load_last_values(self):
        """Pre-fill fields from last session."""
        last_data = self.settings.get("last_data_dir", "")
        if last_data and Path(last_data).exists():
            self.data_dir_entry.insert(0, last_data)

        last_deriv = self.settings.get("last_derivatives_dir", "")
        if last_deriv:
            self.deriv_dir_entry.insert(0, last_deriv)
            deriv_path = Path(last_deriv)
            if deriv_path.exists():
                self._refresh_reviewer_list(deriv_path)

        last_reviewer = self.settings.get("reviewer_id", "")
        if last_reviewer:
            ids = list(self.reviewer_option.cget("values"))
            if last_reviewer in ids:
                self.reviewer_var.set(last_reviewer)

    def _confirm(self):
        data_dir_str = self.data_dir_entry.get().strip()
        if not data_dir_str or not Path(data_dir_str).exists():
            self.status_label.configure(
                text="Please select a valid data directory.", text_color="orange"
            )
            return

        deriv_dir_str = self.deriv_dir_entry.get().strip()
        if not deriv_dir_str:
            self.status_label.configure(
                text="Please select a derivatives directory.", text_color="orange"
            )
            return

        reviewer_id = self.reviewer_var.get()
        if not reviewer_id or reviewer_id == "— select —":
            self.status_label.configure(
                text="Please select or create a reviewer.", text_color="orange"
            )
            return

        deriv_dir = Path(deriv_dir_str)
        deriv_dir.mkdir(parents=True, exist_ok=True)

        self.result_data_dir = Path(data_dir_str)
        self.result_derivatives_dir = deriv_dir
        self.result_reviewer_id = reviewer_id

        self.settings.set("last_data_dir", str(self.result_data_dir))
        self.settings.set("last_derivatives_dir", str(self.result_derivatives_dir))
        self.settings.set("reviewer_id", self.result_reviewer_id)
        self.settings.save_settings()

        self.grab_release()
        self.destroy()


class FaceDietApp(ctk.CTk):
    """Main application with tabbed interface."""

    def __init__(self,
                 data_dir: Optional[Path] = None,
                 derivatives_dir: Optional[Path] = None,
                 reviewer_id: Optional[str] = None,
                 settings: Optional[SettingsManager] = None):
        super().__init__()

        self.title("Face-Diet: Comprehensive Face Processing Pipeline")
        self.geometry("1600x1000")

        self.settings = settings or SettingsManager()
        self.restart_to_setup = False

        if data_dir is not None and derivatives_dir is not None and reviewer_id is not None:
            self.data_dir = Path(data_dir)
            self.derivatives_dir = Path(derivatives_dir)
            self.reviewer_id = str(reviewer_id)
        else:
            self.withdraw()
            dialog = StartupDialog(self, self.settings)
            self.wait_window(dialog)

            result_data_dir = getattr(dialog, "result_data_dir", None)
            result_derivatives_dir = getattr(dialog, "result_derivatives_dir", None)
            result_reviewer_id = getattr(dialog, "result_reviewer_id", None)

            if result_data_dir is None:
                self.quit()
                self.destroy()
                return

            self.data_dir = Path(result_data_dir)
            self.derivatives_dir = Path(result_derivatives_dir)
            self.reviewer_id = str(result_reviewer_id or "")

            self.deiconify()
            self.update_idletasks()
            self.lift()
            self.focus_force()

        # Top bar
        self.top_bar = ctk.CTkFrame(self, fg_color=("gray90", "gray17"))
        self.top_bar.pack(fill="x", padx=10, pady=(10, 0))

        ctk.CTkLabel(
            self.top_bar, text="Data:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=(12, 4), pady=8)
        self.top_bar_data_label = ctk.CTkLabel(
            self.top_bar,
            text=str(self.data_dir),
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.top_bar_data_label.pack(side="left", padx=(0, 16), pady=8)

        ctk.CTkLabel(
            self.top_bar, text="Derivatives:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=(4, 4), pady=8)
        self.top_bar_deriv_label = ctk.CTkLabel(
            self.top_bar,
            text=str(self.derivatives_dir),
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.top_bar_deriv_label.pack(side="left", padx=(0, 16), pady=8)

        ctk.CTkLabel(
            self.top_bar, text="Reviewer:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=(4, 4), pady=8)
        self.top_bar_reviewer_label = ctk.CTkLabel(
            self.top_bar,
            text=self.reviewer_id or "—",
            font=ctk.CTkFont(size=12),
            text_color="#3b8ed0"
        )
        self.top_bar_reviewer_label.pack(side="left", padx=(0, 20), pady=8)

        ctk.CTkButton(
            self.top_bar,
            text="Back to setup",
            command=self._on_back_to_setup,
            width=120,
            height=28,
            font=ctk.CTkFont(size=12)
        ).pack(side="right", padx=10, pady=6)

        # Tabs
        self.tabview = ctk.CTkTabview(self, width=1580, height=980)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        self.tabview.add("Face Detection")
        self.tabview.add("Face Instance Review")
        self.tabview.add("Resolve Mismatches")
        self.tabview.add("Face ID Clustering")
        self.tabview.add("Face ID Review")

        self.tab1 = VideoProcessingTab(
            self.tabview.tab("Face Detection"),
            self.settings, self.data_dir, self.derivatives_dir, self.reviewer_id
        )
        self.tab1.pack(fill="both", expand=True)

        self.tab2 = FaceInstanceReviewTab(
            self.tabview.tab("Face Instance Review"),
            self.settings, self.data_dir, self.derivatives_dir, self.reviewer_id
        )
        self.tab2.pack(fill="both", expand=True)

        self.tab_mismatch = MismatchResolutionTab(
            self.tabview.tab("Resolve Mismatches"),
            self.settings, self.data_dir, self.derivatives_dir, self.reviewer_id
        )
        self.tab_mismatch.pack(fill="both", expand=True)

        self.tab3 = FaceIDAssignmentTab(
            self.tabview.tab("Face ID Clustering"),
            self.settings, self.data_dir, self.derivatives_dir, self.reviewer_id
        )
        self.tab3.pack(fill="both", expand=True)

        self.tab4 = ManualReviewTab(
            self.tabview.tab("Face ID Review"),
            self.settings, self.data_dir, self.derivatives_dir, self.reviewer_id
        )
        self.tab4.pack(fill="both", expand=True)

        self.tab_mismatch.bind("<Map>", self._on_resolve_mismatches_tab_shown)
        self.tab3.bind("<Map>", self._on_face_id_clustering_tab_shown)
        self.tab4.bind("<Map>", self._on_face_id_review_tab_shown)

    def _on_resolve_mismatches_tab_shown(self, event=None):
        if hasattr(self.tab_mismatch, "_load_session_list"):
            self.tab_mismatch._load_session_list()

    def _on_face_id_clustering_tab_shown(self, event=None):
        if hasattr(self.tab3, "_load_participants_and_sessions"):
            self.tab3._load_participants_and_sessions()

    def _on_face_id_review_tab_shown(self, event=None):
        if hasattr(self.tab4, "_populate_participants_tab4"):
            self.tab4._populate_participants_tab4()

    def _on_back_to_setup(self):
        self.restart_to_setup = True
        self.destroy()


def main():
    """Main entry point."""
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    settings = SettingsManager()
    while True:
        dialog_root = ctk.CTk()
        dialog_root.withdraw()
        dialog = StartupDialog(dialog_root, settings)
        dialog_root.wait_window(dialog)

        result_data_dir = getattr(dialog, "result_data_dir", None)
        result_derivatives_dir = getattr(dialog, "result_derivatives_dir", None)
        result_reviewer_id = getattr(dialog, "result_reviewer_id", None)

        try:
            ids_str = dialog_root.tk.eval("after info")
            for id_str in ids_str.split():
                try:
                    dialog_root.after_cancel(id_str)
                except (tkinter.TclError, ValueError):
                    pass
        except Exception:
            pass
        dialog_root.destroy()

        if result_data_dir is None:
            return

        try:
            app = FaceDietApp(
                data_dir=result_data_dir,
                derivatives_dir=result_derivatives_dir,
                reviewer_id=result_reviewer_id,
                settings=settings,
            )
        except Exception:
            import traceback
            print("Face-Diet failed to start:", file=sys.stderr)
            traceback.print_exc()
            raise
        app.mainloop()
        if not getattr(app, "restart_to_setup", False):
            break


__all__ = ["StartupDialog", "FaceDietApp", "main"]
