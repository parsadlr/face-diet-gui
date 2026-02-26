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
    Modal dialog shown at startup to select / create a reviewer and project directory.
    Blocks the main window until confirmed.
    """

    def __init__(self, master, settings: SettingsManager):
        super().__init__(master)
        self.settings = settings
        self.result_project_dir: Optional[Path] = None
        self.result_reviewer_id: Optional[str] = None

        self.title("Face-Diet — Setup")
        self.geometry("720x580")
        self.minsize(620, 520)
        self.resizable(True, True)
        self.grab_set()   # make modal
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
            text="Select your project directory and reviewer identity to begin.",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(pady=(0, 20))

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

        self.status_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=11),
                                          text_color="orange")
        self.status_label.pack(pady=4)

        ctk.CTkButton(
            self,
            text="Continue →",
            width=180,
            height=42,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._confirm
        ).pack(pady=16)

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
        self.new_rev_frame.pack(fill="x", padx=30, pady=8)
        self.new_id_entry.focus()

    def _create_reviewer(self):
        project_dir_str = self.dir_entry.get().strip()
        if not project_dir_str or not Path(project_dir_str).exists():
            self.status_label.configure(text="Please select a valid project directory first.")
            return

        raw_id = self.new_id_entry.get().strip()
        if not raw_id:
            self.status_label.configure(text="Reviewer ID cannot be empty.")
            return

        reviewer_id = ReviewerRegistry.sanitize_id(raw_id)

        registry = ReviewerRegistry(Path(project_dir_str))
        if registry.reviewer_exists(reviewer_id):
            self.status_label.configure(text=f"Reviewer '{reviewer_id}' already exists.")
            return

        registry.add_reviewer(reviewer_id, reviewer_id)
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

        self.settings.set("last_project_dir", str(self.result_project_dir))
        self.settings.set("reviewer_id", self.result_reviewer_id)
        self.settings.save_settings()

        self.grab_release()
        self.destroy()


class FaceDietApp(ctk.CTk):
    """Main application with tabbed interface."""

    def __init__(self, project_dir: Optional[Path] = None, reviewer_id: Optional[str] = None,
                 settings: Optional[SettingsManager] = None):
        super().__init__()

        self.title("Face-Diet: Comprehensive Face Processing Pipeline")
        self.geometry("1600x1000")

        self.settings = settings or SettingsManager()
        self.restart_to_setup = False

        if project_dir is not None and reviewer_id is not None:
            self.project_dir = Path(project_dir)
            self.reviewer_id = str(reviewer_id)
        else:
            self.withdraw()
            dialog = StartupDialog(self, self.settings)
            self.wait_window(dialog)

            result_project_dir = getattr(dialog, "result_project_dir", None)
            result_reviewer_id = getattr(dialog, "result_reviewer_id", None)

            if result_project_dir is None:
                self.quit()
                self.destroy()
                return

            self.project_dir = Path(result_project_dir)
            self.reviewer_id = str(result_reviewer_id or "")

            self.deiconify()
            self.update_idletasks()
            self.lift()
            self.focus_force()

        self.top_bar = ctk.CTkFrame(self, fg_color=("gray90", "gray17"))
        self.top_bar.pack(fill="x", padx=10, pady=(10, 0))
        ctk.CTkLabel(
            self.top_bar,
            text="Project:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=(12, 4), pady=8)
        self.top_bar_dir_label = ctk.CTkLabel(
            self.top_bar,
            text=str(self.project_dir) if self.project_dir else "—",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.top_bar_dir_label.pack(side="left", padx=(0, 20), pady=8)
        ctk.CTkLabel(
            self.top_bar,
            text="Reviewer:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=(10, 4), pady=8)
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

        self.tabview = ctk.CTkTabview(self, width=1580, height=980)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        self.tabview.add("Face Detection")
        self.tabview.add("Face Instance Review")
        self.tabview.add("Resolve Mismatches")
        self.tabview.add("Face ID Clustering")
        self.tabview.add("Face ID Review")

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

        self.tab_mismatch = MismatchResolutionTab(
            self.tabview.tab("Resolve Mismatches"),
            self.settings, self.project_dir, self.reviewer_id
        )
        self.tab_mismatch.pack(fill="both", expand=True)

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

        result_project_dir = getattr(dialog, "result_project_dir", None)
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

        if result_project_dir is None:
            return

        try:
            app = FaceDietApp(project_dir=result_project_dir, reviewer_id=result_reviewer_id, settings=settings)
        except Exception:
            import traceback
            print("Face-Diet failed to start:", file=sys.stderr)
            traceback.print_exc()
            raise
        app.mainloop()
        if not getattr(app, "restart_to_setup", False):
            break


__all__ = ["StartupDialog", "FaceDietApp", "main"]
