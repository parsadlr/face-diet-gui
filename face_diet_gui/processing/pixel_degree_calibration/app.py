"""
Calibration mini-GUI — main window.

Run with:
    python -m face_diet_gui.processing.pixel_degree_calibration

Two tabs:
    Segment Targets  – click each calibration frame to segment the target
    Fit Mapping      – compute samples from masks and fit the PPD mapping
"""
import customtkinter as ctk


def main() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = CalibrationApp()
    app.mainloop()


class CalibrationApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Pixel-Degree Calibration")
        self.geometry("1280x820")
        self.minsize(900, 600)

        # Imports are deferred so the module loads fast without heavy deps
        from face_diet_gui.processing.pixel_degree_calibration.tab_segment import SegmentTab
        from face_diet_gui.processing.pixel_degree_calibration.tab_fit import FitTab

        # ---- header ----
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(14, 0))

        ctk.CTkLabel(
            header,
            text="Pixel-Degree Calibration",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(side="left")

        ctk.CTkLabel(
            header,
            text="Tobii glasses · FOV calibration tool",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        ).pack(side="left", padx=(12, 0))

        # ---- tab view ----
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=(10, 14))

        seg_frame = self.tabview.add("  Segment Targets  ")
        fit_frame = self.tabview.add("  Fit Mapping  ")

        self.seg_tab = SegmentTab(seg_frame)
        self.seg_tab.pack(fill="both", expand=True)

        self.fit_tab = FitTab(fit_frame, seg_tab=self.seg_tab)
        self.fit_tab.pack(fill="both", expand=True)

        # Pre-fill masks dir in fit tab whenever user switches to it
        self.tabview.configure(command=self._on_tab_change)

    def _on_tab_change(self) -> None:
        selected = self.tabview.get()
        if "Fit" in selected:
            self.fit_tab.sync_masks_dir_from_seg()
