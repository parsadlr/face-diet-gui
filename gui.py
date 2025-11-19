import os
from tkinter import Canvas, filedialog, messagebox, ttk
from typing import List, Optional, Tuple

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from db import initialize_schema, upsert_image, insert_faces
from face_utils import (
    initialize_detector,
    detect_faces,
    draw_boxes,
    extract_landmarks,
    draw_landmarks_colored,
    extract_poses,
)


class FaceGUI:
    def __init__(self, master: ctk.CTk) -> None:
        self.master = master
        master.title("Face-Diet: Face Detector")
        
        # Custom colors matching HTML design
        primary_color = "#007ACC"
        
        # Main container
        self.main_frame = ctk.CTkFrame(master, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=16, pady=16)

        # Canvas for image display (using tkinter Canvas as CustomTkinter doesn't have one)
        self.canvas = Canvas(self.main_frame, width=800, height=600, bg="#2c2c2c", highlightthickness=0)
        self.canvas.pack(pady=(0, 12))
        self.canvas.bind("<Motion>", self.on_mouse_move)

        # Button container
        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.button_frame.pack(fill="x", pady=(0, 8))

        self.open_button = ctk.CTkButton(
            self.button_frame, 
            text="Open Image", 
            command=self.open_image,
            width=140,
            height=40,
            corner_radius=8,
            fg_color="#007ACC",
            hover_color="#0066AA",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.open_button.pack(side="left", padx=8)

        self.detect_button = ctk.CTkButton(
            self.button_frame,
            text="Detect Faces",
            command=self.run_detection,
            state="disabled",
            width=140,
            height=40,
            corner_radius=8,
            fg_color="#007ACC",
            hover_color="#0066AA",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.detect_button.pack(side="left", padx=8)

        self.landmarks_button = ctk.CTkButton(
            self.button_frame,
            text="Landmarks",
            command=self.show_landmarks,
            state="disabled",
            width=140,
            height=40,
            corner_radius=8,
            fg_color="#3c3c3c",
            hover_color="#4c4c4c",
            text_color="#E0E0E0",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.landmarks_button.pack(side="left", padx=8)

        self.save_button = ctk.CTkButton(
            self.button_frame,
            text="Save to DB",
            command=self.save_to_db,
            state="disabled",
            width=140,
            height=40,
            corner_radius=8,
            fg_color="#3c3c3c",
            hover_color="#4c4c4c",
            text_color="#E0E0E0",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.save_button.pack(side="left", padx=8)

        # Status label
        self.status = ctk.CTkLabel(
            self.main_frame,
            text="Ready",
            anchor="w",
            font=ctk.CTkFont(size=13),
            text_color="#AAAAAA"
        )
        self.status.pack(fill="x", pady=(0, 8))

        self.detector = initialize_detector(use_gpu=False)
        initialize_schema()

        self.image_bgr: Optional[np.ndarray] = None
        self.image_path: Optional[str] = None
        self.display_image_tk: Optional[ImageTk.PhotoImage] = None
        self.boxes: List[Tuple[int, int, int, int]] = []
        self.scores: List[float] = []
        self.face_infos: List[dict] = []
        self.poses: List[Optional[dict]] = []
        # attributes removed for insightface-only mode
        self.display_scale: float = 1.0
        self.display_offset_x: int = 0
        self.display_offset_y: int = 0
        self._tooltip_items: List[int] = []

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)
        self.master.update_idletasks()

    def open_image(self) -> None:
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Open image", filetypes=filetypes)
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Could not read image: {path}")
            return
        self.image_bgr = img
        self.image_path = path
        self.boxes = []
        self.scores = []
        self.poses = []
        self.attributes = []
        self._render_image(img)
        self.detect_button.configure(state="normal")
        self.landmarks_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.face_infos = []
        self._clear_tooltip()
        self.set_status(f"Loaded {os.path.basename(path)}")

    def run_detection(self) -> None:
        if self.image_bgr is None:
            return
        self.set_status("Running detection...")
        boxes, scores, _ = detect_faces(self.detector, self.image_bgr, return_embeddings=False)
        self.boxes = boxes
        self.scores = scores
        self.poses = extract_poses(self.detector, self.image_bgr)
        self._generate_demo_infos()
        vis = draw_boxes(self.image_bgr, boxes)
        self._render_image(vis)
        self.set_status(f"Detected {len(boxes)} face(s)")
        self.landmarks_button.configure(state="normal")
        self.save_button.configure(state="normal")

    def show_landmarks(self) -> None:
        if self.image_bgr is None:
            return
        self.set_status("Extracting landmarks...")
        landmarks_list = extract_landmarks(self.detector, self.image_bgr)
        vis = draw_landmarks_colored(self.image_bgr, landmarks_list)
        # If boxes are already detected, overlay them as well for context
        if self.boxes:
            vis = draw_boxes(vis, self.boxes)
        self._render_image(vis)
        self.set_status(f"Plotted landmarks for {len(landmarks_list)} face(s)")

    def save_to_db(self) -> None:
        if self.image_bgr is None or self.image_path is None:
            return
        h, w = self.image_bgr.shape[:2]
        try:
            image_id = upsert_image(self.image_path, width=int(w), height=int(h))
            insert_faces(image_id, self.boxes, self.scores)
            messagebox.showinfo("Saved", f"Saved {len(self.boxes)} face(s) for image id {image_id}")
            self.set_status("Saved to DB")
        except Exception as e:
            messagebox.showerror("DB Error", str(e))

    def _render_image(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        max_w, max_h = 800, 600
        scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.display_image_tk = ImageTk.PhotoImage(image=image)
        self.canvas.delete("all")
        ox = max(0, (max_w - new_w) // 2)
        oy = max(0, (max_h - new_h) // 2)
        self.canvas.create_image(ox, oy, anchor="nw", image=self.display_image_tk)
        self.display_scale = scale
        self.display_offset_x = ox
        self.display_offset_y = oy
        self._clear_tooltip()

    def _generate_demo_infos(self) -> None:
        """Generate initial face info with just detection data."""
        import random
        self.face_infos = []
        for i, (_x, _y, _w, _h) in enumerate(self.boxes):
            pose = self.poses[i] if i < len(self.poses) and self.poses[i] else None
            info = {
                "face_id": f"F-{i+1:03d}",
                "score": self.scores[i] if i < len(self.scores) else 0.0,
                "pose": pose,
                "has_attributes": False,
            }
            self.face_infos.append(info)
    
    def _update_face_infos(self) -> None:
        """Update face infos with analyzed attributes."""
        for i in range(len(self.face_infos)):
            if i < len(self.attributes) and self.attributes[i] is not None:
                attr = self.attributes[i]
                self.face_infos[i].update({
                    "age": attr.get("age", "Unknown"),
                    "gender": attr.get("gender", "Unknown"),
                    "emotion": attr.get("emotion", "Unknown"),
                    "race": attr.get("race", "Unknown"),
                    "has_attributes": True,
                })
            else:
                self.face_infos[i]["has_attributes"] = False

    def _image_coords_from_canvas(self, mx: int, my: int) -> Optional[Tuple[int, int]]:
        ix = mx - self.display_offset_x
        iy = my - self.display_offset_y
        if ix < 0 or iy < 0:
            return None
        if self.display_scale <= 0:
            return None
        return int(ix / self.display_scale), int(iy / self.display_scale)

    def _hover_face_index(self, mx: int, my: int) -> int:
        mapped = self._image_coords_from_canvas(mx, my)
        if not mapped:
            return -1
        px, py = mapped
        for idx, (x, y, w, h) in enumerate(self.boxes):
            if x <= px <= x + w and y <= py <= y + h:
                return idx
        return -1

    def _clear_tooltip(self) -> None:
        if self._tooltip_items:
            for item in self._tooltip_items:
                try:
                    self.canvas.delete(item)
                except Exception:
                    pass
            self._tooltip_items = []

    def _show_tooltip(self, mx: int, my: int, lines: List[str]) -> None:
        self._clear_tooltip()
        text = "\n".join(lines)
        pad = 6
        # Create text first to measure bbox, then draw a rectangle behind
        text_id = self.canvas.create_text(mx + 12, my + 12, text=text, anchor="nw", fill="#ffffff", font=("TkDefaultFont", 9))
        bbox = self.canvas.bbox(text_id)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            rect_id = self.canvas.create_rectangle(x1 - pad, y1 - pad, x2 + pad, y2 + pad, fill="#000000", outline="#cccccc")
            # Raise text above rect
            self.canvas.tag_raise(text_id, rect_id)
            self._tooltip_items = [rect_id, text_id]
        else:
            self._tooltip_items = [text_id]

    def on_mouse_move(self, event) -> None:
        if not self.boxes:
            self._clear_tooltip()
            return
        idx = self._hover_face_index(event.x, event.y)
        if idx < 0:
            self._clear_tooltip()
            return
        info = self.face_infos[idx] if 0 <= idx < len(self.face_infos) else None
        if info is None:
            self._clear_tooltip()
            return
        
        # Build tooltip lines
        lines = [f"Face ID: {info.get('face_id', 'N/A')}"]
        
        if info.get("has_attributes", False):
            lines.append(f"Age: {info.get('age', 'N/A')}")
            lines.append(f"Gender: {info.get('gender', 'N/A')}")
            lines.append(f"Emotion: {info.get('emotion', 'N/A')}")
            lines.append(f"Ethnicity: {info.get('race', 'N/A')}")
        else:
            lines.append("Score: {:.3f}".format(info.get('score', 0.0)))
            lines.append("(Click 'Analyze Attributes' for details)")
        
        pose = info.get('pose')
        if pose:
            # Display angles with correct axis labels
            lines.append(f"Pose: yaw={pose['yaw']:.1f}°, pitch={pose['pitch']:.1f}°, roll={pose['roll']:.1f}°")
        
        self._show_tooltip(event.x, event.y, lines)


def main() -> None:
    # Set appearance mode and default theme
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    root = ctk.CTk()
    root.geometry("900x750")
    root.configure(fg_color="#1a1a1a")  # Background color from HTML
    
    gui = FaceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()


