"""
Face-Diet GUI: Face ID Review Tool

A GUI for manually reviewing and merging face IDs detected by the clustering algorithm.
Allows loading a participant directory, viewing face IDs with representative images,
and merging IDs that belong to the same person.
"""

import os
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import customtkinter as ctk
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox


class FaceIDReviewGUI:
    """Main GUI for reviewing and merging face IDs."""
    
    def __init__(self, master: ctk.CTk):
        self.master = master
        master.title("Face-Diet: Face ID Review")
        master.geometry("1400x900")
        
        # Data storage
        self.participant_dir: Optional[Path] = None
        self.df: Optional[pd.DataFrame] = None
        self.df_full: Optional[pd.DataFrame] = None
        self.face_groups: Dict = {}  # {face_id: {'count': int, 'representative': dict, 'thumbnail': Image}}
        self.face_groups_all: Dict = {}  # Keep all groups before filtering
        self.merge_groups: Dict[str, List[str]] = {}  # {merged_id: [original_ids]}
        self.face_id_to_merged: Dict[str, str] = {}  # {original_id: merged_id}
        self.session_bbox_stats: Dict[str, Dict[str, float]] = {}  # {session_name: stats}
        self.representative_sample_size = 30
        
        # UI components
        self.row_widgets: Dict[str, Dict] = {}  # {display_key: {'frame': Frame, 'checkbox': Checkbox, ...}}
        self.selected_ids: set = set()
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the main UI structure."""
        # Main container with padding
        main_container = ctk.CTkFrame(self.master, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            main_container,
            text="Face ID Review & Merging Tool",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Folder selection panel
        self._create_folder_selection_panel(main_container)
        
        # Progress bar (initially hidden)
        self.progress_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="Loading...",
            font=ctk.CTkFont(size=12)
        )
        self.progress_label.pack()
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, width=400)
        self.progress_bar.pack(pady=5)
        self.progress_bar.set(0)
        
        # Merge controls
        self._create_merge_controls(main_container)
        
        # Face ID list (scrollable)
        self._create_face_list(main_container)
        
        # Save button
        self._create_save_button(main_container)
    
    def _create_folder_selection_panel(self, parent):
        """Create the folder selection panel."""
        panel = ctk.CTkFrame(parent)
        panel.pack(fill="x", pady=(0, 15))
        
        # Label
        label = ctk.CTkLabel(
            panel,
            text="Participant Folder:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        label.pack(side="left", padx=(10, 10))
        
        # Path display
        self.path_entry = ctk.CTkEntry(
            panel,
            placeholder_text="Select participant directory containing faces_combined.csv",
            width=500,
            state="readonly"
        )
        self.path_entry.pack(side="left", padx=(0, 10))
        
        # Browse button
        browse_btn = ctk.CTkButton(
            panel,
            text="Browse",
            command=self._browse_folder,
            width=120,
            height=35,
            font=ctk.CTkFont(size=13, weight="bold")
        )
        browse_btn.pack(side="left", padx=(0, 10))
        
        # Min instances filter
        ctk.CTkLabel(
            panel,
            text="Min Instances:",
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=(20, 5))
        
        self.min_instances_var = ctk.IntVar(value=1)
        self.min_instances_spinbox = ctk.CTkEntry(
            panel,
            width=60,
            textvariable=self.min_instances_var
        )
        self.min_instances_spinbox.pack(side="left", padx=(0, 5))
        
        # Min confidence filter
        ctk.CTkLabel(
            panel,
            text="Min Confidence:",
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=(15, 5))

        self.min_confidence_var = ctk.DoubleVar(value=0.0)
        self.min_confidence_entry = ctk.CTkEntry(
            panel,
            width=70,
            textvariable=self.min_confidence_var
        )
        self.min_confidence_entry.pack(side="left", padx=(0, 5))

        # Review button (load + filter)
        self.review_btn = ctk.CTkButton(
            panel,
            text="Review",
            command=self._review_with_filters,
            width=80,
            height=35,
            font=ctk.CTkFont(size=13),
            state="disabled"
        )
        self.review_btn.pack(side="left", padx=(0, 10))
    
    def _create_merge_controls(self, parent):
        """Create merge/unmerge control buttons."""
        controls = ctk.CTkFrame(parent)
        controls.pack(fill="x", pady=(10, 10))
        
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
    
    def _create_face_list(self, parent):
        """Create the scrollable face ID list."""
        # Container with border
        list_container = ctk.CTkFrame(parent)
        list_container.pack(fill="both", expand=True, pady=(0, 15))
        
        # Header
        header = ctk.CTkFrame(list_container, height=40)
        header.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(header, text="Select", width=60, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)
        ctk.CTkLabel(header, text="Face Preview", width=100, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)
        ctk.CTkLabel(header, text="Face ID(s)", width=300, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)
        ctk.CTkLabel(header, text="Instances", width=100, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)
        ctk.CTkLabel(header, text="Actions", width=120, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)
        
        # Scrollable frame for face list
        self.face_list_frame = ctk.CTkScrollableFrame(list_container, height=500)
        self.face_list_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))
    
    def _create_save_button(self, parent):
        """Create the save button."""
        save_frame = ctk.CTkFrame(parent, fg_color="transparent")
        save_frame.pack(fill="x")
        
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
        
        # Check for session folders
        session_dirs = [d for d in self.participant_dir.iterdir() if d.is_dir()]
        if not session_dirs:
            messagebox.showwarning(
                "Warning",
                "No session directories found. Face thumbnails may not be available."
            )
        
        # Load data in background thread
        self.progress_frame.pack(pady=10)
        self.merge_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        
        thread = threading.Thread(target=self._load_data_thread, args=(csv_path,))
        thread.daemon = True
        thread.start()
    
    def _load_data_thread(self, csv_path: Path):
        """Load and process data in background thread."""
        try:
            print("\n" + "="*80)
            print("LOADING FACE DATA")
            print("="*80)
            
            # Update progress
            print(f"[1/6] Loading CSV: {csv_path}")
            self._update_progress("Loading CSV...", 0.1)
            
            # Load CSV
            self.df_full = pd.read_csv(csv_path)
            self.df = self.df_full.copy()
            print(f"      ✓ Loaded {len(self.df)} rows")
            
            # Validate required columns
            required_cols = ['face_id', 'session_name', 'frame_number', 'x', 'y', 'w', 'h', 'embedding']
            print(f"\n[2/6] Validating columns...")
            print(f"      Required: {', '.join(required_cols)}")
            print(f"      Found: {', '.join(self.df.columns)}")
            
            missing = [col for col in required_cols if col not in self.df.columns]
            if missing:
                error_msg = f"Missing required columns: {', '.join(missing)}"
                print(f"      ✗ ERROR: {error_msg}")
                self.master.after(0, lambda: messagebox.showerror("Error", error_msg))
                self.master.after(0, lambda: self.progress_frame.pack_forget())
                return
            print(f"      ✓ All required columns present")
            
            # Initialize merge tracking for ALL IDs (even if filtered out)
            all_face_ids = self.df['face_id'].unique()
            self.face_id_to_merged = {fid: fid for fid in all_face_ids}

            # Apply confidence filter (if available)
            min_conf = float(self.min_confidence_var.get())
            if 'confidence' in self.df.columns and min_conf > 0.0:
                before = len(self.df)
                self.df = self.df[self.df['confidence'] >= min_conf].reset_index(drop=True)
                print(f"\n[2.5/6] Applied min confidence filter: {min_conf}")
                print(f"      Rows: {before} → {len(self.df)}")
            else:
                print(f"\n[2.5/6] Min confidence filter skipped (confidence column missing or 0.0)")

            # Compute counts and apply min instances BEFORE any heavy processing
            min_instances = int(self.min_instances_var.get())
            face_counts = self.df['face_id'].value_counts()
            eligible_ids = face_counts[face_counts >= min_instances].index.tolist()
            print(f"\n[2.7/6] Applied min instances filter: {min_instances}")
            print(f"      IDs: {len(face_counts)} → {len(eligible_ids)}")

            if not eligible_ids:
                print("      ⚠️  No face IDs meet the filter criteria.")
                self.master.after(0, lambda: messagebox.showwarning(
                    "No Results",
                    "No face IDs meet the selected filters. Try lowering thresholds."
                ))
                self.master.after(0, lambda: self.progress_frame.pack_forget())
                return

            # Group by face_id (eligible only)
            print(f"\n[3/6] Grouping face instances...")
            self._update_progress("Grouping face instances...", 0.2)
            face_ids = eligible_ids
            total_ids = len(face_ids)
            print(f"      Processing {total_ids} eligible face IDs")
            
            # Compute per-session bbox extent (for potential scaling)
            print(f"\n[3.5/6] Computing per-session bbox extents...")
            self.session_bbox_stats = {}
            for session_name, session_df in self.df.groupby('session_name'):
                max_x2 = float((session_df['x'] + session_df['w']).max())
                max_y2 = float((session_df['y'] + session_df['h']).max())
                min_x = float(session_df['x'].min())
                min_y = float(session_df['y'].min())
                self.session_bbox_stats[session_name] = {
                    'max_x2': max_x2,
                    'max_y2': max_y2,
                    'min_x': min_x,
                    'min_y': min_y,
                }
                print(f"      {session_name}: inferred size ≈ {int(max_x2)}x{int(max_y2)} (min x,y={int(min_x)},{int(min_y)})")
            
            self.face_groups = {}
            
            print(f"\n[4/6] Computing representatives and extracting thumbnails...")
            
            # Sample some IDs to debug
            debug_indices = [0, 1, 2, len(face_ids)//2, len(face_ids)-1] if len(face_ids) > 5 else range(len(face_ids))
            
            for idx, face_id in enumerate(face_ids):
                # Update progress periodically
                if idx % 10 == 0:
                    progress = 0.2 + (0.6 * idx / total_ids)
                    self._update_progress(f"Processing face {idx+1}/{total_ids}...", progress)
                    print(f"      [{idx+1}/{total_ids}] Processing {face_id}...", end='\r')
                
                face_instances = self.df[self.df['face_id'] == face_id]
                count = len(face_instances)
                
                # Find representative face
                representative = self._find_representative_face(face_instances)
                
                # Extract thumbnail (debug select faces)
                thumbnail = None
                if representative:
                    debug_mode = (idx in debug_indices)
                    thumbnail = self._extract_face_crop(representative, debug=debug_mode)
                
                self.face_groups[face_id] = {
                    'count': count,
                    'representative': representative,
                    'thumbnail': thumbnail,
                    'original_ids': [face_id]  # Track original IDs for merging
                }
                
                # Initialize merge tracking
                self.face_id_to_merged[face_id] = face_id
            
            print(f"\n      ✓ Processed {total_ids} face IDs")
            
            # Sort by count (descending)
            print(f"\n[5/6] Sorting by instance count...")
            self.face_groups = dict(
                sorted(self.face_groups.items(), key=lambda x: x[1]['count'], reverse=True)
            )
            print(f"      ✓ Sorted")
            
            # Display results
            print(f"\n[6/6] Rendering UI...")
            self._update_progress("Rendering UI...", 0.9)
            self.master.after(0, self._display_face_list)
            
            # Store all groups for filtering (already filtered)
            self.face_groups_all = self.face_groups.copy()
            
            # Hide progress and enable controls
            self.master.after(0, lambda: self.progress_frame.pack_forget())
            self.master.after(0, lambda: self.save_btn.configure(state="normal"))
            self.master.after(0, lambda: self.review_btn.configure(state="normal"))
            
            print(f"      ✓ UI rendered")
            print("\n" + "="*80)
            print("LOADING COMPLETE")
            print("="*80)
            print(f"Total face IDs: {len(self.face_groups)}")
            print(f"Total face instances: {len(self.df)}")
            print("="*80 + "\n")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\n✗ ERROR during data loading:")
            print(error_details)
            
            self.master.after(0, lambda: messagebox.showerror(
                "Error",
                f"Failed to load data:\n{str(e)}\n\nCheck terminal for details."
            ))
            self.master.after(0, lambda: self.progress_frame.pack_forget())
    
    def _update_progress(self, text: str, value: float):
        """Update progress bar and label."""
        self.master.after(0, lambda: self.progress_label.configure(text=text))
        self.master.after(0, lambda: self.progress_bar.set(value))
    
    def _find_representative_face(self, face_instances: pd.DataFrame) -> Optional[dict]:
        """Find the face closest to centroid while also confident and sharp."""
        try:
            if face_instances.empty:
                return None
            df = face_instances.copy()

            # Build candidate pool using confidence+sharpness (fast)
            if 'confidence' in df.columns and 'sharpness' in df.columns:
                conf = df['confidence'].astype(float)
                sharp = df['sharpness'].astype(float)
                conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
                sharp_norm = (sharp - sharp.min()) / (sharp.max() - sharp.min() + 1e-8)
                df['rep_score'] = conf_norm * sharp_norm
                candidate_df = df.nlargest(self.representative_sample_size, 'rep_score')
            elif 'confidence' in df.columns:
                candidate_df = df.nlargest(self.representative_sample_size, 'confidence')
            elif 'sharpness' in df.columns:
                candidate_df = df.nlargest(self.representative_sample_size, 'sharpness')
            else:
                candidate_df = df.head(self.representative_sample_size)

            # Parse embeddings only for candidates (fast)
            embeddings = []
            row_indices = []
            for idx, row in candidate_df.iterrows():
                embedding_str = row.get('embedding', '')
                if embedding_str and not pd.isna(embedding_str):
                    try:
                        embedding = np.array(json.loads(embedding_str), dtype=np.float32)
                        embeddings.append(embedding)
                        row_indices.append(idx)
                    except Exception:
                        continue

            if embeddings:
                emb = np.array(embeddings)
                centroid = emb.mean(axis=0)
                emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
                centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
                similarity = np.dot(emb_norm, centroid_norm)

                # Combine centroid similarity with confidence+sharpness (if available)
                if 'confidence' in candidate_df.columns and 'sharpness' in candidate_df.columns:
                    conf = candidate_df.loc[row_indices, 'confidence'].astype(float)
                    sharp = candidate_df.loc[row_indices, 'sharpness'].astype(float)
                    conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
                    sharp_norm = (sharp - sharp.min()) / (sharp.max() - sharp.min() + 1e-8)
                    combined = similarity * conf_norm * sharp_norm
                else:
                    combined = similarity

                best_local = int(np.argmax(combined))
                best_idx = row_indices[best_local]
                return df.loc[best_idx].to_dict()

            # Fallback: confidence + sharpness
            if 'confidence' in df.columns and 'sharpness' in df.columns:
                conf = df['confidence'].astype(float)
                sharp = df['sharpness'].astype(float)
                conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
                sharp_norm = (sharp - sharp.min()) / (sharp.max() - sharp.min() + 1e-8)
                df['rep_score'] = conf_norm * sharp_norm
                best_idx = df['rep_score'].idxmax()
                return df.loc[best_idx].to_dict()

            if 'confidence' in df.columns:
                return df.sort_values(['confidence'], ascending=False).iloc[0].to_dict()
            if 'sharpness' in df.columns:
                return df.sort_values(['sharpness'], ascending=False).iloc[0].to_dict()

            return df.iloc[0].to_dict()
        except Exception as e:
            print(f"        Error in _find_representative_face: {e}")
            import traceback
            traceback.print_exc()
            return face_instances.iloc[0].to_dict() if len(face_instances) > 0 else None
    
    def _extract_face_crop(self, face_info: dict, debug: bool = False) -> Optional[Image.Image]:
        """Extract face crop from video frame."""
        try:
            session_name = face_info['session_name']
            frame_number = int(face_info['frame_number'])
            time_seconds = face_info.get('time_seconds', None)
            
            # Read coordinates as floats first to check if they need conversion
            x_raw = face_info['x']
            y_raw = face_info['y']
            w_raw = face_info['w']
            h_raw = face_info['h']
            
            x, y, w, h = int(x_raw), int(y_raw), int(w_raw), int(h_raw)
            
            if debug:
                print(f"\n        [DEBUG] Extracting crop:")
                print(f"          Session: {session_name}")
                print(f"          Frame: {frame_number}")
                print(f"          BBox (raw): x={x_raw}, y={y_raw}, w={w_raw}, h={h_raw}")
                print(f"          BBox (int): x={x}, y={y}, w={w}, h={h}")
            
            # Find session directory
            session_dir = self.participant_dir / session_name
            if not session_dir.exists():
                print(f"        Warning: Session directory not found: {session_dir}")
                return self._create_placeholder_image()
            
            # Find video file - try multiple patterns
            video_patterns = ["scenevideo.*", "scene_video.*", "*.mp4", "*.avi", "*.mov"]
            video_candidates = []
            for pattern in video_patterns:
                video_candidates.extend(list(session_dir.glob(pattern)))

            # Deduplicate while preserving order
            video_files = []
            seen = set()
            for vf in video_candidates:
                if vf.name.startswith("._"):
                    continue
                if vf not in seen:
                    seen.add(vf)
                    video_files.append(vf)

            if not video_files:
                print(f"        Warning: No video file found in {session_dir}")
                print(f"          Tried patterns: {video_patterns}")
                return self._create_placeholder_image()

            # Prefer the video whose resolution best matches inferred bbox extents
            inferred_w = None
            inferred_h = None
            if session_name in self.session_bbox_stats:
                inferred_w = self.session_bbox_stats[session_name].get('max_x2')
                inferred_h = self.session_bbox_stats[session_name].get('max_y2')

            best_video = video_files[0]
            best_score = None
            for vf in video_files:
                cap_tmp = cv2.VideoCapture(str(vf))
                if not cap_tmp.isOpened():
                    cap_tmp.release()
                    continue
                vw = cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH)
                vh = cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cap_tmp.release()
                if inferred_w and inferred_h:
                    score = abs(vw - inferred_w) + abs(vh - inferred_h)
                else:
                    score = 0
                if best_score is None or score < best_score:
                    best_score = score
                    best_video = vf

            video_path = best_video

            if debug:
                print(f"          Video: {video_path.name}")
            
            # Open video and extract frame
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"        Warning: Could not open video: {video_path}")
                return self._create_placeholder_image()
            
            # Get video properties for debugging
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if debug:
                print(f"          Video properties:")
                print(f"            Size: {video_width}x{video_height}")
                print(f"            Total frames: {total_frames}")
                print(f"            FPS: {fps}")
            
            # Seek frame: prefer time-based seek if time_seconds is available
            frame = None
            if time_seconds is not None and not pd.isna(time_seconds):
                cap.set(cv2.CAP_PROP_POS_MSEC, float(time_seconds) * 1000.0)
                ret, frame = cap.read()
                if debug:
                    print(f"          Seeking by time_seconds: {time_seconds:.3f}s")
            else:
                if frame_number >= total_frames:
                    print(f"        Warning: Frame {frame_number} exceeds video length ({total_frames} frames)")
                    cap.release()
                    return self._create_placeholder_image()
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if debug:
                    print(f"          Seeking by frame_number: {frame_number}")
            cap.release()
            
            if not ret:
                print(f"        Warning: Could not read frame from {video_path}")
                return self._create_placeholder_image()
            
            # Get actual frame dimensions
            h_img, w_img = frame.shape[:2]
            
            if debug:
                print(f"          Actual frame size: {w_img}x{h_img}")
                print(f"          Requested crop: ({x},{y}) to ({x+w},{y+h})")
            
            # No scaling applied; use original bbox coordinates
            
            # Check if coordinates are severely out of bounds (data issue)
            if (x < -w or y < -h or x > w_img + w or y > h_img + h):
                print(f"        ERROR: Crop coordinates severely out of bounds!")
                print(f"          Frame: {w_img}x{h_img}, Crop: x={x}, y={y}, w={w}, h={h}")
                print(f"          This face instance appears to have invalid coordinates.")
                return self._create_placeholder_image()
            
            # Warn about boundary issues
            if x < 0 or y < 0 or x + w > w_img or y + h > h_img:
                if debug:
                    print(f"          Note: Clamping crop to frame boundaries")
                    print(f"            Original: ({x},{y}) to ({x+w},{y+h})")
            
            # Add padding to avoid too-tight crops
            pad = int(max(w, h) * 0.1)

            # Crop face region (clamp to image bounds)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)
            
            if debug:
                print(f"            Clamped: ({x1},{y1}) to ({x2},{y2})")
            
            # Ensure we have valid crop dimensions
            if x2 <= x1 or y2 <= y1:
                print(f"        Warning: Invalid crop dimensions after clamping")
                print(f"          x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                return self._create_placeholder_image()
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                print(f"        Warning: Empty face crop for session {session_name}, frame {frame_number}")
                return self._create_placeholder_image()
            
            if debug:
                print(f"          Crop size: {face_crop.shape[1]}x{face_crop.shape[0]}")
            
            # Resize to thumbnail size
            face_crop = cv2.resize(face_crop, (80, 80), interpolation=cv2.INTER_AREA)
            
            # Convert BGR to RGB
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            if debug:
                print(f"          ✓ Crop extracted successfully")
            
            return Image.fromarray(face_crop_rgb)
            
        except Exception as e:
            print(f"        Error extracting face crop: {e}")
            import traceback
            traceback.print_exc()
            return self._create_placeholder_image()
    
    def _create_placeholder_image(self) -> Image.Image:
        """Create a placeholder image when face crop extraction fails."""
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
        """Create a row for a face ID (or merged group)."""
        row_frame = ctk.CTkFrame(self.face_list_frame)
        row_frame.pack(fill="x", padx=5, pady=3)
        
        # Checkbox
        checkbox_var = ctk.BooleanVar()
        checkbox = ctk.CTkCheckBox(
            row_frame,
            text="",
            variable=checkbox_var,
            width=60,
            command=lambda: self._on_checkbox_toggle(face_id, checkbox_var.get())
        )
        checkbox.pack(side="left", padx=5)
        
        # Thumbnail(s) container
        thumbnail_container = ctk.CTkFrame(row_frame, width=100)
        thumbnail_container.pack(side="left", padx=5)
        
        # Display thumbnail(s)
        original_ids = info.get('original_ids', [face_id])
        
        # For merged groups, show one thumbnail per original ID
        if len(original_ids) > 1:
            for orig_id in original_ids[:5]:  # Show up to 5 thumbnails
                # Get thumbnail from face_groups_all (the unfiltered data)
                if orig_id in self.face_groups_all:
                    thumb = self.face_groups_all[orig_id].get('thumbnail')
                elif orig_id in self.face_groups:
                    thumb = self.face_groups[orig_id].get('thumbnail')
                else:
                    thumb = None
                
                if thumb:
                    thumb_tk = ImageTk.PhotoImage(thumb)
                    thumb_label = ctk.CTkLabel(thumbnail_container, image=thumb_tk, text="")
                    thumb_label.image = thumb_tk  # Keep reference
                    thumb_label.pack(side="left", padx=2)
            
            if len(original_ids) > 5:
                # Show "+N more" label
                more_label = ctk.CTkLabel(
                    thumbnail_container,
                    text=f"+{len(original_ids)-5}",
                    font=ctk.CTkFont(size=10),
                    width=30
                )
                more_label.pack(side="left", padx=2)
        else:
            # Single thumbnail
            thumb = info.get('thumbnail')
            if thumb:
                thumb_tk = ImageTk.PhotoImage(thumb)
                thumb_label = ctk.CTkLabel(thumbnail_container, image=thumb_tk, text="")
                thumb_label.image = thumb_tk  # Keep reference
                thumb_label.pack(side="left", padx=2)
        
        # Face ID label(s)
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
    
    def _apply_filter(self):
        """Apply minimum instances filter."""
        try:
            min_instances = self.min_instances_var.get()
            
            print(f"\n{'='*60}")
            print(f"APPLYING FILTER: Min instances = {min_instances}")
            print(f"{'='*60}")
            
            # Filter face_groups
            filtered_groups = {
                fid: info for fid, info in self.face_groups_all.items()
                if info['count'] >= min_instances
            }
            
            print(f"Before filter: {len(self.face_groups_all)} IDs")
            print(f"After filter: {len(filtered_groups)} IDs")
            print(f"Filtered out: {len(self.face_groups_all) - len(filtered_groups)} IDs")
            
            self.face_groups = filtered_groups
            
            # Clear selection
            self.selected_ids.clear()
            
            # Refresh display
            self._display_face_list()
            
            print(f"✓ Filter applied")
            print(f"{'='*60}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Invalid minimum instances value: {e}")
    
    def _merge_selected(self):
        """Merge selected face IDs into a single group."""
        if len(self.selected_ids) < 2:
            return
        
        selected_list = list(self.selected_ids)
        
        print("\n" + "="*60)
        print("MERGING FACE IDs")
        print("="*60)
        print(f"Selected IDs: {', '.join(selected_list)}")
        
        # Find the ID with most instances
        counts = {fid: self.face_groups[fid]['count'] for fid in selected_list}
        merged_id = max(counts, key=counts.get)
        print(f"Merged ID (most instances): {merged_id} ({counts[merged_id]} instances)")
        
        # Collect all original IDs
        all_original_ids = []
        total_count = 0
        
        for fid in selected_list:
            original_ids = self.face_groups[fid].get('original_ids', [fid])
            all_original_ids.extend(original_ids)
            total_count += self.face_groups[fid]['count']
            print(f"  - {fid}: {len(original_ids)} original ID(s), {self.face_groups[fid]['count']} instances")
        
        print(f"Total merged count: {total_count} instances from {len(all_original_ids)} original IDs")
        
        # Create merged group
        merged_info = {
            'count': total_count,
            'representative': self.face_groups[merged_id]['representative'],
            'thumbnail': self.face_groups[merged_id]['thumbnail'],
            'original_ids': all_original_ids
        }
        
        # Update merge tracking
        for orig_id in all_original_ids:
            self.face_id_to_merged[orig_id] = merged_id
        
        # Remove selected IDs from face_groups and add merged
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
        print("Refreshing display...")
        self._display_face_list()
        
        print("✓ Merge complete")
        print("="*60 + "\n")
        
        messagebox.showinfo(
            "Merged",
            f"Merged {len(selected_list)} face IDs into {merged_id}\n\n"
            f"Total instances: {total_count}\n"
            f"Original IDs: {len(all_original_ids)}"
        )
    
    def _unmerge_group(self, merged_id: str):
        """Unmerge a merged group back into separate IDs."""
        if merged_id not in self.face_groups:
            return
        
        merged_info = self.face_groups[merged_id]
        original_ids = merged_info.get('original_ids', [])
        
        if len(original_ids) <= 1:
            return  # Nothing to unmerge
        
        print("\n" + "="*60)
        print("UNMERGING FACE ID")
        print("="*60)
        print(f"Merged ID: {merged_id}")
        print(f"Original IDs to restore: {', '.join(original_ids)}")
        
        # Remove merged group from current view
        if merged_id in self.face_groups:
            del self.face_groups[merged_id]
        
        # Restore individual groups from face_groups_all
        print("Restoring individual groups...")
        for orig_id in original_ids:
            if orig_id in self.face_groups_all:
                # Restore from cached data
                self.face_groups[orig_id] = self.face_groups_all[orig_id].copy()
                print(f"  - {orig_id}: {self.face_groups_all[orig_id]['count']} instances (restored from cache)")
            else:
                # Fallback: recompute if not in cache
                face_instances = self.df[self.df['face_id'] == orig_id]
                count = len(face_instances)
                print(f"  - {orig_id}: {count} instances (recomputed)")
                
                representative = self._find_representative_face(face_instances)
                thumbnail = self._extract_face_crop(representative) if representative else None
                
                self.face_groups[orig_id] = {
                    'count': count,
                    'representative': representative,
                    'thumbnail': thumbnail,
                    'original_ids': [orig_id]
                }
                self.face_groups_all[orig_id] = self.face_groups[orig_id].copy()
            
            # Update merge tracking
            self.face_id_to_merged[orig_id] = orig_id
        
        # Apply current filter
        min_instances = self.min_instances_var.get()
        self.face_groups = {
            fid: info for fid, info in self.face_groups.items()
            if info['count'] >= min_instances
        }
        
        # Re-sort by count
        self.face_groups = dict(
            sorted(self.face_groups.items(), key=lambda x: x[1]['count'], reverse=True)
        )
        
        # Refresh display
        print("Refreshing display...")
        self._display_face_list()
        
        print("✓ Unmerge complete")
        print("="*60 + "\n")
        
        messagebox.showinfo(
            "Unmerged",
            f"Unmerged {merged_id} into {len(original_ids)} separate IDs"
        )
    
    def _save_results(self):
        """Save merged results to CSV with new merged_face_id column."""
        if self.df_full is None:
            return
        
        # Confirm save
        response = messagebox.askyesno(
            "Confirm Save",
            "This will add a 'merged_face_id' column to faces_combined.csv.\n"
            "The original file will be backed up. Continue?"
        )
        
        if not response:
            print("Save cancelled by user")
            return
        
        try:
            print("\n" + "="*60)
            print("SAVING MERGED RESULTS")
            print("="*60)
            
            # Create backup
            csv_path = self.participant_dir / "faces_combined.csv"
            backup_path = self.participant_dir / "faces_combined.backup.csv"
            
            print(f"Creating backup: {backup_path}")
            import shutil
            shutil.copy2(csv_path, backup_path)
            print("✓ Backup created")
            
            # Add merged_face_id column to full data
            print(f"Adding 'merged_face_id' column...")
            df_out = self.df_full.copy()
            df_out['merged_face_id'] = df_out['face_id'].map(self.face_id_to_merged)
            
            # Count merges
            num_merged = len([k for k, v in self.face_id_to_merged.items() if k != v])
            unique_merged_ids = df_out['merged_face_id'].nunique()
            original_unique_ids = df_out['face_id'].nunique()
            
            print(f"  Original unique IDs: {original_unique_ids}")
            print(f"  Merged unique IDs: {unique_merged_ids}")
            print(f"  IDs affected by merging: {num_merged}")
            
            # Save updated CSV
            print(f"Saving to: {csv_path}")
            df_out.to_csv(csv_path, index=False)
            print("✓ CSV saved successfully")
            
            print("="*60)
            print("SAVE COMPLETE")
            print("="*60 + "\n")
            
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
            error_details = traceback.format_exc()
            print(f"\n✗ ERROR during save:")
            print(error_details)
            
            messagebox.showerror(
                "Error",
                f"Failed to save results:\n{str(e)}\n\nCheck terminal for details."
            )


def main():
    """Main entry point."""
    # Set appearance
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    # Create main window
    root = ctk.CTk()
    
    # Create GUI
    gui = FaceIDReviewGUI(root)
    
    # Run
    root.mainloop()


if __name__ == "__main__":
    main()
