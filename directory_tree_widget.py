"""
Directory Tree Widget for Face-Diet GUI

Custom tree view for displaying project/participant/session hierarchy.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import customtkinter as ctk


class DirectoryTreeWidget(ctk.CTkFrame):
    """
    Custom directory tree widget with checkboxes for selection.
    
    Shows hierarchy: Project > Participants > Sessions
    """
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.project_dir: Optional[Path] = None
        self.tree_data: Dict = {}  # {path_str: node_data}
        self.participant_frames: Dict = {}  # {participant_name: frame}
        self.session_widgets: Dict = {}  # {(participant, session): widgets}
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the tree UI."""
        # Controls at top
        controls_frame = ctk.CTkFrame(self)
        controls_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkButton(
            controls_frame,
            text="Expand All",
            command=self.expand_all,
            width=100,
            height=28
        ).pack(side="left", padx=2)
        
        ctk.CTkButton(
            controls_frame,
            text="Collapse All",
            command=self.collapse_all,
            width=100,
            height=28
        ).pack(side="left", padx=2)
        
        ctk.CTkButton(
            controls_frame,
            text="Select All",
            command=self.select_all,
            width=100,
            height=28
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            controls_frame,
            text="Deselect All",
            command=self.deselect_all,
            width=100,
            height=28
        ).pack(side="left", padx=2)
        
        # Scrollable tree area
        self.tree_frame = ctk.CTkScrollableFrame(self, height=400)
        self.tree_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    def build_tree(self, project_dir: str):
        """
        Build directory tree from project directory.
        
        Expected structure:
        project_dir/
            participant1/
                session1/
                    scenevideo.*
                    eye_tracking.tsv
                session2/
                    ...
            participant2/
                ...
        """
        self.project_dir = Path(project_dir)
        
        # Clear existing tree
        for widget in self.tree_frame.winfo_children():
            widget.destroy()
        
        self.tree_data.clear()
        self.participant_frames.clear()
        self.session_widgets.clear()
        
        # Scan directory structure
        if not self.project_dir.exists():
            ctk.CTkLabel(
                self.tree_frame,
                text=f"Directory not found: {project_dir}",
                text_color="red"
            ).pack(pady=20)
            return
        
        participants = sorted([
            d for d in self.project_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.') and not d.name.startswith('_')
        ])
        
        if not participants:
            ctk.CTkLabel(
                self.tree_frame,
                text="No participant directories found",
                text_color="orange"
            ).pack(pady=20)
            return
        
        # Build tree UI
        for participant_dir in participants:
            self._create_participant_node(participant_dir)
    
    def _create_participant_node(self, participant_dir: Path):
        """Create a participant node with its sessions."""
        participant_name = participant_dir.name
        
        # Find sessions
        sessions = sorted([
            d for d in participant_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        if not sessions:
            return  # Skip participants with no sessions
        
        # Participant frame (collapsible)
        participant_frame = ctk.CTkFrame(self.tree_frame)
        participant_frame.pack(fill="x", pady=2, padx=5)
        
        # Header frame for participant
        header_frame = ctk.CTkFrame(participant_frame)
        header_frame.pack(fill="x", padx=2, pady=2)
        
        # Expand/collapse button
        expand_var = ctk.BooleanVar(value=True)
        expand_btn = ctk.CTkButton(
            header_frame,
            text="▼",
            width=30,
            height=25,
            command=lambda: self._toggle_participant(participant_name, expand_var)
        )
        expand_btn.pack(side="left", padx=2)
        
        # Participant checkbox (controls all sessions)
        participant_var = ctk.BooleanVar(value=True)
        participant_cb = ctk.CTkCheckBox(
            header_frame,
            text=f"{participant_name} ({len(sessions)} sessions)",
            variable=participant_var,
            command=lambda: self._on_participant_checkbox(participant_name, participant_var),
            font=ctk.CTkFont(size=13, weight="bold"),
            checkbox_width=18,
            checkbox_height=18
        )
        participant_cb.pack(side="left", padx=5)
        
        # Sessions container (collapsible)
        sessions_container = ctk.CTkFrame(participant_frame)
        sessions_container.pack(fill="x", padx=(40, 5), pady=(0, 5))
        
        # Store participant data
        self.participant_frames[participant_name] = {
            'frame': participant_frame,
            'sessions_container': sessions_container,
            'expand_btn': expand_btn,
            'expand_var': expand_var,
            'checkbox_var': participant_var,
        }
        
        # Create session nodes
        for session_dir in sessions:
            self._create_session_node(participant_name, session_dir, sessions_container)
    
    def _create_session_node(self, participant_name: str, session_dir: Path, container):
        """Create a session node."""
        session_name = session_dir.name
        
        # Check for required files
        video_files = list(session_dir.glob("scenevideo.*"))
        eye_tracking_file = session_dir / "eye_tracking.tsv"
        
        has_video = len(video_files) > 0
        has_eye_tracking = eye_tracking_file.exists()
        
        # Session frame
        session_frame = ctk.CTkFrame(container)
        session_frame.pack(fill="x", pady=1, padx=2)
        
        # Session checkbox
        session_var = ctk.BooleanVar(value=True)
        session_cb = ctk.CTkCheckBox(
            session_frame,
            text=session_name,
            variable=session_var,
            font=ctk.CTkFont(size=12),
            checkbox_width=18,
            checkbox_height=18
        )
        session_cb.pack(side="left", padx=5, pady=2)
        
        # File status indicators
        status_text = []
        if has_video:
            status_text.append("✓ video")
        else:
            status_text.append("✗ video")
        
        if has_eye_tracking:
            status_text.append("✓ eye tracking")
        else:
            status_text.append("✗ eye tracking")
        
        status_label = ctk.CTkLabel(
            session_frame,
            text=" | ".join(status_text),
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        status_label.pack(side="left", padx=10)
        
        # Store session data
        self.session_widgets[(participant_name, session_name)] = {
            'frame': session_frame,
            'checkbox_var': session_var,
            'has_video': has_video,
            'has_eye_tracking': has_eye_tracking,
            'path': session_dir,
        }
    
    def _toggle_participant(self, participant_name: str, expand_var: ctk.BooleanVar):
        """Toggle participant expansion."""
        is_expanded = expand_var.get()
        new_state = not is_expanded
        expand_var.set(new_state)
        
        # Update button text
        btn = self.participant_frames[participant_name]['expand_btn']
        btn.configure(text="▼" if new_state else "▶")
        
        # Show/hide sessions container
        container = self.participant_frames[participant_name]['sessions_container']
        if new_state:
            container.pack(fill="x", padx=(40, 5), pady=(0, 5))
        else:
            container.pack_forget()
    
    def _on_participant_checkbox(self, participant_name: str, participant_var: ctk.BooleanVar):
        """Handle participant checkbox toggle - update all child sessions."""
        is_selected = participant_var.get()
        
        # Update all sessions under this participant
        for (p_name, s_name), widgets in self.session_widgets.items():
            if p_name == participant_name:
                widgets['checkbox_var'].set(is_selected)
    
    def expand_all(self):
        """Expand all participants."""
        for participant_name, data in self.participant_frames.items():
            data['expand_var'].set(True)
            data['expand_btn'].configure(text="▼")
            data['sessions_container'].pack(fill="x", padx=(40, 5), pady=(0, 5))
    
    def collapse_all(self):
        """Collapse all participants."""
        for participant_name, data in self.participant_frames.items():
            data['expand_var'].set(False)
            data['expand_btn'].configure(text="▶")
            data['sessions_container'].pack_forget()
    
    def select_all(self):
        """Select all sessions."""
        # Select all participants
        for data in self.participant_frames.values():
            data['checkbox_var'].set(True)
        
        # Select all sessions
        for widgets in self.session_widgets.values():
            widgets['checkbox_var'].set(True)
    
    def deselect_all(self):
        """Deselect all sessions."""
        # Deselect all participants
        for data in self.participant_frames.values():
            data['checkbox_var'].set(False)
        
        # Deselect all sessions
        for widgets in self.session_widgets.values():
            widgets['checkbox_var'].set(False)
    
    def get_selected_sessions(self) -> List[Tuple[str, str, Path]]:
        """
        Get list of selected sessions.
        
        Returns
        -------
        List[Tuple[str, str, Path]]
            List of (participant_name, session_name, session_path) tuples
        """
        selected = []
        
        for (participant_name, session_name), widgets in self.session_widgets.items():
            if widgets['checkbox_var'].get():
                selected.append((participant_name, session_name, widgets['path']))
        
        return selected
    
    def get_participants_and_sessions(self) -> Dict[str, List[Tuple[str, Path]]]:
        """
        Get selected sessions grouped by participant.
        
        Returns
        -------
        Dict[str, List[Tuple[str, Path]]]
            {participant_name: [(session_name, session_path), ...]}
        """
        result = {}
        
        for (participant_name, session_name), widgets in self.session_widgets.items():
            if widgets['checkbox_var'].get():
                if participant_name not in result:
                    result[participant_name] = []
                result[participant_name].append((session_name, widgets['path']))
        
        return result
