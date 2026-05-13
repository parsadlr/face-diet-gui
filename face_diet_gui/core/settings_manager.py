"""
Settings Manager for Face-Diet GUI

Handles persistence of GUI settings to/from JSON config file,
and manages the per-project reviewer registry stored in
{derivatives_dir}/annotations/reviewers.json.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class ReviewerRegistry:
    """
    Manage the list of reviewers for a project.

    The registry is stored as a small JSON file at:
        {derivatives_dir}/annotations/reviewers.json

    It is completely independent of the GUI settings file so that
    multiple users sharing the same derivatives directory all see the
    same reviewer list regardless of their local machine config.
    """

    ANNOTATIONS_DIR = "annotations"
    REGISTRY_FILE = "reviewers.json"

    # ------------------------------------------------------------------ #
    # Construction / loading                                               #
    # ------------------------------------------------------------------ #

    def __init__(self, derivatives_dir: Path):
        self.derivatives_dir = Path(derivatives_dir)
        self._registry_path = self.derivatives_dir / self.ANNOTATIONS_DIR / self.REGISTRY_FILE
        self._data: Dict = {"reviewers": []}
        self._load()

    def _load(self):
        if self._registry_path.exists():
            try:
                with open(self._registry_path, "r", encoding="utf-8") as fh:
                    self._data = json.load(fh)
            except Exception as e:
                print(f"Warning: could not load reviewer registry: {e}")

    def _save(self):
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._registry_path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def sanitize_id(raw: str) -> str:
        """
        Convert a free-text name into a filesystem-safe reviewer ID.
        E.g. "Alice Smith" -> "alice_smith"
        """
        return re.sub(r"[^\w\-]", "_", raw).lower().strip("_")

    def get_reviewers(self) -> List[Dict]:
        """Return the list of reviewer dicts (id, display_name, created_at)."""
        return list(self._data.get("reviewers", []))

    def get_reviewer_ids(self) -> List[str]:
        return [r["id"] for r in self.get_reviewers()]

    def reviewer_exists(self, reviewer_id: str) -> bool:
        return reviewer_id in self.get_reviewer_ids()

    def add_reviewer(self, reviewer_id: str, display_name: str) -> bool:
        """
        Add a new reviewer to the registry.

        Returns True on success, False if the ID already exists.
        """
        if self.reviewer_exists(reviewer_id):
            return False
        self._data.setdefault("reviewers", []).append({
            "id": reviewer_id,
            "display_name": display_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        self._save()
        return True

    def get_reviewer_dir(self, reviewer_id: str) -> Path:
        """
        Return the root annotation directory for this reviewer:
            {derivatives_dir}/annotations/{reviewer_id}/
        """
        return self.derivatives_dir / self.ANNOTATIONS_DIR / reviewer_id

    def get_annotations_base_dir(self) -> Path:
        """Return {derivatives_dir}/annotations/"""
        return self.derivatives_dir / self.ANNOTATIONS_DIR

    # ------------------------------------------------------------------ #
    # BIDS-named path helpers                                              #
    # ------------------------------------------------------------------ #

    def get_face_detections_path(self, participant: str, session: str) -> Path:
        """
        {derivatives_dir}/{participant}/{session}/{participant}_{session}_face-detections.csv
        """
        return self.derivatives_dir / participant / session / f"{participant}_{session}_face-detections.csv"

    def get_is_face_annotation_path(self, reviewer_id: str, participant: str, session: str) -> Path:
        """
        {derivatives_dir}/annotations/{reviewer_id}/{participant}/{session}/{participant}_{session}_is-face.csv
        """
        return (
            self.get_reviewer_dir(reviewer_id)
            / participant
            / session
            / f"{participant}_{session}_is-face.csv"
        )

    def get_review_status_path(self, reviewer_id: str, participant: str, session: str) -> Path:
        """
        {derivatives_dir}/annotations/{reviewer_id}/{participant}/{session}/{participant}_{session}_review-status.json
        """
        return (
            self.get_reviewer_dir(reviewer_id)
            / participant
            / session
            / f"{participant}_{session}_review-status.json"
        )

    def get_merges_path(self, reviewer_id: str, participant: str) -> Path:
        """
        {derivatives_dir}/annotations/{reviewer_id}/{participant}/merges.csv
        """
        return self.get_reviewer_dir(reviewer_id) / participant / "merges.csv"

    def get_face_id_review_status_path(self, reviewer_id: str, participant: str) -> Path:
        """
        {derivatives_dir}/annotations/{reviewer_id}/{participant}/face-id-review-status.json
        """
        return self.get_reviewer_dir(reviewer_id) / participant / "face-id-review-status.json"

    def get_face_ids_path(self, participant: str) -> Path:
        """
        {derivatives_dir}/{participant}/{participant}_face-ids.csv
        """
        return self.derivatives_dir / participant / f"{participant}_face-ids.csv"

    def get_clustering_stats_path(self, participant: str) -> Path:
        """
        {derivatives_dir}/{participant}/{participant}_clustering-stats.txt
        """
        return self.derivatives_dir / participant / f"{participant}_clustering-stats.txt"

    def get_consensus_dir(self, participant: str, session: str) -> Path:
        """
        {derivatives_dir}/annotations/consensus/{participant}/{session}/
        """
        return self.derivatives_dir / self.ANNOTATIONS_DIR / "consensus" / participant / session

    def get_consensus_annotation_path(self, participant: str, session: str) -> Path:
        """
        {derivatives_dir}/annotations/consensus/{participant}/{session}/{participant}_{session}_consensus-is-face.csv
        """
        return self.get_consensus_dir(participant, session) / f"{participant}_{session}_consensus-is-face.csv"

    def get_mismatches_resolved_path(self, participant: str, session: str) -> Path:
        """
        {derivatives_dir}/annotations/consensus/{participant}/{session}/{participant}_{session}_mismatches-resolved.json
        """
        return self.get_consensus_dir(participant, session) / f"{participant}_{session}_mismatches-resolved.json"


class SettingsManager:
    """Manage GUI settings with JSON persistence."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path.home() / ".face_diet_config.json"

        self.config_path = Path(config_path)
        self.settings: Dict[str, Any] = self._default_settings()
        self.load_settings()

    def _default_settings(self) -> Dict[str, Any]:
        """Return default settings."""
        return {
            "last_data_dir": "",
            "last_derivatives_dir": "",
            "reviewer_id": "",
            "stage1": {
                "sampling_rate": 30,
                "use_gpu": False,
                "use_original_fps": True,
                "min_confidence": 0.0,
                "start_time": None,
                "end_time": None,
            },
            "stage2": {
                "batch_size": 32,
            },
            "stage3": {
                "algorithm": "leiden",
                "similarity_threshold": 0.6,
                "k_neighbors": 50,
                "min_confidence": 0.0,
                "enable_refinement": True,
                "min_cluster_size": 5,
                "k_voting": 10,
                "min_votes": 5,
                "reassign_threshold": None,
            },
            "tab3": {
                "min_instances": 1,
                "min_confidence": 0.0,
            }
        }

    def load_settings(self):
        """Load settings from config file."""
        if not self.config_path.exists():
            print(f"Config file not found at {self.config_path}, using defaults")
            return

        try:
            with open(self.config_path, 'r') as f:
                loaded = json.load(f)
            self._merge_settings(self.settings, loaded)
            print(f"Loaded settings from {self.config_path}")
        except Exception as e:
            print(f"Error loading settings from {self.config_path}: {e}")
            print("Using default settings")

    def _merge_settings(self, defaults: dict, loaded: dict):
        """Recursively merge loaded settings into defaults."""
        for key, value in loaded.items():
            if key in defaults:
                if isinstance(value, dict) and isinstance(defaults[key], dict):
                    self._merge_settings(defaults[key], value)
                else:
                    defaults[key] = value

    def save_settings(self):
        """Save current settings to config file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
            print(f"Saved settings to {self.config_path}")
        except Exception as e:
            print(f"Error saving settings to {self.config_path}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value using dot notation."""
        keys = key.split('.')
        value = self.settings
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """Set a setting value using dot notation."""
        keys = key.split('.')
        current = self.settings
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire settings section."""
        return self.settings.get(section, {})

    def set_section(self, section: str, values: Dict[str, Any]):
        """Set an entire settings section."""
        self.settings[section] = values
