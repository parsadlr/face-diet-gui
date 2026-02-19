"""
Settings Manager for Face-Diet GUI

Handles persistence of GUI settings to/from JSON config file.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class SettingsManager:
    """Manage GUI settings with JSON persistence."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize settings manager.
        
        Parameters
        ----------
        config_path : Path, optional
            Path to config file. If None, uses default location in user home.
        """
        if config_path is None:
            config_path = Path.home() / ".face_diet_config.json"
        
        self.config_path = Path(config_path)
        self.settings: Dict[str, Any] = self._default_settings()
        self.load_settings()
    
    def _default_settings(self) -> Dict[str, Any]:
        """Return default settings."""
        return {
            "last_project_dir": "",
            "last_session_dir_review": "",
            "reviewer_name": "",
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
            
            # Merge with defaults (in case new settings were added)
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
            # Create parent directory if needed
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
            
            print(f"Saved settings to {self.config_path}")
        
        except Exception as e:
            print(f"Error saving settings to {self.config_path}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value using dot notation.
        
        Examples
        --------
        settings.get("stage1.sampling_rate")
        settings.get("last_project_dir")
        """
        keys = key.split('.')
        value = self.settings
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set a setting value using dot notation.
        
        Examples
        --------
        settings.set("stage1.sampling_rate", 60)
        settings.set("last_project_dir", "/path/to/project")
        """
        keys = key.split('.')
        current = self.settings
        
        # Navigate to the parent dict
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the final value
        current[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire settings section."""
        return self.settings.get(section, {})
    
    def set_section(self, section: str, values: Dict[str, Any]):
        """Set an entire settings section."""
        self.settings[section] = values
