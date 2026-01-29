"""
Launcher for Face-Diet Multi-Tab GUI

Run this script to start the comprehensive face processing GUI.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the multi-tab GUI
from gui_multitab import main

if __name__ == "__main__":
    main()
