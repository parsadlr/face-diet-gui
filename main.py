"""
Face-Diet GUI entry point.

Run from project root: python main.py
"""

import sys
from pathlib import Path

# Ensure project root is on path so the package resolves
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from face_diet_gui.gui.app import main

if __name__ == "__main__":
    main()
