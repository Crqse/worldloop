"""Pytest configuration for worldloop-adapters tests."""
import sys
from pathlib import Path

# Ensure src/ is on sys.path so `worldloop_adapters` is importable without install.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
