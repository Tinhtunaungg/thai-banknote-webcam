"""Paths shared by the desktop and web applications."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"


def reference_path(name):
    """Use an existing reference from the original layout when present."""
    legacy_path = PROJECT_ROOT / name
    return legacy_path if legacy_path.exists() else ASSETS_DIR / name
