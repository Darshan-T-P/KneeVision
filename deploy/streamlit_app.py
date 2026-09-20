"""Streamlit Community Cloud entrypoint.

Lives in its own directory (with its own requirements.txt, no uv.lock/pyproject.toml nearby)
so Community Cloud installs the lean CPU-only dependencies below instead of the full
GPU-oriented uv.lock at the repo root (used for local training on a real GPU) — Cloud
prefers uv.lock over requirements.txt whenever both sit in the entrypoint's own directory.
Runs the actual app unmodified; __file__ inside it still resolves to its real repo-root path.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "streamlit_app.py"), run_name="__main__")
