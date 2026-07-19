"""Patch MLflow for Python 3.14 compatibility.

Run after `uv sync` upgrades MLflow:
    uv run python scripts/fix_mlflow_py314.py
"""
from pathlib import Path
import mlflow.assistant.skill_installer as _

installer_path = Path(_.__file__)
content = installer_path.read_text()
old = "from importlib.abc import Traversable"
new = "from importlib.resources.abc import Traversable"
if old in content:
    content = content.replace(old, new)
    installer_path.write_text(content)
    print(f"Patched: {installer_path}")
elif new in content:
    print("Already patched.")
else:
    print(f"Could not find import line in {installer_path}")
