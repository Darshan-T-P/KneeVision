"""Start MLflow tracking UI with Python 3.14 compatibility patch.

Usage:
    uv run python scripts/mlflow_server.py server --port 5000
"""
import importlib.abc as abc
import importlib.resources.abc as resources_abc

if not hasattr(abc, "Traversable"):
    abc.Traversable = resources_abc.Traversable

from mlflow.cli import cli

if __name__ == "__main__":
    cli()
