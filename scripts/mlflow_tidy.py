"""Tidy the local MLflow store so it is clean and presentable.

Dry-run by default; pass --apply to change anything (the DB is backed up first).

What it does:
  1. Deletes leftover `test_*` experiments (written by the test suite before it was
     isolated) and purges them, including their mlruns/<id> artifact folders.
  2. Marks runs stuck in RUNNING (no live training process) as KILLED.
  3. Tags every run in the project experiment so the UI can filter/group:
       run_type    training | evaluation | clinical | fusion-evaluation
       task        5-class | binary | ordinal | multitask | grouped | ...
       arch        backbone name, where it can be read from the run name
       run_health  complete | no_metrics | interrupted
  4. Sets a description on the experiment.
  5. Writes a markdown leaderboard of training runs (--summary).

Real runs are never deleted. Usage:
    uv run python scripts/mlflow_tidy.py                  # preview
    uv run python scripts/mlflow_tidy.py --apply --summary reports/mlflow_summary.md
"""
import argparse
import importlib.resources.abc as resources_abc
import shutil
import subprocess
import sys
import time
from importlib import abc
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
if not hasattr(abc, "Traversable"):  # Python 3.14 + MLflow
    abc.Traversable = resources_abc.Traversable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlflow.tracking import MlflowClient

from kneevision.config.settings import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, PROJECT_ROOT

EXPERIMENT_NOTE = (
    "KneeVision++ — explainable multimodal AI for knee osteoarthritis (KL grading). "
    "Training runs (DenseNet/EfficientNet/ConvNeXt; standard, binary, ordinal, multitask), "
    "clinical-text (BioClinicalBERT) runs, and evaluation / fusion-evaluation runs. "
    "Filter with tags: run_type, task, arch, run_health."
)
ARCHS = ("densenet121", "efficientnet-b4", "convnext_small", "convnext_tiny", "resnet50", "swin", "vit")
KAPPA_KEYS = ("best_kappa", "val_kappa")


def classify(name: str, tags: dict) -> dict:
    """Derive run_type / task / arch from the run name and existing tags."""
    n = name.lower()
    if n.startswith("clinical"):
        run_type, task = "clinical", "text-regression"
    elif n.startswith("eval_fusion"):
        run_type, task = "fusion-evaluation", "multimodal-fusion"
    elif n.startswith(("binary_", "grouped_", "evaluate_")):
        run_type, task = "evaluation", n.split("_")[0].replace("evaluate", "5-class")
    else:
        run_type = "training"
        if "multitask" in n:
            task = "multitask"
        elif "binary" in n:
            task = "binary"
        elif "ordinal" in n:
            task = "ordinal"
        else:
            task = "5-class"
    arch = next((a for a in ARCHS if n.startswith(a) or f"_{a}" in n), tags.get("model") or "")
    out = {"run_type": run_type, "task": task}
    if arch and run_type == "training":
        out["arch"] = arch
    return out


def best_kappa(client: MlflowClient, run) -> float | None:
    m = run.data.metrics
    if "best_kappa" in m:
        return m["best_kappa"]
    if "val_kappa" in m:
        hist = client.get_metric_history(run.info.run_id, "val_kappa")
        return max((h.value for h in hist), default=m["val_kappa"])
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="Actually modify the store (default: preview only)")
    p.add_argument("--summary", type=Path, default=None, help="Write a markdown leaderboard here")
    args = p.parse_args()

    client = MlflowClient(MLFLOW_TRACKING_URI)
    dry = not args.apply
    say = (lambda m: print(f"[dry-run] {m}")) if dry else (lambda m: print(m))

    if not dry and MLFLOW_TRACKING_URI.startswith("sqlite:///"):
        db = Path(MLFLOW_TRACKING_URI.removeprefix("sqlite:///"))
        backup = db.with_name(f"{db.name}.bak-{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(db, backup)
        print(f"Backed up {db.name} -> {backup.name}")

    # 1. test experiments ------------------------------------------------------
    junk = [e for e in client.search_experiments() if e.name.startswith("test_")]
    for e in junk:
        say(f"delete experiment {e.experiment_id} '{e.name}'")
        if not dry:
            client.delete_experiment(e.experiment_id)
    if junk and not dry:
        ids = ",".join(e.experiment_id for e in junk)
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/mlflow_server.py"), "gc",
                        "--backend-store-uri", MLFLOW_TRACKING_URI, "--experiment-ids", ids], check=True)

    # 2-4. real experiment -----------------------------------------------------
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        sys.exit(f"experiment '{MLFLOW_EXPERIMENT_NAME}' not found")
    say(f"set description on experiment '{exp.name}'")
    if not dry:
        client.set_experiment_tag(exp.experiment_id, "mlflow.note.content", EXPERIMENT_NOTE)

    runs = client.search_runs([exp.experiment_id], max_results=1000, order_by=["attributes.start_time ASC"])
    killed = 0
    leaderboard = []
    for run in runs:
        name = run.info.run_name or run.info.run_id[:8]
        tags = classify(name, run.data.tags)
        if run.info.status == "RUNNING":
            health = "interrupted"
            killed += 1
            say(f"mark KILLED: {name}")
            if not dry:
                client.set_terminated(run.info.run_id, "KILLED", end_time=run.info.start_time)
        else:
            health = "complete" if run.data.metrics else "no_metrics"
        tags["run_health"] = health
        if not dry:
            for k, v in tags.items():
                client.set_tag(run.info.run_id, k, v)
        if tags["run_type"] == "training":
            k = best_kappa(client, run)
            if k is not None and health == "complete":
                leaderboard.append((k, name, tags.get("arch", ""), tags["task"], run.info.run_id))
    say(f"tag {len(runs)} runs with run_type/task/arch/run_health ({killed} stale RUNNING -> KILLED)")

    # 5. leaderboard -----------------------------------------------------------
    if args.summary:
        leaderboard.sort(reverse=True)
        lines = ["# MLflow training-run leaderboard", "",
                 f"Experiment `{exp.name}` — completed training runs, ranked by best validation QWK (kappa).",
                 "", "| # | Run | Arch | Task | Best val kappa |", "|---|---|---|---|---|"]
        for i, (k, name, arch, task, _rid) in enumerate(leaderboard, 1):
            lines.append(f"| {i} | {name} | {arch} | {task} | {k:.4f} |")
        text = "\n".join(lines) + "\n"
        if dry:
            say(f"would write {args.summary} ({len(leaderboard)} runs)")
        else:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(text)
            print(f"Wrote {args.summary} ({len(leaderboard)} runs)")
    if dry:
        print("\nPreview only — re-run with --apply to make these changes.")


if __name__ == "__main__":
    main()
