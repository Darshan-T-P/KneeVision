"""Build a public, sanitized snapshot of the MLflow store for a Hugging Face Space.

Copies mlflow.db to deploy/mlflow_space/mlflow.db (the original is never modified) and
scrubs what should not be public: the OS username, local absolute paths, artifact
locations, and the model registry (its artifacts are not published). Params, metrics
and tags are kept.

Usage:
    uv run python scripts/build_mlflow_space.py                       # build only
    uv run python scripts/build_mlflow_space.py --push Darshan13/KneeVision-MLflow   # needs `hf auth login`
"""
import argparse
import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "deploy" / "mlflow_space"
NO_ARTIFACTS = "file:///tmp/artifacts-not-published"


def sanitize(db: Path) -> dict:
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute("UPDATE tags SET value='kneevision' WHERE key='mlflow.user'")
    cur.execute("UPDATE runs SET artifact_uri = ? || '/' || run_uuid", (NO_ARTIFACTS,))
    cur.execute("UPDATE experiments SET artifact_location = ? || '/' || experiment_id", (NO_ARTIFACTS,))
    # source.name is a repo-relative script path, keep it; drop anything absolute.
    cur.execute("UPDATE tags SET value=REPLACE(value, ?, '') WHERE key='mlflow.source.name'", (str(ROOT) + "/",))
    for table in ("model_version_tags", "model_versions", "registered_model_aliases",
                  "registered_model_tags", "registered_models",
                  "logged_model_metrics", "logged_model_params", "logged_model_tags", "logged_models"):
        cur.execute(f"DELETE FROM {table}")
    con.commit()
    leaks = {}
    for table, col in [("tags", "value"), ("params", "value"), ("runs", "artifact_uri"),
                       ("experiments", "artifact_location"), ("experiment_tags", "value")]:
        n = cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} LIKE '/home/%' OR {col} LIKE '%/darshan%'").fetchone()[0]
        if n:
            leaks[f"{table}.{col}"] = n
    cur.execute("VACUUM")
    con.close()
    return leaks


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=ROOT / "mlflow.db")
    p.add_argument("--push", metavar="REPO_ID", help="Upload deploy/mlflow_space to this HF Space (created if missing)")
    args = p.parse_args()

    out = SPACE / "mlflow.db"
    shutil.copy2(args.src, out)
    leaks = sanitize(out)
    if leaks:
        out.unlink()
        raise SystemExit(f"refusing to publish, private strings remain: {leaks}")
    print(f"Built {out} ({out.stat().st_size / 1e6:.1f} MB), no private paths found.")

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(folder_path=str(SPACE), repo_id=args.push, repo_type="space",
                          allow_patterns=["Dockerfile", "start.sh", "README.md", "mlflow.db"])
        print(f"Pushed -> https://huggingface.co/spaces/{args.push}")


if __name__ == "__main__":
    main()
