"""Upload the app's checkpoints to the Hugging Face model repo so Streamlit Cloud can download them.

Run once, after logging in with a token that has WRITE access to the repo:
    uv run hf auth login        # or: uv run huggingface-cli login
    uv run python scripts/upload_models_to_hf.py            # preview (dry run)
    uv run python scripts/upload_models_to_hf.py --upload   # actually upload

Uploads only these files (the models the cloud app lists first). Swin-T / ConvNeXt-Tiny (~115 MB each) are not uploaded.
The app downloads whatever is listed in REQUIRED_CHECKPOINTS in streamlit_app.py on first start.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO = "Darshan13/KneeVision-models"
MODELS = Path(__file__).resolve().parents[1] / "models"
FILES = [
    "best_densenet121_multitask_ordinal_soft_mixup_a04.pt", "best_densenet121_multitask_ordinal_soft_mixup_a04.json",
    "best_densenet121_ordinal_ordinal_soft_mixup_a04.pt", "best_densenet121_ordinal_ordinal_soft_mixup_a04.json",
    "best_densenet121_ordinal_short40_robust_seed42.pt", "best_densenet121_ordinal_short40_robust_seed42.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true", help="really upload (default is a dry-run preview)")
    args = ap.parse_args()
    api = HfApi()
    total = 0
    for name in FILES:
        path = MODELS / name
        if not path.exists():
            print(f"MISSING locally: {name}")
            continue
        size = path.stat().st_size
        total += size
        print(f"{'upload' if args.upload else 'would upload'}: {name} ({size / 1e6:.1f} MB)")
        if args.upload:
            api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=REPO, repo_type="model",
                            commit_message=f"Add {name}")
    print(f"total {total / 1e6:.1f} MB -> https://huggingface.co/{REPO}")
    if not args.upload:
        print("Dry run only. Re-run with --upload after `hf auth login`.")


if __name__ == "__main__":
    main()
