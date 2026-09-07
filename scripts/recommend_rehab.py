"""Retrieval-augmented rehabilitation recommendation for a knee OA KL grade.

Retrieves relevant excerpts from data/guidelines/ and asks a local Ollama
model to synthesize them into a short summary. Falls back to showing the
retrieved excerpts directly if Ollama isn't running / the model isn't pulled.

Setup (one-time):
    ollama serve &                  # start the local Ollama server
    ollama pull llama3.2:1b         # pull the small default model

Usage:
    uv run python scripts/recommend_rehab.py --kl 3
    uv run python scripts/recommend_rehab.py --kl 2 --context "68yo, BMI 31, mild pain on stairs"
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kneevision.config.settings import GUIDELINES_DIR
from kneevision.rag import RehabRecommender


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kl", type=int, required=True, choices=range(5), help="KL grade 0-4")
    parser.add_argument("--context", default="", help="Optional free-text patient context")
    parser.add_argument("--guidelines-dir", type=Path, default=GUIDELINES_DIR)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    recommender = RehabRecommender.from_guidelines_dir(args.guidelines_dir)
    result = recommender.recommend(args.kl, patient_context=args.context, top_k=args.top_k)

    print("=" * 60)
    print(f"KL grade {result.kl_grade} — {'LLM-synthesized' if result.used_llm else 'retrieval-only fallback'}")
    print("=" * 60)
    print(result.synthesis)
    print()
    print("-" * 60)
    print(f"Retrieved {len(result.retrieved_chunks)} guideline excerpt(s):")
    for chunk in result.retrieved_chunks:
        print(f"  - [{chunk.source_path}] {chunk.heading}")
    if result.used_llm:
        print()
        print(result.disclaimer)


if __name__ == "__main__":
    main()
