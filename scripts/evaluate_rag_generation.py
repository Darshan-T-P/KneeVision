"""Measure how faithful the local LLM's rehab summaries are to the retrieved excerpts.

Runs the real RehabRecommender (needs `ollama serve` + the model) over the labelled query
sets and reports, per query:
  first-draft flagged  - faithfulness check found an unsupported claim in the 1st draft
  retried / fell back  - what the pipeline then did (retry once, else show excerpts only)
  final LLM text       - the summary shown to the user came from the LLM and passed the check

Usage:
    uv run python scripts/evaluate_rag_generation.py --out reports/rag_generation_eval.md
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kneevision.config.settings import GUIDELINES_DIR
from kneevision.rag import RehabRecommender
from kneevision.rag.llm import OllamaClient

SETS = {"tuning": "tests/fixtures/rag_eval_queries.json", "held-out": "tests/fixtures/rag_eval_heldout.json"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="Only the first N queries per set (smoke test)")
    args = p.parse_args()

    rec = RehabRecommender.from_guidelines_dir(GUIDELINES_DIR, llm=OllamaClient(timeout=180))
    rows = []
    for set_name, rel in SETS.items():
        queries = json.loads((ROOT / rel).read_text())["queries"][: args.limit]
        for i, q in enumerate(queries, 1):
            t0 = time.time()
            r = rec.recommend(q["kl"], patient_context=q["context"])
            rows.append({"set": set_name, "kl": q["kl"], "context": q["context"], "llm": r.used_llm,
                         "attempts": r.llm_attempts, "flags": r.faithfulness_flags or [],
                         "secs": time.time() - t0})
            print(f"[{set_name} {i}/{len(queries)}] KL{q['kl']} attempts={r.llm_attempts} "
                  f"llm={r.used_llm} flags={len(r.faithfulness_flags or [])} {rows[-1]['secs']:.0f}s", flush=True)

    if not any(r["attempts"] for r in rows):
        sys.exit("The LLM never answered — is `ollama serve` running?")

    lines = ["# RAG generation faithfulness (real llama3.2:1b, retrieval + check + retry/fallback)", "",
             "| Set | Queries | First draft flagged | Fixed by retry | Fell back to excerpts | Final summary from LLM |",
             "|---|---|---|---|---|---|"]
    for name in SETS:
        s = [r for r in rows if r["set"] == name]
        flagged = [r for r in s if r["flags"]]
        fell = [r for r in s if r["attempts"] and not r["llm"]]
        n = len(s)
        lines.append(f"| {name} | {n} | {len(flagged)} ({len(flagged)/n:.0%}) | {len(flagged) - len(fell)} | "
                     f"{len(fell)} ({len(fell)/n:.0%}) | {sum(r['llm'] for r in s)} ({sum(r['llm'] for r in s)/n:.0%}) |")
    kinds = Counter(f.split(" ")[0] if not f.startswith("mentions") else "mentions-unsupported-intervention"
                    for r in rows for f in r["flags"])
    lines += ["", "## What gets flagged in first drafts", ""] + [f"- {k}: {v}" for k, v in kinds.most_common()]
    lines += ["", f"Mean time per query: {sum(r['secs'] for r in rows)/len(rows):.1f}s "
              "(includes the retry when it happens)."]
    report = "\n".join(lines)
    print("\n" + report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")


if __name__ == "__main__":
    main()
