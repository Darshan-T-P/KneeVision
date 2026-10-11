"""Evaluate RAG guideline retrieval against a labelled query set.

Compares three configurations on tests/fixtures/rag_eval_queries.json:
  A  original hand-written corpus,     original retriever (equal +1 boost for grade-specific and "all")
  B  corpus + AAOS CPG excerpts,       original retriever
  C  corpus + AAOS CPG excerpts,       current retriever (grade-specific +2, "all" +1, patient context matched separately)

Metrics (per query, then averaged), for the top-k the pipeline would return:
  grade_hit  - the KL grade's own guideline document is in the top-k
  hit        - at least one labelled-relevant chunk is in the top-k
  recall     - fraction of labelled-relevant chunks in the top-k
  MRR        - reciprocal rank of the first labelled-relevant chunk (0 if none in top-k)
  topic_hit / topic_recall - same hit/recall but ignoring the KL grade's main section, i.e. only the
               context-specific labels. The grade section is always labelled relevant and always ranks
               first under grade boosting, which inflates the headline numbers; these are the honest ones.

Usage:
    uv run python scripts/evaluate_rag_retrieval.py
    uv run python scripts/evaluate_rag_retrieval.py --top-k 6 --out reports/rag_retrieval_eval.md
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics.pairwise import cosine_similarity

from kneevision.config.settings import GUIDELINES_DIR
from kneevision.rag.corpus import load_guideline_chunks
from kneevision.rag.pipeline import KL_LABELS
from kneevision.rag.retriever import GuidelineRetriever, _grade_matches

GRADE_DOCS = {0: "kl0_kl1_normal_doubtful.md", 1: "kl0_kl1_normal_doubtful.md", 2: "kl2_mild.md",
              3: "kl3_moderate.md", 4: "kl4_severe.md"}
MAIN_HEADING = {0: "Recommendations", 1: "Recommendations", 2: "Recommendations (core treatments)",
                3: "Recommendations", 4: "Recommendations"}
AAOS_FILE = "aaos_oak3_recommendations.md"


def legacy_retrieve(retriever: GuidelineRetriever, query: str, kl_grade: int, top_k: int):
    """The retriever as it was before grade-specific chunks got a larger boost."""
    scores = cosine_similarity(retriever.vectorizer.transform([query]), retriever._matrix)[0]
    for i, c in enumerate(retriever.chunks):
        if c.kl_grade == "all" or _grade_matches(c.kl_grade, str(kl_grade)):
            scores[i] += 1.0
    ranked = sorted(range(len(retriever.chunks)), key=lambda i: scores[i], reverse=True)
    return [retriever.chunks[i] for i in ranked[:top_k] if scores[i] > 0]


def build_query(kl: int, context: str) -> str:
    # Mirrors RehabRecommender.recommend()
    return f"knee osteoarthritis KL grade {kl} ({KL_LABELS[kl]}) rehabilitation. {context}".strip()


def evaluate(name, retrieve, queries, top_k):
    rows = []
    for item in queries:
        got = retrieve(build_query(item["kl"], item["context"]), item["kl"], top_k, item["context"])
        ids = [f"{c.source_path}::{c.heading}" for c in got]
        rel = set(item["relevant"])
        topic_rel = rel - {f"{GRADE_DOCS[item['kl']]}::{MAIN_HEADING[item['kl']]}"}
        first = next((r for r, cid in enumerate(ids, 1) if cid in rel), None)
        rows.append({
            "topic": item["topic"], "kl": item["kl"], "got": ids,
            "grade_hit": any(c.source_path == GRADE_DOCS[item["kl"]] for c in got),
            "hit": first is not None,
            "recall": len(rel & set(ids)) / len(rel),
            "mrr": 1 / first if first else 0.0,
            "topic_hit": (bool(topic_rel & set(ids)) if topic_rel else None),
            "topic_recall": (len(topic_rel & set(ids)) / len(topic_rel) if topic_rel else None),
            "topic_mrr": (1 / next((r for r, cid in enumerate(ids, 1) if cid in topic_rel), float("inf"))
                          if topic_rel else None),
        })
    return name, rows


def mean(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def sweep(chunks, queries, k):
    import itertools
    ret = GuidelineRetriever(chunks)
    rows = []
    for cw, gb, ab in itertools.product([0, 1, 2, 4, 8], [0.5, 1.0, 1.5, 2.0], [0.0, 0.25, 0.5, 1.0]):
        if ab > gb:
            continue
        res = []
        for item in queries:
            got = ret.retrieve(build_query(item["kl"], item["context"]), kl_grade=item["kl"], top_k=k,
                               context=item["context"], context_weight=cw, grade_boost=gb, all_boost=ab)
            ids = [f"{c.source_path}::{c.heading}" for c in got]
            topic_rel = set(item["relevant"]) - {f"{GRADE_DOCS[item['kl']]}::{MAIN_HEADING[item['kl']]}"}
            first = next((i for i, x in enumerate(ids, 1) if x in topic_rel), None)
            res.append({"grade": any(c.source_path == GRADE_DOCS[item["kl"]] for c in got),
                        "main": f"{GRADE_DOCS[item['kl']]}::{MAIN_HEADING[item['kl']]}" in ids,
                        "t_hit": (first is not None) if topic_rel else None,
                        "t_rec": len(topic_rel & set(ids)) / len(topic_rel) if topic_rel else None,
                        "t_mrr": (1 / first if first else 0.0) if topic_rel else None})
        rows.append((cw, gb, ab, mean(res, "main"), mean(res, "t_hit"), mean(res, "t_rec"), mean(res, "t_mrr")))
    rows.sort(key=lambda r: (r[3] == 1.0, r[5] + r[6]), reverse=True)
    print(f"top-{k}; sorted by (grade main section always present, topic recall+MRR)")
    print("ctx_w grade_b all_b | main  t_hit t_rec t_mrr")
    for r in rows[:15]:
        print(f"{r[0]:>5} {r[1]:>7} {r[2]:>5} | {r[3]:.2f}  {r[4]:.2f}  {r[5]:.2f}  {r[6]:.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queries", type=Path, default=ROOT / "tests/fixtures/rag_eval_queries.json")
    p.add_argument("--guidelines-dir", type=Path, default=GUIDELINES_DIR)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--sweep", action="store_true", help="Grid-search context_weight / grade_boost / all_boost")
    p.add_argument("--out", type=Path, default=None, help="Optional markdown report path")
    args = p.parse_args()

    queries = json.loads(args.queries.read_text())["queries"]
    all_chunks = load_guideline_chunks(args.guidelines_dir)
    base_chunks = [c for c in all_chunks if c.source_path != AAOS_FILE]
    if len(base_chunks) == len(all_chunks):
        sys.exit(f"{AAOS_FILE} not found in {args.guidelines_dir}; run scripts/ingest_aaos_guideline.py first.")

    if args.sweep:
        return sweep(all_chunks, queries, args.top_k)

    ret_base, ret_full = GuidelineRetriever(base_chunks), GuidelineRetriever(all_chunks)
    k = args.top_k
    results = [
        evaluate("A: base corpus, original retriever", lambda q, g, n, c="": legacy_retrieve(ret_base, q, g, n), queries, k),
        evaluate("B: +AAOS, original retriever", lambda q, g, n, c="": legacy_retrieve(ret_full, q, g, n), queries, k),
        evaluate("C: +AAOS, current retriever (+context match)", lambda q, g, n, c="": ret_full.retrieve(q, kl_grade=g, top_k=n, context=c), queries, k),
    ]
    # Note: A cannot retrieve AAOS-labelled chunks, so its hit/recall/MRR on those queries is capped by construction.

    lines = [f"# RAG retrieval evaluation (top-{k}, {len(queries)} queries)", "",
             "| Configuration | grade_hit | hit | recall | MRR |", "|---|---|---|---|---|"]
    for name, rows in results:
        lines.append(f"| {name} | {mean(rows,'grade_hit'):.2f} | {mean(rows,'hit'):.2f} | {mean(rows,'recall'):.2f} | {mean(rows,'mrr'):.2f} |")

    n_topic = sum(1 for r in results[0][1] if r["topic_hit"] is not None)
    lines += ["", f"## Topic-specific labels only ({n_topic} queries with context-specific targets)", "",
              "| Configuration | topic_hit | topic_recall | topic_MRR |", "|---|---|---|---|"]
    for name, rows in results:
        lines.append(f"| {name} | {mean(rows,'topic_hit'):.2f} | {mean(rows,'topic_recall'):.2f} | {mean(rows,'topic_mrr'):.2f} |")
    lines += ["", "_A cannot retrieve AAOS chunks, so its topic scores are capped for AAOS-labelled queries._"]

    lines += ["", "## Config C: queries whose topic-specific target was NOT retrieved", ""]
    misses = [(q, r) for q, r in zip(queries, results[2][1]) if r["topic_hit"] is False]
    if not misses:
        lines.append("None.")
    for q, r in misses:
        lines.append(f"- KL{q['kl']} [{q['topic']}] \"{q['context']}\" — wanted {q['relevant']}; got {r['got']}")

    report = "\n".join(lines)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
