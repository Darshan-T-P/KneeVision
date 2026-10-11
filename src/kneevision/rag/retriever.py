from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from kneevision.rag.corpus import GuidelineChunk, load_guideline_chunks

# Chosen from a plateau of equally good settings in `scripts/evaluate_rag_retrieval.py --sweep`
# (not the single best cell, to avoid overfitting the small labelled set).
CONTEXT_WEIGHT = 4.0
GRADE_BOOST = 2.0
ALL_BOOST = 1.0


class GuidelineRetriever:
    """TF-IDF retrieval over the guideline corpus. Deliberately lightweight
    (no vector DB / embedding model) since the corpus is a small, curated set
    of clinical guideline excerpts — scikit-learn (already a dependency) is
    enough for good keyword-level retrieval at this scale.
    """

    def __init__(self, chunks: list[GuidelineChunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self.vectorizer.fit_transform([c.text for c in chunks]) if chunks else None

    @classmethod
    def from_dir(cls, guidelines_dir: Path) -> "GuidelineRetriever":
        return cls(load_guideline_chunks(guidelines_dir))

    def retrieve(
        self, query: str, kl_grade: int | None = None, top_k: int = 4, context: str = "",
        context_weight: float = CONTEXT_WEIGHT, grade_boost: float = GRADE_BOOST,
        all_boost: float = ALL_BOOST,
    ) -> list[GuidelineChunk]:
        """Return the top_k most relevant chunks for `query`. If `kl_grade`
        is given, chunks tagged for that grade (or "all") are boosted ahead
        of other chunks without excluding them entirely — general lifestyle
        guidance is still often relevant. Grade-specific chunks get a larger
        boost than "all" chunks.

        `context` is free-text patient context. It is matched on its own and
        added with `context_weight`: inside `query` it is diluted by the fixed
        "knee osteoarthritis KL grade ... rehabilitation" prefix, which would
        otherwise dominate the TF-IDF similarity."""
        if not self.chunks:
            return []

        scores = cosine_similarity(self.vectorizer.transform([query]), self._matrix)[0]
        if context.strip():
            scores = scores + context_weight * cosine_similarity(
                self.vectorizer.transform([context]), self._matrix)[0]

        if kl_grade is not None:
            target = str(kl_grade)
            for i, chunk in enumerate(self.chunks):
                if chunk.kl_grade == "all":
                    scores[i] += all_boost  # boost, not a hard filter
                elif _grade_matches(chunk.kl_grade, target):
                    scores[i] += grade_boost

        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:top_k] if scores[i] > 0]


def _grade_matches(chunk_grade: str, target: str) -> bool:
    """chunk_grade may be a single grade ("2") or a range ("0-1")."""
    if "-" in chunk_grade:
        lo, hi = chunk_grade.split("-")
        return lo.isdigit() and hi.isdigit() and int(lo) <= int(target) <= int(hi)
    return chunk_grade == target
