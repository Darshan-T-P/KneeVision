from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from kneevision.rag.corpus import GuidelineChunk, load_guideline_chunks


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
        self, query: str, kl_grade: int | None = None, top_k: int = 4
    ) -> list[GuidelineChunk]:
        """Return the top_k most relevant chunks for `query`. If `kl_grade`
        is given, chunks tagged for that grade (or "all") are boosted ahead
        of chunks tagged for a different grade, without excluding them
        entirely — general lifestyle guidance is still often relevant."""
        if not self.chunks:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]

        if kl_grade is not None:
            target = str(kl_grade)
            for i, chunk in enumerate(self.chunks):
                if chunk.kl_grade == "all" or _grade_matches(chunk.kl_grade, target):
                    scores[i] += 1.0  # boost, not a hard filter

        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:top_k] if scores[i] > 0]


def _grade_matches(chunk_grade: str, target: str) -> bool:
    """chunk_grade may be a single grade ("2") or a range ("0-1")."""
    if "-" in chunk_grade:
        lo, hi = chunk_grade.split("-")
        return lo.isdigit() and hi.isdigit() and int(lo) <= int(target) <= int(hi)
    return chunk_grade == target
