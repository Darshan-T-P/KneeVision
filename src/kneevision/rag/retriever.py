import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from kneevision.rag.corpus import GuidelineChunk, load_guideline_chunks

# Chosen from a plateau of equally good settings in `scripts/evaluate_rag_retrieval.py --sweep`
# (not the single best cell, to avoid overfitting the small labelled set).
CONTEXT_WEIGHT = 4.0
GRADE_BOOST = 2.0
ALL_BOOST = 1.0
# An on-demand chunk must match the patient context at least this much (cosine, context-only)
# to be returned. Without this, every chunk shares the fixed query prefix and unrelated chunks
# enter the top-k on their boost alone, which a small LLM then repeats in its summary.
MIN_RELEVANCE = 0.0
# Words in more than this fraction of chunks ("knee", "osteoarthritis", "patients", ...) carry
# no discriminating signal in this single-topic corpus and are ignored by the vectorizer.
MAX_DF = 0.5
MIN_CHUNKS_FOR_MAX_DF = 8


# Lay / shorthand terms in patient context -> vocabulary the guidelines use. TF-IDF only
# matches shared words, so without this "BMI 33" never reaches "weight loss" and "gel"
# never reaches "topical NSAIDs". Deliberately small and explicit; extend as needed.
CONTEXT_SYNONYMS = {
    "bmi": "weight loss overweight obese", "obese": "weight loss overweight obese",
    "obesity": "weight loss overweight obese", "overweight": "weight loss obese",
    "gel": "topical NSAIDs", "cream": "topical NSAIDs", "ointment": "topical NSAIDs",
    "diclofenac": "topical NSAIDs", "patch": "topical NSAIDs",
    "tablets": "oral NSAIDs acetaminophen", "pills": "oral NSAIDs acetaminophen",
    "paracetamol": "acetaminophen", "ibuprofen": "oral NSAIDs", "opioid": "narcotics tramadol",
    "opioids": "narcotics tramadol", "codeine": "narcotics",
    "tai": "exercise tai chi", "yoga": "exercise mind-body", "swimming": "aquatic exercise",
    "gym": "exercise supervised", "walking": "exercise", "cycling": "exercise", "running": "exercise",
    "physio": "supervised exercise physical therapist", "physiotherapy": "supervised exercise physical therapist",
    "injection": "intra-articular", "injections": "intra-articular", "steroid": "corticosteroids",
    "knee replacement": "arthroplasty surgery", "replacement": "arthroplasty surgery",
    "insomnia": "sleep", "stress": "sleep stress", "brace": "braces", "cane": "canes", "stick": "canes",
}


def expand_context(context: str) -> str:
    """Append guideline vocabulary for known lay terms found in `context`."""
    low = context.lower()
    extra = [v for k, v in CONTEXT_SYNONYMS.items() if re.search(rf"\b{re.escape(k)}\b", low)]
    return f"{context} {' '.join(extra)}".strip() if extra else context


class GuidelineRetriever:
    """TF-IDF retrieval over the guideline corpus. Deliberately lightweight
    (no vector DB / embedding model) since the corpus is a small, curated set
    of clinical guideline excerpts — scikit-learn (already a dependency) is
    enough for good keyword-level retrieval at this scale.
    """

    def __init__(self, chunks: list[GuidelineChunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            # Only meaningful (and only valid) once there are enough chunks to define "common".
            max_df=MAX_DF if len(chunks) >= MIN_CHUNKS_FOR_MAX_DF else 1.0,
        )
        self._matrix = self.vectorizer.fit_transform([c.text for c in chunks]) if chunks else None

    @classmethod
    def from_dir(cls, guidelines_dir: Path) -> "GuidelineRetriever":
        return cls(load_guideline_chunks(guidelines_dir))

    def retrieve(
        self, query: str, kl_grade: int | None = None, top_k: int = 4, context: str = "",
        context_weight: float = CONTEXT_WEIGHT, grade_boost: float = GRADE_BOOST,
        all_boost: float = ALL_BOOST, min_relevance: float = MIN_RELEVANCE,
    ) -> list[GuidelineChunk]:
        """Return the top_k most relevant chunks for `query`. If `kl_grade`
        is given, chunks tagged for that grade (or "all") are boosted ahead
        of general chunks, and chunks for other grades are excluded.
        Grade-specific chunks get a larger boost than "all" chunks. "On-demand"
        reference chunks are returned only when `context` matches them.

        `context` is free-text patient context. It is matched on its own and
        added with `context_weight`: inside `query` it is diluted by the fixed
        "knee osteoarthritis KL grade ... rehabilitation" prefix, which would
        otherwise dominate the TF-IDF similarity."""
        if not self.chunks:
            return []

        scores = cosine_similarity(self.vectorizer.transform([query]), self._matrix)[0]
        has_context = bool(context.strip())
        context = expand_context(context)
        ctx_sim = [0.0] * len(self.chunks)
        if has_context:
            ctx_sim = cosine_similarity(self.vectorizer.transform([context]), self._matrix)[0]
            scores = scores + context_weight * ctx_sim

        eligible = []
        for i, chunk in enumerate(self.chunks):
            if chunk.kl_grade == "all":
                # On-demand reference chunks need a genuine match with the patient context.
                if chunk.on_demand and not (has_context and ctx_sim[i] > min_relevance):
                    continue
                if kl_grade is not None:
                    scores[i] += all_boost
            elif kl_grade is not None:
                # Guidance written for a different KL grade is never returned.
                if not _grade_matches(chunk.kl_grade, str(kl_grade)):
                    continue
                scores[i] += grade_boost
            eligible.append(i)

        ranked = sorted(eligible, key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:top_k] if scores[i] > 0]


def _grade_matches(chunk_grade: str, target: str) -> bool:
    """chunk_grade may be a single grade ("2") or a range ("0-1")."""
    if "-" in chunk_grade:
        lo, hi = chunk_grade.split("-")
        return lo.isdigit() and hi.isdigit() and int(lo) <= int(target) <= int(hi)
    return chunk_grade == target
