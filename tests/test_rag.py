from pathlib import Path

import pytest

from kneevision.rag.corpus import load_guideline_chunks, _parse_frontmatter, _split_sections
from kneevision.rag.retriever import GuidelineRetriever
from kneevision.rag.llm import OllamaClient, OllamaUnavailableError
from kneevision.rag.pipeline import RehabRecommender, DISCLAIMER


SAMPLE_KL2 = """---
kl_grade: 2
topic: mild osteoarthritis
---

# KL 2 (Mild)

## Exercise

Structured exercise programs including strengthening and aquatic exercise
are core treatments for mild knee osteoarthritis.

## Weight Management

Weight loss of 5% of body weight is associated with symptom improvement.
"""

SAMPLE_KL4 = """---
kl_grade: 4
topic: severe osteoarthritis
---

# KL 4 (Severe)

## Surgical Referral

Total knee arthroplasty should be considered once conservative treatment
has been adequately tried and pain/function are significantly limited.
"""


@pytest.fixture
def guidelines_dir(tmp_path):
    (tmp_path / "kl2.md").write_text(SAMPLE_KL2)
    (tmp_path / "kl4.md").write_text(SAMPLE_KL4)
    return tmp_path


def test_parse_frontmatter():
    meta, body = _parse_frontmatter(SAMPLE_KL2)
    assert meta == {"kl_grade": "2", "topic": "mild osteoarthritis"}
    assert body.startswith("# KL 2 (Mild)")


def test_parse_frontmatter_no_frontmatter():
    meta, body = _parse_frontmatter("# Just a heading\n\nSome text.")
    assert meta == {}
    assert body == "# Just a heading\n\nSome text."


def test_split_sections():
    _, body = _parse_frontmatter(SAMPLE_KL2)
    sections = _split_sections(body)
    assert [h for h, _ in sections] == ["Exercise", "Weight Management"]
    assert "Structured exercise" in sections[0][1]


def test_load_guideline_chunks(guidelines_dir):
    chunks = load_guideline_chunks(guidelines_dir)
    assert len(chunks) == 3  # 2 sections in kl2.md + 1 in kl4.md
    kl2_chunks = [c for c in chunks if c.kl_grade == "2"]
    kl4_chunks = [c for c in chunks if c.kl_grade == "4"]
    assert len(kl2_chunks) == 2
    assert len(kl4_chunks) == 1
    assert kl4_chunks[0].heading == "Surgical Referral"


def test_load_guideline_chunks_skips_sources_topic(guidelines_dir):
    (guidelines_dir / "sources.md").write_text(
        "---\nkl_grade: all\ntopic: sources\n---\n\n# Sources\n\n## Disclaimer\n\nNot medical advice.\n"
    )
    chunks = load_guideline_chunks(guidelines_dir)
    assert all(c.source_path != "sources.md" for c in chunks)
    assert len(chunks) == 3  # unchanged from kl2.md + kl4.md only


def test_retriever_boosts_matching_kl_grade(guidelines_dir):
    retriever = GuidelineRetriever.from_dir(guidelines_dir)
    results = retriever.retrieve("knee osteoarthritis treatment", kl_grade=4, top_k=1)
    assert len(results) == 1
    assert results[0].kl_grade == "4"

    results = retriever.retrieve("knee osteoarthritis treatment", kl_grade=2, top_k=1)
    assert len(results) == 1
    assert results[0].kl_grade == "2"


def test_retriever_ranks_relevant_text_higher(guidelines_dir):
    retriever = GuidelineRetriever.from_dir(guidelines_dir)
    results = retriever.retrieve("surgical arthroplasty referral", top_k=1)
    assert results[0].heading == "Surgical Referral"


def test_retriever_empty_corpus():
    retriever = GuidelineRetriever([])
    assert retriever.retrieve("anything") == []


def test_recommender_falls_back_when_llm_unavailable(guidelines_dir):
    class BrokenLLM(OllamaClient):
        def generate(self, prompt: str) -> str:
            raise OllamaUnavailableError("no server")

    recommender = RehabRecommender.from_guidelines_dir(guidelines_dir, llm=BrokenLLM())
    result = recommender.recommend(kl_grade=4, patient_context="high pain")

    assert result.used_llm is False
    assert "Surgical Referral" in result.synthesis
    assert DISCLAIMER in result.synthesis
    assert len(result.retrieved_chunks) > 0


def test_recommender_uses_llm_when_available(guidelines_dir):
    class FakeLLM(OllamaClient):
        def generate(self, prompt: str) -> str:
            assert "Kellgren-Lawrence grade: 2" in prompt
            return "Mock synthesized recommendation."

    recommender = RehabRecommender.from_guidelines_dir(guidelines_dir, llm=FakeLLM())
    result = recommender.recommend(kl_grade=2)

    assert result.used_llm is True
    assert result.synthesis == "Mock synthesized recommendation."


def test_recommender_rejects_invalid_kl_grade(guidelines_dir):
    recommender = RehabRecommender.from_guidelines_dir(guidelines_dir)
    with pytest.raises(ValueError):
        recommender.recommend(kl_grade=7)


def test_ollama_client_raises_when_server_unreachable():
    client = OllamaClient(base_url="http://localhost:1")  # nothing listens here
    with pytest.raises(OllamaUnavailableError):
        client.generate("hello")


def test_real_guideline_corpus_loads_and_retrieves():
    """Sanity check against the actual data/guidelines/ corpus shipped with the repo."""
    real_dir = Path(__file__).resolve().parents[1] / "data" / "guidelines"
    chunks = load_guideline_chunks(real_dir)
    assert len(chunks) > 0
    retriever = GuidelineRetriever(chunks)
    results = retriever.retrieve("knee osteoarthritis rehabilitation exercise", kl_grade=3, top_k=3)
    assert len(results) > 0
