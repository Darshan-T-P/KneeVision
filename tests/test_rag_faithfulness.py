"""Faithfulness checks, retry/fallback in the pipeline, and the retrieval gating rules."""
from kneevision.rag.corpus import GuidelineChunk
from kneevision.rag.faithfulness import find_unsupported_claims
from kneevision.rag.pipeline import RehabRecommender
from kneevision.rag.retriever import GuidelineRetriever, expand_context


def chunk(text, heading="H", grade="all", source="x.md", on_demand=False):
    return GuidelineChunk(text=text, source_path=source, kl_grade=grade, topic="t", heading=heading,
                          on_demand=on_demand)


EXERCISE = chunk("Structured exercise and weight loss are core treatments.", "Exercise")


# ── find_unsupported_claims: the failures seen from llama3.2:1b ────────────────

def test_flags_limited_rating_read_as_therapy():
    answer = "The AAOS recommends limited therapy modalities for patients with knee osteoarthritis."
    assert any("evidence rating" in p for p in find_unsupported_claims(answer, [EXERCISE]))


def test_flags_intervention_not_in_excerpts():
    problems = find_unsupported_claims("Acupuncture and canes may help.", [EXERCISE])
    assert any("acupuncture" in p for p in problems) and any("canes" in p for p in problems)


def test_flags_invented_patient_state():
    problems = find_unsupported_claims("Given the patient's normal knee joint and no symptoms, stay active.", [EXERCISE])
    assert any("no symptoms" in p for p in problems)


def test_clean_answer_has_no_flags():
    answer = "Structured exercise and weight loss help. This is not medical advice; consult a clinician."
    assert find_unsupported_claims(answer, [EXERCISE]) == []


def test_patient_context_can_support_a_mention():
    answer = "You asked about braces; they could be used."
    assert find_unsupported_claims(answer, [EXERCISE], patient_context="considering braces") == []


def test_disclaimer_text_never_flags():
    answer = "Exercise helps. This is not medical advice and a clinician (physical therapist, surgeon) should be consulted."
    assert find_unsupported_claims(answer, [EXERCISE]) == []


# ── pipeline retry / fallback ──────────────────────────────────────────────────

class ScriptedLLM:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def recommender(llm, tmp_path):
    (tmp_path / "kl2.md").write_text(
        "---\nkl_grade: 2\ntopic: mild\n---\n\n## Exercise\n\nStructured exercise and weight loss are core treatments.\n")
    return RehabRecommender.from_guidelines_dir(tmp_path, llm=llm)


def test_faithful_draft_is_used_once(tmp_path):
    llm = ScriptedLLM("Exercise and weight loss help.")
    rec = recommender(llm, tmp_path).recommend(2)
    assert rec.used_llm and rec.llm_attempts == 1 and rec.faithfulness_flags is None


def test_unfaithful_draft_is_retried_with_the_problems_named(tmp_path):
    llm = ScriptedLLM("Acupuncture is great.", "Exercise helps.")
    rec = recommender(llm, tmp_path).recommend(2)
    assert rec.used_llm and rec.llm_attempts == 2 and rec.faithfulness_flags
    assert "acupuncture" in llm.prompts[1] and "rejected" in llm.prompts[1]
    assert "Exercise helps." in rec.synthesis


def test_still_unfaithful_after_retry_falls_back_to_excerpts(tmp_path):
    llm = ScriptedLLM("Acupuncture is great.", "Acupuncture works.")
    rec = recommender(llm, tmp_path).recommend(2)
    assert not rec.used_llm and rec.llm_attempts == 2
    assert "Acupuncture" not in rec.synthesis and "not supported by the excerpts" in rec.synthesis


# ── retrieval gating ───────────────────────────────────────────────────────────

def test_on_demand_chunk_needs_matching_context():
    chunks = [chunk("Kl2 core care.", "Core", grade="2", source="kl2.md"),
              chunk("Topical NSAIDs are recommended for knee pain.", "Topical", on_demand=True)]
    r = GuidelineRetriever(chunks)
    assert [c.heading for c in r.retrieve("knee", kl_grade=2)] == ["Core"]
    assert "Topical" in [c.heading for c in r.retrieve("knee", kl_grade=2, context="wants topical relief")]


def test_other_grade_chunks_are_excluded():
    chunks = [chunk("Kl2 exercise.", "Two", grade="2"), chunk("Kl4 exercise.", "Four", grade="4")]
    got = GuidelineRetriever(chunks).retrieve("exercise", kl_grade=2, context="exercise")
    assert [c.heading for c in got] == ["Two"]


def test_expand_context_maps_lay_terms():
    assert "topical NSAIDs" in expand_context("prefers a gel")
    assert "weight loss" in expand_context("BMI 33")
    assert expand_context("no known terms here") == "no known terms here"
