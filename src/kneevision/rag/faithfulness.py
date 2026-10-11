"""Lightweight faithfulness checks for LLM-written rehab summaries.

A small local model can (a) mention interventions that are not in the retrieved
guideline excerpts, (b) read an evidence label such as AAOS "Limited" as a kind of
therapy, and (c) invent patient facts. These checks are lexical and deliberately
conservative: they flag likely problems for a retry / fallback, they do not prove
an answer is correct.
"""
import re

from kneevision.rag.corpus import GuidelineChunk

# Canonical intervention -> regex that detects a mention of it.
INTERVENTIONS: dict[str, str] = {
    "acupuncture": r"acupuncture",
    "TENS / PENS (electrical nerve stimulation)": r"\b(tens|pens)\b|nerve stimulation",
    "PEMF": r"pulsed electromagnetic|\bpemf\b",
    "shockwave therapy": r"shock ?wave",
    "laser treatment": r"\blaser\b",
    "massage": r"\bmassage\b",
    "manual therapy": r"manual therapy",
    "dry needling": r"dry needling",
    "braces": r"\bbrac(e|es|ing)\b",
    "canes": r"\bcanes?\b",
    "lateral wedge insoles": r"wedge insoles?|\binsoles?\b",
    "hyaluronic acid": r"hyaluronic|viscosupplement",
    "corticosteroid injection": r"corticosteroid|cortisone|steroid injection",
    "platelet-rich plasma": r"platelet[- ]rich|\bprp\b",
    "denervation": r"denervation",
    "arthroscopy / lavage / debridement": r"arthroscop|lavage|debridement",
    "meniscectomy": r"meniscectomy",
    "osteotomy": r"osteotomy",
    "knee replacement / arthroplasty": r"arthroplasty|knee replacement|joint replacement",
    "oral NSAIDs": r"oral nsaids?|\bnsaids?\b|ibuprofen|naproxen",
    "topical NSAIDs": r"topical",
    "acetaminophen": r"acetaminophen|paracetamol",
    "opioids / tramadol": r"opioid|narcotic|tramadol|codeine",
    "supplements": r"supplement|glucosamine|chondroitin|turmeric|ginger|vitamin d",
    "tai chi / yoga": r"tai chi|yoga",
    "aquatic exercise": r"aquatic|swimming|hydrotherapy",
    "weight loss": r"weight loss|lose weight|weight management|body weight",
    "physical therapy": r"physical therap|physiotherap",
}

# Evidence ratings read as therapy names, e.g. "limited therapy modalities".
MISREAD_RATING = re.compile(r"\blimited (therap(y|ies)|modalit(y|ies)|treatments?|options?)\b", re.I)
# Patient facts that no excerpt or context supplied.
INVENTED_STATE = re.compile(
    r"\b(no symptoms|asymptomatic|symptom[- ]free|normal (knee|joint)s?|healthy knee)\b", re.I)


def _mentions(text: str) -> set[str]:
    return {name for name, pat in INTERVENTIONS.items() if re.search(pat, text, re.I)}


def find_unsupported_claims(answer: str, chunks: list[GuidelineChunk], patient_context: str = "") -> list[str]:
    """Return human-readable problems found in `answer` (empty list = nothing flagged)."""
    # Strip the boilerplate disclaimer so "consult a clinician" cannot trigger anything.
    answer = re.sub(r"not medical advice.*", "", answer, flags=re.I | re.S)
    source = " ".join(c.text for c in chunks) + " " + patient_context
    problems = [f"mentions '{t}' but no retrieved excerpt does" for t in sorted(_mentions(answer) - _mentions(source))]
    if MISREAD_RATING.search(answer) and not MISREAD_RATING.search(source):
        problems.append("treats an evidence rating ('Limited') as a type of therapy")
    for m in INVENTED_STATE.finditer(answer):
        if not re.search(re.escape(m.group(0)), source, re.I):
            problems.append(f"states '{m.group(0)}' which neither the excerpts nor the context support")
    return problems
