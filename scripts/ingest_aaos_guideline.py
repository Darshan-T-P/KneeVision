"""Convert the AAOS knee OA CPG (3rd ed., 2021) 'Summary of Recommendations'
into a retrievable guideline document under data/guidelines/.

Each recommendation becomes one `## <topic>` section containing the AAOS
statement and its strength of recommendation, so the RAG pipeline can cite and
retrieve it like the hand-written guideline files.

One-time setup (the PDF is untrusted input; text extraction only):
    mkdir -p data/guidelines_raw/aaos && cd data/guidelines_raw/aaos
    curl -L -o oak3cpg.pdf https://www.aaos.org/globalassets/quality-and-practice-resources/osteoarthritis-of-the-knee/oak3cpg.pdf
    pdftotext -layout oak3cpg.pdf oak3cpg.txt

Usage:
    uv run python scripts/ingest_aaos_guideline.py
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START_RE = re.compile(r"^SUMMARY OF RECOMMENDATIONS\s*$")
END_RE = re.compile(r"^DEVELOPMENT GROUP ROSTER")
NOISE_RE = re.compile(r"^(View (background|data)|\s*\d+\s*$)")
STRENGTH_RE = re.compile(r"^(?:Strength of Recommendation:\s*)(Strong|Moderate|Limited|Consensus)")
DESC_RE = re.compile(r"^(Description:|Evidence |Evidence from)")


def parse_recommendations(text: str) -> list[dict]:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if START_RE.match(l) and i > 100)
    end = next(i for i, l in enumerate(lines) if i > start and END_RE.match(l))

    recs, block = [], []
    for line in lines[start + 1:end]:
        if NOISE_RE.match(line):
            continue
        block.append(line.rstrip())
    # A recommendation = statement lines, then a strength line, then AAOS's
    # evidence-grade boilerplate; it ends at the first blank line after the
    # strength line. The consensus-section intro paragraph is skipped.
    cur, skipping_intro = None, False
    for line in block:
        s = line.strip()
        if s.startswith("Summary of Consensus Statement"):
            skipping_intro = True
            continue
        if not s:
            if skipping_intro:
                skipping_intro = False
            elif cur and cur["strength"]:
                recs.append(cur)
                cur = None
            continue
        if skipping_intro:
            continue
        m = STRENGTH_RE.match(s)
        if m and cur is not None:
            cur["strength"] = m.group(1)
        elif cur and cur["strength"]:
            continue  # evidence-grade boilerplate
        elif cur is None:
            cur = {"topic": s, "body": [], "strength": None}
        else:
            cur["body"].append(s)
    if cur and cur["strength"]:
        recs.append(cur)
    return recs


def render(recs: list[dict]) -> str:
    out = [
        "---", "kl_grade: all", "topic: AAOS 2021 non-arthroplasty knee OA recommendations", "---", "",
        "# AAOS Knee OA CPG (3rd ed., 2021) — Summary of Recommendations", "",
        "Recommendation statements excerpted from AAOS, *Management of Osteoarthritis of the Knee "
        "(Non-Arthroplasty), Evidence-Based Clinical Practice Guideline, 3rd Edition* (2021). "
        "Each entry carries AAOS's own strength-of-recommendation rating.", "",
    ]
    for r in recs:
        # Join with spaces, but undo PDF hyphen line-breaks and keep bullet items on their own lines.
        body = []
        for ln in r["body"]:
            if ln.startswith(("•", "a.", "b.", "c.")):
                body.append("\n- " + ln.lstrip("•").strip())
            else:
                body.append(ln)
        text = " ".join(body).replace("\n- ", "\n- ").replace(" \n", "\n")
        text = re.sub(r"performance- based", "performance-based", text)
        out += [f"## {r['topic']}", "", f"{text}", "", f"AAOS strength of recommendation: {r['strength']}.", ""]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--txt", type=Path, default=ROOT / "data/guidelines_raw/aaos/oak3cpg.txt")
    p.add_argument("--out", type=Path, default=ROOT / "data/guidelines/aaos_oak3_recommendations.md")
    args = p.parse_args()
    recs = parse_recommendations(args.txt.read_text())
    args.out.write_text(render(recs))
    print(f"Wrote {len(recs)} recommendations -> {args.out}")
    for r in recs:
        print(f"  [{r['strength']:<9}] {r['topic']}")


if __name__ == "__main__":
    main()
