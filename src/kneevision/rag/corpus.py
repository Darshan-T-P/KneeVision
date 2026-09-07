from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GuidelineChunk:
    """One retrievable unit of guideline text: a markdown section (## heading)
    from a document under data/guidelines/, tagged with that document's
    frontmatter metadata."""
    text: str
    source_path: str
    kl_grade: str  # e.g. "0-1", "2", "3", "4", or "all"
    topic: str
    heading: str


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Split a leading `---\\nkey: value\\n---` block from the rest of the file."""
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    _, front, body = parts
    meta = {}
    for line in front.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, body.strip()


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split a markdown body into (heading, text) sections on '## ' headings.
    Content before the first '##' (e.g. the '# Title' line) is dropped."""
    sections = []
    current_heading = None
    current_lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if current_heading is not None and current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line[3:].strip()
            current_lines = []
        elif current_heading is not None:
            current_lines.append(line)
    if current_heading is not None and current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return sections


def load_guideline_chunks(guidelines_dir: Path) -> list[GuidelineChunk]:
    """Load every *.md file under `guidelines_dir` and split it into
    per-section retrievable chunks. Files with topic "sources" (citations /
    disclaimer metadata, not clinical guidance) are skipped — the disclaimer
    they contain is always surfaced separately by RehabRecommender."""
    chunks = []
    for path in sorted(guidelines_dir.glob("*.md")):
        content = path.read_text()
        meta, body = _parse_frontmatter(content)
        if meta.get("topic") == "sources":
            continue
        kl_grade = meta.get("kl_grade", "all")
        topic = meta.get("topic", path.stem)
        for heading, text in _split_sections(body):
            if not text:
                continue
            chunks.append(GuidelineChunk(
                text=text, source_path=path.name, kl_grade=kl_grade, topic=topic, heading=heading,
            ))
    return chunks
