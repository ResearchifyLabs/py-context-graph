"""Chunk markdown documents by ## headings, prepending the # title as context."""

import re
from typing import List, TypedDict


class MarkdownChunk(TypedDict):
    title: str
    section: str
    text: str


def chunk_markdown(text: str) -> List[MarkdownChunk]:
    lines = text.split("\n")

    title = ""
    for line in lines:
        if re.match(r"^#\s+", line):
            title = re.sub(r"^#\s+", "", line).strip()
            break

    section_pattern = re.compile(r"^##\s+(.+)")
    chunks: List[MarkdownChunk] = []
    current_section = ""
    current_lines: list = []

    for line in lines:
        m = section_pattern.match(line)
        if m:
            if current_section and current_lines:
                chunks.append(_build_chunk(title, current_section, current_lines))
            current_section = m.group(1).strip()
            current_lines = [line]
        elif current_section:
            current_lines.append(line)

    if current_section and current_lines:
        chunks.append(_build_chunk(title, current_section, current_lines))

    return chunks


def _build_chunk(title: str, section: str, section_lines: list) -> MarkdownChunk:
    body = "\n".join(section_lines)
    text = (
        f"Following section is part of a larger document. Use the title to understand the context of the section. {title}\n\n{body}"
        if title
        else body
    )
    return MarkdownChunk(title=title, section=section, text=text)
