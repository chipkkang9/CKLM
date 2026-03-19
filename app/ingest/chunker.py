from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    section: str
    index: int


def _is_heading(line: str) -> bool:
    s = line.strip()
    return s.startswith("#") and len(s) > 1


def _flush_buffer(buffer: list[str], section: str) -> str | None:
    payload = "\n".join([ln for ln in buffer if ln.strip()]).strip()
    if not payload:
        return None
    if section:
        return f"{section}\n\n{payload}".strip()
    return payload


def split_markdown_into_chunks(
    markdown_text: str,
    *,
    min_chars: int = 500,
    max_chars: int = 800,
) -> list[Chunk]:
    """Split markdown into section-aware chunks around 500-800 chars."""
    if not markdown_text or not markdown_text.strip():
        return []

    chunks: list[Chunk] = []
    current_section = ""
    buffer: list[str] = []

    def emit_from_buffer() -> None:
        nonlocal buffer
        full = _flush_buffer(buffer, current_section)
        buffer = []
        if not full:
            return

        # Hard-wrap oversized blocks while keeping the section title attached.
        remaining = full
        while len(remaining) > max_chars:
            cut = remaining.rfind("\n", min_chars, max_chars)
            if cut == -1:
                cut = remaining.rfind(" ", min_chars, max_chars)
            if cut == -1:
                cut = max_chars
            part = remaining[:cut].strip()
            if part:
                chunks.append(Chunk(text=part, section=current_section, index=len(chunks)))
            remaining = remaining[cut:].strip()

        if remaining:
            chunks.append(Chunk(text=remaining, section=current_section, index=len(chunks)))

    for raw in markdown_text.splitlines():
        line = raw.rstrip()

        if _is_heading(line):
            if buffer:
                emit_from_buffer()
            current_section = line.strip()
            continue

        if not line.strip():
            if buffer and len("\n".join(buffer)) >= min_chars:
                emit_from_buffer()
            continue

        candidate_size = len("\n".join(buffer + [line]))
        if buffer and candidate_size > max_chars:
            emit_from_buffer()
        buffer.append(line)

    if buffer:
        emit_from_buffer()

    return chunks
