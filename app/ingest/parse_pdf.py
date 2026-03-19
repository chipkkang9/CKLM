from __future__ import annotations

from pathlib import Path

import pymupdf4llm


def parse_pdf_to_markdown(pdf_path: str | Path) -> str:
    """Parse a PDF file into markdown text."""
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path}")

    markdown = pymupdf4llm.to_markdown(str(path))
    if not isinstance(markdown, str) or not markdown.strip():
        raise RuntimeError(f"No markdown content extracted from: {path}")
    return markdown
