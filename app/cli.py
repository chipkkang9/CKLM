from __future__ import annotations

import argparse
from pathlib import Path

from app.ingest.chunker import split_markdown_into_chunks
from app.ingest.parse_pdf import parse_pdf_to_markdown
from app.retrieval.embedder import TextEmbedder
from app.retrieval.vector_store import ChromaVectorStore


def ingest_pdf(pdf_path: str, persist_dir: str = "data/chroma") -> None:
    path = Path(pdf_path).expanduser().resolve()
    markdown = parse_pdf_to_markdown(path)

    parsed_dir = Path("data/parsed")
    parsed_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = parsed_dir / f"{path.stem}.md"
    parsed_path.write_text(markdown, encoding="utf-8")

    chunks = split_markdown_into_chunks(markdown)
    if not chunks:
        raise RuntimeError("No chunks created from parsed markdown")

    texts = [c.text for c in chunks]
    metadatas = [
        {
            "source": str(path),
            "filename": path.name,
            "section": c.section,
            "chunk_index": c.index,
        }
        for c in chunks
    ]

    embedder = TextEmbedder()
    embeddings = embedder.embed(texts)

    store = ChromaVectorStore(persist_path=persist_dir)
    stored = store.upsert(texts=texts, embeddings=embeddings, metadatas=metadatas)

    print(f"Stored chunks: {stored}")
    print("Example chunk:")
    print(texts[0][:500])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CKLM terminal CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest a PDF into Chroma")
    ingest.add_argument("pdf_path", help="Path to a PDF file")
    ingest.add_argument(
        "--persist-dir",
        default="data/chroma",
        help="Persistent Chroma directory (default: data/chroma)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        ingest_pdf(pdf_path=args.pdf_path, persist_dir=args.persist_dir)


if __name__ == "__main__":
    main()
