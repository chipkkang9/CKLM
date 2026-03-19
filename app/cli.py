from __future__ import annotations

import argparse
from pathlib import Path

def ingest_pdf(pdf_path: str, persist_dir: str = "data/chroma") -> None:
    from app.ingest.chunker import split_markdown_into_chunks
    from app.ingest.parse_pdf import parse_pdf_to_markdown
    from app.retrieval.embedder import TextEmbedder
    from app.retrieval.vector_store import ChromaVectorStore

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

    chat = sub.add_parser("chat", help="Start terminal chat with RAG retrieval")
    chat.add_argument(
        "--persist-dir",
        default="data/chroma",
        help="Persistent Chroma directory (default: data/chroma)",
    )
    chat.add_argument(
        "--collection",
        default="documents",
        help="Chroma collection name (default: documents)",
    )
    chat.add_argument(
        "--model-id",
        default="google/gemma-3-4b-it",
        help="HF model id (default: google/gemma-3-4b-it)",
    )
    chat.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of retrieved chunks per turn (default: 4)",
    )
    chat.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max generated tokens per turn (default: 256)",
    )

    return parser


def run_chat(
    *,
    persist_dir: str,
    collection: str,
    model_id: str,
    top_k: int,
    max_new_tokens: int,
) -> None:
    from app.rag.pipeline import RagChatPipeline

    pipeline = RagChatPipeline(
        model_id=model_id,
        persist_dir=persist_dir,
        collection_name=collection,
    )
    chat_history: list[dict] = []

    print("RAG 채팅을 시작합니다. 종료하려면 'exit' 또는 'quit'를 입력하세요.")
    while True:
        user_input = input("\n나> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("종료합니다.")
            break

        answer, retrieved = pipeline.answer(
            question=user_input,
            chat_history=chat_history,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
        )

        print("\n어시스턴트>")
        print(answer)
        print("\n검색된 문맥:")
        for idx, chunk in enumerate(retrieved, start=1):
            src = chunk.metadata.get("filename", "unknown")
            section = chunk.metadata.get("section", "")
            print(f"{idx}. {src} | {section} | distance={chunk.distance:.4f}")

        chat_history.append(
            {"role": "user", "content": [{"type": "text", "text": user_input}]}
        )
        chat_history.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        ingest_pdf(pdf_path=args.pdf_path, persist_dir=args.persist_dir)
    elif args.command == "chat":
        run_chat(
            persist_dir=args.persist_dir,
            collection=args.collection,
            model_id=args.model_id,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
