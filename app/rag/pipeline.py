from __future__ import annotations

from dataclasses import dataclass

from app.model.chat_model import GemmaChatModel
from app.retrieval.embedder import TextEmbedder
from app.retrieval.vector_store import ChromaVectorStore


SYSTEM_PROMPT = (
    "You are a research assistant. "
    "Match the user's language: reply in Korean when the user writes in Korean, "
    "and reply in English when the user writes in English, unless the user explicitly requests another language. "
    "Use retrieved context when relevant, and state uncertainty clearly when evidence is insufficient. "
    "Keep answers concise and accurate."
)


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    distance: float


class RagChatPipeline:
    def __init__(
        self,
        *,
        model_id: str = "google/gemma-3-4b-it",
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_dir: str = "data/chroma",
        collection_name: str = "documents",
    ) -> None:
        self.embedder = TextEmbedder(model_name=embed_model_name)
        self.store = ChromaVectorStore(
            persist_path=persist_dir,
            collection_name=collection_name,
        )
        self.model = GemmaChatModel(model_id=model_id)

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        q_emb = self.embedder.embed([query])[0]
        results = self.store.query(query_embedding=q_emb, top_k=top_k)
        return [
            RetrievedChunk(
                text=item["document"],
                metadata=item["metadata"],
                distance=item["distance"],
            )
            for item in results
        ]

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "(검색된 문맥 없음)"

        lines: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("filename", "unknown")
            section = chunk.metadata.get("section", "")
            lines.append(f"[Chunk {i}] source={source} section={section}")
            lines.append(chunk.text)
            lines.append("")
        return "\n".join(lines).strip()

    def answer(
        self,
        *,
        question: str,
        chat_history: list[dict],
        top_k: int = 4,
        max_new_tokens: int = 256,
    ) -> tuple[str, list[RetrievedChunk]]:
        retrieved = self.retrieve(question, top_k=top_k)
        context = self._build_context(retrieved)

        prompt_text = (
            "아래 검색 문맥을 우선 활용해 사용자 질문에 답하라.\n\n"
            f"검색 문맥:\n{context}\n\n"
            f"사용자 질문:\n{question}"
        )

        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": [{"type": "text", "text": prompt_text}]})

        answer_text = self.model.generate(messages, max_new_tokens=max_new_tokens)
        return answer_text, retrieved
