from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import chromadb


class ChromaVectorStore:
    def __init__(
        self,
        persist_path: str | Path = "data/chroma",
        collection_name: str = "documents",
    ) -> None:
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_path))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    @staticmethod
    def _id_for(text: str, metadata: dict) -> str:
        base = f"{metadata.get('source', '')}:{metadata.get('chunk_index', -1)}:{text}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def upsert(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict],
    ) -> int:
        if not (len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError("texts, embeddings, and metadatas must have equal lengths")
        if not texts:
            return 0

        ids = [self._id_for(texts[i], metadatas[i]) for i in range(len(texts))]
        self.collection.upsert(
            ids=ids,
            documents=list(texts),
            embeddings=[list(e) for e in embeddings],
            metadatas=list(metadatas),
        )
        return len(ids)
