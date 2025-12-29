from __future__ import annotations

import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from embedops.errors import ConfigError, VectorStoreError, EmbeddingError
from embedops.vector_store.pinecone_client import get_index
from embedops.retrieval.chunk_store import load_chunk_by_keys

load_dotenv()


def _env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if not v:
        raise ConfigError(f"Missing required environment variable: {name}")
    return v


def query_vectors(
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        raise ValueError("Query text must be non-empty.")

    model_name = _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    namespace = _env("EMBEDDING_NAMESPACE", "emb_v1")

    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        raise EmbeddingError(f"Failed to load embedding model: {e}") from e

    try:
        query_embedding = model.encode(query, normalize_embeddings=True)
    except Exception as e:
        raise EmbeddingError(f"Failed to embed query: {e}") from e

    dim = model.get_sentence_embedding_dimension()
    index = get_index(dimension=dim)

    try:
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
    except Exception as e:
        raise VectorStoreError(f"Pinecone query failed: {e}") from e

    matches = results.get("matches", []) if isinstance(results, dict) else results.matches

    formatted = []
    for m in matches:
        score = m["score"] if isinstance(m, dict) else m.score
        md = m["metadata"] if isinstance(m, dict) else m.metadata

        row = load_chunk_by_keys(md["doc_id"], md["chunk_id"])
        text = (row["text"][:500] + "...") if row and row.get("text") else None

        formatted.append(
            {"score": float(score), "doc_id": md["doc_id"], "chunk_id": md["chunk_id"], "source": md["source"],
                "text_preview": text, })
    return formatted


if __name__ == "__main__":
    query = "What is model drift in MLOps?"
    hits = query_vectors(query)
    for h in hits:
        print(h)