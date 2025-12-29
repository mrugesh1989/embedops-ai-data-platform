from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Any, Dict, List

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from embedops.errors import ConfigError, EmbeddingError, VectorStoreError
from embedops.vector_store.pinecone_client import get_index
from embedops.retrieval.chunk_store import load_chunk_by_keys

load_dotenv()


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None or str(v).strip() == "":
        raise ConfigError(f"Missing required environment variable: {name}")
    return str(v).strip()


@dataclass
class RetrievalResources:
    model_name: str
    namespace: str
    model: SentenceTransformer
    dim: int
    index: Any


def init_resources(
    model_name: Optional[str] = None,
    namespace: Optional[str] = None,
) -> RetrievalResources:
    """
    Initialize and cache heavy resources: embedding model + Pinecone index handle.
    """
    mn = (model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")).strip()
    ns = (namespace or os.getenv("EMBEDDING_NAMESPACE", "emb_v1")).strip()
    if not mn:
        raise ConfigError("EMBEDDING_MODEL is empty.")
    if not ns:
        raise ConfigError("EMBEDDING_NAMESPACE is empty.")

    try:
        model = SentenceTransformer(mn)
    except Exception as e:
        raise EmbeddingError(f"Failed to load embedding model '{mn}': {e}") from e

    dim = model.get_sentence_embedding_dimension()
    if not isinstance(dim, int) or dim <= 0:
        raise EmbeddingError(f"Invalid embedding dimension returned by model: {dim}")

    index = get_index(dimension=dim)
    return RetrievalResources(model_name=mn, namespace=ns, model=model, dim=dim, index=index)


def retrieve(
    resources: RetrievalResources,
    query: str,
    top_k: int = 5,
    score_threshold: Optional[float] = None,
    doc_id: Optional[str] = None,
    source: Optional[str] = None,
    version: Optional[int] = None,
    namespace: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Perform semantic retrieval from Pinecone and resolve chunk text from chunk store.
    Supports metadata filters for enterprise usage.
    """
    if not query or not query.strip():
        raise ValueError("Query text must be non-empty.")
    if top_k <= 0 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20.")

    # Embed query (cached model)
    try:
        q_emb = resources.model.encode(query, normalize_embeddings=True)
    except Exception as e:
        raise EmbeddingError(f"Failed to embed query: {e}") from e

    # Metadata filter
    flt: Dict[str, Any] = {}
    if doc_id:
        flt["doc_id"] = {"$eq": doc_id}
    if source:
        flt["source"] = {"$eq": source}
    if version is not None:
        flt["version"] = {"$eq": int(version)}

    ns = (namespace or resources.namespace).strip()

    try:
        res = resources.index.query(
            vector=q_emb.tolist(),
            top_k=top_k,
            namespace=ns,
            include_metadata=True,
            filter=flt if flt else None,
        )
    except Exception as e:
        raise VectorStoreError(f"Pinecone query failed: {e}") from e

    matches = res.get("matches", []) if isinstance(res, dict) else res.matches

    out: List[Dict[str, Any]] = []
    for m in matches:
        score = float(m["score"] if isinstance(m, dict) else m.score)
        md = m["metadata"] if isinstance(m, dict) else m.metadata

        # Optional score threshold
        if score_threshold is not None and score < float(score_threshold):
            continue

        row = load_chunk_by_keys(md["doc_id"], md["chunk_id"])
        text_preview = None
        if row and row.get("text"):
            txt = row["text"]
            text_preview = (txt[:500] + "...") if len(txt) > 500 else txt

        out.append(
            {
                "score": score,
                "doc_id": md["doc_id"],
                "chunk_id": int(md["chunk_id"]),
                "source": md.get("source", ""),
                "version": md.get("version"),
                "namespace": ns,
                "text_preview": text_preview,
            }
        )

    return out