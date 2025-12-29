from __future__ import annotations

import os
import uuid
import time
from typing import Any
import json
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from embedops.errors import ConfigError, EmbeddingError, IngestionError, VectorStoreError
from embedops.ingestion.ingest_pdfs import ingest_pdfs
from embedops.processing.chunking import chunk_text, ChunkConfig
from embedops.vector_store.pinecone_client import get_index

load_dotenv()


def _env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or str(v).strip() == "":
        raise ConfigError(f"Missing required environment variable: {name}")
    return str(v).strip()


def _retry_upsert(index, vectors, namespace: str, max_retries: int = 3, base_sleep: float = 0.5):
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            index.upsert(vectors=vectors, namespace=namespace)
            return
        except Exception as e:
            last_err = e
            # Simple exponential backoff
            time.sleep(base_sleep * (2 ** (attempt - 1)))
    raise VectorStoreError(f"Pinecone upsert failed after {max_retries} retries: {last_err}") from last_err


def main():
    try:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
        namespace = os.getenv("EMBEDDING_NAMESPACE", "emb_v1").strip()

        if not model_name:
            raise ConfigError("EMBEDDING_MODEL is empty.")
        if not namespace:
            raise ConfigError("EMBEDDING_NAMESPACE is empty.")

        # Ingest
        docs = ingest_pdfs()
        if not docs:
            raise IngestionError("No documents ingested (unexpected).")

        # Load embedding model
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model '{model_name}': {e}") from e

        dim = model.get_sentence_embedding_dimension()
        if not isinstance(dim, int) or dim <= 0:
            raise EmbeddingError(f"Invalid embedding dimension returned by model: {dim}")

        # Pinecone index
        index = get_index(dimension=dim)

        cfg = ChunkConfig(chunk_words=400, overlap_words=80)
        # Chunk store (local) to map retrieval results back to text
        chunk_store_path = Path("data/processed/chunks.jsonl")
        chunk_store_path.parent.mkdir(parents=True, exist_ok=True)

        vectors: list[tuple[str, list[float], dict[str, Any]]] = []
        skipped_chunks = 0

        # Build vectors + write chunk store
        with chunk_store_path.open("w", encoding="utf-8") as chunk_store_file:
            for doc in tqdm(docs, desc="Docs"):
                chunks = chunk_text(doc["text"], cfg)
                if not chunks:
                    continue

                try:
                    embs = model.encode(chunks, normalize_embeddings=True)
                except Exception as e:
                    raise EmbeddingError(f"Embedding generation failed for source={doc.get('source')}: {e}") from e

                for i, emb in enumerate(embs):
                    if emb is None or len(emb) != dim:
                        skipped_chunks += 1
                        continue

                    vector_id = str(uuid.uuid4())
                    metadata = {"doc_id": doc["doc_id"], "chunk_id": i, "source": doc["source"],
                        "version": doc["version"], }

                    # Persist chunk text locally (one JSON per line)
                    chunk_store_file.write(json.dumps(
                        {"vector_id": vector_id, "doc_id": doc["doc_id"], "chunk_id": i, "source": doc["source"],
                            "version": doc["version"], "text": chunks[i], }, ensure_ascii=False, ) + "\n")

                    vectors.append((vector_id, emb.tolist(), metadata))
        if not vectors:
            raise EmbeddingError("No vectors produced. Check PDFs for extractable text and chunking settings.")

        # Upsert in batches
        batch_size = int(os.getenv("UPSERT_BATCH_SIZE", "200"))
        if batch_size <= 0:
            raise ConfigError("UPSERT_BATCH_SIZE must be a positive integer.")

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            _retry_upsert(index=index, vectors=batch, namespace=namespace)

        print(f"Upserted {len(vectors)} vectors into Pinecone namespace='{namespace}'.")
        if skipped_chunks:
            print(f"Warning: skipped {skipped_chunks} chunks due to invalid embeddings.")

    except (ConfigError, IngestionError, EmbeddingError, VectorStoreError) as e:
        # Clean, user-actionable errors
        print(f"[EmbedOps ERROR] {type(e).__name__}: {e}")
        raise
    except Exception as e:
        # Unexpected errors with minimal leakage
        print(f"[EmbedOps ERROR] Unexpected failure: {e}")
        raise


if __name__ == "__main__":
    main()