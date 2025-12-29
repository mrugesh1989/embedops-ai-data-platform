from __future__ import annotations

import os
import time
from typing import Optional

from dotenv import load_dotenv
from pinecone import Pinecone

from embedops.errors import ConfigError, VectorStoreError

load_dotenv()


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None or str(v).strip() == "":
        raise ConfigError(f"Missing required environment variable: {name}")
    return str(v).strip()


def _safe_list_index_names(pc: Pinecone) -> set[str]:
    try:
        # Newer clients often support list_indexes().names()
        li = pc.list_indexes()
        if hasattr(li, "names"):
            return set(li.names())
        # Fallback: list of dicts
        return set([i.get("name") for i in li if isinstance(i, dict) and i.get("name")])
    except Exception as e:
        raise VectorStoreError(f"Failed to list Pinecone indexes: {e}") from e


def get_index(dimension: int):
    """
    Returns a Pinecone Index handle, creating the index if needed.
    Includes defensive validation and clearer errors for common setup issues.
    """
    if not isinstance(dimension, int) or dimension <= 0:
        raise ConfigError(f"Invalid embedding dimension: {dimension}. Must be a positive integer.")

    api_key = _env("PINECONE_API_KEY")
    index_name = _env("PINECONE_INDEX_NAME", "embedops-rag-docs")
    cloud = _env("PINECONE_CLOUD", "aws")
    region = _env("PINECONE_REGION", "us-east-1")

    try:
        pc = Pinecone(api_key=api_key)
    except Exception as e:
        raise VectorStoreError(f"Failed to initialize Pinecone client: {e}") from e

    try:
        existing = _safe_list_index_names(pc)

        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec={"serverless": {"cloud": cloud, "region": region}},
            )

            # Index creation may be asynchronous; short wait helps avoid immediate query errors
            # without introducing long sleeps.
            for _ in range(10):
                if index_name in _safe_list_index_names(pc):
                    break
                time.sleep(0.5)

        return pc.Index(index_name)

    except ConfigError:
        raise
    except Exception as e:
        raise VectorStoreError(
            f"Failed to create/get Pinecone index '{index_name}' (cloud={cloud}, region={region}): {e}"
        ) from e