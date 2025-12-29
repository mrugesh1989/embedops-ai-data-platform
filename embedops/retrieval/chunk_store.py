from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from embedops.errors import ProcessingError

REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNK_STORE = REPO_ROOT / "data" / "processed" / "chunks.jsonl"

# Simple in-memory index: (doc_id, chunk_id) -> text record
_CACHE: Dict[Tuple[str, int], Dict[str, Any]] | None = None


def _load_cache() -> Dict[Tuple[str, int], Dict[str, Any]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    if not CHUNK_STORE.exists():
        raise ProcessingError(f"Chunk store not found at {CHUNK_STORE}. Run pipeline first.")

    cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
    try:
        with CHUNK_STORE.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                doc_id = row.get("doc_id")
                chunk_id = row.get("chunk_id")
                if doc_id is None or chunk_id is None:
                    continue
                cache[(str(doc_id), int(chunk_id))] = row
    except Exception as e:
        raise ProcessingError(f"Failed reading chunk store at {CHUNK_STORE}: {e}") from e

    _CACHE = cache
    return _CACHE


def load_chunk_by_keys(doc_id: str, chunk_id: int) -> Optional[Dict[str, Any]]:
    cache = _load_cache()
    return cache.get((str(doc_id), int(chunk_id)))