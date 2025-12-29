from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any

from embedops.errors import ProcessingError

CHUNK_STORE = Path("data/processed/chunks.jsonl")

def load_chunk_by_keys(doc_id: str, chunk_id: int) -> Optional[Dict[str, Any]]:
    if not CHUNK_STORE.exists():
        raise ProcessingError(f"Chunk store not found at {CHUNK_STORE}. Run the pipeline first.")

    try:
        with CHUNK_STORE.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("doc_id") == doc_id and int(row.get("chunk_id")) == int(chunk_id):
                    return row
    except Exception as e:
        raise ProcessingError(f"Failed reading chunk store: {e}") from e

    return None