from __future__ import annotations

import os
import hashlib
from datetime import datetime, timezone
from typing import Any

from pypdf import PdfReader

from embedops.errors import IngestionError


RAW_DIR = os.path.join("data", "raw")


def _checksum(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _extract_pdf_text(path: str) -> str:
    try:
        reader = PdfReader(path)
        text_parts: list[str] = []
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                # Skip problematic page, don't fail the entire file.
                continue
        return "\n".join(text_parts).strip()
    except Exception as e:
        raise IngestionError(f"Failed to read PDF '{path}': {e}") from e


def ingest_pdfs(raw_dir: str = RAW_DIR) -> list[dict[str, Any]]:
    """
    Ingest PDFs under data/raw and return a list of documents:
      {doc_id, source, text, version, ingested_at}

    - Skips non-PDF files
    - Skips PDFs with no extractable text
    - Continues on per-file failure; surfaces a summary error if nothing usable is ingested
    """
    if not os.path.isdir(raw_dir):
        raise IngestionError(f"Raw directory not found: '{raw_dir}'. Create it and add PDFs.")

    docs: list[dict[str, Any]] = []
    errors: list[str] = []

    for fname in sorted(os.listdir(raw_dir)):
        if not fname.lower().endswith(".pdf"):
            continue

        path = os.path.join(raw_dir, fname)

        try:
            full_text = _extract_pdf_text(path)
            if not full_text:
                errors.append(f"{fname}: no extractable text (scanned PDF or empty).")
                continue

            doc_id = _checksum(full_text)
            docs.append(
                {
                    "doc_id": doc_id,
                    "source": fname,
                    "text": full_text,
                    "version": 1,
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        except IngestionError as e:
            errors.append(f"{fname}: {e}")
            continue

    if not docs:
        msg = "No ingestible PDFs found."
        if errors:
            msg += " Issues encountered:\n- " + "\n- ".join(errors[:10])
            if len(errors) > 10:
                msg += f"\n- ... and {len(errors) - 10} more"
        raise IngestionError(msg)

    return docs