from __future__ import annotations

import os
import hashlib
from datetime import datetime, timezone
from typing import Any

from pypdf import PdfReader

from embedops.errors import IngestionError


RAW_DIR = os.path.join("data", "raw")


def _checksum(text: str) -> str:
    """ 
    Calculates and returns the MD5 checksum of the input text.

    Args:
        text (str): The input text to hash.

    Returns:
        str: The hexadecimal MD5 checksum of the input text.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _extract_pdf_text(path: str) -> str:
    """
    Extracts all text from a PDF file at the given path.

    Returns the text from every page concatenated with newlines.
    Skips problematic pages without raising, and raises IngestionError if the entire file read fails.

    Args:
        path (str): The filesystem path to the PDF file.

    Returns:
        str: The concatenated text of all successfully extracted pages.

    Raises:
        IngestionError: If the PDF file cannot be read or processed at all.
    """
    try:
        # Open the PDF with PdfReader
        reader = PdfReader(path)
        text_parts: list[str] = []
        # Iterate through each page in the PDF
        for page in reader.pages:
            try:
                # Try to extract the text from the page
                text_parts.append(page.extract_text() or "")
            except Exception:
                # Skip problematic page, don't fail the entire file.
                continue
        # Concatenate all the extracted text parts with newlines and strip any extra whitespace
        return "\n".join(text_parts).strip()
    except Exception as e:
        # If the PDF cannot be read at all, raise an ingestion-specific error
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