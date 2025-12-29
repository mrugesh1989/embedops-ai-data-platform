from __future__ import annotations

from dataclasses import dataclass

from embedops.errors import ProcessingError


@dataclass(frozen=True)
class ChunkConfig:
    chunk_words: int = 400
    overlap_words: int = 80


def _validate_cfg(cfg: ChunkConfig) -> None:
    if not isinstance(cfg.chunk_words, int) or cfg.chunk_words <= 0:
        raise ProcessingError("chunk_words must be a positive integer.")
    if not isinstance(cfg.overlap_words, int) or cfg.overlap_words < 0:
        raise ProcessingError("overlap_words must be a non-negative integer.")
    if cfg.overlap_words >= cfg.chunk_words:
        raise ProcessingError("overlap_words must be strictly less than chunk_words.")


def chunk_text(text: str, cfg: ChunkConfig) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    Raises ProcessingError on invalid config.
    Returns [] for empty/whitespace-only text.
    """
    _validate_cfg(cfg)

    if text is None:
        raise ProcessingError("Input text is None.")
    text = text.strip()
    if not text:
        return []

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + cfg.chunk_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        # Move start with overlap
        start = max(0, end - cfg.overlap_words)

        # Safety: ensure progress even if overlap misconfigured (shouldn't happen after validation)
        if start >= end:
            start = end

    return chunks