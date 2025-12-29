from __future__ import annotations

class EmbedOpsError(Exception):
    """Base exception for EmbedOps."""

class ConfigError(EmbedOpsError):
    """Invalid or missing configuration/environment variables."""

class IngestionError(EmbedOpsError):
    """Document ingestion failures."""

class ProcessingError(EmbedOpsError):
    """Text processing/chunking failures."""

class EmbeddingError(EmbedOpsError):
    """Embedding model or embedding generation failures."""

class VectorStoreError(EmbedOpsError):
    """Vector database (Pinecone) errors."""