from __future__ import annotations

import time
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from embedops.errors import (
    ConfigError,
    EmbeddingError,
    VectorStoreError,
    ProcessingError,
)
from embedops.retrieval.service import init_resources, retrieve, RetrievalResources
from embedops.llm.llm_client import LLMClient
from embedops.rag.rag_service import answer_question

app = FastAPI(
    title="EmbedOps Retrieval API",
    version="0.3.0",
    description="Semantic retrieval service (Pinecone + chunk store) for RAG pipelines. Retrieval and LLM are cleanly separated.",
)

RESOURCES: Optional[RetrievalResources] = None
LLM: Optional[LLMClient] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question or search query.")
    top_k: int = Field(5, ge=1, le=20, description="Number of matches to return.")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score.")

    # Enterprise filters (metadata)
    doc_id: Optional[str] = Field(None, description="Filter by a single document id.")
    source: Optional[str] = Field(None, description="Filter by source filename.")
    version: Optional[int] = Field(None, ge=0, description="Filter by document/version integer.")

    # Optional override (useful for embedding version rollouts)
    namespace: Optional[str] = Field(None, description="Override Pinecone namespace (default from env).")


class Hit(BaseModel):
    score: float
    doc_id: str
    chunk_id: int
    source: str
    version: Optional[int] = None
    namespace: str
    text_preview: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    top_k: int
    score_threshold: Optional[float]
    filters: Dict[str, Any]
    latency_ms: int
    hits: List[Hit]


class RagRequest(QueryRequest):
    max_context_chars: int = Field(
        3500, ge=500, le=12000, description="Max characters of retrieved context to send to the LLM."
    )
    temperature: float = Field(0.2, ge=0.0, le=1.0, description="LLM sampling temperature.")
    max_tokens: int = Field(300, ge=50, le=2000, description="Max tokens to generate.")


class RagResponse(BaseModel):
    query: str
    answer: str
    llm_model: str
    used_context_chars: int
    latency_ms: int
    hits: List[Hit]


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.on_event("startup")
def startup():
    """
    Initialize heavy resources once:
    - Retrieval resources (embedding model + Pinecone index handle)
    - Optional LLM client (enabled only if LLM is configured)
    """
    global RESOURCES, LLM

    # Retrieval resources must be available for both /query and /rag/answer
    RESOURCES = init_resources()

    # LLM is optional; /rag/answer will return 501 if not configured
    try:
        LLM = LLMClient()
    except Exception:
        LLM = None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "retrieval_ready": RESOURCES is not None,
        "llm_enabled": LLM is not None,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    global RESOURCES
    if RESOURCES is None:
        raise HTTPException(status_code=503, detail="Service not ready (resources not initialized).")

    start = time.time()
    try:
        hits = retrieve(
            resources=RESOURCES,
            query=req.query,
            top_k=req.top_k,
            score_threshold=req.score_threshold,
            doc_id=req.doc_id,
            source=req.source,
            version=req.version,
            namespace=req.namespace,
        )

        latency_ms = int((time.time() - start) * 1000)

        normalized = [
            Hit(
                score=float(h["score"]),
                doc_id=str(h["doc_id"]),
                chunk_id=int(h["chunk_id"]),
                source=str(h["source"]),
                version=h.get("version"),
                namespace=str(h["namespace"]),
                text_preview=h.get("text_preview"),
            )
            for h in hits
        ]

        return QueryResponse(
            query=req.query,
            top_k=req.top_k,
            score_threshold=req.score_threshold,
            filters={
                "doc_id": req.doc_id,
                "source": req.source,
                "version": req.version,
                "namespace": req.namespace,
            },
            latency_ms=latency_ms,
            hits=normalized,
        )

    except (ConfigError, EmbeddingError, VectorStoreError, ProcessingError) as e:
        msg = f"{type(e).__name__}: {e}"
        status = 400 if isinstance(e, (ConfigError, ProcessingError)) else 500
        raise HTTPException(status_code=status, detail=msg) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e


@app.post("/rag/answer", response_model=RagResponse)
def rag_answer(req: RagRequest) -> RagResponse:
    global RESOURCES, LLM
    if RESOURCES is None:
        raise HTTPException(status_code=503, detail="Service not ready (resources not initialized).")
    if LLM is None:
        raise HTTPException(
            status_code=501,
            detail="LLM not configured. Set LLM_API_KEY (and optionally LLM_BASE_URL, LLM_MODEL) to enable /rag/answer.",
        )

    start = time.time()
    try:
        result = answer_question(
            resources=RESOURCES,
            llm=LLM,
            query=req.query,
            top_k=req.top_k,
            score_threshold=req.score_threshold,
            doc_id=req.doc_id,
            source=req.source,
            version=req.version,
            namespace=req.namespace,
            max_context_chars=req.max_context_chars,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )

        latency_ms = int((time.time() - start) * 1000)

        hits = result.get("hits", [])
        normalized_hits = [
            Hit(
                score=float(h["score"]),
                doc_id=str(h["doc_id"]),
                chunk_id=int(h["chunk_id"]),
                source=str(h["source"]),
                version=h.get("version"),
                namespace=str(h["namespace"]),
                text_preview=h.get("text_preview"),
            )
            for h in hits
        ]

        return RagResponse(
            query=req.query,
            answer=str(result.get("answer", "")).strip(),
            llm_model=str(result.get("llm_model", "")),
            used_context_chars=int(result.get("used_context_chars", 0)),
            latency_ms=latency_ms,
            hits=normalized_hits,
        )

    except (ConfigError, EmbeddingError, VectorStoreError, ProcessingError) as e:
        msg = f"{type(e).__name__}: {e}"
        status = 400 if isinstance(e, (ConfigError, ProcessingError)) else 500
        raise HTTPException(status_code=status, detail=msg) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {e}") from e