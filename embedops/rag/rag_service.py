from __future__ import annotations

from typing import Dict, Any, List, Optional

from embedops.retrieval.service import RetrievalResources, retrieve
from embedops.llm.llm_client import LLMClient


def _format_context(hits: List[Dict[str, Any]], max_chars: int = 3500) -> str:
    parts: List[str] = []
    used = 0

    for h in hits:
        txt = (h.get("text_preview") or "").strip()
        if not txt:
            continue

        cite = f"[source={h.get('source')} doc_id={h.get('doc_id')} chunk_id={h.get('chunk_id')}]"
        block = f"{cite}\n{txt}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)

    return "\n".join(parts).strip()


def answer_question(
    resources: RetrievalResources,
    llm: LLMClient,
    query: str,
    top_k: int = 5,
    score_threshold: Optional[float] = None,
    doc_id: Optional[str] = None,
    source: Optional[str] = None,
    version: Optional[int] = None,
    namespace: Optional[str] = None,
    max_context_chars: int = 3500,
    temperature: float = 0.2,
    max_tokens: int = 300,
) -> Dict[str, Any]:
    hits = retrieve(
        resources=resources,
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        doc_id=doc_id,
        source=source,
        version=version,
        namespace=namespace,
    )

    context = _format_context(hits, max_chars=max_context_chars)

    system = (
        "You are a concise assistant. Answer using ONLY the provided context. "
        "If the context is insufficient, say: 'I don't have enough context to answer that.' "
        "Cite sources using the bracketed citation lines."
    )
    user = f"Question:\n{query}\n\nContext:\n{context if context else '[no context]'}"

    resp = llm.generate(
        prompt=f"{system}\n\n{user}",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = llm.extract_text(resp)

    return {
        "answer": content,
        "hits": hits,
        "used_context_chars": len(context),
        "llm_model": llm.model,
    }