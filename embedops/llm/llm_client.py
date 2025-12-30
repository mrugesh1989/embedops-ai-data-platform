from __future__ import annotations

import os
import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, List

from embedops.errors import ConfigError


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None or str(v).strip() == "":
        raise ConfigError(f"Missing required environment variable: {name}")
    return str(v).strip()


class LLMClient:
    """
    Minimal, provider-agnostic client for OpenAI-compatible chat completions APIs.
    Works with OpenAI, Azure OpenAI (with compatible gateway), and many self-hosted servers
    (vLLM, Ollama w/ OpenAI adapter, etc.) if they expose /v1/chat/completions.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: int = 60,
    ):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.timeout_s = timeout_s

        if not self.api_key:
            raise ConfigError("OPENAI_API_KEY is missing. Set it in .env to enable /rag/answer.")

    def chat_completions(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 300,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"LLM HTTPError {e.code}: {body}") from e
        except Exception as e:
            raise RuntimeError(f"LLM request failed: {e}") from e