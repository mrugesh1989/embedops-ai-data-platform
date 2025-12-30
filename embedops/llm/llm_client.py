from __future__ import annotations

import os
import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

from embedops.errors import ConfigError


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None or str(v).strip() == "":
        raise ConfigError(f"Missing required environment variable: {name}")
    return str(v).strip()


class LLMClient:
    """
    Minimal Ollama client using the native REST endpoint (/api/generate).
    Zero Python deps beyond stdlib.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: int = 120,
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "None")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "None")
        self.timeout_s = timeout_s

        if not self.model:
            raise ConfigError("OLLAMA_MODEL is empty.")
        if not self.base_url:
            raise ConfigError("OLLAMA_BASE_URL is empty.")

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 300) -> Dict[str, Any]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"Ollama HTTPError {e.code}: {body}") from e
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e

    @staticmethod
    def extract_text(resp: Dict[str, Any]) -> str:
        return str(resp.get("response", "")).strip()