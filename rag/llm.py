"""Minimal OpenAI-compatible chat client.

Works with any server that speaks the OpenAI ``/chat/completions`` protocol:
OpenAI itself, Azure OpenAI, Ollama, LM Studio, vLLM, llama.cpp, Jan, etc.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from .config import RagConfig

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class LLMClient:
    def __init__(self, config: RagConfig):
        self.config = config
        self.base_url = (config.llm_base_url or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = config.llm_api_key or os.environ.get("OPENAI_API_KEY")

    def _headers(self) -> dict[str, str]:
        if self.base_url == DEFAULT_BASE_URL and not self.api_key:
            raise ValueError(
                "No LLM API key configured. Either set RAG_LLM_API_KEY (or "
                "OPENAI_API_KEY), or point RAG_LLM_BASE_URL at a local "
                "OpenAI-compatible server such as Ollama "
                "(http://localhost:11434/v1)."
            )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": (
                self.config.temperature if temperature is None else temperature
            ),
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        response = requests.post(
            url, headers=self._headers(), json=payload, timeout=120
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
