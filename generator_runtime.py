from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


GENERATIVE_ROUTES = {"general", "code", "summarize"}


@dataclass
class GenerationResult:
    used: bool
    provider: str
    model: str | None
    answer: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GeneratorRuntime:
    """Small provider-agnostic generation layer for FlyGPT v0.4.

    The server stays usable with no generator configured. When
    FLYGPT_GENERATOR_URL and FLYGPT_GENERATOR_MODEL are set, requests are sent
    to an OpenAI-compatible chat-completions endpoint.
    """

    def __init__(self) -> None:
        self.url = os.environ.get("FLYGPT_GENERATOR_URL", "").strip()
        self.model = os.environ.get("FLYGPT_GENERATOR_MODEL", "").strip()
        self.api_key = os.environ.get("FLYGPT_GENERATOR_API_KEY", "").strip()
        self.timeout = float(os.environ.get("FLYGPT_GENERATOR_TIMEOUT", "45"))

    @property
    def configured(self) -> bool:
        return bool(self.url and self.model)

    @property
    def provider_name(self) -> str:
        if not self.configured:
            return "fallback"
        return os.environ.get("FLYGPT_GENERATOR_PROVIDER", "compatible-http").strip() or "compatible-http"

    def status(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "provider": self.provider_name,
            "model": self.model or None,
            "url_configured": bool(self.url),
            "api_key_configured": bool(self.api_key),
        }

    def _system_prompt(self, route: str) -> str:
        common = (
            "You are FlyGPT v0.4, a concise experimental assistant. "
            "Answer the user's request directly in the user's language. "
            "Do not claim that you searched the web or remembered prior chats unless "
            "that information was explicitly provided in the current request. "
        )

        route_prompts = {
            "general": (
                "The router selected GENERAL. Give a clear factual explanation. "
                "If uncertain, say what is uncertain instead of inventing facts."
            ),
            "code": (
                "The router selected CODE. Give practical programming help. "
                "Prefer a small correct example over a huge code dump. "
                "Mention assumptions when the request is underspecified."
            ),
            "summarize": (
                "The router selected SUMMARIZE. Summarize only material present in "
                "the user's request. Do not add outside facts."
            ),
        }

        return common + route_prompts.get(route, route_prompts["general"])

    def generate(self, message: str, route: str) -> GenerationResult:
        if route not in GENERATIVE_ROUTES:
            return GenerationResult(
                used=False,
                provider=self.provider_name,
                model=self.model or None,
                answer=None,
            )

        if not self.configured:
            return GenerationResult(
                used=False,
                provider="fallback",
                model=None,
                answer=None,
            )

        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self._system_prompt(route),
                },
                {
                    "role": "user",
                    "content": message,
                },
            ],
            "temperature": 0.35,
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            self.url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))

            choices = payload.get("choices") or []
            if not choices:
                raise ValueError("generator response did not contain choices")

            message_obj = choices[0].get("message") or {}
            answer = message_obj.get("content")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("generator response did not contain message content")

            return GenerationResult(
                used=True,
                provider=self.provider_name,
                model=self.model,
                answer=answer.strip(),
            )

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            return GenerationResult(
                used=False,
                provider=self.provider_name,
                model=self.model,
                answer=None,
                error=f"{type(exc).__name__}: {exc}",
            )
