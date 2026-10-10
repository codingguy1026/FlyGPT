"""Loopback-only policy for self-hosted FlyGPT language generation.

This intentionally protects *LLM generation* traffic only. neuPrint and the
optional research connector are separate network services.
"""

from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlsplit

DEFAULT_LOCAL_GENERATOR_URL = "http://127.0.0.1:11434/v1/chat/completions"
DEFAULT_LOCAL_MODEL = "qwen2.5:0.5b-instruct"


def enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def is_local_generator_url(url: str) -> bool:
    """Accept only direct HTTP requests to literal loopback or localhost.

    Reject cloud endpoints, HTTPS, userinfo, invalid ports and unusual URLs.
    In local-only mode, the caller must also disable proxies and redirects.
    """
    try:
        parts = urlsplit(url.strip())
        if (
            parts.scheme.lower() != "http"
            or not parts.netloc
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.fragment
        ):
            return False
        port = parts.port
        if port is not None and not (1 <= port <= 65535):
            return False
        host = parts.hostname.lower()
        if host == "localhost":
            return True
        return ip_address(host).is_loopback
    except (ValueError, AttributeError):
        return False
