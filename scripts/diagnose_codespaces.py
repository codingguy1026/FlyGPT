#!/usr/bin/env python3
"""Check FlyGPT local health and Codespaces forwarded access without secrets.

Run `make doctor` in the *same Codespace* as `make run`. The public URL
probe is unauthenticated by design: private Codespaces ports need an authenticated
browser session and a 404 there is NOT proof of an application-level 404.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    status: int | None
    body: str = ""
    error: str = ""


def probe(url: str, *, host: str | None = None) -> Probe:
    headers = {"User-Agent": "FlyGPT-codespaces-doctor/1.0"}
    if host:
        headers["Host"] = host
    request = urllib.request.Request(url, headers=headers)
    # Avoid a workstation/CI proxy affecting localhost results.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=4) as response:
            return Probe(response.status, response.read(32000).decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return Probe(exc.code, error="HTTP " + str(exc.code))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Never print URL credentials, tokens, or private environment contents.
        return Probe(None, error=type(exc).__name__)


def is_backend_ok(result: Probe) -> bool:
    if result.status != 200:
        return False
    try:
        return json.loads(result.body).get("status") == "ok"
    except (ValueError, AttributeError):
        return False


def forwarded_address(environment: dict[str, str]) -> str | None:
    name = environment.get("CODESPACE_NAME", "").strip()
    domain = environment.get(
        "GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN", ""
    ).strip()
    if not name:
        return None
    # GitHub's documented default only if Codespaces omitted the domain.
    domain = domain or "app.github.dev"
    # Values come from GitHub's process environment, not untrusted user input.
    if not all(part.isalnum() or part == "-" for part in name):
        return None
    if not all(part.isalnum() or part in ".-" for part in domain):
        return None
    return f"https://{name}-3000.{domain}/"


def diagnose(*, external: bool = True) -> bool:
    print("🪰 FlyGPT Codespaces doctor")
    frontend = probe("http://127.0.0.1:3000/")
    backend = probe("http://127.0.0.1:8000/api/health")
    proxied = probe("http://127.0.0.1:3000/api/health")

    front_ok = frontend.status == 200 and "FlyGPT" in frontend.body
    back_ok = is_backend_ok(backend)
    proxy_ok = is_backend_ok(proxied)
    for name, result, ok in (
        ("Frontend /", frontend, front_ok),
        ("Backend /api/health", backend, back_ok),
        ("Frontend /api/health rewrite", proxied, proxy_ok),
    ):
        print(f" {'✅' if ok else '❌'} {name}: HTTP {result.status or 'unreachable'}")
    if frontend.status == 200 and not front_ok:
        print(" ⚠️ Port 3000 answered, but the HTML did not look like FlyGPT.")
    if front_ok and back_ok and not proxy_ok:
        print(" ⚠️ Frontend API rewrite failed; check frontend/next.config.mjs.")

    url = forwarded_address(os.environ)
    if not url:
        print(" ℹ️ Codespace address unavailable; open port 3000 from VS Code PORTS.")
    else:
        print(f" 🔗 Expected Codespaces frontend: {url}")
        host = urllib.request.urlsplit(url).hostname
        host_probe = probe("http://127.0.0.1:3000/", host=host)
        host_ok = host_probe.status == 200
        print(
            f" {'✅' if host_ok else '⚠️'} Frontend with forwarded Host header: "
            f"HTTP {host_probe.status or 'unreachable'}"
        )
        if external:
            remote = probe(url)
            print(
                " ℹ️ Unauthenticated forwarded address: "
                f"HTTP {remote.status or 'unreachable'}"
            )
            if remote.status == 404 and front_ok:
                print(
                    " ⚠️ Local frontend is healthy but the forwarded link returned 404. "
                    "This may be the Codespaces forwarding layer, an old URL, "
                    "or a PRIVATE port requiring a logged-in GitHub browser."
                )
                print(
                    "    In VS Code PORTS, remove/re-add port 3000 and use "
                    "Open in Browser / Copy Forwarded Address."
                )
            elif remote.status is None:
                print(
                    " ⚠️ External probe was unreachable from this terminal. "
                    "Use the authenticated link shown in PORTS."
                )
        print(
            " 🔒 Keep your port private unless you deliberately want to share "
            "the running app. Never paste GITHUB_TOKEN into chat."
        )

    if front_ok and back_ok and proxy_ok:
        print(" ✅ FlyGPT's local serving path is healthy.")
    else:
        print(" ❌ Local serving needs attention; inspect the make run terminal.")
    return front_ok and back_ok and proxy_ok


if __name__ == "__main__":
    ok = diagnose(external="--local-only" not in sys.argv[1:])
    sys.exit(0 if ok else 1)
