#!/usr/bin/env python3
"""Sample local FlyGPT readiness repeatedly to catch intermittent failures.

Run while `make run` or `make stable` is active in a separate terminal.
No authentication tokens or external/private Codespaces URLs are probed.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

from scripts.diagnose_codespaces import Probe, is_backend_ok, probe

FRONTEND_URL = "http://127.0.0.1:3000/"
BACKEND_URL = "http://127.0.0.1:8000/api/health"
PROXY_URL = "http://127.0.0.1:3000/api/health"


def check_stability(
    *,
    samples: int = 6,
    interval_seconds: float = 1.0,
    prober: Callable[[str], Probe] = probe,
    sleeper: Callable[[float], None] = time.sleep,
) -> bool:
    if samples <= 0 or interval_seconds < 0:
        raise ValueError("samples must be positive and interval nonnegative")

    failures = {"frontend": 0, "backend": 0, "proxy": 0}
    print(f"🩺 FlyGPT local stability check ({samples} samples)")

    for i in range(samples):
        frontend = prober(FRONTEND_URL)
        backend = prober(BACKEND_URL)
        proxy = prober(PROXY_URL)
        states = {
            "frontend": frontend.status == 200 and "FlyGPT" in frontend.body,
            "backend": is_backend_ok(backend),
            "proxy": is_backend_ok(proxy),
        }
        for name, ok in states.items():
            if not ok:
                failures[name] += 1
        display = " ".join(
            f"{name}={'✅' if ok else '❌'}" for name, ok in states.items()
        )
        print(f" {i + 1}/{samples} {display}", flush=True)
        if i + 1 < samples:
            sleeper(interval_seconds)

    total = sum(failures.values())
    if total:
        print(
            " ❌ Intermittent or persistent local failures: "
            + ", ".join(f"{name}={count}/{samples}" for name, count in failures.items())
        )
        print("    Frontend failures: check port 3000 and Next.js logs.")
        print("    Backend failures: check port 8000 and Uvicorn logs.")
        print("    Proxy-only failures: check frontend/next.config.mjs and backend URL.")
        return False

    print(" ✅ All local probes succeeded. This does not test the private Codespaces gateway.")
    return True


if __name__ == "__main__":
    sys.exit(0 if check_stability() else 1)
