"""Preserve solver authority while optionally running a bounded language stage."""
import os
import re
from time import perf_counter


def realize_verified(answer, generator):
    meta = {"stage": "verified_language_realization_v1", "attempted": False,
            "accepted": False, "fallback": True, "model": getattr(generator, "model", None)}
    if not answer or not getattr(generator, "configured", False):
        meta["reason"] = "generator_unavailable"
        return answer, meta
    if getattr(generator, "local_only", False) and not getattr(generator, "local_endpoint_allowed", False):
        meta["reason"] = "endpoint_blocked"
        return answer, meta
    try:
        seconds = max(1.0, min(60.0, float(os.getenv("FLYGPT_VERIFIED_REALIZER_TIMEOUT", "12"))))
    except ValueError:
        seconds = 12.0
    meta["attempted"] = True
    started = perf_counter()
    try:
        result = generator._generate_once(
            model=generator.model,
            messages=[
                {"role": "system", "content": "Only adjust whitespace and line breaks of the provided verified Korean text. Never change, add, or remove any word, fact, number, equation or punctuation. Output only the result."},
                {"role": "user", "content": answer},
            ],
            timeout_seconds=seconds,
            max_tokens_override=256,
        )
        candidate = result.answer if result.used else None
        meta["latency_ms"] = round((perf_counter() - started) * 1000)
        if candidate and re.sub(r"\s+", "", candidate) == re.sub(r"\s+", "", answer):
            meta.update({"accepted": True, "fallback": False, "reason": "unchanged_content"})
            return candidate, meta
        meta["reason"] = "content_changed" if candidate else "generator_failed"
    except Exception:
        meta["reason"] = "generator_failed"
    return answer, meta
