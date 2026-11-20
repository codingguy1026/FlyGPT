"""Isolated local worker bridge for Papiti v0.6.1 brain-conditioned speech.

No external endpoint, pretrained model, answer table, or silent downloads.
The subprocess uses a trusted local checkpoint and receives ONLY the upstream
completed utterance plan; it never receives a calculator implementation.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import re
import subprocess
import sys
from threading import Lock, Thread
import time
import unicodedata
from typing import Any

from papiti_lm.v0_6_1_data import (
    EVERYDAY_SPEECH_ACTS, SOCIAL_SPEECH_ACTS, is_korean_greeting, packet_text,
)
from semantic_planner import INTENT_FACT_IDS

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "papiti_language_v0_6_1.pt"
_REPEATED_WORD = re.compile(r"(?<!\w)([가-힣A-Za-z]{2,})\s+\1(?!\w)")
REPLY_HISTORY_LIMIT = 2

# These speech acts have supervised examples in the small v0.6.1 corpus.
# Other plans (notably live-search status and arbitrary sourced excerpts)
# must go to a separately configured language model, not this tiny speaker.
V061_TRAINED_SPEECH_ACTS = SOCIAL_SPEECH_ACTS.union(INTENT_FACT_IDS, ("state_exact_math_result",))


def reply_key(text: str) -> str:
    """Treat changes in case, punctuation, and spacing as the same wording."""
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    words = "".join(char for char in normalized if char.isalnum())
    return hashlib.sha256((words or normalized).encode("utf-8")).hexdigest()


def recent_reply_keys(
    message: str, history: list[dict[str, Any]] | None,
) -> tuple[str, ...]:
    """Use only recent replies to this question from the current chat."""
    def question_key(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", text).casefold().split()).rstrip(" .!?~")

    key = question_key(message)
    turns = history or []
    avoided: dict[str, None] = {}
    for index in range(len(turns) - 2, -1, -1):
        previous, reply = turns[index:index + 2]
        if (previous.get("role") == "user" and reply.get("role") == "assistant"
                and question_key(str(previous.get("content", ""))) == key
                and isinstance(reply.get("content"), str) and reply["content"].strip()):
            avoided[reply_key(reply["content"])] = None
            if len(avoided) == REPLY_HISTORY_LIMIT:
                break
    return tuple(avoided)


def supports_v061_plan(plan: dict[str, Any]) -> bool:
    """Use the scratch speaker only for supervised, verifiable meaning types."""
    if not isinstance(plan, dict) or plan.get("speech_act") not in V061_TRAINED_SPEECH_ACTS:
        return False
    try:
        packet_text(plan)
    except ValueError:
        return False
    return True


def _preserves_missing_status(speech_act: str, answer: str) -> bool:
    """Conservative language-aware guard for evidence/memory absence replies.

    This is deliberately not a general semantic classifier. It rejects obvious
    unrelated text, including the model's memorized project introduction.
    """
    if speech_act == "report_no_external_evidence":
        subject_ko = r"자료|출처|정보|검색|근거|확인|검증|증거|알\s*수"
        subject_en = r"source|evidence|information|search|verif|data|answer|know"
    elif speech_act == "report_no_memory":
        subject_ko = r"기억|이전|대화|기록|찾|과거"
        subject_en = r"memory|previous|earlier|conversation|recall|find|found"
    else:
        return True
    missing_ko = r"없|못|않|어렵|불가|부족|미확인|불가능|모르|제공할\s*수"
    missing_en = r"\b(?:not|no|cannot|can't|unavailable|lack|insufficient|unable|don't|unknown)\b"
    return bool(
        (re.search(subject_ko, answer) and re.search(missing_ko, answer))
        or (re.search(subject_en, answer, re.I) and re.search(missing_en, answer, re.I))
    )


def checkpoint_path() -> Path:
    path = os.environ.get("FLYGPT_HYBRID_SPEECH_CHECKPOINT", "").strip()
    return Path(path).expanduser() if path else DEFAULT_CHECKPOINT


def _speaker_checkpoints(plan: dict[str, Any]) -> tuple[Path, ...]:
    base = checkpoint_path()
    act = plan.get("speech_act")
    prose = base.with_stem(base.stem + "_prose")
    if (act in SOCIAL_SPEECH_ACTS or act in INTENT_FACT_IDS) and prose.is_file():
        return (prose,)
    greetings = base.with_stem(base.stem + "_greetings")
    if is_korean_greeting(plan) and greetings.is_file():
        return (greetings,)
    dialogue = base.with_name(f"{base.stem}_dialogue{base.suffix}")
    if act in EVERYDAY_SPEECH_ACTS:
        return (dialogue,)
    if act in SOCIAL_SPEECH_ACTS or act in INTENT_FACT_IDS:
        social = base.with_name(f"{base.stem}_social{base.suffix}")
        if social.is_file():
            return (social, dialogue) if dialogue.is_file() else (social,)
        if dialogue.is_file():
            return (base, dialogue) if base.is_file() else (dialogue,)
    return (base,)


def checkpoint_present(plan: dict[str, Any] | None = None) -> bool:
    path = checkpoint_path() if plan is None else _speaker_checkpoints(plan)[0]
    return path.is_file()


def speech_python() -> str:
    configured = os.environ.get("FLYGPT_HYBRID_SPEECH_PYTHON", "").strip()
    if configured:
        return configured
    for virtual_name in (".venv-papiti", ".venv-train"):
        virtual = ROOT / virtual_name / "bin" / "python"
        if virtual.is_file():
            return str(virtual)
    return sys.executable


def check_candidate(
    plan: dict[str, Any], answer: str, *, avoid_reply_keys: tuple[str, ...] = (),
) -> str | None:
    """Reject unsupported/hallucinated outputs; never manufacture a reply."""
    try:
        packet_text(plan)
    except ValueError:
        return "invalid_semantic_packet"
    if not answer or not answer.strip():
        return "empty_speech"
    if len(answer) > 900 or "\ufffd" in answer or answer.lstrip().startswith(("{", "[", chr(96))):
        return "malformed_speech"
    if _REPEATED_WORD.search(answer):
        return "repetitive_speech"
    speech_act = plan.get("speech_act")
    units = plan["content_units"]
    if not _preserves_missing_status(str(speech_act), answer):
        return "missing_status_semantics"
    if speech_act == "state_exact_math_result":
        if len(units) != 1 or units[0]["kind"] != "exact_result":
            return "invalid_exact_result"
        source = units[0]["value"]
        prefix = "Exact math tool result: "
        if not source.startswith(prefix) or answer.strip() != source[len(prefix):].strip():
            return "exact_result_not_preserved"
    protected = {"grounded_project_fact", "verified_knowledge",
                 "user_asserted_context", "retrieved_memory",
                 "retrieved_web_excerpt"}
    for unit in units:
        if unit["kind"] in protected and unit["value"] not in answer:
            return "semantic_content_not_preserved"
    # The caller can only show retrieved material with source attribution.
    if speech_act == "report_retrieved_web_evidence":
        for unit in units:
            url = unit.get("url")
            if (not isinstance(url, str) or not url.startswith("https://")
                    or url not in answer):
                return "web_source_attribution_missing"
    language = str((plan.get("style") or {}).get("language") or "")
    if language == "ko" and speech_act in SOCIAL_SPEECH_ACTS and re.search(r"[A-Za-z]", answer):
        return "wrong_language"
    if language == "ko" and not re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", answer):
        if speech_act != "state_exact_math_result":
            return "wrong_language"
    if language == "en" and re.search(r"[가-힣]", answer):
        return "wrong_language"
    if (speech_act != "state_exact_math_result" and avoid_reply_keys
            and reply_key(answer) in avoid_reply_keys):
        return "repeated_reply"
    return None


def _worker_command(python: str) -> list[str]:
    return [python, "-m", "papiti_lm.v0_6_1", "infer-worker"]


class _SpeechWorker:
    def __init__(self, python: str):
        self.python = python
        self.process = subprocess.Popen(
            _worker_command(python), cwd=ROOT, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        self.requests: SimpleQueue[str | None] = SimpleQueue()
        self.replies: SimpleQueue[str | Exception] = SimpleQueue()
        Thread(target=self._exchange, name="papiti-speech-io", daemon=True).start()

    def _exchange(self) -> None:
        # Pipe reads *and* writes happen off the request thread so slow startup
        # or a wedged child cannot block the caller beyond its deadline.
        assert self.process.stdin is not None and self.process.stdout is not None
        try:
            with self.process.stdin as source, self.process.stdout as output:
                for request in iter(self.requests.get, None):
                    source.write(request + "\n")
                    source.flush()
                    reply = output.readline()
                    if not reply:
                        raise RuntimeError("v0.6.1 speech worker exited")
                    self.replies.put(reply)
        except (OSError, RuntimeError, ValueError) as exc:
            self.replies.put(exc)

    def request(self, payload: dict[str, Any], deadline: float) -> dict[str, Any]:
        self.requests.put(json.dumps(payload, ensure_ascii=False))
        try:
            reply = self.replies.get(timeout=max(0.0, deadline - time.monotonic()))
        except Empty as exc:
            raise TimeoutError("v0.6.1 speech worker exceeded its deadline") from exc
        if isinstance(reply, Exception):
            raise reply
        result = json.loads(reply)
        if not isinstance(result, dict):
            raise RuntimeError("Invalid v0.6.1 speech worker response")
        return result

    def close(self) -> None:
        self.requests.put(None)
        self.process.kill()  # Interrupt even a timed-out inference.
        try:
            self.process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            Thread(target=self.process.wait, daemon=True).start()


_worker_lock = Lock()
_worker: _SpeechWorker | None = None


def _stop_worker() -> None:
    global _worker
    worker, _worker = _worker, None
    if worker is not None:
        worker.close()


def _get_worker() -> _SpeechWorker:
    global _worker
    python = speech_python()
    if _worker is None or _worker.python != python or _worker.process.poll() is not None:
        _stop_worker()
        _worker = _SpeechWorker(python)
    return _worker


def warm_speech_worker() -> None:
    """Start importing PyTorch before the first chat, if weights are installed."""
    base = checkpoint_path()
    candidates = (base, base.with_stem(base.stem + "_social"),
                  base.with_stem(base.stem + "_dialogue"),
                  base.with_stem(base.stem + "_greetings"), base.with_stem(base.stem + "_prose"))
    if any(path.is_file() for path in candidates):
        with _worker_lock:
            _get_worker()


@atexit.register
def stop_speech_worker() -> None:
    with _worker_lock:
        _stop_worker()


def _request_speech(payload: dict[str, Any], deadline: float) -> str:
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not _worker_lock.acquire(timeout=remaining):
        raise TimeoutError("v0.6.1 speech worker did not become available in time")
    try:
        if time.monotonic() >= deadline:
            raise TimeoutError("v0.6.1 speech budget exhausted")
        try:
            result = _get_worker().request(payload, deadline)
        except (OSError, RuntimeError, ValueError):
            # Includes TimeoutError. Never let a late reply become another
            # request's answer. The next call starts a clean worker.
            _stop_worker()
            raise
    finally:
        _worker_lock.release()

    if result.get("error") == "FileNotFoundError":
        raise FileNotFoundError("v0.6.1 speech checkpoint disappeared")
    answer = result.get("answer")
    if not isinstance(answer, str):
        raise RuntimeError("v0.6.1 neural speech worker could not generate a reply")
    return answer.strip()


def generate_speech(
    plan: dict[str, Any], timeout_seconds: float = 20.0, *,
    avoid_reply_keys: tuple[str, ...] = (),
) -> str:
    """Run only the trained Transformer, reusing weights across messages."""
    deadline = time.monotonic() + min(30.0, max(0.0, float(timeout_seconds)))
    packet_text(plan)  # Validate before sending to the isolated speaker.
    path, *alternatives = _speaker_checkpoints(plan)
    if not path.is_file():
        raise FileNotFoundError(f"v0.6.1 checkpoint missing: {path}")
    base = checkpoint_path()
    return _request_speech({
        "plan": plan,
        "checkpoint": str(path.resolve()),
        "base_checkpoint": str(base.resolve()) if path != base and base.is_file() else None,
        "alternate_checkpoint": str(alternatives[0].resolve()) if alternatives else None,
        "avoid_reply_keys": avoid_reply_keys[:REPLY_HISTORY_LIMIT],
    }, deadline)
