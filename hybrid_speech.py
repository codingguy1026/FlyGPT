"""Isolated local subprocess bridge for Papiti v0.6.1 brain-conditioned speech.

No external endpoint, pretrained model, answer table, or silent downloads.
The subprocess uses a trusted local checkpoint and receives ONLY the upstream
completed utterance plan; it never receives a calculator implementation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from papiti_lm.v0_6_1_data import packet_text

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "papiti_language_v0_6_1.pt"
_REPEATED_WORD = re.compile(r"(?<!\w)([가-힣A-Za-z]{2,})\s+\1(?!\w)")

# These speech acts have supervised examples in the small v0.6.1 corpus.
# Other plans (notably live-search status and arbitrary sourced excerpts)
# must go to a separately configured language model, not this tiny speaker.
V061_TRAINED_SPEECH_ACTS = frozenset({
    "return_greeting",
    "acknowledge_thanks",
    "return_farewell",
    "return_welcome",
    "request_specific_task",
    "introduce_flygpt",
    "describe_supported_features",
    "explain_malecns",
    "state_exact_math_result",
})


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


def checkpoint_present() -> bool:
    return checkpoint_path().is_file()


def speech_python() -> str:
    configured = os.environ.get("FLYGPT_HYBRID_SPEECH_PYTHON", "").strip()
    if configured:
        return configured
    for virtual_name in (".venv-papiti", ".venv-train"):
        virtual = ROOT / virtual_name / "bin" / "python"
        if virtual.is_file():
            return str(virtual)
    return sys.executable


def check_candidate(plan: dict[str, Any], answer: str) -> str | None:
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
    if language == "ko" and not re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", answer):
        if speech_act != "state_exact_math_result":
            return "wrong_language"
    if language == "en" and re.search(r"[가-힣]", answer):
        return "wrong_language"
    return None


def generate_speech(plan: dict[str, Any], timeout_seconds: float = 20.0) -> str:
    """Run *only* the trained Transformer, no Native realizer inside speaker."""
    packet_text(plan)  # Validate before serializing/untrusted subprocess input.
    path = checkpoint_path()
    if not path.is_file():
        raise FileNotFoundError(f"v0.6.1 checkpoint missing: {path}")
    seconds = min(30.0, max(1.0, float(timeout_seconds)))
    runner = subprocess.run(
        [speech_python(), "-m", "papiti_lm.v0_6_1", "infer",
         "--checkpoint", str(path), "--max-new-tokens", "96"],
        input=json.dumps(plan, ensure_ascii=False),
        text=True, capture_output=True, timeout=seconds, cwd=ROOT,
        check=False,
    )
    if runner.returncode:
        raise RuntimeError(
            "v0.6.1 neural speech subprocess failed: "
            + runner.stderr.strip()[-300:]
        )
    return runner.stdout.strip()
