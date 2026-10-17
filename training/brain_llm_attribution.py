"""FlyGPT causal-role ablation with a REAL trained MaleCNS population router.

Three arms on the exact same *evaluation-only* questions:
  1. fly_only: real 166700-neuron router + existing deterministic brain/dispatch.
     No LLM is called; output is only what a completed evidence plan can state.
  2. llm_only: raw question directly to a fixed local Ollama model.
  3. fly_plus_llm: SAME fixed Ollama, but the exact FlyGPT generator message
     construction after genuine routing, dispatcher and brain-state planning.

This is NOT a statistically powered language benchmark; output quality on
non-math tasks needs independent human adjudication. Score task-route accuracy
and deterministic-math answer markers separately from fluency. Do not mistake
this for full connectome emergent reasoning or live fly action potentials.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain_runtime import FlyBrainRuntime
from dispatcher import dispatch
from generator_runtime import GeneratorRuntime
from full_population_runtime import FullPopulationRouterRuntime

DEFAULT_QUESTIONS = ROOT / "training/brain_llm_ablation_questions.json"
BASE_PROMPT = (
    "You are a helpful AI assistant. Answer the user's request directly, "
    "in the same language, with clear assumptions. Do not invent facts."
)


def ollama_answer(
    messages: list[dict[str, str]],
    *, model: str, endpoint: str, timeout: float,
) -> dict[str, Any]:
    """Call the SAME local model for both LLM arms. No external API keys."""
    payload = {
        "model": model, "messages": messages, "stream": False,
        "options": {"temperature": 0, "seed": 43, "num_predict": 176},
        "keep_alive": "5m",
    }
    start = time.monotonic()
    req = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as response:
            data = json.load(response)
        answer = str((data.get("message") or {}).get("content") or "").strip()
        return {
            "called": True, "answer": answer[:2500],
            "empty_answer": not bool(answer),
            "latency_ms": round((time.monotonic() - start) * 1000),
            "error": None,
        }
    except Exception as exc:
        return {
            "called": True, "answer": None, "empty_answer": True,
            "latency_ms": round((time.monotonic() - start) * 1000),
            "error": f"{type(exc).__name__}: {str(exc)[:150]}",
        }


def run_arm_brain_only(state: dict[str, Any], dispatcher: dict[str, Any]):
    """Extract only genuine authorized deterministic content, NOT fake prose."""
    if dispatcher["status"] == "uncertain":
        return {
            "answer": dispatcher["answer"], "answer_origin": "router_gate",
            "solved_open_ended_question": False, "llm_called": False,
        }
    plan = state.get("utterance_plan") or {}
    if not plan.get("ready"):
        return {
            "answer": None, "answer_origin": "unresolved_no_language_model",
            "solved_open_ended_question": False, "llm_called": False,
        }
    units = plan.get("content_units") or []
    fact_values = [str(u.get("value", "")).strip() for u in units
                   if str(u.get("value", "")).strip()]
    return {
        "answer": " ".join(fact_values) or None,
        "answer_origin": (
            "exact_tool" if plan.get("speech_act") == "state_exact_math_result"
            else "verified_or_template_plan"
        ),
        "speech_act": plan.get("speech_act"),
        "solved_open_ended_question": False,
        "llm_called": False,
    }


def evaluate_case(question: dict, router, brain, generator, args):
    prompt = str(question["prompt"])
    routed = router.predict(prompt, top_k=len(router.routes))
    dispatched = dispatch(prompt, routed).to_dict()
    initial = brain.plan(route_info=routed, dispatch=dispatched, message=prompt)
    state = brain.finalize(initial, tool_context=dispatched.get("tool_context"))
    task = state.get("task_plan") or {}
    plan = state.get("utterance_plan") or {}
    mouth_ready = bool(plan.get("ready"))
    bounded_ready = bool(task.get("ready") and task.get("contract") == "bounded_task_v1")
    gate_open = dispatched["status"] != "uncertain"

    # This is what the *existing deployed pipeline* would authorize, not a
    # manually invented system prompt to favor the connectome branch.
    combined_allowed = bool(gate_open and (mouth_ready or bounded_ready))
    if combined_allowed:
        try:
            combined_messages = generator._messages(
                prompt, dispatched["route"], memory_context=[],
                tool_context=dispatched.get("tool_context"),
                knowledge_context=None, brain_state=state,
            )
        except Exception as exc:
            combined_allowed = False
            combined_error = str(exc)[:150]
        else:
            combined_error = None
    else:
        combined_messages = []
        combined_error = "not authorized by frozen dispatch/brain gate"

    brain_only = run_arm_brain_only(state, dispatched)
    if args.skip_llm:
        llm_only = {"called": False, "answer": None, "error": "skip_llm mode"}
        combined = {"called": False, "answer": None, "error": "skip_llm mode"}
    else:
        llm_only = ollama_answer(
            [{"role": "system", "content": BASE_PROMPT},
             {"role": "user", "content": prompt}],
            model=args.model, endpoint=args.endpoint, timeout=args.timeout,
        )
        if combined_allowed:
            combined = ollama_answer(
                combined_messages,
                model=args.model, endpoint=args.endpoint, timeout=args.timeout,
            )
        else:
            combined = {"called": False, "answer": None,
                        "error": combined_error}

    marker = question.get("expected_exact")
    def observed_marker(value):
        answer = str((value or {}).get("answer") or "")
        return bool(marker and str(marker) in answer)

    return {
        "case": question["id"], "family": question["category"],
        "question": prompt,
        "expected_route": question["category"],
        "route": routed["route"], "route_match": routed["route"] == question["category"],
        "router_accepted": bool(routed.get("accepted")),
        "dispatch_status": dispatched["status"],
        "confidence": round(routed["confidence"], 5),
        "brain_objective": state.get("objective"),
        "brain_response_mode": state.get("response_mode"),
        "brain_evidence_available": bool((state.get("evidence") or {}).get("available")),
        "utterance_ready": mouth_ready,
        "bounded_task_delegated_to_llm": bounded_ready,
        "same_original_user_question_passed_to_combined_llm":
            bool(bounded_ready and
                 any(m.get("content") == prompt for m in combined_messages)),
        "brain_only": brain_only,
        "llm_only": llm_only,
        "brain_plus_llm": combined,
        "expected_math_marker": marker,
        "math_marker_observed": {
            "brain_only": observed_marker(brain_only),
            "llm_only": observed_marker(llm_only),
            "brain_plus_llm": observed_marker(combined),
        } if marker else None,
    }


def summarize(rows):
    n = len(rows)
    counts = {
        "questions": n,
        "routing_match": sum(r["route_match"] for r in rows),
        "routing_accepted": sum(r["router_accepted"] for r in rows),
        "delegated_bounded_task": sum(r["bounded_task_delegated_to_llm"] for r in rows),
        "combined_passed_original_question": sum(
            r["same_original_user_question_passed_to_combined_llm"] for r in rows
        ),
        "brain_only_exact_tool": sum(
            r["brain_only"]["answer_origin"] == "exact_tool" for r in rows
        ),
        "brain_only_open_ended_solved": sum(
            r["brain_only"]["solved_open_ended_question"] for r in rows
        ),
        "llm_only_successful_response": sum(
            bool(r["llm_only"]["answer"]) for r in rows
        ),
        "combined_successful_response": sum(
            bool(r["brain_plus_llm"]["answer"]) for r in rows
        ),
        "combined_blocked_by_router": sum(
            r["dispatch_status"] == "uncertain" for r in rows
        ),
    }
    math_rows = [r for r in rows if r["expected_math_marker"]]
    counts["math_marker_checks"] = len(math_rows)
    counts["math_marker_hits"] = {
        k: sum(bool(r["math_marker_observed"][k]) for r in math_rows)
        for k in ("brain_only", "llm_only", "brain_plus_llm")
    }
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--report", type=Path, default=Path("/tmp/brain-llm-attribution.json"))
    p.add_argument("--model", default="qwen2.5:0.5b-instruct")
    p.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--skip-llm", action="store_true",
                   help="Only perform causal/provenance audit; no generative claims")
    args = p.parse_args()
    if args.timeout <= 0:
        p.error("timeout must be positive")
    import torch
    torch.set_num_threads(2)
    questions = json.loads(args.questions.read_text(encoding="utf-8"))["questions"]
    if not questions or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Questions must have unique IDs")
    if any(q["category"] not in {"general", "code", "summarize", "math"}
           for q in questions):
        raise ValueError("This no-external-evidence study only supports four route families")
    with __import__("unittest").mock.patch.dict(
        os.environ, {"FLYGPT_ENABLE_FULL_POPULATION_ROUTER": "1"}
    ):
        router = FullPopulationRouterRuntime(args.checkpoint)
    brain = FlyBrainRuntime()
    generator = GeneratorRuntime()
    results = []
    for i, question in enumerate(questions):
        result = evaluate_case(question, router, brain, generator, args)
        results.append(result)
        print(
            f"CASE {i+1}/{len(questions)} {result['case']} "
            f"route={result['route']} accept={result['router_accepted']} "
            f"delegated={result['bounded_task_delegated_to_llm']} "
            f"brain_only_origin={result['brain_only']['answer_origin']} "
            f"llm_answered={bool(result['llm_only']['answer'])} "
            f"combined_answered={bool(result['brain_plus_llm']['answer'])}",
            flush=True,
        )
    summary = summarize(results)
    data = {
        "study": "Actual FlyGPT 166700-neuron router vs direct LLM vs router+LLM",
        "model": args.model if not args.skip_llm else None,
        "llm_mode": "ollama local zero temperature, deterministic seed and 176 token cap" if not args.skip_llm else "disabled",
        "language_generation_benchmark": not args.skip_llm,
        "same_llm_both_generated_arms": not args.skip_llm,
        "model_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "full_population_neurons": router.n_nodes,
        "full_graph_eligible_edges": router.n_edges,
        "summary": summary,
        "per_question": results,
        "limitations": [
            "18 manually authored same-corpus-independent evaluation questions is an exploratory sample, not an independent human-blind benchmark.",
            "Direct LLM and FlyGPT+LLM system instructions differ by design, so this cannot isolate pure topology effects.",
            "FlyGPT+LLM uses actual generator message construction, but experiments use local Ollama transport, not the hosted Gemini configured by an individual user.",
            "The 166700-neuron router aggregates frozen full-graph structural embeddings; it does NOT simulate neuronal spike propagation per question.",
            "Brain-only unresolved outputs are correctly counted as NO answer, not coerced into chatbot replies.",
            "Math marker presence is a narrow lexical check, not answer correctness; other answer qualities are UNGRADED.",
            "No random-connectome and text-only control for this particular generative ablation. Earlier 93.23% score is six-way route macro recall only.",
            "No private chat history, provider secrets, or personal data used. No weights updated or deployed.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SUMMARY "+json.dumps(summary, ensure_ascii=False), flush=True)
    print("REPORT="+str(args.report), flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as w:
            w.write("## FlyGPT 166,700-neuron brain vs LLM attribution (exploratory)\n\n")
            w.write("| Observation | Cases |\n|---|---:|\n")
            for key, val in summary.items():
                if isinstance(val, int):
                    w.write(f"| {key} | {val}/{len(results)} |\n")
            w.write("\n**Only 18 authored evaluation prompts. Math markers are not full correctness scores. Non-math generation is ungraded.**\n")


if __name__ == "__main__":
    main()
