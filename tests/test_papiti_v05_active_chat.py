"""Integration coverage for the real v0.5 main route and reversible opt-in.

No network, API key, live neuPrint, or biological firing simulation is used.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from papiti_v05_chat import run_v05_chat, v05_enabled


class FakePolicy:
    name = "papiti-v0.5-category-free-166700"
    n_neurons = 166700
    capabilities = ("web_search", "memory_lookup", "calculator")

    def __init__(self, selected=None):
        self.selected = selected if selected is not None else []
        self.requests = []

    def decide(self, message):
        self.requests.append(message)
        return {
            "contract": "papiti_v0_5_tool_policy_v1",
            "selected_capabilities": list(self.selected),
            "capability_scores": {
                "web_search": 0.86 if "web_search" in self.selected else 0.02,
                "memory_lookup": 0.94 if "memory_lookup" in self.selected else 0.01,
                "calculator": 0.91 if "calculator" in self.selected else 0.04,
            },
            "language_model_instructions": [
                "Answer the actual user request.",
                "Do not invent prior messages or facts.",
            ],
        }


class FakeMemory:
    def __init__(self, hits=None):
        self.hits = hits if hits is not None else [
            {"role": "user", "content": "작년 프로젝트 이름은 은하야"}
        ]
        self.search_calls = []
        self.add_calls = []

    def search(self, session, message, **kwargs):
        self.search_calls.append((session, message, kwargs))
        return self.hits

    def recent(self, session, limit=12):
        return []

    def add(self, session, role, message):
        self.add_calls.append((session, role, message))


class FakeResearch:
    def __init__(self, used=True):
        self.used = used
        self.calls = []

    def search(self, message):
        self.calls.append(message)
        return SimpleNamespace(
            used=self.used,
            context=lambda: "Official source: https://example.org/verified",
            to_dict=lambda: {"provider": "fixture", "sources": [{"url": "https://example.org/verified"}]},
        )


class FakeGenerator:
    configured = True
    hybrid_mode = False
    native_mode = False
    local_only = False
    api_key = "fake-key-in-tests-only"
    model = "mock-gemini"

    def __init__(self):
        self.messages = None
        self.calls = 0

    def _generate_once(self, *, model, messages, timeout_seconds):
        self.calls += 1
        self.messages = messages
        return SimpleNamespace(
            used=True, provider="test-api", model=model,
            answer="사용자 질문에 대한 답변입니다.",
            error_code=None, finish_reason="stop",
            latency_ms=4,
        )


def execute(text, *, selected=None, session_id="user-session", generator=None,
            research=None, memory=None):
    policy = FakePolicy(selected=selected)
    memory = memory or FakeMemory()
    research = research or FakeResearch()
    generator = generator or FakeGenerator()
    response = run_v05_chat(
        text, policy=policy, memory=memory, research=research,
        generator=generator, session_id=session_id,
        recent_context=[], knowledge_context=None, budget_seconds=20.0,
    )
    return response, memory, research, generator, policy


class MainV05PolicyTests(unittest.TestCase):
    def test_no_old_category_and_no_unneeded_tools(self):
        text = "사과와 바나나의 공통점이 뭐야?"
        result, memory, research, generator, policy = execute(text, selected=[])
        self.assertEqual(policy.requests, [text])
        self.assertEqual(memory.search_calls, [])
        self.assertEqual(research.calls, [])
        self.assertEqual(generator.calls, 1)
        self.assertEqual(result["route"]["model_version"], "v0.5")
        self.assertTrue(result["route"]["category_free"])
        self.assertEqual(result["route"]["route"], "multi_action")
        self.assertEqual(result["route"]["confidence_kind"], "not_applicable")
        self.assertEqual(result["route"]["selected_capabilities"], [])
        self.assertEqual(result["data"]["brain_state"]["tool_reports"], [])
        self.assertEqual(result["answer"], "사용자 질문에 대한 답변입니다.")

    def test_all_three_tools_are_executed_in_one_request(self):
        query = "전에 말한 비용과 현재 가격을 찾아서 17 × 23은 몇인지 계산해 줘"
        result, memory, research, generator, _ = execute(
            query, selected=["memory_lookup", "web_search", "calculator"]
        )
        self.assertEqual(len(memory.search_calls), 1)
        self.assertEqual(research.calls, [query])
        self.assertEqual([x["tool"] for x in result["data"]["brain_state"]["tool_reports"]],
                         ["memory_lookup", "web_search", "calculator"])
        self.assertEqual([x["status"] for x in result["data"]["brain_state"]["tool_reports"]],
                         ["retrieved", "retrieved", "verified"])
        rendered = "\n".join(x["content"] for x in generator.messages)
        self.assertIn("391", rendered)
        self.assertIn("https://example.org/verified", rendered)
        self.assertEqual(result["route"]["n_nodes"], 166700)

    def test_session_scope_is_not_bypassed(self):
        response, memory, _, _, _ = execute(
            "전에 뭐라고 했지?", selected=["memory_lookup"], session_id=None
        )
        self.assertEqual(memory.search_calls, [])
        self.assertEqual(response["data"]["brain_state"]["tool_reports"][0]["status"], "no_evidence")

    def test_missing_hosted_api_is_explicit(self):
        generator = FakeGenerator()
        generator.configured = False
        result, _, _, generator, _ = execute(
            "흥미로운 역사 이야기 들려줘", generator=generator
        )
        self.assertEqual(generator.calls, 0)
        self.assertEqual(result["data"]["v05_error_code"], "external_language_api_unconfigured")
        self.assertIn("GEMINI_API_KEY", result["answer"])
        self.assertEqual(result["route"]["model_version"], "v0.5")

    def test_search_unavailable_not_fabricated(self):
        research = FakeResearch(used=False)
        result, _, _, generator, _ = execute(
            "오늘 경기 결과를 알려줘", selected=["web_search"], research=research
        )
        self.assertEqual(result["data"]["brain_state"]["tool_reports"][0]["status"], "no_evidence")
        evidence = "\n".join(x["content"] for x in generator.messages)
        self.assertIn("Do not invent today's results", evidence)

    def test_invalid_tool_selection_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported tool"):
            execute("안녕", selected=["unsupported_tool"])

    def test_switch_is_opt_in(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(v05_enabled())
        with patch.dict("os.environ", {"FLYGPT_V05_ACTIVE": "1"}):
            self.assertTrue(v05_enabled())
        with patch.dict("os.environ", {"FLYGPT_V05_ACTIVE": "0"}):
            self.assertFalse(v05_enabled())

    def test_unverified_checkpoint_is_rejected(self):
        import torch
        from papiti_v05_runtime import PapitiV05PolicyRuntime
        with tempfile.TemporaryDirectory() as tmp:
            filename = Path(tmp) / "fake.pt"
            torch.save({"format": "papiti-v0.5-category-free-full-malecns",
                        "n_neurons": 256, "model_state": {}}, filename)
            with self.assertRaisesRegex(ValueError, "invariants"):
                PapitiV05PolicyRuntime(filename)

    def test_actual_app_does_not_invoke_old_categorical_router(self):
        import app
        memory = FakeMemory()
        policy = FakePolicy()
        generator = FakeGenerator()
        knowledge = SimpleNamespace(
            context_for_query=lambda user, text, limit: (None, []),
            learn_from_user_text=lambda user, text: [],
        )
        request = SimpleNamespace(cookies={})
        with (
            patch.object(app, "_require_user", return_value={"id": "test-v05"}),
            patch.object(app, "_scoped_session_id", return_value="session-v05"),
            patch.object(app, "_memory", memory),
            patch.object(app, "_knowledge", knowledge),
            patch.object(app, "_research", FakeResearch()),
            patch.object(app, "_generator", generator),
            patch.object(app, "v05_enabled", return_value=True),
            patch.object(app, "get_v05_policy", return_value=policy),
            patch.object(app, "predict_route", side_effect=AssertionError("old router called")),
        ):
            result = app.chat_endpoint(
                app.ChatRequest(message="사과와 바나나의 공통점이 뭐야?"),
                request,
            )
        self.assertEqual(result["router"]["model_version"], "v0.5")
        self.assertTrue(result["router"]["category_free"])
        self.assertEqual(generator.calls, 1)
        self.assertEqual(len(memory.add_calls), 2)


if __name__ == "__main__":
    unittest.main()
