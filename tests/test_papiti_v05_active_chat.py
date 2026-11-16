"""Integration coverage for the real v0.5 main route and reversible opt-in.

No network, API key, live neuPrint, or biological firing simulation is used.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from generator_runtime import GeneratorRuntime
from papiti_v05_chat import run_v05_chat
from papiti_v05_runtime import v05_enabled


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


def execute(text, *, selected=None, session_id="user-session", generator=None,
            research=None, memory=None):
    policy = FakePolicy(selected=selected)
    memory = memory or FakeMemory()
    research = research or FakeResearch()
    generator = generator or GeneratorRuntime()
    response = run_v05_chat(
        text, policy=policy, memory=memory, research=research,
        generator=generator, session_id=session_id,
        budget_seconds=20.0,
    )
    return response, memory, research, generator, policy


class MainV05PolicyTests(unittest.TestCase):
    def test_greeting_sends_only_the_plan_to_local_speaker(self):
        text = "안녕"
        with (
            patch("generator_runtime.speech_checkpoint_present", return_value=True),
            patch("generator_runtime.generate_speech", return_value="안녕!") as speaker,
            patch("generator_runtime.urllib.request.urlopen") as http,
        ):
            result, memory, research, _, policy = execute(text, selected=[])
        plan = speaker.call_args.args[0]
        self.assertEqual(plan["speech_act"], "return_greeting")
        self.assertNotIn("message", plan)
        self.assertEqual(policy.requests, [text])
        self.assertEqual(memory.search_calls, [])
        self.assertEqual(research.calls, [])
        http.assert_not_called()
        self.assertEqual(result["route"]["model_version"], "v0.5")
        self.assertTrue(result["route"]["category_free"])
        self.assertEqual(result["route"]["selected_capabilities"], [])
        self.assertEqual(result["data"]["generation"]["model"], "papiti-language-v0.6.1")
        self.assertEqual(result["answer"], "안녕!")

    def test_calculator_passes_verified_result_to_speaker(self):
        with (
            patch("generator_runtime.speech_checkpoint_present", return_value=True),
            patch("generator_runtime.generate_speech", return_value="42") as speaker,
        ):
            result, *_ = execute("6 × 7은?", selected=["calculator"])
        self.assertEqual(speaker.call_args.args[0]["content_units"], [
            {"kind": "exact_result", "value": "Exact math tool result: 42"}
        ])
        self.assertEqual(result["data"]["brain_state"]["tool_reports"],
                         [{"tool": "calculator", "status": "verified"}])
        self.assertEqual(result["answer"], "42")

    def test_standalone_greeting_bypasses_spurious_tool_selections(self):
        with (
            patch("generator_runtime.speech_checkpoint_present", return_value=True),
            patch("generator_runtime.generate_speech", return_value="Hi!?!") as speaker,
        ):
            result, _, research, _, _ = execute("Hi!?!", selected=["web_search", "calculator"])
        self.assertEqual(research.calls, [])
        self.assertEqual(result["route"]["selected_capabilities"], [])
        self.assertEqual(speaker.call_args.args[0]["greeting"], "Hi!?!")
        self.assertTrue(result["data"]["generation"]["used"])

    def test_thanks_skips_spurious_calculator(self):
        with (
            patch("generator_runtime.speech_checkpoint_present", return_value=True),
            patch("generator_runtime.generate_speech", return_value="천만에!") as speaker,
        ):
            result, *_ = execute("고마워", selected=["calculator"])
        self.assertEqual(speaker.call_args.args[0]["speech_act"], "acknowledge_thanks")
        self.assertEqual(result["data"]["brain_state"]["tool_reports"], [])
        self.assertEqual(result["route"]["selected_capabilities"], [])
        self.assertTrue(result["data"]["generation"]["used"])

    def test_everyday_conversation_skips_spurious_tools(self):
        cases = (
            ("잘 지내요?", "acknowledge_presence", "응, 이야기할 준비 됐어."),
            ("오늘 좋은 일이 있었어요!", "acknowledge_good_news", "오, 좋은 소식이네!"),
            ("좀 피곤해요.", "acknowledge_tiredness", "많이 피곤했겠다."),
            ("ㅎㅎㅎㅎㅎ", "respond_to_laughter", "ㅋㅋㅋ"),
        )
        for prompt, act, answer in cases:
            with (
                self.subTest(prompt=prompt),
                patch("generator_runtime.speech_checkpoint_present", return_value=True),
                patch("generator_runtime.generate_speech", return_value=answer) as speaker,
            ):
                result, _, research, _, _ = execute(prompt, selected=["web_search", "calculator"])
                self.assertEqual(research.calls, [])
                self.assertEqual(result["route"]["selected_capabilities"], [])
                self.assertEqual(speaker.call_args.args[0]["speech_act"], act)
                self.assertTrue(result["data"]["generation"]["used"])

    def test_all_three_tools_still_execute_without_speaking_a_partial_plan(self):
        query = "전에 말한 비용과 현재 가격을 찾아서 17 × 23은 몇인지 계산해 줘"
        with patch("generator_runtime.generate_speech") as speaker:
            result, memory, research, _, _ = execute(
                query, selected=["memory_lookup", "web_search", "calculator"]
            )
        self.assertEqual(len(memory.search_calls), 1)
        self.assertEqual(research.calls, [query])
        self.assertEqual([x["tool"] for x in result["data"]["brain_state"]["tool_reports"]],
                         ["memory_lookup", "web_search", "calculator"])
        self.assertEqual([x["status"] for x in result["data"]["brain_state"]["tool_reports"]],
                         ["retrieved", "retrieved", "verified"])
        self.assertEqual(result["data"]["v05_error_code"], "hybrid_speaker_unsupported")
        self.assertEqual(result["route"]["n_nodes"], 166700)
        speaker.assert_not_called()

    def test_session_scope_is_not_bypassed(self):
        response, memory, _, _, _ = execute(
            "전에 뭐라고 했지?", selected=["memory_lookup"], session_id=None
        )
        self.assertEqual(memory.search_calls, [])
        self.assertEqual(response["data"]["brain_state"]["tool_reports"][0]["status"], "no_evidence")

    def test_v05_ignores_stale_hosted_and_loopback_settings(self):
        with patch.dict("os.environ", {
            "FLYGPT_V05_ACTIVE": "1",
            "FLYGPT_GENERATOR_MODE": "http",
            "FLYGPT_GENERATOR_URL": "https://example.org/chat/completions",
            "FLYGPT_GENERATOR_API_KEY": "test-only",
            "FLYGPT_HYBRID_URL": "http://localhost:9000/v1/chat/completions",
            "FLYGPT_HYBRID_MODEL": "old-local-model",
            "FLYGPT_HYBRID_SPEAKER": "legacy",
        }, clear=True):
            runtime = GeneratorRuntime()
        self.assertEqual(runtime.url, "")
        self.assertFalse(runtime.status()["api_key_configured"])
        self.assertEqual(runtime.status()["model"], "papiti-language-v0.6.1")
        self.assertEqual(runtime.hybrid_speaker, "v0.6.1")

    def test_missing_local_checkpoint_is_explicit(self):
        with patch("generator_runtime.speech_checkpoint_present", return_value=False):
            result, *_ = execute("안녕")
        self.assertEqual(result["data"]["v05_error_code"], "hybrid_speaker_missing")
        self.assertIn("make papiti-lm-train-v0.6.1", result["answer"])

    def test_untrained_questions_and_search_plans_report_unsupported(self):
        with patch("generator_runtime.generate_speech") as speaker:
            general, *_ = execute("사과와 바나나의 공통점이 뭐야?")
            search, *_ = execute(
                "오늘 경기 결과를 알려줘", selected=["web_search"], research=FakeResearch(used=False)
            )
        self.assertFalse(general["data"]["generation"]["used"])
        self.assertEqual(general["data"]["v05_error_code"], "hybrid_speaker_unsupported")
        self.assertEqual(search["data"]["v05_error_code"], "hybrid_speaker_unsupported")
        self.assertEqual(search["data"]["brain_state"]["tool_reports"][0]["status"], "no_evidence")
        speaker.assert_not_called()

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
        generator = GeneratorRuntime()
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
            patch("generator_runtime.speech_checkpoint_present", return_value=True),
            patch("generator_runtime.generate_speech", return_value="안녕!") as speaker,
            patch.object(app, "v05_enabled", return_value=True),
            patch.object(app, "get_v05_policy", return_value=policy),
            patch.object(app, "predict_route", side_effect=AssertionError("old router called")),
        ):
            result = app.chat_endpoint(
                app.ChatRequest(message="안녕"),
                request,
            )
        self.assertEqual(result["router"]["model_version"], "v0.5")
        self.assertTrue(result["router"]["category_free"])
        speaker.assert_called_once()
        self.assertEqual(len(memory.add_calls), 2)


if __name__ == "__main__":
    unittest.main()
