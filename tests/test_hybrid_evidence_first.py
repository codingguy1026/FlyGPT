"""Hybrid evidence-first conversations without categorical routing gates."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch, Mock

import app as api
from brain_runtime import FlyBrainRuntime
from evidence_need import decide_evidence, no_gate_route
from generator_runtime import GeneratorRuntime
from hybrid_speech import check_candidate
from native_realizer import realize_native
from research_runtime import ResearchResult, WebSource


def advisory(route="research", confidence=0.459, accepted=False):
    return {
        "route": route, "confidence": confidence,
        "margin": 0.19, "accepted": accepted,
        "top_routes": [
            {"route": "research", "confidence": 0.459},
            {"route": "memory", "confidence": 0.267},
            {"route": "general", "confidence": 0.183},
        ],
        "model": "fly_router_malecns_v0_4.pt",
    }


def research_result():
    return ResearchResult(
        query="왜 하늘이 파래?",
        provider="brave-llm-context",
        sources=[WebSource(
            url="https://example.org/science/sky",
            title="Why the sky is blue",
            hostname="example.org",
            root_domain="example.org",
            snippets=["햇빛이 공기 분자에 산란되면서 파란빛이 더 많이 퍼집니다."],
            age=[], authority=0.6,
        )],
        retrieved_at=1234,
    )


class EvidenceNeedTests(unittest.TestCase):
    def test_statement_is_not_a_search(self):
        for message in (
            "오늘 야구에서 역전승했어!",
            "기분이 정말 좋아 ㅋㅋㅋㅋ",
            "우리 학교 오늘 진짜 힘들었어.",
        ):
            with self.subTest(message=message):
                self.assertFalse(decide_evidence(message).search)

    def test_public_unanswered_questions_can_search(self):
        for question in ("왜 하늘이 파래?", "최근 대구 뉴스 알려줘", "When was X discovered?"):
            with self.subTest(question=question):
                self.assertTrue(decide_evidence(question).search)

    def test_grounded_story_invitation_searches_without_question_mark(self):
        cases = (
            "흥미로운 역사 이야기를 하나 들려줘.",
            "재미있는 세계사 이야기 들려주세요",
            "조선 시대의 일화 하나 소개해 줘",
            "Tell me an interesting historical story.",
        )
        for request in cases:
            with self.subTest(request=request):
                decision = decide_evidence(request)
                self.assertTrue(decision.search)
                self.assertEqual(decision.reason, "unanswered_grounded_story_request")

    def test_creative_stories_and_personal_anecdotes_do_not_leak_to_web(self):
        for request in (
            "재미있는 이야기 하나 들려줘",
            "무서운 소설 이야기 지어 줘",
            "가상의 역사 이야기를 만들어 줘",
            "우리 가족의 역사 이야기를 들려줘",
            "내 친구의 실제 이야기를 들려줘",
        ):
            with self.subTest(request=request):
                self.assertFalse(decide_evidence(request).search)

    def test_already_grounded_or_user_banned_search(self):
        self.assertFalse(decide_evidence("왜 하늘이 파래?", ready_plan=True).search)
        self.assertFalse(decide_evidence("왜 하늘이 파래?", verified_local_evidence=True).search)
        self.assertFalse(decide_evidence("인터넷 검색하지 마. 왜 하늘이 파래?").search)
        self.assertFalse(decide_evidence("my private password is 123?", ready_plan=False).search)
        self.assertFalse(decide_evidence("내가 오늘 무엇을 먹었을까?").search)

    def test_previous_conversation_uses_memory_not_public_web(self):
        result = decide_evidence("내가 전에 정한 이름 기억해?")
        self.assertFalse(result.search)
        self.assertTrue(result.memory)

    def test_route_model_is_advisory_not_dispatch_gate(self):
        raw = advisory()
        adapted = no_gate_route(raw)
        self.assertFalse(raw["accepted"])
        self.assertEqual(raw["route"], "research")
        self.assertTrue(adapted["accepted"])
        self.assertEqual(adapted["route"], "general")
        self.assertAlmostEqual(adapted["confidence"], 0.183)
        self.assertEqual(adapted["advisory_route"]["route"], "research")
        self.assertEqual(adapted["semantic_override"]["source"], "hybrid_evidence_first")


class EvidencePlanTests(unittest.TestCase):
    def test_unverified_web_excerpts_have_sources_and_are_not_auto_trusted(self):
        brain = FlyBrainRuntime()
        info = {"route": "general", "confidence": 0.18, "margin": 0, "accepted": True}
        plan = brain.plan(route_info=info, dispatch={"route": "general"}, message="왜 하늘이 파래?")
        plan["retrieval"] = "research"
        result = brain.finalize(plan, research_result=research_result().to_dict())
        packet = result["utterance_plan"]
        self.assertTrue(packet["ready"])
        self.assertEqual(packet["speech_act"], "report_retrieved_web_evidence")
        self.assertEqual(packet["content_units"][0]["provenance"], "unverified_web_snippet")
        spoken = realize_native("왜 하늘이 파래?", "general", brain_state=result)
        self.assertIn("햇빛이 공기 분자에 산란", spoken)
        self.assertIn("https://example.org/science/sky", spoken)
        self.assertIn("아직 사실 전체를 독립 검증한 것은 아니야", spoken)
        self.assertEqual(
            check_candidate(packet, "하늘은 분명히 파란색이다."),
            "semantic_content_not_preserved",
        )

    def test_missing_or_untrusted_search_cannot_invent_a_claim(self):
        brain = FlyBrainRuntime()
        state = brain.plan(
            route_info={"route": "general", "accepted": True, "confidence": 0.1, "margin": 0},
            dispatch={"route": "general"}, message="왜 하늘이 파래?",
        )
        state["retrieval"] = "research"
        result = brain.finalize(
            state, research_result={
                "sources": [{
                    "url": "javascript:alert(1)",
                    "title": "untrusted",
                    "snippets": ["This should never be displayed."],
                }],
                "used": True,
            },
        )
        self.assertEqual(result["utterance_plan"]["speech_act"], "report_no_external_evidence")
        answer = realize_native("왜 하늘이 파래?", "general", brain_state=result)
        self.assertNotIn("This should never", answer)


class EvidenceFirstEndToEndTests(unittest.TestCase):
    def call_hybrid(self, message: str, result: ResearchResult | None = None):
        with patch.dict(os.environ, {
            "FLYGPT_GENERATOR_MODE": "hybrid",
            "FLYGPT_HYBRID_SPEAKER": "v0.6.1",
            "FLYGPT_HYBRID_EVIDENCE_FIRST": "1",
            "FLYGPT_HYBRID_URL": "",
            "FLYGPT_HYBRID_MODEL": "",
        }, clear=False):
            runtime = GeneratorRuntime()
            with patch.object(api, "_require_user", return_value={"id": "fake-user"}), \
                 patch.object(api, "_generator", runtime), \
                 patch.object(api, "predict_route", return_value=advisory()), \
                 patch.object(api, "_knowledge") as knowledge, \
                 patch.object(api, "_memory") as memory, \
                 patch.object(api, "_research") as research, \
                 patch("generator_runtime.speech_checkpoint_present", return_value=False):
                knowledge.context_for_query.return_value = (None, [])
                knowledge.learn_from_user_text.return_value = []
                memory.recent.return_value = []
                if result is not None:
                    research.search.return_value = result
                answer = api.chat_endpoint(api.ChatRequest(message=message), object())
                return answer, research.search.call_count

    def test_personal_victory_not_research_or_confidence_blocked(self):
        response, count = self.call_hybrid("오늘 야구에서 역전승했어!")
        self.assertEqual(count, 0)
        self.assertNotIn("confidence≥", response["answer"])
        self.assertEqual(response["router"]["route"], "general")
        self.assertEqual(response["data"]["evidence_decision"]["reason"],
                         "personal_statement_or_nonquestion")

    def test_unknown_public_question_fetches_sources_even_if_router_rejects(self):
        response, count = self.call_hybrid("왜 하늘이 파래?", research_result())
        self.assertEqual(count, 1)
        self.assertIn("언어 모델", response["answer"])
        self.assertIn("https://example.org/science/sky", str(response["data"]["research"]))
        self.assertEqual(response["data"]["evidence_decision"]["reason"],
                         "unanswered_public_question")
        self.assertEqual(response["data"]["generation"]["error_code"],
                         "hybrid_speaker_unsupported")

    def test_historical_story_invitation_retrieves_sourced_material(self):
        historically_grounded = ResearchResult(
            query="흥미로운 역사 이야기를 하나 들려줘.",
            provider="brave-llm-context",
            sources=[WebSource(
                url="https://example.org/history/incident",
                title="Historical event",
                hostname="example.org",
                root_domain="example.org",
                snippets=["한 역사적 사건에 관한 출처가 있는 설명입니다."],
                age=[], authority=0.7,
            )],
            retrieved_at=1234,
        )
        response, search_count = self.call_hybrid(
            "흥미로운 역사 이야기를 하나 들려줘.",
            historically_grounded,
        )
        self.assertEqual(search_count, 1)
        self.assertIn("언어 모델", response["answer"])
        self.assertIn("https://example.org/history/incident", str(response["data"]["research"]))
        self.assertNotIn("이 자유형 질문은 아직 직접 풀 수 없어", response["answer"])
        self.assertEqual(
            response["data"]["evidence_decision"]["reason"],
            "unanswered_grounded_story_request",
        )
        # Search still runs, but no Native excerpt impersonates neural speech.
        self.assertEqual(response["data"]["generation"]["error_code"], "hybrid_speaker_unsupported")

    def test_no_web_sources_reports_missing_evidence(self):
        unavailable = ResearchResult(
            query="왜 하늘이 파래?", provider="brave-llm-context",
            sources=[], retrieved_at=1234, error="BRAVE_SEARCH_API_KEY not configured",
        )
        response, count = self.call_hybrid("왜 하늘이 파래?", unavailable)
        self.assertEqual(count, 1)
        self.assertEqual(response["data"]["generation"]["error_code"],
                         "hybrid_speaker_unsupported")
        self.assertIn("언어 모델", response["answer"])


if __name__ == "__main__":
    unittest.main()
