"""No canned replies, no planner JSON for greetings, safe correction on retry."""
import os
import unittest
from unittest.mock import patch

from brain_runtime import FlyBrainRuntime
from generator_runtime import (
    GenerationResult, GeneratorRuntime,
    _contains_internal_plan, _wrong_output_language,
)


def planned(text):
    return FlyBrainRuntime().plan(
        route_info={
            "route": "general", "confidence": 1.0, "margin": 0.5,
            "accepted": True, "trace": [],
        },
        dispatch={"route": "general", "status": "ready"},
        message=text,
    )


def runtime():
    with patch.dict(os.environ, {
        "FLYGPT_GENERATOR_URL": "https://example.invalid/v1/chat/completions",
        "FLYGPT_GENERATOR_MODEL": "test-language-model",
        "FLYGPT_GENERATOR_RETRY_DELAY": "0",
    }, clear=True):
        return GeneratorRuntime()


def model_says(text):
    return GenerationResult(
        used=True, provider="compatible-http",
        model="test-language-model", answer=text, http_status=200,
    )


class NaturalSocialGenerationTests(unittest.TestCase):
    def test_social_prompt_does_not_echo_brain_planner_json(self):
        gen = runtime()
        state = planned("안녕")
        messages = gen._messages(
            "안녕", "general",
            [{"role": "assistant", "content": "old cached answer"}],
            tool_context="secret tool", knowledge_context="secret knowledge",
            brain_state=state,
        )
        self.assertEqual(len(messages), 2)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("한국어", messages[0]["content"])
        self.assertIn("Return the user's greeting", messages[1]["content"])
        joined = " ".join(m["content"] for m in messages)
        for forbidden in (
            "semantic_authority", "speech_act", "content_units", "utterance_plan",
            "old cached answer", "secret tool", "secret knowledge", '"ready"',
        ):
            self.assertNotIn(forbidden, joined)
        self.assertNotIn("{", joined)

    def test_no_predefined_answer_and_natural_one_syllable_reply(self):
        brain = planned("안녕")
        self.assertFalse(_wrong_output_language("응!", brain["utterance_plan"]))
        self.assertFalse(_wrong_output_language("네.", brain["utterance_plan"]))
        self.assertFalse(_wrong_output_language("ㅎㅇ", brain["utterance_plan"]))
        self.assertTrue(_wrong_output_language("Hello!", brain["utterance_plan"]))
        gen = runtime()
        with patch.object(gen, "_generate_once", return_value=model_says("응!")) as send:
            result = gen.generate("안녕", "general", brain_state=brain)
        self.assertTrue(result.used)
        self.assertEqual(result.answer, "응!")
        send.assert_called_once()

    def test_corrected_retry_changes_prompt_instead_of_repeating_leak(self):
        gen = runtime()
        with patch.object(gen, "_generate_once", side_effect=[
            model_says('{"speech_act":"return_greeting"}'),
            model_says("반가워! 오늘도 잘 지냈어?"),
        ]) as send:
            response = gen.generate("안녕", "general", brain_state=planned("안녕"))
        self.assertTrue(response.used)
        self.assertEqual(response.answer, "반가워! 오늘도 잘 지냈어?")
        self.assertEqual(send.call_count, 2)
        first = send.call_args_list[0].kwargs["messages"]
        second = send.call_args_list[1].kwargs["messages"]
        self.assertNotEqual(first, second)
        self.assertIn("새로운 한국어", second[1]["content"])
        self.assertNotIn("speech_act", str(second))
        self.assertNotIn("return_greeting", str(second))
        self.assertEqual(response.attempts, 2)

    def test_second_invalid_output_stays_blocked_without_canned_prose(self):
        gen = runtime()
        with patch.object(gen, "_generate_once", return_value=model_says(
            '{"speech_act":"return_greeting"}'
        )) as send:
            response = gen.generate("안녕", "general", brain_state=planned("안녕"))
        self.assertFalse(response.used)
        self.assertIsNone(response.answer)
        self.assertEqual(response.error_code, "unsafe_generated_content")
        self.assertEqual(send.call_count, 2)

    def test_wrong_language_retry_uses_korean_guidance(self):
        gen = runtime()
        with patch.object(gen, "_generate_once", side_effect=[
            model_says("Hello."),
            model_says("네!"),
        ]) as send:
            response = gen.generate("안녕", "general", brain_state=planned("안녕"))
        self.assertTrue(response.used)
        self.assertEqual(response.answer, "네!")
        self.assertEqual(send.call_count, 2)

    def test_english_social_act_works_without_planner_json(self):
        state = planned("Hello!")
        gen = runtime()
        messages = gen._messages("Hello!", "general", None, brain_state=state)
        self.assertIn("Speak to the user as Papiti in English", messages[0]["content"])
        self.assertNotIn("utterance_plan", str(messages))
        with patch.object(gen, "_generate_once", return_value=model_says("Hi there!")):
            answer = gen.generate("Hello!", "general", brain_state=state)
        self.assertTrue(answer.used)
        self.assertEqual(answer.answer, "Hi there!")

    def test_hidden_plan_still_blocked(self):
        self.assertTrue(_contains_internal_plan(
            'Here is the plan: {"semantic_steps":[{"fact_id":"x"}]}'
        ))
        self.assertFalse(_contains_internal_plan("안녕, 반가워!"))


if __name__ == "__main__":
    unittest.main()
