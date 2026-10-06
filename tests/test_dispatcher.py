import unittest

from dispatcher import dispatch, dispatch_math_fast_path


def route_info(route: str, confidence: float = 0.95, second: float = 0.03):
    return {
        "route": route,
        "confidence": confidence,
        "top_routes": [
            {"route": route, "confidence": confidence},
            {"route": "general" if route != "general" else "code", "confidence": second},
            {"route": "summarize", "confidence": max(0.0, 1.0 - confidence - second)},
        ],
    }


class DispatcherTests(unittest.TestCase):
    def test_math_route_prepares_exact_tool_context(self):
        result = dispatch("3x+7=22에서 x는?", route_info("math"))
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.handler, "math_tool+generator")
        self.assertEqual(result.answer, "")
        self.assertIn("x = 5", result.tool_context or "")

    def test_math_fraction_prepares_tool_context(self):
        result = dispatch("0.75를 분수로 바꾸면?", route_info("math"))
        self.assertIn("3/4", result.tool_context or "")

    def test_code_route_does_not_author_canned_answer(self):
        result = dispatch(
            "검색하지 말고, 파이썬 리스트 정렬 방법만 알려줘",
            route_info("code"),
        )
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.handler, "generator")
        self.assertEqual(result.answer, "")
        self.assertIsNone(result.tool_context)

    def test_research_is_generator_ready(self):
        result = dispatch("최신 AI 연구 동향 찾아줘", route_info("research"))
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.handler, "research_context+generator")
        self.assertEqual(result.answer, "")

    def test_memory_is_generator_ready(self):
        result = dispatch("내가 전에 정한 이름 뭐였지?", route_info("memory"))
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.handler, "memory_context+generator")
        self.assertEqual(result.answer, "")

    def test_math_fast_path_remains_deterministic(self):
        result = dispatch_math_fast_path("17*23")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.handler, "math_fast_path")
        self.assertIn("391", result.answer)

    def test_calibrated_checkpoint_thresholds_override_defaults(self):
        info = route_info("general", confidence=0.60, second=0.20)
        info["min_confidence"] = 0.70
        info["min_margin"] = 0.15
        info["accepted"] = False
        result = dispatch("애매한 질문", info)
        self.assertEqual(result.status, "uncertain")
        self.assertIn("confidence≥70%", result.answer)

    def test_zero_feature_signal_is_held(self):
        info = route_info("general", confidence=0.95, second=0.02)
        info["feature_signal"] = 0.0
        info["accepted"] = False
        result = dispatch("???", info)
        self.assertEqual(result.status, "uncertain")

    def test_uncertain_route_is_not_executed(self):
        info = {
            "route": "general",
            "confidence": 0.422,
            "top_routes": [
                {"route": "general", "confidence": 0.422},
                {"route": "code", "confidence": 0.359},
                {"route": "summarize", "confidence": 0.185},
            ],
        }
        result = dispatch("애매한 질문", info)
        self.assertEqual(result.status, "uncertain")
        self.assertEqual(result.handler, "confidence_gate")


if __name__ == "__main__":
    unittest.main()
