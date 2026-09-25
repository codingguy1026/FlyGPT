import unittest

from dispatcher import dispatch


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
    def test_math_multiplication(self):
        result = dispatch("17 × 23은?", route_info("math"))
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.handler, "math")
        self.assertIn("391", result.answer)

    def test_math_linear_equation(self):
        result = dispatch("3x+7=22에서 x는?", route_info("math"))
        self.assertEqual(result.status, "completed")
        self.assertIn("x = 5", result.answer)

    def test_math_fraction(self):
        result = dispatch("0.75를 분수로 바꾸면?", route_info("math"))
        self.assertIn("3/4", result.answer)

    def test_code_python_sort(self):
        result = dispatch(
            "검색하지 말고, 파이썬 리스트 정렬 방법만 알려줘",
            route_info("code"),
        )
        self.assertEqual(result.handler, "code")
        self.assertIn("sorted(values)", result.answer)
        self.assertIn("values.sort()", result.answer)

    def test_research_is_pending(self):
        result = dispatch("최신 AI 연구 동향 찾아줘", route_info("research"))
        self.assertEqual(result.status, "pending")
        self.assertEqual(result.handler, "research")

    def test_memory_is_pending(self):
        result = dispatch("내가 전에 정한 이름 뭐였지?", route_info("memory"))
        self.assertEqual(result.status, "pending")
        self.assertEqual(result.handler, "memory")

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
