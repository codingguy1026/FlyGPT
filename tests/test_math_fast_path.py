import unittest

from dispatcher import dispatch_math_fast_path, is_math_fast_path


class MathFastPathTests(unittest.TestCase):
    def test_plain_multiplication(self):
        self.assertTrue(is_math_fast_path("11*11"))
        result = dispatch_math_fast_path("11*11")
        self.assertEqual(result.route, "math")
        self.assertEqual(result.handler, "math_fast_path")
        self.assertIn("121", result.answer)

    def test_parenthesized_expression(self):
        self.assertTrue(is_math_fast_path("(8+4)*3"))
        result = dispatch_math_fast_path("(8+4)*3")
        self.assertIn("36", result.answer)

    def test_korean_math_suffix(self):
        self.assertTrue(is_math_fast_path("11*11은?"))
        self.assertTrue(is_math_fast_path("(8+4)*3은 얼마야?"))

    def test_malecns_body_id_is_not_a_math_operand(self):
        self.assertFalse(is_math_fast_path("뉴런 12781의 출력 연결 5개"))

    def test_exponent_with_korean_question(self):
        self.assertTrue(is_math_fast_path("1972^2는 뭐야"))
        self.assertIn("3888784", dispatch_math_fast_path("1972^2는 뭐야").answer)

    def test_linear_equation_no_generator(self):
        self.assertTrue(is_math_fast_path("8x=8"))
        self.assertIn("x = 1", dispatch_math_fast_path("8x=8").answer)

    def test_unsupported_math_expression_is_not_fast_path(self):
        self.assertFalse(is_math_fast_path("2^100"))

    def test_square_of_expression_not_last_token(self):
        # (1972^2)^2 is not 2^2 and must bypass the LLM entirely.
        self.assertTrue(is_math_fast_path("1972^2의 제곱은?"))
        result = dispatch_math_fast_path("1972^2의 제곱은?")
        self.assertIn("15122640998656", result.answer)
        self.assertEqual(result.handler, "math_fast_path")

    def test_simple_korean_square(self):
        self.assertTrue(is_math_fast_path("1972의 제곱은?"))
        self.assertIn("3888784", dispatch_math_fast_path("1972의 제곱은?").answer)

    def test_date_is_not_math_fast_path(self):
        self.assertFalse(is_math_fast_path("2026-09-26"))

    def test_sentence_still_uses_router(self):
        self.assertFalse(is_math_fast_path("파이썬 리스트 어떻게 정렬해?"))


if __name__ == "__main__":
    unittest.main()
