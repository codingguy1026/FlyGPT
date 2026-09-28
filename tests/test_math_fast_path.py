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

    def test_date_is_not_math_fast_path(self):
        self.assertFalse(is_math_fast_path("2026-09-26"))

    def test_sentence_still_uses_router(self):
        self.assertFalse(is_math_fast_path("파이썬 리스트 어떻게 정렬해?"))


if __name__ == "__main__":
    unittest.main()
