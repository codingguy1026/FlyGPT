"""Contract tests for the standalone LLM-free capability audit."""
from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from scripts.audit_native_reasoning import ForbiddenGenerator, run_audit


class NativeReasoningAuditTests(unittest.TestCase):
    def test_fixed_proofs_and_guardrails_pass_without_llm(self) -> None:
        report = run_audit()
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["generator_calls"], 0)
        self.assertEqual(report["proof_cases"], 8)
        self.assertEqual(report["proof_cases_passed"], 8)
        self.assertEqual(len(report["cases"]), 10)
        self.assertTrue(all(item["passed"] for item in report["cases"]))
        json.dumps(report, ensure_ascii=False)

    def test_provenance_never_overclaims_neural_reasoning(self) -> None:
        report = run_audit()
        self.assertFalse(report["malecns_checkpoint_loaded"])
        self.assertFalse(report["neural_reasoning_demonstrated"])
        self.assertTrue(all(not case["is_connectome_inference"] for case in report["cases"]))
        proofs = [c for c in report["cases"] if c["category"] == "proof"]
        self.assertTrue(all(c["origin"] == "deterministic_solver" for c in proofs))
        self.assertTrue(all(c["answer"] for c in proofs))

    def test_changed_input_requires_changed_result(self) -> None:
        by_name = {c["name"]: c for c in run_audit()["cases"]}
        self.assertNotEqual(
            by_name["quadratic_roots"]["answer"],
            by_name["changed_coefficients_change_roots"]["answer"],
        )
        self.assertNotEqual(
            by_name["forward_chain_entails"]["answer"],
            by_name["missing_premise_does_not_entail"]["answer"],
        )

    def test_even_forced_realizer_mode_cannot_call_llm(self) -> None:
        # The audit overrides an accidentally enabled model before realization.
        with (
            patch.dict(os.environ, {"FLYGPT_VERIFIED_REALIZER_MODE": "always"}),
            patch.object(ForbiddenGenerator, "_generate_once",
                         side_effect=AssertionError("LLM invoked")) as attempted,
        ):
            report = run_audit()
        self.assertEqual(report["status"], "PASS", report)
        attempted.assert_not_called()

    def test_unsolved_prompts_do_not_masquerade_as_proofs(self) -> None:
        by_name = {c["name"]: c for c in run_audit()["cases"]}
        self.assertEqual(by_name["open_ended_not_proven"]["category"], "abstention")
        self.assertEqual(
            by_name["greeting_is_only_a_semantic_plan"]["category"], "semantic_only"
        )
        self.assertIsNone(by_name["open_ended_not_proven"]["answer"])
        self.assertIsNone(by_name["greeting_is_only_a_semantic_plan"]["answer"])


if __name__ == "__main__":
    unittest.main()
