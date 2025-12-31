"""
End-to-End Stress Tests (Suite E) for v2.3.1
"""
import pytest
from legacycodebench.evaluators_v231 import EvaluatorV231

class TestE2EStress:
    """Suite E: Stress Testing."""

    def test_e2e_01_golden_test(self):
        """E2E-01: The Golden Test"""
        evaluator = EvaluatorV231()
        # Minimal valid input
        doc = "This program calculates the premium."
        source = "PROGRAM-ID. TEST."
        gt = {
            "business_rules": {"rules": []},
            "data_structures": {"structures": []}
        }
        
        result = evaluator.evaluate("LCB-TEST-01", "test-model", doc, source, gt)
        assert result.lcb_score >= 0
        assert result.version == "2.3.1"

    def test_e2e_05_monkey_test(self):
        """E2E-05: The Monkey Test (Random bytes)"""
        evaluator = EvaluatorV231()
        # 1KB of random garbage text (simulated)
        doc = "x8s7df6 g876dfg 87dfg6 87sdgf876sgdf g" * 100
        source = "GARBAGE"
        gt = {"invalid": "structure"}
        
        # Should not crash
        try:
            result = evaluator.evaluate("LCB-MONKEY-01", "monkey", doc, source, gt)
            assert result is not None
        except Exception as e:
            pytest.fail(f"Evaluator crashed on garbage input: {e}")

    def test_e2e_06_empty_test(self):
        """E2E-06: The Empty Test"""
        evaluator = EvaluatorV231()
        doc = ""
        source = ""
        gt = {}
        
        result = evaluator.evaluate("LCB-EMPTY-01", "empty", doc, source, gt)
        assert result.lcb_score >= 0 # Should probably be 0, but definitely handled
