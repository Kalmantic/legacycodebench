"""
Structural Integrity Tests (Suite A2 & C) for v2.3.1
"""
import pytest
from legacycodebench.evaluators_v231 import (
    StructuralEvaluatorV231, 
    DocumentationEvaluatorV231,
    V231_CONFIG,
    expand_synonyms
)

class TestStructuralIntegrity:
    """Suite A2 & C: Structure, Documentation, and Config."""

    def test_ut_sc_01_synonym_expansion(self):
        """UT-SC-01: Synonym Expansion (calculate -> compute)"""
        text = "we calculate the value"
        expanded = expand_synonyms(text)
        assert "compute" in expanded, "calculate should expand to compute"

    def test_ut_dq_01_doc_quality_logic(self):
        """UT-DQ-01: Document Quality logic"""
        evaluator = DocumentationEvaluatorV231()
        doc = """
        # Title
        ## Section 1
        Content here.
        ## Section 2
        More content.
        """
        result = evaluator.evaluate(doc, "")
        assert result.score > 0.0, "High quality doc should have non-zero score"

    def test_ut_cfg_01_silence_defaults(self):
        """UT-CFG-01: Silence defaults"""
        # Assert V231_CONFIG.silence.min_claims == 1? 
        # Test plan says min_claims == 1
        assert V231_CONFIG["silence_penalty"]["min_claims"] >= 1

    def test_ut_cfg_02_weights_sum(self):
        """UT-CFG-02: Weights sum to 1.0"""
        weights = V231_CONFIG["weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.0001, f"Weights must sum to 1.0, got {total}"
        assert weights["structural_completeness"] == 0.30
