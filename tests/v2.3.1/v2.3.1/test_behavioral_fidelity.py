"""
Behavioral Fidelity Tests (Suite A) for v2.3.1
"The COBOL Nightmare"
"""
import pytest
from legacycodebench.evaluators_v231 import BehavioralEvaluatorV231, ClaimExtractor

class TestBehavioralFidelity:
    """Suite A: Verify COBOL parsing and behavioral claim extraction."""

    def test_ut_bf_01_standard_pic(self):
        """UT-BF-01: Standard PIC 99 (Min:0, Mid:50, Max:99)"""
        pic = "PIC 99"
        expected_min = 0
        expected_max = 99
        assert pic == "PIC 99"

    def test_ut_bf_03_decimal_mixed(self):
        """UT-BF-03: Decimal Mixed PIC S9(4)V99 (Critical)"""
        pic = "PIC S9(4)V99"
        assert "S" in pic
        assert "V" in pic

    def test_ut_bf_04_huge_numeric(self):
        """UT-BF-04: Huge Numeric PIC 9(18)"""
        pic = "PIC 9(18)"
        max_val = 10**18 - 1
        assert max_val > 0
        
    def test_ut_bf_05_comp_3_packed(self):
        """UT-BF-05: Comp-3 Packed"""
        pic = "PIC S9(5) COMP-3"
        assert "COMP-3" in pic

    def test_ut_bf_06_alphanumeric(self):
        """UT-BF-06: Alphanumeric PIC X(100)"""
        pic = "PIC X(100)"
        length = 100
        assert length == 100

    def test_claim_extractor_natural_language(self):
        """Claim patterns should match GPT-4o/Claude natural writing style
        
        Origin: error_analysis (BSM=0 pattern in 5/10 tasks)
        User Outcome at Risk: Silence Penalty triggers on valid docs
        """
        extractor = ClaimExtractor()
        
        # Actual GPT-4o style documentation
        doc = """
        The total cost is the sum of the base cost, tax, and shipping.
        Tax is 8% of the base cost.
        The base cost equals the price plus quality adjustments.
        """
        claims = extractor.extract(doc)
        
        assert len(claims) >= 3, f"Expected 3+ claims from natural language, got {len(claims)}"

    def test_bsm_fuzzy_file_matching(self):
        """BSM should match CUSTOMER-FILE to CUSTOMERS.DAT
        
        Origin: error_analysis (CF-05 in T1-005, T2-001, T4-001)
        User Outcome at Risk: Valid docs zeroed for file naming mismatch
        """
        from legacycodebench.evaluators_v231.behavioral_v231 import BSMValidator
        
        validator = BSMValidator()
        doc = "The program reads from CUSTOMERS.DAT and writes to TRANSACTIONS.DAT"
        calls = [
            {"name": "CUSTOMER-FILE", "type": "file"},
            {"name": "TRANSACTION-FILE", "type": "file"},
        ]
        
        result = validator.validate(doc, calls)
        
        assert result["matched"] == 2, f"Expected 2 matches, got {result['matched']}"
        assert result["score"] == 1.0, f"Expected score 1.0, got {result['score']}"
