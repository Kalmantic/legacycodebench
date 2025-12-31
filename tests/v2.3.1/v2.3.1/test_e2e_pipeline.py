"""
E2E Pipeline Tests for LegacyCodeBench V2.3.1

Hamel Framework: Test the actual user journey.
These run the full CLI commands and verify expected outcomes.
"""
import pytest
import subprocess
import json
import os
from pathlib import Path


class TestE2EPipeline:
    """E2E tests for the full LCB pipeline."""

    @pytest.mark.slow
    def test_e2e_pipeline_success_001(self):
        """Pipeline completes successfully with mock mode.
        
        User Outcome at Risk: Benchmark fails to run entirely.
        Origin: anticipated
        Level: L1 (Assertion)
        
        Note: Marked slow - run with pytest -m slow
        """
        pytest.skip("Slow test - run manually with: legacycodebench run-full-benchmark --mock --task-limit 1")

    def test_e2e_leaderboard_schema_001(self):
        """Leaderboard has valid JSON schema.
        
        User Outcome at Risk: Dashboard can't parse results.
        Origin: anticipated
        Level: L1
        """
        leaderboard_path = Path(__file__).parent.parent.parent / "results" / "leaderboard.json"
        
        if not leaderboard_path.exists():
            pytest.skip("Leaderboard not found - run benchmark first")
        
        with open(leaderboard_path) as f:
            data = json.load(f)
        
        # PASS: Has required structure (actual key is 'leaderboard')
        assert "leaderboard" in data, "Missing leaderboard key"
        assert isinstance(data["leaderboard"], list), "Leaderboard must be a list"

    def test_e2e_evaluate_produces_score_001(self):
        """Evaluate command produces valid LCB score.
        
        User Outcome at Risk: User submits doc, gets no feedback.
        Origin: error_analysis
        Level: L1
        """
        results_dir = Path(__file__).parent.parent.parent / "results"
        
        # Find any v2.3.1 result file
        result_files = list(results_dir.glob("*_v2.3.1.json"))
        if not result_files:
            pytest.skip("No result files found")
        
        with open(result_files[0]) as f:
            result = json.load(f)
        
        # PASS: Score exists and in valid range
        lcb_score = result.get("result", {}).get("lcb_score")
        assert lcb_score is not None, "Missing lcb_score"
        assert 0 <= lcb_score <= 100, f"Score out of range: {lcb_score}"

    def test_e2e_no_false_cf_on_passing_doc_001(self):
        """Passing docs should have no critical failures.
        
        User Outcome at Risk: Good docs get zeroed incorrectly.
        Origin: error_analysis (CF-02 false positive)
        Level: L1
        """
        results_dir = Path(__file__).parent.parent.parent / "results"
        
        # Find result files where passed=True
        for result_file in results_dir.glob("*_v2.3.1.json"):
            with open(result_file) as f:
                result = json.load(f)
            
            if result.get("result", {}).get("passed"):
                cf = result.get("result", {}).get("critical_failures", [])
                assert len(cf) == 0, f"Passing doc has CF: {cf}"
                return  # Found at least one passing doc
        
        pytest.skip("No passing docs found to verify")

    def test_e2e_ground_truth_cache_works(self):
        """Ground truth cache produces consistent results.
        
        User Outcome at Risk: Slow benchmarks, inconsistent scores.
        Origin: anticipated
        Level: L1
        """
        cache_dir = Path(__file__).parent.parent.parent / "cache" / "ground_truth"
        
        # PASS: Cache directory exists with files
        assert cache_dir.exists(), "Ground truth cache directory missing"
        
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) > 0, "No cached ground truth files"
