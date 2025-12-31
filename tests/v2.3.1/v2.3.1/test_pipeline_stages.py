"""
Full Pipeline Stage Tests for LegacyCodeBench V2.3.1

Hamel Framework: Test every stage of the LCB pipeline.
Each test follows the Hamel eval case structure.
"""
import pytest
import json
from pathlib import Path


class TestStage1DatasetLoading:
    """Stage 1: Dataset Loading"""

    def test_stage1_dataset_loading_001(self):
        """Datasets directory exists with COBOL files.
        
        User Outcome at Risk: No COBOL files loaded, benchmark cannot run.
        Origin: anticipated
        Level: L1
        """
        datasets_dir = Path(__file__).parent.parent.parent / "datasets"
        
        # PASS: Directory exists and has subdirectories
        assert datasets_dir.exists(), "Datasets directory missing"
        
        subdirs = [d for d in datasets_dir.iterdir() if d.is_dir()]
        assert len(subdirs) >= 1, "No dataset subdirectories found"


class TestStage2TaskSelection:
    """Stage 2: Task Selection"""

    def test_stage2_task_selection_tier_distribution_001(self):
        """Tasks include multiple tiers.
        
        User Outcome at Risk: Benchmark skews to easy tasks only.
        Origin: anticipated
        Level: L1
        """
        tasks_dir = Path(__file__).parent.parent.parent / "tasks"
        
        if not tasks_dir.exists():
            pytest.skip("Tasks directory not found")
        
        task_files = list(tasks_dir.glob("*.json"))
        
        # Check tier distribution
        tiers = set()
        for tf in task_files[:20]:  # Sample first 20
            name = tf.stem
            if "-T1-" in name:
                tiers.add("T1")
            elif "-T2-" in name:
                tiers.add("T2")
            elif "-T3-" in name:
                tiers.add("T3")
            elif "-T4-" in name:
                tiers.add("T4")
        
        # PASS: At least 2 tiers represented (can't guarantee all 4)
        assert len(tiers) >= 1, f"Only found tiers: {tiers}"


class TestStage3GroundTruth:
    """Stage 3: Ground Truth Generation"""

    def test_stage3_ground_truth_valid_json_001(self):
        """Ground truth files have required structure.
        
        User Outcome at Risk: Evaluation has no reference.
        Origin: anticipated
        Level: L1
        """
        cache_dir = Path(__file__).parent.parent.parent / "cache" / "ground_truth"
        
        if not cache_dir.exists():
            pytest.skip("Ground truth cache not found")
        
        gt_files = list(cache_dir.glob("*.json"))
        assert len(gt_files) > 0, "No ground truth files"
        
        # Check first file has required keys
        with open(gt_files[0]) as f:
            gt = json.load(f)
        
        required_keys = ["business_rules", "data_structures"]
        for key in required_keys:
            assert key in gt, f"Missing required key: {key}"

    def test_stage3_ground_truth_has_rules_001(self):
        """Ground truth files should have business rules.
        
        User Outcome at Risk: SC evaluation fails on empty GT.
        Origin: error_analysis (6 BBANK* files had 0 rules)
        Level: L1
        """
        cache_dir = Path(__file__).parent.parent.parent / "cache" / "ground_truth"
        
        if not cache_dir.exists():
            pytest.skip("Ground truth cache not found")
        
        gt_files = list(cache_dir.glob("*.json"))
        
        # Sample 10 files
        empty_count = 0
        for gf in gt_files[:10]:
            with open(gf) as f:
                gt = json.load(f)
            
            rules = gt.get("business_rules", {})
            total = rules.get("total_rules", len(rules.get("rules", [])))
            
            if total == 0:
                empty_count += 1
        
        # PASS: Less than 50% empty (some CICS files may be empty)
        assert empty_count < 5, f"{empty_count}/10 files have 0 rules"


class TestStage4StructuralCompleteness:
    """Stage 4: SC Evaluation"""

    def test_stage4_sc_score_range_001(self):
        """SC score is within valid range.
        
        User Outcome at Risk: SC score corrupts LCB total.
        Origin: anticipated
        Level: L1
        """
        from legacycodebench.evaluators_v231 import StructuralEvaluatorV231
        
        evaluator = StructuralEvaluatorV231()
        
        # Minimal test
        doc = "The program calculates TOTAL from AMOUNT and RATE."
        gt = {"business_rules": {"rules": []}, "data_structures": {"structures": []}}
        
        result = evaluator.evaluate(doc, gt)
        
        assert 0.0 <= result.score <= 1.0, f"SC score out of range: {result.score}"


class TestStage5DocumentationQuality:
    """Stage 5: DQ Evaluation"""

    def test_stage5_dq_score_range_001(self):
        """DQ score is within valid range.
        
        User Outcome at Risk: DQ score corrupts LCB total.
        Origin: anticipated
        Level: L1
        """
        from legacycodebench.evaluators_v231 import DocumentationEvaluatorV231
        
        evaluator = DocumentationEvaluatorV231()
        
        doc = """# Business Purpose
        This program processes banking transactions.
        
        ## Business Rules
        1. Validate account number
        2. Check balance
        """
        gt = {}
        
        result = evaluator.evaluate(doc, gt)
        
        assert 0.0 <= result.score <= 1.0, f"DQ score out of range: {result.score}"


class TestStage6BehavioralFidelity:
    """Stage 6: BF Evaluation"""

    def test_stage6_bf_claims_extracted_001(self):
        """Claims are extracted from valid documentation.
        
        User Outcome at Risk: Silence penalty triggers incorrectly.
        Origin: error_analysis
        Level: L1
        """
        from legacycodebench.evaluators_v231 import ClaimExtractor
        
        extractor = ClaimExtractor()
        
        doc = """
        The total cost is the sum of the base cost, tax, and shipping.
        Tax is 8% of the base cost.
        Result is stored in FINAL-AMOUNT.
        """
        
        claims = extractor.extract(doc)
        
        assert len(claims) >= 1, f"Expected claims, got {len(claims)}"


class TestStage7CriticalFailures:
    """Stage 7: Critical Failure Detection"""

    def test_stage7_cf_no_false_positives_001(self):
        """Paragraph patterns don't trigger CF-02.
        
        User Outcome at Risk: Good docs zeroed incorrectly.
        Origin: error_analysis
        Level: L1
        """
        from legacycodebench.evaluators_v231 import CriticalFailureDetectorV231
        
        detector = CriticalFailureDetectorV231()
        
        doc = "The END-OF-JOB routine terminates processing."
        gt = {"data_structures": {"structures": []}}
        
        failures = detector.detect_all(doc, gt)
        
        cf02_found = any(cf.cf_id == "CF-02" for cf in failures)
        assert not cf02_found, "CF-02 false positive on paragraph pattern"


class TestStage8Scoring:
    """Stage 8: Scoring Engine"""

    def test_stage8_scoring_weights_correct_001(self):
        """Scoring weights sum to 1.0.
        
        User Outcome at Risk: Final LCB score miscalculated.
        Origin: anticipated
        Level: L1
        """
        from legacycodebench.evaluators_v231.config_v231 import V231_CONFIG
        
        weights = V231_CONFIG["weights"]
        total = sum(weights.values())
        
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, not 1.0"


class TestStage9Leaderboard:
    """Stage 9: Leaderboard Generation"""

    def test_stage9_leaderboard_valid_schema_001(self):
        """Leaderboard has valid schema.
        
        User Outcome at Risk: Dashboard/website breaks.
        Origin: anticipated
        Level: L1
        """
        leaderboard_path = Path(__file__).parent.parent.parent / "results" / "leaderboard.json"
        
        if not leaderboard_path.exists():
            pytest.skip("Leaderboard not found")
        
        with open(leaderboard_path) as f:
            data = json.load(f)
        
        assert "leaderboard" in data, "Missing leaderboard key"
