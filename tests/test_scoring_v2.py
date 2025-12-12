"""Test v2.0 Scoring System

Tests the v2.0 scoring formula and critical failure detection.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from legacycodebench.scoring import ScoringSystem


def test_lcb_v2_formula():
    """Test LCB v2.0 score calculation"""
    print("\n" + "=" * 70)
    print("Test: LCB v2.0 Formula")
    print("=" * 70)

    scorer = ScoringSystem()

    # Perfect scores
    score = scorer.calculate_lcb_v2_score(
        sc=100.0,
        bf=100.0,
        sq=100.0,
        tr=100.0,
        critical_failures=None
    )

    print(f"Perfect scores: {score:.2f}")
    assert score == 100.0, "Perfect scores should yield 100.0"

    # Weighted average
    # SC:30%, BF:35%, SQ:25%, TR:10%
    score = scorer.calculate_lcb_v2_score(
        sc=80.0,   # 80 * 0.30 = 24
        bf=90.0,   # 90 * 0.35 = 31.5
        sq=70.0,   # 70 * 0.25 = 17.5
        tr=60.0,   # 60 * 0.10 = 6
        critical_failures=None
    )
    expected = 24 + 31.5 + 17.5 + 6  # 79.0

    print(f"Weighted scores: {score:.2f} (expected: {expected:.2f})")
    assert abs(score - expected) < 0.1, f"Should calculate {expected}, got {score}"

    print("[PASS] LCB v2.0 formula test passed")
    return True


def test_critical_failure_auto_fail():
    """Test that critical failures result in score of 0"""
    print("\n" + "=" * 70)
    print("Test: Critical Failure Auto-Fail")
    print("=" * 70)

    scorer = ScoringSystem()

    # Even with perfect component scores, CF should result in 0
    score = scorer.calculate_lcb_v2_score(
        sc=100.0,
        bf=100.0,
        sq=100.0,
        tr=100.0,
        critical_failures=["CF-03"]  # Critical failure present
    )

    print(f"Score with CF-03: {score:.2f}")
    assert score == 0.0, "Critical failure should result in score of 0"

    print("[PASS] Critical failure auto-fail test passed")
    return True


def test_cf01_detection():
    """Test CF-01: Missing primary calculation"""
    print("\n" + "=" * 70)
    print("Test: CF-01 Detection")
    print("=" * 70)

    scorer = ScoringSystem()

    # Create evaluation result with missing critical calculation
    result = {
        'structural_completeness': {
            'missing_elements': {
                'business_rules': [
                    {'description': 'Calculate interest on principal'}
                ]
            }
        }
    }

    failures = scorer.detect_critical_failures(result)

    print(f"Critical failures: {failures}")
    assert "CF-01" in failures, "Should detect missing primary calculation"

    print("[PASS] CF-01 detection test passed")
    return True


def test_cf02_detection():
    """Test CF-02: Hallucinated module"""
    print("\n" + "=" * 70)
    print("Test: CF-02 Detection")
    print("=" * 70)

    scorer = ScoringSystem()

    # Create evaluation result with fabricated references
    result = {
        'traceability': {
            'invalid_references': [
                {'type': 'fabricated', 'name': 'NONEXISTENT-MODULE'}
            ]
        }
    }

    failures = scorer.detect_critical_failures(result)

    print(f"Critical failures: {failures}")
    assert "CF-02" in failures, "Should detect hallucinated module"

    print("[PASS] CF-02 detection test passed")
    return True


def test_cf03_detection():
    """Test CF-03: Wrong data transformation"""
    print("\n" + "=" * 70)
    print("Test: CF-03 Detection")
    print("=" * 70)

    scorer = ScoringSystem()

    # Create evaluation result with high output mismatch
    result = {
        'behavioral_fidelity': {
            'output_mismatch_rate': 0.15  # 15% > 10% threshold
        }
    }

    failures = scorer.detect_critical_failures(result)

    print(f"Critical failures: {failures}")
    print(f"Mismatch rate: 15%")
    assert "CF-03" in failures, "Should detect wrong data transformation"

    print("[PASS] CF-03 detection test passed")
    return True


def test_cf04_detection():
    """Test CF-04: Missing error handler"""
    print("\n" + "=" * 70)
    print("Test: CF-04 Detection")
    print("=" * 70)

    scorer = ScoringSystem()

    # Create evaluation result with missing error handlers
    result = {
        'structural_completeness': {
            'missing_elements': {
                'error_handlers': ['ON SIZE ERROR', 'INVALID KEY']
            }
        },
        'ground_truth': {
            'error_handlers': {
                'on_size_error': 1,
                'invalid_key': 1,
                'at_end': 0,
                'file_status_checks': 0
            }
        }
    }

    failures = scorer.detect_critical_failures(result)

    print(f"Critical failures: {failures}")
    print(f"Missing: 2/2 error handlers (100%)")
    assert "CF-04" in failures, "Should detect missing error handlers"

    print("[PASS] CF-04 detection test passed")
    return True


def test_cf05_detection():
    """Test CF-05: Broken traceability"""
    print("\n" + "=" * 70)
    print("Test: CF-05 Detection")
    print("=" * 70)

    scorer = ScoringSystem()

    # Create evaluation result with high broken reference rate
    result = {
        'traceability': {
            'broken_reference_rate': 0.25  # 25% > 20% threshold
        }
    }

    failures = scorer.detect_critical_failures(result)

    print(f"Critical failures: {failures}")
    print(f"Broken reference rate: 25%")
    assert "CF-05" in failures, "Should detect broken traceability"

    print("[PASS] CF-05 detection test passed")
    return True


def test_cf06_detection():
    """Test CF-06: False positive"""
    print("\n" + "=" * 70)
    print("Test: CF-06 Detection")
    print("=" * 70)

    scorer = ScoringSystem()

    # Create evaluation result where tests pass but gaps present
    result = {
        'behavioral_fidelity': {
            'tests_passed_rate': 0.97,  # 97% passed
            'gap_markers': 5  # But gaps are present!
        }
    }

    failures = scorer.detect_critical_failures(result)

    print(f"Critical failures: {failures}")
    print(f"Test pass rate: 97%")
    print(f"Gap markers: 5")
    assert "CF-06" in failures, "Should detect false positive"

    print("[PASS] CF-06 detection test passed")
    return True


def test_aggregate_v2_results():
    """Test aggregation of v2.0 results"""
    print("\n" + "=" * 70)
    print("Test: Aggregate v2.0 Results")
    print("=" * 70)

    scorer = ScoringSystem()

    results = [
        {
            'structural_completeness_score': 80.0,
            'behavioral_fidelity_score': 85.0,
            'semantic_quality_score': 75.0,
            'traceability_score': 90.0,
            'lcb_score': 82.0,
            'critical_failures': []
        },
        {
            'structural_completeness_score': 90.0,
            'behavioral_fidelity_score': 95.0,
            'semantic_quality_score': 85.0,
            'traceability_score': 88.0,
            'lcb_score': 90.0,
            'critical_failures': []
        },
        {
            'structural_completeness_score': 70.0,
            'behavioral_fidelity_score': 75.0,
            'semantic_quality_score': 65.0,
            'traceability_score': 80.0,
            'lcb_score': 72.0,
            'critical_failures': ["CF-03"]  # 1 task with CF
        }
    ]

    aggregated = scorer.aggregate_v2_results(results)

    print(f"Total tasks: {aggregated['total_tasks']}")
    print(f"LCB v2.0 avg: {aggregated['lcb_v2_avg']:.2f}")
    print(f"SC avg: {aggregated['structural_completeness_avg']:.2f}")
    print(f"BF avg: {aggregated['behavioral_fidelity_avg']:.2f}")
    print(f"SQ avg: {aggregated['semantic_quality_avg']:.2f}")
    print(f"TR avg: {aggregated['traceability_avg']:.2f}")
    print(f"Critical failure rate: {aggregated['critical_failure_rate']:.1%}")

    assert aggregated['total_tasks'] == 3
    assert aggregated['critical_failure_count'] == 1
    assert abs(aggregated['critical_failure_rate'] - 1/3) < 0.01, f"CF rate should be ~0.333, got {aggregated['critical_failure_rate']}"

    # Check averages
    expected_lcb_avg = (82.0 + 90.0 + 72.0) / 3
    assert abs(aggregated['lcb_v2_avg'] - expected_lcb_avg) < 0.1

    print("[PASS] Aggregate v2.0 results test passed")
    return True


def test_no_critical_failures():
    """Test that clean result has no CFs"""
    print("\n" + "=" * 70)
    print("Test: No Critical Failures")
    print("=" * 70)

    scorer = ScoringSystem()

    # Create clean evaluation result
    result = {
        'structural_completeness': {
            'missing_elements': {
                'business_rules': [],
                'error_handlers': []
            }
        },
        'behavioral_fidelity': {
            'output_mismatch_rate': 0.02,  # 2% < 10%
            'tests_passed_rate': 0.98,
            'gap_markers': 0  # No gaps
        },
        'traceability': {
            'broken_reference_rate': 0.05,  # 5% < 20%
            'invalid_references': []
        },
        'ground_truth': {
            'error_handlers': {
                'on_size_error': 0,
                'invalid_key': 0,
                'at_end': 0,
                'file_status_checks': 0
            }
        }
    }

    failures = scorer.detect_critical_failures(result)

    print(f"Critical failures: {failures}")
    assert len(failures) == 0, "Clean result should have no critical failures"

    print("[PASS] No critical failures test passed")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("v2.0 Scoring System Test Suite")
    print("=" * 70)

    tests = [
        ("LCB v2.0 Formula", test_lcb_v2_formula),
        ("Critical Failure Auto-Fail", test_critical_failure_auto_fail),
        ("CF-01 Detection", test_cf01_detection),
        ("CF-02 Detection", test_cf02_detection),
        ("CF-03 Detection", test_cf03_detection),
        ("CF-04 Detection", test_cf04_detection),
        ("CF-05 Detection", test_cf05_detection),
        ("CF-06 Detection", test_cf06_detection),
        ("Aggregate v2.0 Results", test_aggregate_v2_results),
        ("No Critical Failures", test_no_critical_failures),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                skipped += 1
        except AssertionError as e:
            print(f"[FAIL] {test_name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test_name} ERROR: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed == 0 and passed > 0:
        print("[PASS] ALL TESTS PASSED")
    print("=" * 70)
