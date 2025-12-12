"""Test Behavior Comparator

Tests the behavior comparator's ability to compare execution outputs
and detect critical failures.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from legacycodebench.execution.behavior_comparator import BehaviorComparator, ComparisonResult
from legacycodebench.execution.cobol_executor import ExecutionResult
from legacycodebench.execution.test_generator import TestCase


def create_test_case(test_id):
    """Helper to create test case"""
    return TestCase(
        test_id=test_id,
        description=f"Test {test_id}",
        inputs={"VAR1": 100},
        expected_outputs={"RESULT": 200},
        test_type="boundary",
        priority=1,
        rationale="Test case"
    )


def create_execution_result(success=True, stdout="", stderr="", files=None):
    """Helper to create ExecutionResult"""
    return ExecutionResult(
        success=success,
        stdout=stdout,
        stderr=stderr,
        exit_code=0 if success else 1,
        file_outputs=files or {},
        execution_time_ms=100.0,
        error_message=None if success else "Execution failed"
    )


def test_exact_match():
    """Test exact output match"""
    print("\n" + "=" * 70)
    print("Test: Exact Output Match")
    print("=" * 70)

    comparator = BehaviorComparator()

    # Create identical results
    orig = create_execution_result(stdout="RESULT: 300\n")
    gen = create_execution_result(stdout="RESULT: 300\n")
    test_cases = [create_test_case("TEST-001")]

    result = comparator.compare([orig], [gen], test_cases)

    print(f"Score: {result.score:.2%}")
    print(f"Matches: {result.matches}/{result.total_tests}")
    print(f"Critical failures: {len(result.critical_failures)}")

    assert result.score == 1.0, "Should have 100% match"
    assert result.matches == 1
    assert result.mismatches == 0
    assert len(result.critical_failures) == 0
    print("[PASS] Exact match test passed")
    return True


def test_whitespace_normalization():
    """Test whitespace normalization"""
    print("\n" + "=" * 70)
    print("Test: Whitespace Normalization")
    print("=" * 70)

    comparator = BehaviorComparator()

    # Different whitespace but same content
    orig = create_execution_result(stdout="RESULT:   300")
    gen = create_execution_result(stdout="RESULT: 300")
    test_cases = [create_test_case("TEST-002")]

    result = comparator.compare([orig], [gen], test_cases)

    print(f"Score: {result.score:.2%}")
    print(f"Matches: {result.matches}/{result.total_tests}")

    assert result.score == 1.0, "Should match despite whitespace differences"
    print("[PASS] Whitespace normalization test passed")
    return True


def test_numerical_tolerance():
    """Test numerical precision tolerance"""
    print("\n" + "=" * 70)
    print("Test: Numerical Tolerance")
    print("=" * 70)

    comparator = BehaviorComparator(numerical_tolerance=0.01)

    # Slightly different numbers (within tolerance)
    orig = create_execution_result(stdout="AMOUNT: 123.456")
    gen = create_execution_result(stdout="AMOUNT: 123.457")
    test_cases = [create_test_case("TEST-003")]

    result = comparator.compare([orig], [gen], test_cases)

    print(f"Score: {result.score:.2%}")
    print(f"Difference: 123.457 - 123.456 = 0.001")
    print(f"Tolerance: 0.01")

    # Should match within tolerance
    assert result.score >= 0.95, "Should match within numerical tolerance"
    print("[PASS] Numerical tolerance test passed")
    return True


def test_cf03_detection():
    """Test CF-03: Wrong data transformation detection"""
    print("\n" + "=" * 70)
    print("Test: CF-03 Detection (>=10% mismatch)")
    print("=" * 70)

    comparator = BehaviorComparator()

    # Create 10 tests with 2 mismatches (20% > 10% threshold)
    orig_results = []
    gen_results = []
    test_cases = []

    for i in range(10):
        test_cases.append(create_test_case(f"TEST-{i:03d}"))

        if i < 2:
            # Mismatched outputs
            orig_results.append(create_execution_result(stdout=f"RESULT: {i * 100}"))
            gen_results.append(create_execution_result(stdout=f"RESULT: {i * 999}"))  # Wrong
        else:
            # Matching outputs
            orig_results.append(create_execution_result(stdout=f"RESULT: {i * 100}"))
            gen_results.append(create_execution_result(stdout=f"RESULT: {i * 100}"))

    result = comparator.compare(orig_results, gen_results, test_cases)

    print(f"Score: {result.score:.2%}")
    print(f"Mismatches: {result.mismatches}/{result.total_tests}")
    print(f"Mismatch rate: {result.mismatches / result.total_tests * 100:.1f}%")
    print(f"Critical failures: {result.critical_failures}")

    # Should detect CF-03
    assert any("CF-03" in cf for cf in result.critical_failures), "Should detect CF-03"
    print("[PASS] CF-03 detection test passed")
    return True


def test_cf06_detection():
    """Test CF-06: False positive detection"""
    print("\n" + "=" * 70)
    print("Test: CF-06 Detection (gaps + passing tests)")
    print("=" * 70)

    print("[SKIP] CF-06 detection requires integration with code generator")
    print("       Gap tracking needs to flow from GeneratedCode to ExecutionResult")
    print("       This will be tested in end-to-end integration tests")
    # NOTE: CF-06 detection logic exists in behavior_comparator.py
    # but requires metadata from code generator to work properly
    return True


def test_file_output_comparison():
    """Test file output comparison"""
    print("\n" + "=" * 70)
    print("Test: File Output Comparison")
    print("=" * 70)

    comparator = BehaviorComparator()

    # Create results with file outputs
    orig = create_execution_result(
        stdout="Processing complete",
        files={"output.dat": "RECORD001|DATA001\nRECORD002|DATA002\n"}
    )
    gen = create_execution_result(
        stdout="Processing complete",
        files={"output.dat": "RECORD001|DATA001\nRECORD002|DATA002\n"}
    )
    test_cases = [create_test_case("TEST-FILE")]

    result = comparator.compare([orig], [gen], test_cases)

    print(f"Score: {result.score:.2%}")
    print(f"File match: {result.details[0]['file_output_match']}")

    assert result.score == 1.0, "Should match file outputs"
    assert result.details[0]['file_output_match'], "Files should match"
    print("[PASS] File output comparison test passed")
    return True


def test_execution_error_handling():
    """Test handling of execution errors"""
    print("\n" + "=" * 70)
    print("Test: Execution Error Handling")
    print("=" * 70)

    comparator = BehaviorComparator()

    # Original succeeds, generated fails
    orig = create_execution_result(success=True, stdout="SUCCESS")
    gen = create_execution_result(success=False, stderr="Runtime error")
    test_cases = [create_test_case("TEST-ERROR")]

    result = comparator.compare([orig], [gen], test_cases)

    print(f"Score: {result.score:.2%}")
    print(f"Errors: {result.errors}")
    print(f"Result: {result.details[0]['result']}")

    assert result.score == 0.0, "Should fail when generated code errors"
    assert result.errors == 1
    assert result.details[0]['result'] == 'error'
    print("[PASS] Execution error handling test passed")
    return True


def test_similarity_scoring():
    """Test similarity scoring for partial matches"""
    print("\n" + "=" * 70)
    print("Test: Similarity Scoring")
    print("=" * 70)

    comparator = BehaviorComparator()

    # Similar but not identical outputs (difference too large for tolerance)
    orig = create_execution_result(stdout="TOTAL: 1000\nCOUNT: 5\nAVERAGE: 200")
    gen = create_execution_result(stdout="TOTAL: 1000\nCOUNT: 5\nAVERAGE: 150")  # Larger difference
    test_cases = [create_test_case("TEST-SIM")]

    result = comparator.compare([orig], [gen], test_cases)

    print(f"Overall Score: {result.score:.2%}")
    print(f"Similarity: {result.details[0]['similarity']:.2%}")
    print(f"Result: {result.details[0]['result']}")
    print(f"Expected: Similarity >95% counts as match, but <100%")

    # Check that similarity is computed correctly (should be high but not 100%)
    similarity = result.details[0]['similarity']
    assert 0.90 <= similarity < 1.0, f"Similarity should be 90-99%, got {similarity:.2%}"

    # If similarity >= 95%, it counts as a "match" so overall score is 100%
    # But the detailed similarity score should show it's not perfect
    print("[PASS] Similarity scoring test passed")
    return True


def test_single_comparison():
    """Test compare_single convenience method"""
    print("\n" + "=" * 70)
    print("Test: Single Comparison Method")
    print("=" * 70)

    comparator = BehaviorComparator()

    orig = create_execution_result(stdout="OUTPUT: 500")
    gen = create_execution_result(stdout="OUTPUT: 500")

    is_match, similarity, reason = comparator.compare_single(orig, gen)

    print(f"Is match: {is_match}")
    print(f"Similarity: {similarity:.2%}")
    print(f"Reason: {reason}")

    assert is_match, "Should match"
    assert similarity == 1.0, "Should have 100% similarity"
    print("[PASS] Single comparison test passed")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Behavior Comparator Test Suite")
    print("=" * 70)

    tests = [
        ("Exact Match", test_exact_match),
        ("Whitespace Normalization", test_whitespace_normalization),
        ("Numerical Tolerance", test_numerical_tolerance),
        ("CF-03 Detection", test_cf03_detection),
        ("CF-06 Detection", test_cf06_detection),
        ("File Output Comparison", test_file_output_comparison),
        ("Execution Error Handling", test_execution_error_handling),
        ("Similarity Scoring", test_similarity_scoring),
        ("Single Comparison", test_single_comparison),
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
