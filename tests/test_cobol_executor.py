"""Test COBOL Executor Implementation

Tests the complete execution pipeline:
1. Docker image availability
2. COBOL compilation
3. Program execution
4. Output capture
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from legacycodebench.execution.cobol_executor import COBOLExecutor, ExecutionResult


def test_docker_available():
    """Test that Docker is available and image exists"""
    try:
        executor = COBOLExecutor()
        assert executor is not None
        print("[PASS] Docker is available")
        print(f"[INFO] Image: {executor.docker_image}")
        return True
    except RuntimeError as e:
        print(f"[SKIP] Docker not available: {e}")
        return False


def test_simple_hello_world():
    """Test simple HELLO WORLD program"""
    try:
        executor = COBOLExecutor()
    except RuntimeError as e:
        print(f"[SKIP] Docker not available: {e}")
        return False

    cobol_code = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. HELLO.
       PROCEDURE DIVISION.
           DISPLAY "Hello from GnuCOBOL".
           STOP RUN.
    """

    result = executor.execute(cobol_code, test_inputs={})

    print(f"\n=== Test: Simple Hello World ===")
    print(f"Success: {result.success}")
    print(f"Exit Code: {result.exit_code}")
    print(f"Output: {result.stdout}")
    print(f"Errors: {result.stderr}")
    print(f"Execution Time: {result.execution_time_ms}ms")

    assert result.success, f"Execution failed: {result.error_message}"
    assert "Hello from GnuCOBOL" in result.stdout
    assert result.exit_code == 0
    print("[PASS] Simple Hello World passed")
    return True


def test_arithmetic_calculation():
    """Test arithmetic operations"""
    try:
        executor = COBOLExecutor()
    except RuntimeError as e:
        print(f"[SKIP] Docker not available: {e}")
        return False

    cobol_code = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. CALC.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  NUM1            PIC 9(5) VALUE 100.
       01  NUM2            PIC 9(5) VALUE 200.
       01  RESULT          PIC 9(5).
       PROCEDURE DIVISION.
           COMPUTE RESULT = NUM1 + NUM2.
           DISPLAY "RESULT: " RESULT.
           STOP RUN.
    """

    result = executor.execute(cobol_code, test_inputs={})

    print(f"\n=== Test: Arithmetic Calculation ===")
    print(f"Success: {result.success}")
    print(f"Output: {result.stdout}")

    assert result.success, f"Execution failed: {result.error_message}"
    assert "300" in result.stdout
    print("[PASS] Arithmetic calculation passed")
    return True


def test_accept_input():
    """Test ACCEPT statement with input data"""
    try:
        executor = COBOLExecutor()
    except RuntimeError as e:
        print(f"[SKIP] Docker not available: {e}")
        return False

    cobol_code = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. INPUT-TEST.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  USER-INPUT      PIC 9(5).
       01  DOUBLED         PIC 9(5).
       PROCEDURE DIVISION.
           ACCEPT USER-INPUT.
           COMPUTE DOUBLED = USER-INPUT * 2.
           DISPLAY "INPUT: " USER-INPUT.
           DISPLAY "DOUBLED: " DOUBLED.
           STOP RUN.
    """

    # Test with input value 50
    result = executor.execute(cobol_code, test_inputs={"USER-INPUT": 50})

    print(f"\n=== Test: ACCEPT Input ===")
    print(f"Success: {result.success}")
    print(f"Output: {result.stdout}")

    assert result.success, f"Execution failed: {result.error_message}"
    assert "50" in result.stdout  # Input value
    assert "100" in result.stdout  # Doubled value
    print("[PASS] ACCEPT input passed")
    return True


def test_compilation_error():
    """Test handling of compilation errors"""
    try:
        executor = COBOLExecutor()
    except RuntimeError as e:
        print(f"[SKIP] Docker not available: {e}")
        return False

    # Invalid COBOL code (missing period)
    invalid_code = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. INVALID
       PROCEDURE DIVISION.
           DISPLAY "This will fail".
           STOP RUN.
    """

    result = executor.execute(invalid_code, test_inputs={})

    print(f"\n=== Test: Compilation Error ===")
    print(f"Success: {result.success}")
    print(f"Error: {result.error_message}")

    assert not result.success, "Should fail with compilation error"
    assert result.exit_code != 0
    print("[PASS] Compilation error handling passed")
    return True


def test_batch_execution():
    """Test batch execution with multiple test cases"""
    try:
        executor = COBOLExecutor()
    except RuntimeError as e:
        print(f"[SKIP] Docker not available: {e}")
        return False

    cobol_code = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. MULTIPLY.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  INPUT-NUM       PIC 9(5).
       01  RESULT          PIC 9(5).
       PROCEDURE DIVISION.
           ACCEPT INPUT-NUM.
           COMPUTE RESULT = INPUT-NUM * 3.
           DISPLAY "RESULT: " RESULT.
           STOP RUN.
    """

    # Multiple test cases
    test_cases = [
        {"INPUT-NUM": 10},   # Expected: 30
        {"INPUT-NUM": 20},   # Expected: 60
        {"INPUT-NUM": 100},  # Expected: 300
    ]

    results = executor.execute_batch(cobol_code, test_cases)

    print(f"\n=== Test: Batch Execution ===")
    print(f"Total test cases: {len(results)}")

    for i, result in enumerate(results):
        print(f"  Test {i+1}: {'PASS' if result.success else 'FAIL'} - {result.stdout.strip()}")

    assert all(r.success for r in results), "All batch executions should succeed"
    assert len(results) == 3
    print("[PASS] Batch execution passed")
    return True


def test_execution_timeout():
    """Test timeout handling"""
    try:
        executor = COBOLExecutor(timeout_seconds=2)
    except RuntimeError as e:
        print(f"[SKIP] Docker not available: {e}")
        return False

    # Infinite loop - will timeout
    timeout_code = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. TIMEOUT-TEST.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  COUNTER         PIC 9(5) VALUE 0.
       PROCEDURE DIVISION.
           PERFORM FOREVER
               ADD 1 TO COUNTER
           END-PERFORM.
           STOP RUN.
    """

    result = executor.execute(timeout_code, test_inputs={})

    print(f"\n=== Test: Execution Timeout ===")
    print(f"Success: {result.success}")
    print(f"Timeout: {result.timeout}")
    print(f"Error: {result.error_message}")

    assert not result.success
    assert result.timeout or "timed out" in result.stderr.lower()
    print("[PASS] Timeout handling passed")
    return True


def test_program_name_extraction():
    """Test PROGRAM-ID extraction"""
    try:
        executor = COBOLExecutor()
    except RuntimeError as e:
        print(f"[SKIP] Docker not available: {e}")
        return False

    cobol_code = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. MY-CUSTOM-PROGRAM.
       PROCEDURE DIVISION.
           DISPLAY "Program name test".
           STOP RUN.
    """

    # Should extract "MY_CUSTOM_PROGRAM" (hyphens replaced)
    program_name = executor._extract_program_name(cobol_code)

    print(f"\n=== Test: Program Name Extraction ===")
    print(f"Extracted: {program_name}")

    assert program_name == "MY_CUSTOM_PROGRAM"
    print("[PASS] Program name extraction passed")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("COBOL Executor Test Suite")
    print("=" * 70)

    # Run tests and track results
    tests = [
        ("Docker Available", test_docker_available),
        ("Simple Hello World", test_simple_hello_world),
        ("Arithmetic Calculation", test_arithmetic_calculation),
        ("ACCEPT Input", test_accept_input),
        ("Compilation Error", test_compilation_error),
        ("Batch Execution", test_batch_execution),
        ("Execution Timeout", test_execution_timeout),
        ("Program Name Extraction", test_program_name_extraction),
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
            print(f"[FAIL] {test_name} ERROR: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed == 0 and passed > 0:
        print("[PASS] ALL TESTS PASSED")
    print("=" * 70)
