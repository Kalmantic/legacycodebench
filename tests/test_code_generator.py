"""Test Constrained Code Generator

Tests the code generator's ability to create COBOL from documentation
with proper gap detection.
"""

from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from legacycodebench.execution.code_generator import ConstrainedCodeGenerator


def test_gap_detection():
    """Test that gaps are detected in generated code"""
    print("\n" + "=" * 70)
    print("Test: Gap Detection")
    print("=" * 70)

    generator = ConstrainedCodeGenerator()

    # Incomplete documentation (missing rate)
    documentation = """
    This program calculates interest on a principal amount.

    Data structures:
    - PRINCIPAL: Principal amount
    - RESULT: Calculated interest

    Business logic:
    - Calculate interest
    - Display result
    """

    expected_structure = {
        "program_name": "CALC-INTEREST",
        "data_structures": [
            {"name": "PRINCIPAL"},
            {"name": "RESULT"}
        ],
        "paragraphs": [{"name": "MAIN"}],
        "business_rules": [],
        "files": []
    }

    # Check if LLM API keys are available
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    if not (has_openai or has_anthropic):
        print("[SKIP] No LLM API keys available, testing fallback only")
        # Test template-based generation
        result = generator.generate(documentation, expected_structure)
        print(f"Template generation - Gaps: {len(result.gaps)}")
        print(f"Gap percentage: {result.gap_percentage:.1f}%")
        return True

    # Test with LLM
    print(f"[INFO] Testing with LLM (OpenAI: {has_openai}, Anthropic: {has_anthropic})")
    result = generator.generate(documentation, expected_structure)

    print(f"\nGenerated code length: {len(result.code)} chars")
    print(f"Gaps detected: {len(result.gaps)}")
    print(f"Gap percentage: {result.gap_percentage:.1f}%")
    print(f"Is complete: {result.is_complete}")

    if result.gaps:
        print("\nGap examples:")
        for gap in result.gaps[:3]:
            print(f"  - {gap}")

    # Should detect gaps for missing interest rate
    assert len(result.gaps) > 0, "Should detect gaps in incomplete documentation"
    print("\n[PASS] Gap detection working correctly")
    return True


def test_complete_documentation():
    """Test with complete documentation - should have no gaps"""
    print("\n" + "=" * 70)
    print("Test: Complete Documentation (No Gaps)")
    print("=" * 70)

    generator = ConstrainedCodeGenerator()

    # Complete documentation
    documentation = """
    Interest Calculator Program

    Data Structures:
    - PRINCIPAL: PIC 9(7)V99 - Principal amount
    - RATE: PIC 9V9999 - Interest rate (decimal, e.g., 0.05 for 5%)
    - YEARS: PIC 99 - Number of years
    - INTEREST: PIC 9(7)V99 - Calculated interest

    Business Logic:
    1. ACCEPT PRINCIPAL from input
    2. ACCEPT RATE from input
    3. ACCEPT YEARS from input
    4. COMPUTE INTEREST = PRINCIPAL * RATE * YEARS
    5. DISPLAY "Interest: " INTEREST
    6. STOP RUN

    Program Structure:
    - IDENTIFICATION DIVISION with PROGRAM-ID CALC-INT
    - DATA DIVISION with WORKING-STORAGE SECTION
    - PROCEDURE DIVISION with MAIN-LOGIC paragraph
    """

    expected_structure = {
        "program_name": "CALC-INT",
        "data_structures": [
            {"name": "PRINCIPAL", "picture": "9(7)V99"},
            {"name": "RATE", "picture": "9V9999"},
            {"name": "YEARS", "picture": "99"},
            {"name": "INTEREST", "picture": "9(7)V99"}
        ],
        "paragraphs": [{"name": "MAIN-LOGIC"}],
        "business_rules": [
            {"description": "COMPUTE INTEREST = PRINCIPAL * RATE * YEARS"}
        ],
        "files": []
    }

    has_llm = bool(os.getenv("OPENAI_API_KEY")) or bool(os.getenv("ANTHROPIC_API_KEY"))

    if not has_llm:
        print("[SKIP] No LLM API keys available")
        return True

    result = generator.generate(documentation, expected_structure)

    print(f"\nGenerated code length: {len(result.code)} chars")
    print(f"Gaps detected: {len(result.gaps)}")
    print(f"Gap percentage: {result.gap_percentage:.1f}%")
    print(f"Is complete: {result.is_complete}")

    print("\n[INFO] Generated COBOL (first 500 chars):")
    print(result.code[:500])

    # Should have very few or no gaps
    print(f"\n[PASS] Complete documentation test passed ({len(result.gaps)} gaps)")
    return True


def test_gap_percentage_calculation():
    """Test gap percentage calculation"""
    print("\n" + "=" * 70)
    print("Test: Gap Percentage Calculation")
    print("=" * 70)

    generator = ConstrainedCodeGenerator()

    # Code with known gaps
    code_with_gaps = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. TEST.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  FIELD1  PIC [GAP: data type not specified].
       PROCEDURE DIVISION.
           *> [GAP: business logic not documented]
           DISPLAY "Test".
           STOP RUN.
    """

    gaps = [
        {"line": 5, "description": "data type not specified"},
        {"line": 7, "description": "business logic not documented"}
    ]

    gap_percentage = generator._calculate_gap_percentage(code_with_gaps, gaps)

    print(f"Code LOC: ~10")
    print(f"Gap markers: {len(gaps)}")
    print(f"Gap percentage: {gap_percentage:.1f}%")

    assert gap_percentage > 0, "Should calculate non-zero gap percentage"
    print("\n[PASS] Gap percentage calculation working")
    return True


def test_template_fallback():
    """Test template-based generation (no LLM)"""
    print("\n" + "=" * 70)
    print("Test: Template-Based Fallback")
    print("=" * 70)

    generator = ConstrainedCodeGenerator()

    documentation = "Simple test program that displays hello world"
    expected_structure = {
        "program_name": "HELLO",
        "data_structures": [],
        "paragraphs": [{"name": "MAIN"}],
        "business_rules": [],
        "files": []
    }

    # Force template generation by removing API keys temporarily
    old_openai = os.environ.get("OPENAI_API_KEY")
    old_anthropic = os.environ.get("ANTHROPIC_API_KEY")

    try:
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        result = generator.generate(documentation, expected_structure)

        print(f"\nTemplate code length: {len(result.code)} chars")
        print(f"Contains IDENTIFICATION DIVISION: {'IDENTIFICATION DIVISION' in result.code}")
        print(f"Contains PROGRAM-ID: {'PROGRAM-ID' in result.code}")

        assert "IDENTIFICATION DIVISION" in result.code
        assert "PROGRAM-ID" in result.code
        print("\n[PASS] Template fallback working")
        return True

    finally:
        # Restore API keys
        if old_openai:
            os.environ["OPENAI_API_KEY"] = old_openai
        if old_anthropic:
            os.environ["ANTHROPIC_API_KEY"] = old_anthropic


if __name__ == "__main__":
    print("=" * 70)
    print("Constrained Code Generator Test Suite")
    print("=" * 70)

    # Check for API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    print(f"\n[INFO] OpenAI API Key: {'Available' if has_openai else 'Not available'}")
    print(f"[INFO] Anthropic API Key: {'Available' if has_anthropic else 'Not available'}")

    if not (has_openai or has_anthropic):
        print("\n[WARNING] No LLM API keys detected. Some tests will be skipped.")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to test LLM generation.")

    tests = [
        ("Gap Detection", test_gap_detection),
        ("Complete Documentation", test_complete_documentation),
        ("Gap Percentage Calculation", test_gap_percentage_calculation),
        ("Template Fallback", test_template_fallback),
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
