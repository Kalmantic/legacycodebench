"""Test Case Generator for Behavioral Fidelity Evaluation

Implements test case generation from ground truth for execution-based validation.

Features:
- Boundary value analysis from PICTURE clauses
- Branch coverage tests from business rules and control flow
- Error handler tests
- Generates 10-20 representative test cases per task

Weight in v2.0 evaluation: 35% (Behavioral Fidelity)
"""

import re
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Represents a single test case for COBOL program execution"""
    test_id: str
    description: str
    inputs: Dict[str, Any]  # Variable name → value
    expected_outputs: Dict[str, Any]  # Variable name → expected value
    test_type: str  # "boundary", "branch", "error", "typical"
    priority: int  # 1=critical, 2=important, 3=nice-to-have
    rationale: str  # Why this test case was generated

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "test_id": self.test_id,
            "description": self.description,
            "inputs": self.inputs,
            "expected_outputs": self.expected_outputs,
            "test_type": self.test_type,
            "priority": self.priority,
            "rationale": self.rationale
        }


class TestGenerator:
    """
    Generate test cases from ground truth.

    Implements boundary value analysis, branch coverage, and error testing
    to validate behavioral fidelity of generated documentation.

    Algorithm:
    1. Extract value ranges from PICTURE clauses
    2. Generate boundary tests (min, max, zero, typical)
    3. Generate branch tests from IF/EVALUATE statements
    4. Generate error tests from error handlers
    5. Prioritize and select 10-20 most important tests
    """

    def __init__(self, max_tests_per_task: int = 15):
        """
        Initialize test generator.

        Args:
            max_tests_per_task: Maximum number of test cases to generate
        """
        self.max_tests_per_task = max_tests_per_task
        self.test_counter = 0

    def generate(self, ground_truth: Dict, task_id: str) -> List[TestCase]:
        """
        Generate test cases from ground truth.

        Args:
            ground_truth: Complete ground truth from static analysis
            task_id: Task identifier for test IDs

        Returns:
            List of TestCase objects (prioritized and limited)
        """
        logger.info(f"Generating test cases for {task_id}")
        self.test_counter = 0

        all_tests = []

        # 1. Generate boundary value tests from PICTURE clauses
        logger.info("  Generating boundary value tests...")
        boundary_tests = self._generate_boundary_tests(
            ground_truth.get("data_structures", {}),
            task_id
        )
        all_tests.extend(boundary_tests)

        # 2. Generate branch coverage tests
        logger.info("  Generating branch coverage tests...")
        branch_tests = self._generate_branch_tests(
            ground_truth.get("business_rules", {}),
            ground_truth.get("control_flow", {}),
            task_id
        )
        all_tests.extend(branch_tests)

        # 3. Generate error handler tests
        logger.info("  Generating error handler tests...")
        error_tests = self._generate_error_tests(
            ground_truth.get("error_handlers", []),
            task_id
        )
        all_tests.extend(error_tests)

        # 4. Generate typical/happy path tests
        logger.info("  Generating typical value tests...")
        typical_tests = self._generate_typical_tests(
            ground_truth.get("data_structures", {}),
            task_id
        )
        all_tests.extend(typical_tests)

        # 5. Prioritize and select top N tests
        selected_tests = self._prioritize_and_select(all_tests)

        logger.info(f"Generated {len(all_tests)} tests, selected {len(selected_tests)} (limit: {self.max_tests_per_task})")

        return selected_tests

    def _generate_boundary_tests(self, data_structures: Dict, task_id: str) -> List[TestCase]:
        """Generate boundary value tests from PICTURE clauses"""
        tests = []

        fields = data_structures.get("fields", [])

        for field in fields:
            # Only test elementary items with PICTURE clauses
            if not field.get("picture") or field.get("level", 0) == 1:
                continue

            picture = field["picture"]
            field_name = field["name"]

            # Parse PICTURE to get data type and range
            data_type, min_val, max_val = self._parse_picture_range(picture)

            if data_type is None:
                continue  # Skip unparseable PICTURE clauses

            # Generate boundary tests for this field
            if data_type == "numeric":
                # Test minimum value
                tests.append(TestCase(
                    test_id=f"{task_id}_BV_{self._next_id()}",
                    description=f"Boundary test: {field_name} at minimum value",
                    inputs={field_name: min_val},
                    expected_outputs={},  # Will be filled by COBOL execution
                    test_type="boundary",
                    priority=2,
                    rationale=f"Test minimum valid value for {field_name} (PIC {picture})"
                ))

                # Test maximum value
                tests.append(TestCase(
                    test_id=f"{task_id}_BV_{self._next_id()}",
                    description=f"Boundary test: {field_name} at maximum value",
                    inputs={field_name: max_val},
                    expected_outputs={},
                    test_type="boundary",
                    priority=2,
                    rationale=f"Test maximum valid value for {field_name} (PIC {picture})"
                ))

                # Test zero (if in range)
                if min_val <= 0 <= max_val:
                    tests.append(TestCase(
                        test_id=f"{task_id}_BV_{self._next_id()}",
                        description=f"Boundary test: {field_name} at zero",
                        inputs={field_name: 0},
                        expected_outputs={},
                        test_type="boundary",
                        priority=2,
                        rationale=f"Test zero value for {field_name} (often a special case)"
                    ))

            elif data_type == "alphanumeric":
                # Test empty string
                tests.append(TestCase(
                    test_id=f"{task_id}_BV_{self._next_id()}",
                    description=f"Boundary test: {field_name} empty",
                    inputs={field_name: ""},
                    expected_outputs={},
                    test_type="boundary",
                    priority=2,
                    rationale=f"Test empty/spaces for {field_name} (PIC {picture})"
                ))

                # Test full length
                tests.append(TestCase(
                    test_id=f"{task_id}_BV_{self._next_id()}",
                    description=f"Boundary test: {field_name} at max length",
                    inputs={field_name: "A" * max_val},
                    expected_outputs={},
                    test_type="boundary",
                    priority=2,
                    rationale=f"Test maximum length for {field_name} (PIC {picture})"
                ))

        return tests

    def _generate_branch_tests(self, business_rules: Dict,
                               control_flow: Dict, task_id: str) -> List[TestCase]:
        """Generate tests to cover different branches/conditions"""
        tests = []

        rules = business_rules.get("rules", [])

        for i, rule in enumerate(rules):
            # Extract condition from rule
            condition = rule.get("condition", "")
            rule_type = rule.get("type", "")

            # Generate tests for IF statements
            if rule_type == "IF":
                # Test true branch
                tests.append(TestCase(
                    test_id=f"{task_id}_BR_{self._next_id()}",
                    description=f"Branch test: {rule.get('description', 'condition')} is TRUE",
                    inputs=self._generate_inputs_for_condition(condition, True),
                    expected_outputs={},
                    test_type="branch",
                    priority=1,  # High priority for business logic
                    rationale=f"Test true branch of: {condition}"
                ))

                # Test false branch
                tests.append(TestCase(
                    test_id=f"{task_id}_BR_{self._next_id()}",
                    description=f"Branch test: {rule.get('description', 'condition')} is FALSE",
                    inputs=self._generate_inputs_for_condition(condition, False),
                    expected_outputs={},
                    test_type="branch",
                    priority=1,
                    rationale=f"Test false branch of: {condition}"
                ))

            # Generate tests for EVALUATE statements
            elif rule_type == "EVALUATE":
                # Extract possible values
                values = rule.get("values", [])
                for value in values[:3]:  # Limit to 3 cases per EVALUATE
                    tests.append(TestCase(
                        test_id=f"{task_id}_BR_{self._next_id()}",
                        description=f"Branch test: EVALUATE case '{value}'",
                        inputs=self._generate_inputs_for_evaluate(rule.get("variable", ""), value),
                        expected_outputs={},
                        test_type="branch",
                        priority=1,
                        rationale=f"Test EVALUATE case: {value}"
                    ))

        return tests

    def _generate_error_tests(self, error_handlers: List[Dict], task_id: str) -> List[TestCase]:
        """Generate tests to trigger error handlers"""
        tests = []

        for i, handler in enumerate(error_handlers):
            error_condition = handler.get("condition", "")
            error_type = handler.get("type", "")

            # Generate test to trigger this error
            tests.append(TestCase(
                test_id=f"{task_id}_ERR_{self._next_id()}",
                description=f"Error test: Trigger {error_type}",
                inputs=self._generate_error_inputs(error_condition, error_type),
                expected_outputs={},
                test_type="error",
                priority=1,  # High priority - error handling is critical
                rationale=f"Test error handler: {error_condition}"
            ))

        return tests

    def _generate_typical_tests(self, data_structures: Dict, task_id: str) -> List[TestCase]:
        """Generate typical/happy path test cases"""
        tests = []

        # Generate 2-3 typical tests with normal values
        fields = data_structures.get("fields", [])

        # Filter to elementary items with PICTURE
        input_fields = [f for f in fields if f.get("picture") and f.get("level", 0) > 1]

        # Generate typical test with mid-range values
        if input_fields:
            typical_inputs = {}
            for field in input_fields[:5]:  # Limit to 5 main fields
                picture = field["picture"]
                data_type, min_val, max_val = self._parse_picture_range(picture)

                if data_type == "numeric":
                    # Use mid-range value
                    typical_inputs[field["name"]] = (min_val + max_val) // 2
                elif data_type == "alphanumeric":
                    # Use typical string
                    typical_inputs[field["name"]] = "TEST"

            if typical_inputs:
                tests.append(TestCase(
                    test_id=f"{task_id}_TYP_{self._next_id()}",
                    description="Typical test: Normal input values",
                    inputs=typical_inputs,
                    expected_outputs={},
                    test_type="typical",
                    priority=2,
                    rationale="Test normal/expected values (happy path)"
                ))

        return tests

    def _parse_picture_range(self, picture: str) -> tuple:
        """
        Parse PICTURE clause to determine data type and valid range.

        Args:
            picture: PICTURE clause (e.g., "9(5)", "X(20)", "S9(3)V99")

        Returns:
            (data_type, min_value, max_value) or (None, None, None)
        """
        # Remove PIC/PICTURE prefix if present
        picture = re.sub(r'^PIC(?:TURE)?\s+', '', picture, flags=re.IGNORECASE).strip()
        
        # Remove trailing period (common in COBOL)
        picture = picture.rstrip('.')

        # Numeric patterns: 9, S9, V, Z (edited numeric), etc.
        # Pattern: [S]9[(n)] or [S]9...9 with optional V and decimal places
        # Also handles Z (zero-suppressed), + (sign), - (sign), . (decimal)
        
        # First try standard numeric: S9(5)V99 or S999V99
        numeric_pattern = r'^(S)?(?:9+|\d*9\(\d+\))(?:V(?:9+|\d*9\(\d+\)))?$'
        # Clean picture for analysis (remove display chars like Z, +, -, .)
        clean_pic = re.sub(r'[Z\+\-\.\$\,]', '9', picture.upper())
        
        match = re.match(numeric_pattern, clean_pic)
        if match:
            signed = match.group(1) is not None or '+' in picture or '-' in picture
            
            # Count integer and decimal digits
            if 'V' in clean_pic.upper():
                parts = clean_pic.upper().split('V')
                integer_part = parts[0].replace('S', '')
                decimal_part = parts[1] if len(parts) > 1 else ''
            else:
                integer_part = clean_pic.replace('S', '')
                decimal_part = ''
            
            # Parse 9(n) or count 9s
            int_match = re.search(r'9\((\d+)\)', integer_part)
            if int_match:
                integer_digits = int(int_match.group(1))
            else:
                integer_digits = integer_part.count('9')
            
            dec_match = re.search(r'9\((\d+)\)', decimal_part)
            if dec_match:
                decimal_digits = int(dec_match.group(1))
            else:
                decimal_digits = decimal_part.count('9')

            # Calculate range
            if integer_digits > 0:
                max_val = (10 ** integer_digits) - 1
                min_val = -(10 ** integer_digits) + 1 if signed else 0
                return ("numeric", min_val, max_val)

        # Alphanumeric: X or X(n)
        alpha_pattern = r'^X+(?:\((\d+)\))?$'
        match = re.match(alpha_pattern, picture.upper())
        if match:
            if match.group(1):
                length = int(match.group(1))
            else:
                length = picture.upper().count('X')
            return ("alphanumeric", 0, length)  # min=0, max=length

        # Alphabetic: A or A(n)
        alpha_pattern = r'^A+(?:\((\d+)\))?$'
        match = re.match(alpha_pattern, picture.upper())
        if match:
            if match.group(1):
                length = int(match.group(1))
            else:
                length = picture.upper().count('A')
            return ("alphabetic", 0, length)

        # Edited numeric patterns (Z, $, etc.) - treat as numeric display
        if re.match(r'^[Z9\$\+\-\.\,\(\)]+$', picture.upper()):
            # Count significant digits
            digit_count = sum(1 for c in picture.upper() if c in 'Z9')
            if digit_count > 0:
                max_val = (10 ** digit_count) - 1
                return ("numeric", 0, max_val)

        # Unable to parse - downgrade to debug level to reduce noise
        logger.debug(f"Unable to parse PICTURE clause: {picture}")
        return (None, None, None)

    def _generate_inputs_for_condition(self, condition: str, should_be_true: bool) -> Dict[str, Any]:
        """
        Generate inputs that make a condition true or false.

        This is a simplified heuristic - real implementation would need
        symbolic execution or constraint solving.
        """
        # Extract variable and comparison from condition
        # Example: "AMOUNT > 1000" or "STATUS = 'A'"

        comparison_match = re.search(r'(\w+)\s*(>|<|=|>=|<=)\s*([\'"]?\w+[\'"]?)', condition)

        if comparison_match:
            var = comparison_match.group(1)
            op = comparison_match.group(2)
            val = comparison_match.group(3).strip('\'"')

            # Convert to appropriate type
            try:
                val = int(val)
            except ValueError:
                pass  # Keep as string

            # Generate value that makes condition true or false
            if should_be_true:
                if op == ">":
                    return {var: val + 1}
                elif op == "<":
                    return {var: val - 1}
                elif op == "=":
                    return {var: val}
                elif op == ">=":
                    return {var: val}
                elif op == "<=":
                    return {var: val}
            else:
                if op == ">":
                    return {var: val - 1}
                elif op == "<":
                    return {var: val + 1}
                elif op == "=":
                    return {var: val + 1 if isinstance(val, int) else "DIFFERENT"}
                elif op == ">=":
                    return {var: val - 1}
                elif op == "<=":
                    return {var: val + 1}

        # Fallback: return empty dict
        return {}

    def _generate_inputs_for_evaluate(self, variable: str, value: Any) -> Dict[str, Any]:
        """Generate inputs for EVALUATE case"""
        if variable:
            return {variable: value}
        return {}

    def _generate_error_inputs(self, condition: str, error_type: str) -> Dict[str, Any]:
        """Generate inputs that trigger an error condition"""
        # Common error scenarios
        if "FILE STATUS" in condition or error_type == "file_status":
            return {"FILE-STATUS": "35"}  # File not found
        elif "ZERO" in condition or "DIV" in error_type:
            return {"DIVISOR": 0}  # Division by zero
        elif "OVERFLOW" in condition:
            return {"AMOUNT": 999999999}  # Overflow
        else:
            # Generic error condition
            return {"ERROR-FLAG": "Y"}

    def _prioritize_and_select(self, tests: List[TestCase]) -> List[TestCase]:
        """
        Prioritize tests and select top N.

        Priority order:
        1. Priority 1 tests (critical business logic, error handlers)
        2. Priority 2 tests (boundary values, important branches)
        3. Priority 3 tests (nice-to-have)
        """
        # Sort by priority (1=highest)
        sorted_tests = sorted(tests, key=lambda t: (t.priority, random.random()))

        # Select up to max_tests_per_task
        selected = sorted_tests[:self.max_tests_per_task]

        # Log distribution
        priority_counts = {}
        for test in selected:
            priority_counts[test.priority] = priority_counts.get(test.priority, 0) + 1

        logger.info(f"  Selected tests by priority: P1={priority_counts.get(1, 0)}, "
                   f"P2={priority_counts.get(2, 0)}, P3={priority_counts.get(3, 0)}")

        return selected

    def _next_id(self) -> int:
        """Get next test ID counter"""
        self.test_counter += 1
        return self.test_counter

    def export_tests(self, tests: List[TestCase], output_file: str):
        """Export test cases to JSON file"""
        import json

        with open(output_file, 'w') as f:
            json.dump([t.to_dict() for t in tests], f, indent=2)

        logger.info(f"Exported {len(tests)} test cases to {output_file}")
