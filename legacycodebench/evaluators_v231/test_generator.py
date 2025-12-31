"""
Test Generator V2.3.1

Generates test cases for claim verification with BOUNDARY VALUE TESTING.

Innovation: For each CALCULATION claim, generate THREE tests:
- Min: Smallest valid value from PIC clause
- Mid: Representative mid-range value
- Max: Largest valid value from PIC clause

This catches edge-case bugs that mid-range tests miss.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import re
import logging

from .config_v231 import V231_CONFIG
from .claim_extractor import Claim, ClaimType


logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A test case for claim verification."""
    test_id: str
    claim_id: str
    test_type: str  # "min", "mid", "max", "condition", "error"
    inputs: Dict[str, any] = field(default_factory=dict)
    expected_outputs: Dict[str, any] = field(default_factory=dict)  # Variable → expected value
    expected_behavior: str = ""
    description: str = ""
    priority: int = 1


class TestGenerator:
    """
    Generate test cases from claims with boundary value analysis.
    
    Pipeline:
    1. Parse claims to identify testable properties
    2. Extract value ranges from ground truth PIC clauses
    3. Generate Min/Mid/Max tests for calculations
    4. Generate condition tests for conditionals
    5. Prioritize and cap at max_tests
    """
    
    def __init__(self):
        self.config = V231_CONFIG["boundary_testing"]
        self.max_tests = self.config["max_tests_total"]
    
    def generate(
        self,
        claims: List[Claim],
        ground_truth: Dict
    ) -> List[TestCase]:
        """
        Generate test cases from claims.
        
        Args:
            claims: Extracted claims
            ground_truth: Ground truth with data structure info
            
        Returns:
            List of TestCases (max 15)
        """
        tests = []
        
        logger.debug(f"Stage 1: Generating tests for {len(claims)} claims [...]")
        
        for claim in claims:
            claim_tests = self._generate_for_claim(claim, ground_truth)
            tests.extend(claim_tests)
        
        logger.debug(f"Stage 1: Generated {len(tests)} raw tests [OK]")
        
        # Prioritize and cap
        logger.debug("Stage 2: Prioritizing tests [...]")
        tests = self._prioritize_tests(tests)
        logger.debug(f"Stage 2: Prioritized to {len(tests)} tests [OK]")
        
        # Assign IDs
        for i, test in enumerate(tests):
            test.test_id = f"TEST-{i+1:03d}"
        
        return tests
    
    def _generate_for_claim(
        self,
        claim: Claim,
        ground_truth: Dict
    ) -> List[TestCase]:
        """
        Generate tests for a single claim.
        """
        if claim.claim_type == ClaimType.CALCULATION:
            return self._generate_calculation_tests(claim, ground_truth)
        
        elif claim.claim_type == ClaimType.CONDITIONAL:
            return self._generate_conditional_tests(claim, ground_truth)
        
        elif claim.claim_type == ClaimType.RANGE:
            return self._generate_range_tests(claim, ground_truth)
        
        elif claim.claim_type == ClaimType.ERROR:
            return self._generate_error_tests(claim, ground_truth)
        
        else:
            # Assignment or unknown: single mid-range test
            return [TestCase(
                test_id="",
                claim_id=claim.claim_id,
                test_type="mid",
                description=f"Verify: {claim.text[:100]}",
                priority=3,
            )]
    
    def _generate_calculation_tests(
        self,
        claim: Claim,
        ground_truth: Dict
    ) -> List[TestCase]:
        """
        Generate Min/Mid/Max tests for calculation claims.
        
        This is the key innovation of V2.3.1 boundary testing.
        """
        if not self.config["enabled"]:
            # Fallback to single test
            return [TestCase(
                test_id="",
                claim_id=claim.claim_id,
                test_type="mid",
                description=f"Calculate: {claim.text[:100]}",
                priority=1,
            )]
        
        tests = []
        
        # Get field info for output variable
        field_info = self._find_field_info(claim.output_var, ground_truth)
        
        if field_info:
            min_val, mid_val, max_val = self._get_boundary_values(field_info)
        else:
            # Default boundaries
            min_val, mid_val, max_val = 0, 50, 100
        
        # Generate input values for each boundary
        for test_type, value in [("min", min_val), ("mid", mid_val), ("max", max_val)]:
            inputs = self._generate_inputs(claim.input_vars, value, ground_truth)

            # Calculate expected output for this test
            expected_outputs = self._calculate_expected_outputs(claim, inputs)

            tests.append(TestCase(
                test_id="",
                claim_id=claim.claim_id,
                test_type=test_type,
                inputs=inputs,
                expected_outputs=expected_outputs,
                expected_behavior=f"output in {claim.output_var or 'result'}",
                description=f"Boundary {test_type}: {claim.text[:80]}",
                priority=1 if test_type in ("min", "max") else 2,
            ))

        return tests
    
    def _generate_conditional_tests(
        self,
        claim: Claim,
        ground_truth: Dict
    ) -> List[TestCase]:
        """
        Generate true/false tests for conditional claims.
        """
        tests = []
        
        condition_var = claim.components.get("condition_var", "")
        field_info = self._find_field_info(condition_var, ground_truth)
        
        if field_info:
            min_val, mid_val, max_val = self._get_boundary_values(field_info)
        else:
            min_val, mid_val, max_val = 0, 50, 100
        
        # True case (exceed threshold)
        tests.append(TestCase(
            test_id="",
            claim_id=claim.claim_id,
            test_type="condition_true",
            inputs={condition_var: max_val} if condition_var else {},
            expected_behavior="condition triggers",
            description=f"Condition true: {claim.text[:80]}",
            priority=1,
        ))
        
        # False case (below threshold)
        tests.append(TestCase(
            test_id="",
            claim_id=claim.claim_id,
            test_type="condition_false",
            inputs={condition_var: min_val} if condition_var else {},
            expected_behavior="condition does not trigger",
            description=f"Condition false: {claim.text[:80]}",
            priority=1,
        ))
        
        return tests
    
    def _generate_range_tests(
        self,
        claim: Claim,
        ground_truth: Dict
    ) -> List[TestCase]:
        """
        Generate boundary tests for range claims.
        """
        tests = []
        
        min_val = claim.components.get("min_value", 0)
        max_val = claim.components.get("max_value", 100)
        
        try:
            min_val = float(min_val)
            max_val = float(max_val)
        except (TypeError, ValueError):
            min_val, max_val = 0, 100
        
        var = claim.output_var or "value"
        
        # At minimum
        tests.append(TestCase(
            test_id="",
            claim_id=claim.claim_id,
            test_type="range_min",
            inputs={var: min_val},
            expected_behavior="valid",
            description=f"Range min: {claim.text[:80]}",
            priority=1,
        ))
        
        # At maximum
        tests.append(TestCase(
            test_id="",
            claim_id=claim.claim_id,
            test_type="range_max",
            inputs={var: max_val},
            expected_behavior="valid",
            description=f"Range max: {claim.text[:80]}",
            priority=1,
        ))
        
        # Below minimum (invalid)
        tests.append(TestCase(
            test_id="",
            claim_id=claim.claim_id,
            test_type="range_below",
            inputs={var: min_val - 1},
            expected_behavior="invalid",
            description=f"Below range: {claim.text[:80]}",
            priority=2,
        ))
        
        return tests
    
    def _generate_error_tests(
        self,
        claim: Claim,
        ground_truth: Dict
    ) -> List[TestCase]:
        """
        Generate tests for error handling claims.
        """
        return [TestCase(
            test_id="",
            claim_id=claim.claim_id,
            test_type="error",
            expected_behavior="error handled",
            description=f"Error handling: {claim.text[:80]}",
            priority=2,
        )]
    
    def _find_field_info(
        self,
        var_name: Optional[str],
        ground_truth: Dict
    ) -> Optional[Dict]:
        """
        Find field information in ground truth data structures.
        """
        if not var_name:
            return None
        
        var_upper = var_name.upper()
        
        # Check data structures
        ds_data = ground_truth.get("data_structures", {})
        if isinstance(ds_data, dict):
            structures = ds_data.get("structures", [])
        elif isinstance(ds_data, list):
            structures = ds_data
        else:
            structures = []
        
        for ds in structures:
            if isinstance(ds, dict):
                # Check if this is the structure
                if ds.get("name", "").upper() == var_upper:
                    return ds
                
                # Check fields
                for field in ds.get("fields", []):
                    if isinstance(field, dict):
                        if field.get("name", "").upper() == var_upper:
                            return field
        
        return None
    
    def _get_boundary_values(
        self,
        field_info: Dict
    ) -> Tuple[float, float, float]:
        """
        Extract min/mid/max values from field PIC clause.
        """
        pic = field_info.get("pic", field_info.get("picture", ""))
        
        if not pic:
            # Default
            return 0, 50, 100
        
        # Parse PIC clause for numeric fields
        # Examples: PIC 9(5), PIC S9(5)V99, PIC 999V99
        
        # Count 9s (integer digits)
        int_digits = pic.count('9')
        int_digits += sum(int(m.group(1)) - 1 for m in re.finditer(r'9\((\d+)\)', pic))
        
        # Check for decimal (V)
        has_decimal = 'V' in pic.upper()
        
        if int_digits == 0:
            return 0, 50, 100
        
        # Calculate max value
        max_val = (10 ** int_digits) - 1
        if has_decimal:
            # Assume 2 decimal places
            dec_match = re.search(r'V9+', pic, re.IGNORECASE)
            if dec_match:
                dec_digits = dec_match.group().count('9')
                max_val = max_val / (10 ** dec_digits)
        
        min_val = 0.01 if has_decimal else 1
        mid_val = max_val / 2
        
        return min_val, mid_val, max_val
    
    def _generate_inputs(
        self,
        input_vars: List[str],
        target_value: float,
        ground_truth: Dict
    ) -> Dict[str, float]:
        """
        Generate input values for test case.
        """
        inputs = {}
        
        for var in input_vars:
            if not var:
                continue
            
            field_info = self._find_field_info(var, ground_truth)
            
            if field_info:
                _, mid, max_val = self._get_boundary_values(field_info)
                # Scale input to achieve target output
                inputs[var] = min(target_value, max_val)
            else:
                inputs[var] = target_value
        
        return inputs
    
    def _calculate_expected_outputs(
        self,
        claim: Claim,
        inputs: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Calculate expected outputs for a claim given input values.

        For calculation claims, attempts to evaluate the formula.
        For other claims, returns empty dict (no specific output expected).

        Args:
            claim: The claim being tested
            inputs: Input variable values

        Returns:
            Dict of output_var → expected_value
        """
        if claim.claim_type != ClaimType.CALCULATION:
            return {}

        if not claim.output_var or not inputs:
            return {}

        # Try to extract and evaluate the calculation from claim text
        expected_value = self._evaluate_calculation(claim, inputs)

        if expected_value is not None:
            return {claim.output_var: expected_value}

        return {}

    def _evaluate_calculation(
        self,
        claim: Claim,
        inputs: Dict[str, float]
    ) -> Optional[float]:
        """
        Evaluate a calculation claim to get expected output.

        Supports basic operations:
        - Addition: "sum of X and Y", "X + Y", "add X to Y"
        - Multiplication: "X multiplied by Y", "X * Y", "product of X and Y"
        - Subtraction: "X minus Y", "X - Y", "subtract Y from X"
        - Division: "X divided by Y", "X / Y"

        Args:
            claim: Calculation claim
            inputs: Input values

        Returns:
            Expected output value, or None if can't evaluate
        """
        text = claim.text.lower()

        # Get input values
        input_values = []
        for var in claim.input_vars:
            if var in inputs:
                input_values.append(inputs[var])

        if len(input_values) < 1:
            return None

        # Detect operation type from claim text
        # Multiplication
        if any(keyword in text for keyword in ['multiply', 'multiplied', 'product', 'times', ' * ']):
            if len(input_values) >= 2:
                result = input_values[0]
                for val in input_values[1:]:
                    result *= val
                return result

        # Division
        elif any(keyword in text for keyword in ['divide', 'divided', 'quotient', ' / ']):
            if len(input_values) >= 2 and input_values[1] != 0:
                return input_values[0] / input_values[1]

        # Subtraction
        elif any(keyword in text for keyword in ['subtract', 'minus', 'difference', ' - ']):
            if len(input_values) >= 2:
                return input_values[0] - input_values[1]

        # Addition (default for calculations)
        elif any(keyword in text for keyword in ['add', 'sum', 'total', 'plus', ' + ', 'combined']):
            return sum(input_values)

        # Percentage
        elif 'percent' in text or '%' in text:
            if len(input_values) >= 2:
                # X percent of Y = (X/100) * Y
                return (input_values[0] / 100) * input_values[1]
            elif len(input_values) == 1:
                # X percent = X/100
                return input_values[0] / 100

        # Default: if it mentions "calculate" or "compute", try sum
        elif any(keyword in text for keyword in ['calculate', 'compute', 'determine']):
            if len(input_values) > 0:
                # Try to detect if it's a simple assignment vs calculation
                if len(input_values) == 1:
                    return input_values[0]  # Simple assignment
                else:
                    return sum(input_values)  # Sum multiple values

        # Can't determine operation
        return None

    def _prioritize_tests(self, tests: List[TestCase]) -> List[TestCase]:
        """
        Prioritize and cap tests to max limit.
        """
        # Sort by priority (lower = higher priority)
        sorted_tests = sorted(tests, key=lambda t: t.priority)

        # Cap at max
        if len(sorted_tests) > self.max_tests:
            sorted_tests = sorted_tests[:self.max_tests]
            logger.info(f"Tests capped at {self.max_tests}")

        return sorted_tests
