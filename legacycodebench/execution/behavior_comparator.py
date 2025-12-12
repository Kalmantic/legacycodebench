"""Behavior Comparator for Behavioral Fidelity Evaluation

Compares execution outputs between original COBOL and generated code.

Features:
- Compares stdout, stderr, and file outputs
- Handles numerical precision differences
- Handles formatting differences
- Calculates behavioral fidelity score
- Detects critical execution failures

Weight in v2.0 evaluation: 35% (Behavioral Fidelity)
"""

import re
import difflib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Results from comparing two execution outputs"""
    score: float  # 0.0 to 1.0
    matches: int  # Number of matching test cases
    mismatches: int  # Number of mismatching test cases
    errors: int  # Number of execution errors
    total_tests: int  # Total test cases
    details: List[Dict]  # Detailed comparison for each test
    critical_failures: List[str]  # Critical execution failures

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "score": self.score,
            "matches": self.matches,
            "mismatches": self.mismatches,
            "errors": self.errors,
            "total_tests": self.total_tests,
            "details": self.details,
            "critical_failures": self.critical_failures,
            "pass_rate": self.matches / self.total_tests if self.total_tests > 0 else 0.0
        }


class BehaviorComparator:
    """
    Compare execution outputs to measure behavioral fidelity.

    Algorithm:
    1. For each test case:
       - Compare stdout
       - Compare stderr
       - Compare file outputs
       - Calculate similarity score
    2. Aggregate results across all tests
    3. Calculate overall behavioral fidelity score
    4. Detect critical failures

    Handles:
    - Whitespace differences
    - Numerical precision (0.001 tolerance)
    - Line ending differences
    - Output formatting variations
    """

    def __init__(self,
                 numerical_tolerance: float = 0.001,
                 whitespace_sensitive: bool = False):
        """
        Initialize behavior comparator.

        Args:
            numerical_tolerance: Tolerance for numerical comparisons
            whitespace_sensitive: Whether to consider whitespace differences
        """
        self.numerical_tolerance = numerical_tolerance
        self.whitespace_sensitive = whitespace_sensitive

    def compare(self,
               original_results: List,  # List of ExecutionResult
               generated_results: List,  # List of ExecutionResult
               test_cases: List) -> ComparisonResult:
        """
        Compare execution results between original and generated code.

        Args:
            original_results: Results from original COBOL execution
            generated_results: Results from generated code execution
            test_cases: Test cases that were executed

        Returns:
            ComparisonResult with aggregated comparison
        """
        logger.info(f"Comparing {len(original_results)} execution results")

        if len(original_results) != len(generated_results):
            logger.error(
                f"Result count mismatch: {len(original_results)} vs {len(generated_results)}"
            )
            return ComparisonResult(
                score=0.0,
                matches=0,
                mismatches=0,
                errors=1,
                total_tests=len(test_cases),
                details=[],
                critical_failures=["Result count mismatch"]
            )

        # Compare each test case
        details = []
        matches = 0
        mismatches = 0
        errors = 0
        critical_failures = []

        for i, (orig, gen, test) in enumerate(zip(original_results, generated_results, test_cases)):
            logger.info(f"  Comparing test case {i+1}/{len(test_cases)}: {test.test_id}")

            # Check for execution errors
            if not orig.success:
                logger.error(f"    Original execution failed: {orig.error_message}")
                errors += 1
                details.append({
                    "test_id": test.test_id,
                    "result": "error",
                    "reason": f"Original execution failed: {orig.error_message}",
                    "similarity": 0.0
                })
                critical_failures.append(
                    f"Original program failed on test {test.test_id}: {orig.error_message}"
                )
                continue

            if not gen.success:
                logger.error(f"    Generated execution failed: {gen.error_message}")
                errors += 1
                details.append({
                    "test_id": test.test_id,
                    "result": "error",
                    "reason": f"Generated execution failed: {gen.error_message}",
                    "similarity": 0.0
                })
                continue

            # Compare outputs
            similarity = self._compare_outputs(orig, gen)

            if similarity >= 0.95:  # 95% similarity threshold
                matches += 1
                result = "match"
                logger.info(f"    ✓ Match (similarity: {similarity:.2%})")
            else:
                mismatches += 1
                result = "mismatch"
                logger.warning(f"    ✗ Mismatch (similarity: {similarity:.2%})")

            details.append({
                "test_id": test.test_id,
                "result": result,
                "similarity": similarity,
                "stdout_match": self._normalize_output(orig.stdout) == self._normalize_output(gen.stdout),
                "file_output_match": self._compare_file_outputs(orig.file_outputs, gen.file_outputs) >= 0.95
            })

        # Calculate overall score
        total_tests = len(test_cases)
        score = matches / total_tests if total_tests > 0 else 0.0

        # Detect critical failures
        critical_failures.extend(self._detect_critical_failures(
            matches, mismatches, errors, total_tests, generated_results
        ))

        logger.info(
            f"Comparison complete: {matches} matches, {mismatches} mismatches, "
            f"{errors} errors out of {total_tests} tests (score: {score:.2%})"
        )

        if critical_failures:
            logger.warning(f"Critical failures detected: {critical_failures}")

        return ComparisonResult(
            score=score,
            matches=matches,
            mismatches=mismatches,
            errors=errors,
            total_tests=total_tests,
            details=details,
            critical_failures=critical_failures
        )

    def _compare_outputs(self, orig_result, gen_result) -> float:
        """
        Compare execution outputs and calculate similarity.

        Args:
            orig_result: ExecutionResult from original
            gen_result: ExecutionResult from generated

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Compare stdout
        stdout_similarity = self._compare_text(
            orig_result.stdout,
            gen_result.stdout
        )

        # Compare stderr (less weight)
        stderr_similarity = self._compare_text(
            orig_result.stderr,
            gen_result.stderr
        )

        # Compare file outputs
        file_similarity = self._compare_file_outputs(
            orig_result.file_outputs,
            gen_result.file_outputs
        )

        # Weighted combination
        # stdout: 50%, files: 40%, stderr: 10%
        overall_similarity = (
            stdout_similarity * 0.5 +
            file_similarity * 0.4 +
            stderr_similarity * 0.1
        )

        return overall_similarity

    def _compare_text(self, text1: str, text2: str) -> float:
        """
        Compare two text outputs.

        Handles:
        - Whitespace normalization
        - Numerical precision
        - Line ending differences

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Normalize texts
        norm1 = self._normalize_output(text1)
        norm2 = self._normalize_output(text2)

        # Exact match check
        if norm1 == norm2:
            return 1.0

        # Use sequence matcher for similarity
        matcher = difflib.SequenceMatcher(None, norm1, norm2)
        similarity = matcher.ratio()

        # Check if difference is only numerical precision
        if similarity > 0.90:
            # Try numerical comparison
            if self._are_numerically_equivalent(text1, text2):
                return 1.0

        return similarity

    def _normalize_output(self, text: str) -> str:
        """
        Normalize output text for comparison.

        Args:
            text: Raw output text

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Handle whitespace
        if not self.whitespace_sensitive:
            # Normalize whitespace (but preserve line structure)
            lines = text.split('\n')
            normalized_lines = [' '.join(line.split()) for line in lines]
            text = '\n'.join(normalized_lines)

        # Strip trailing whitespace
        text = text.strip()

        return text

    def _are_numerically_equivalent(self, text1: str, text2: str) -> bool:
        """
        Check if two texts are equivalent when considering numerical precision.

        Args:
            text1: First text
            text2: Second text

        Returns:
            True if numerically equivalent
        """
        # Extract all numbers from both texts
        numbers1 = self._extract_numbers(text1)
        numbers2 = self._extract_numbers(text2)

        if len(numbers1) != len(numbers2):
            return False

        # Compare each number with tolerance
        for n1, n2 in zip(numbers1, numbers2):
            if abs(n1 - n2) > self.numerical_tolerance:
                return False

        return True

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from text"""
        # Pattern for integers and decimals
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, text)

        numbers = []
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                pass

        return numbers

    def _compare_file_outputs(self, files1: Dict[str, str],
                              files2: Dict[str, str]) -> float:
        """
        Compare file outputs.

        Args:
            files1: Original file outputs
            files2: Generated file outputs

        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not files1 and not files2:
            return 1.0  # Both have no files

        if not files1 or not files2:
            return 0.0  # One has files, other doesn't

        # Check if same files exist
        keys1 = set(files1.keys())
        keys2 = set(files2.keys())

        if keys1 != keys2:
            logger.warning(f"File output mismatch: {keys1} vs {keys2}")
            # Partial credit for intersection
            common_files = keys1 & keys2
            if not common_files:
                return 0.0
            coverage = len(common_files) / max(len(keys1), len(keys2))
        else:
            coverage = 1.0

        # Compare file contents
        similarities = []
        for filename in keys1 & keys2:
            content1 = files1[filename]
            content2 = files2[filename]

            file_similarity = self._compare_text(content1, content2)
            similarities.append(file_similarity)

        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            return coverage * avg_similarity
        else:
            return coverage

    def compare_single(self, orig_result, gen_result) -> Tuple[bool, float, str]:
        """
        Compare single execution result (convenience method).

        Args:
            orig_result: Original ExecutionResult
            gen_result: Generated ExecutionResult

        Returns:
            (is_match: bool, similarity: float, reason: str)
        """
        if not orig_result.success:
            return (False, 0.0, f"Original failed: {orig_result.error_message}")

        if not gen_result.success:
            return (False, 0.0, f"Generated failed: {gen_result.error_message}")

        similarity = self._compare_outputs(orig_result, gen_result)

        is_match = similarity >= 0.95

        if is_match:
            reason = "Outputs match"
        else:
            reason = f"Similarity: {similarity:.2%}"

        return (is_match, similarity, reason)

    def _detect_critical_failures(self,
                                  matches: int,
                                  mismatches: int,
                                  errors: int,
                                  total_tests: int,
                                  generated_results: List) -> List[str]:
        """
        Detect critical failures in behavioral testing.

        Critical Failures:
        - CF-03: Wrong data transformation (≥10% output mismatch)
        - CF-06: False positive (tests pass but gaps present in generated code)

        Args:
            matches: Number of matching tests
            mismatches: Number of mismatching tests
            errors: Number of errors
            total_tests: Total number of tests
            generated_results: Results from generated code execution

        Returns:
            List of critical failure descriptions
        """
        critical_failures = []

        if total_tests == 0:
            return critical_failures

        # CF-03: Wrong data transformation (≥10% output mismatch)
        mismatch_rate = (mismatches + errors) / total_tests
        if mismatch_rate >= 0.10:
            cf_msg = (
                f"CF-03: Wrong data transformation "
                f"({mismatch_rate*100:.1f}% outputs differ, threshold: 10%)"
            )
            critical_failures.append(cf_msg)
            logger.error(cf_msg)

        # CF-06: False positive (≥95% tests pass but [GAP] markers present)
        pass_rate = matches / total_tests
        gaps_present = self._check_for_gaps_in_generated_code(generated_results)

        if pass_rate >= 0.95 and gaps_present:
            cf_msg = (
                "CF-06: False positive "
                f"({pass_rate*100:.1f}% tests pass but [GAP] markers present in generated code)"
            )
            critical_failures.append(cf_msg)
            logger.error(cf_msg)

        return critical_failures

    def _check_for_gaps_in_generated_code(self, generated_results: List) -> bool:
        """
        Check if generated code contains gap markers.

        Gap markers like [GAP: ...] indicate incomplete documentation
        that the code generator couldn't fill in.

        Args:
            generated_results: List of ExecutionResult from generated code

        Returns:
            True if gaps are present
        """
        # Check if any result has gap metadata
        for result in generated_results:
            # Check stderr for gap markers (compilation might show them)
            if result.stderr and '[GAP' in result.stderr:
                return True

            # Check stdout for gap markers
            if result.stdout and '[GAP' in result.stdout:
                return True

        return False
