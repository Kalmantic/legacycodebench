"""Scoring system for LegacyCodeBench"""

from typing import Dict, List, Optional
from pathlib import Path
import json
import logging

from legacycodebench.config import OVERALL_WEIGHTS, PERFORMANCE_TIERS, EVALUATION_WEIGHTS, CRITICAL_FAILURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScoringSystem:
    """Calculate overall scores from evaluation results (v1.0 and v2.0)"""

    def __init__(self):
        # v1.0 weights (legacy)
        self.weights = OVERALL_WEIGHTS
        # v2.0 weights
        self.v2_weights = EVALUATION_WEIGHTS
        self.tiers = PERFORMANCE_TIERS
        self.critical_failures = CRITICAL_FAILURES

    # ===================================================================
    # V1.0 Methods (kept for backward compatibility)
    # ===================================================================

    def calculate_overall_score(self, doc_score: float, und_score: float) -> float:
        """Calculate overall LegacyCodeBench score (v1.0 method)"""
        return (
            self.weights["documentation"] * doc_score +
            self.weights["understanding"] * und_score
        )

    def get_performance_tier(self, score: float) -> str:
        """Get performance tier for a score"""
        if score >= self.tiers["excellent"]:
            return "excellent"
        elif score >= self.tiers["good"]:
            return "good"
        elif score >= self.tiers["fair"]:
            return "fair"
        else:
            return "poor"

    def aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results across multiple tasks (v1.0 method)"""
        doc_scores = []
        und_scores = []
        overall_scores = []

        for result in results:
            if result.get("category") == "documentation":
                doc_scores.append(result.get("score", 0.0))
            elif result.get("category") == "understanding":
                und_scores.append(result.get("score", 0.0))

            overall_scores.append(result.get("overall_score", 0.0))

        doc_avg = sum(doc_scores) / len(doc_scores) if doc_scores else 0.0
        und_avg = sum(und_scores) / len(und_scores) if und_scores else 0.0
        overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

        return {
            "documentation_avg": round(doc_avg, 4),
            "understanding_avg": round(und_avg, 4),
            "overall_avg": round(overall_avg, 4),
            "documentation_tasks": len(doc_scores),
            "understanding_tasks": len(und_scores),
            "total_tasks": len(results),
        }

    # ===================================================================
    # V2.0 Methods (new scoring formula)
    # ===================================================================

    def calculate_lcb_v2_score(self, sc: float, bf: float, sq: float, tr: float,
                               critical_failures: Optional[List[str]] = None) -> float:
        """
        Calculate LCB v2.0 score per PRD Section 7.

        Formula: LCB_Score = (0.30 × SC) + (0.35 × BF) + (0.25 × SQ) + (0.10 × TR)

        Critical failures result in automatic score of 0.

        Args:
            sc: Structural Completeness (0-100)
            bf: Behavioral Fidelity (0-100)
            sq: Semantic Quality (0-100)
            tr: Traceability (0-100)
            critical_failures: List of CF codes (e.g., ["CF-01", "CF-03"])

        Returns:
            LCB score (0-100), or 0 if any critical failure
        """
        # Check for critical failures
        if critical_failures:
            logger.warning(f"Critical failures detected: {critical_failures}")
            logger.warning("LCB score = 0 due to critical failures")
            return 0.0

        # Calculate weighted score
        lcb_score = (
            (self.v2_weights["structural_completeness"] * sc) +
            (self.v2_weights["behavioral_fidelity"] * bf) +
            (self.v2_weights["semantic_quality"] * sq) +
            (self.v2_weights["traceability"] * tr)
        )

        # Ensure score is in valid range
        lcb_score = max(0.0, min(100.0, lcb_score))

        logger.info(f"LCB v2.0 Score: {lcb_score:.2f} (SC={sc:.1f}, BF={bf:.1f}, SQ={sq:.1f}, TR={tr:.1f})")

        return lcb_score

    def detect_critical_failures(self, evaluation_result: Dict) -> List[str]:
        """
        Detect all critical failures from evaluation result.

        Per PRD Section 7.2: Any critical failure → Score = 0

        Args:
            evaluation_result: Full evaluation result dict

        Returns:
            List of CF codes (e.g., ["CF-01", "CF-03"])
        """
        failures = []

        # CF-01: Missing primary calculation
        if self._missing_primary_calculation(evaluation_result):
            failures.append("CF-01")
            logger.warning("CF-01: Missing primary calculation")

        # CF-02: Hallucinated module
        if self._hallucinated_module(evaluation_result):
            failures.append("CF-02")
            logger.warning("CF-02: Hallucinated module (references non-existent code)")

        # CF-03: Wrong data transformation
        bf_result = evaluation_result.get('behavioral_fidelity', {})
        output_mismatch_rate = bf_result.get('output_mismatch_rate', 0)
        if output_mismatch_rate >= 0.10:  # ≥10% outputs differ
            failures.append("CF-03")
            logger.warning(f"CF-03: Wrong data transformation ({output_mismatch_rate:.1%} mismatch)")

        # CF-04: Missing error handler
        if self._missing_error_handler(evaluation_result):
            failures.append("CF-04")
            logger.warning("CF-04: Missing error handler")

        # CF-05: Broken traceability
        tr_result = evaluation_result.get('traceability', {})
        broken_reference_rate = tr_result.get('broken_reference_rate', 0)
        if broken_reference_rate >= 0.20:  # ≥20% invalid references
            failures.append("CF-05")
            logger.warning(f"CF-05: Broken traceability ({broken_reference_rate:.1%} invalid refs)")

        # CF-06: False positive (NEW in v2.0)
        gap_markers = bf_result.get('gap_markers', 0)
        tests_passed_rate = bf_result.get('tests_passed_rate', 0)
        if tests_passed_rate >= 0.95 and gap_markers > 0:
            failures.append("CF-06")
            logger.warning(f"CF-06: False positive (execution passes but {gap_markers} gap markers present)")

        if failures:
            logger.error(f"Total critical failures: {len(failures)}")

        return failures

    def _missing_primary_calculation(self, result: Dict) -> bool:
        """Check if primary calculation is missing from documentation"""
        sc = result.get('structural_completeness', {})
        missing_elements = sc.get('missing_elements', {})

        # Check if critical business rules are missing
        missing_rules = missing_elements.get('business_rules', [])
        critical_keywords = ['calculate', 'compute', 'total', 'sum', 'interest', 'balance']

        for rule in missing_rules:
            rule_desc = rule.get('description', '').lower()
            if any(keyword in rule_desc for keyword in critical_keywords):
                return True

        return False

    def _hallucinated_module(self, result: Dict) -> bool:
        """Check if documentation references non-existent modules"""
        tr = result.get('traceability', {})
        invalid_refs = tr.get('invalid_references', [])

        # Check for fabricated references (references to code that doesn't exist)
        for ref in invalid_refs:
            if ref.get('type') == 'fabricated':
                return True

        return False

    def _missing_error_handler(self, result: Dict) -> bool:
        """Check if error handlers are missing"""
        sc = result.get('structural_completeness', {})
        missing_elements = sc.get('missing_elements', {})

        # Check if error handlers exist in ground truth but are missing in docs
        error_handlers_in_gt = result.get('ground_truth', {}).get('error_handlers', {})
        total_handlers = (
            error_handlers_in_gt.get('on_size_error', 0) +
            error_handlers_in_gt.get('invalid_key', 0) +
            error_handlers_in_gt.get('at_end', 0) +
            error_handlers_in_gt.get('file_status_checks', 0)
        )

        missing_handlers = missing_elements.get('error_handlers', [])

        # If >50% of error handlers are missing, it's critical
        if total_handlers > 0 and len(missing_handlers) / total_handlers > 0.5:
            return True

        return False

    def aggregate_v2_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate v2.0 results across multiple tasks.

        Args:
            results: List of v2.0 evaluation results

        Returns:
            Aggregated statistics
        """
        sc_scores = []
        bf_scores = []
        sq_scores = []
        tr_scores = []
        lcb_scores = []
        critical_failure_count = 0

        for result in results:
            sc_scores.append(result.get('structural_completeness_score', 0.0))
            bf_scores.append(result.get('behavioral_fidelity_score', 0.0))
            sq_scores.append(result.get('semantic_quality_score', 0.0))
            tr_scores.append(result.get('traceability_score', 0.0))
            lcb_scores.append(result.get('lcb_score', 0.0))

            if result.get('critical_failures'):
                critical_failure_count += 1

        return {
            "structural_completeness_avg": round(sum(sc_scores) / len(sc_scores), 2) if sc_scores else 0.0,
            "behavioral_fidelity_avg": round(sum(bf_scores) / len(bf_scores), 2) if bf_scores else 0.0,
            "semantic_quality_avg": round(sum(sq_scores) / len(sq_scores), 2) if sq_scores else 0.0,
            "traceability_avg": round(sum(tr_scores) / len(tr_scores), 2) if tr_scores else 0.0,
            "lcb_v2_avg": round(sum(lcb_scores) / len(lcb_scores), 2) if lcb_scores else 0.0,
            "total_tasks": len(results),
            "critical_failure_count": critical_failure_count,
            "critical_failure_rate": round(critical_failure_count / len(results), 3) if results else 0.0,
        }

