"""Documentation Evaluator v2.0

Main evaluator implementing the complete v2.0 framework.

Implements Section 5 of spec: Revised Scoring Framework
- Structural Completeness (30%)
- Behavioral Fidelity (35%) - placeholder until execution infrastructure ready
- Semantic Quality (25%)
- Traceability (10%)
- Critical Failure Detection

Replaces v1.0's reference-based ROUGE/BLEU evaluation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

from legacycodebench.static_analysis.ground_truth_generator import GroundTruthGenerator
from .structural_completeness import StructuralCompletenessEvaluator
from .semantic_quality import SemanticQualityEvaluator
from .traceability import TraceabilityEvaluator

# Import Behavioral Fidelity components
from legacycodebench.execution.test_generator import TestGenerator
from legacycodebench.execution.cobol_executor import COBOLExecutor
from legacycodebench.execution.code_generator import ConstrainedCodeGenerator
from legacycodebench.execution.behavior_comparator import BehaviorComparator

logger = logging.getLogger(__name__)


class DocumentationEvaluatorV2:
    """
    v2.0 Documentation Evaluator.

    Core innovation: Validates documentation against SOURCE CODE,
    not against human-written reference docs.

    Evaluation Flow:
    1. Load/generate ground truth from source code (static analysis)
    2. Evaluate Structural Completeness (element coverage)
    3. Evaluate Behavioral Fidelity (execution-based) - placeholder for now
    4. Evaluate Semantic Quality (LLM-as-judge)
    5. Evaluate Traceability (reference validation)
    6. Detect critical failures
    7. Calculate weighted LCB score
    """

    def __init__(self, ground_truth_cache_dir: Optional[Path] = None,
                results_dir: Optional[Path] = None,
                enable_execution: bool = True,
                docker_image: str = "legacycodebench-cobol:latest"):
        """
        Initialize v2.0 evaluator.

        Args:
            ground_truth_cache_dir: Directory to cache ground truth
            results_dir: Directory for results and escalations
            enable_execution: Enable execution-based BF evaluation (requires Docker)
            docker_image: Docker image for COBOL execution
        """
        self.ground_truth_gen = GroundTruthGenerator()
        self.sc_evaluator = StructuralCompletenessEvaluator()
        self.sq_evaluator = SemanticQualityEvaluator(results_dir=results_dir)
        self.tr_evaluator = TraceabilityEvaluator()

        # Behavioral Fidelity components (optional, requires Docker)
        self.enable_execution = enable_execution
        if enable_execution:
            try:
                self.test_generator = TestGenerator(max_tests_per_task=15)
                self.cobol_executor = COBOLExecutor(docker_image=docker_image)
                self.code_generator = ConstrainedCodeGenerator()
                self.behavior_comparator = BehaviorComparator()
                logger.info("Behavioral Fidelity evaluation enabled (execution-based)")
            except Exception as e:
                logger.warning(f"Behavioral Fidelity disabled: {e}")
                self.enable_execution = False
        else:
            logger.info("Behavioral Fidelity evaluation disabled (using placeholder)")

        self.ground_truth_cache_dir = ground_truth_cache_dir or Path("ground_truth_cache")
        self.ground_truth_cache_dir.mkdir(parents=True, exist_ok=True)

        # Scoring weights per Section 5.2 of spec
        self.weights = {
            "structural_completeness": 0.30,
            "behavioral_fidelity": 0.35,
            "semantic_quality": 0.25,
            "traceability": 0.10
        }

    def evaluate(self, submission_path: Path, task, **kwargs) -> Dict:
        """
        Evaluate documentation using v2.0 methodology.

        Args:
            submission_path: Path to AI-generated documentation
            task: Task object with source file information
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Evaluation results with v2.0 scores
        """
        logger.info("="*70)
        logger.info(f"EVALUATING: {task.task_id} using v2.0 framework")
        logger.info("="*70)

        # Step 1: Load submission
        with open(submission_path, 'r', encoding='utf-8') as f:
            submission_content = f.read()

        # Step 2: Load or generate ground truth
        logger.info("Loading/generating ground truth from source code...")
        ground_truth = self._load_or_generate_ground_truth(task)

        # Step 3: Load source code
        source_code_paths = task.get_input_files_absolute()
        source_code = self._load_source_code(source_code_paths)

        # Step 4: Evaluate Structural Completeness (30%)
        logger.info("Evaluating Structural Completeness (30%)...")
        sc_result = self.sc_evaluator.evaluate(submission_content, ground_truth)

        # Step 5: Evaluate Behavioral Fidelity (35%)
        logger.info("Evaluating Behavioral Fidelity (35%)...")
        if self.enable_execution:
            bf_result = self._evaluate_behavioral_fidelity(
                submission_content,
                ground_truth,
                source_code_paths,
                task.task_id
            )
        else:
            logger.info("  Using placeholder (execution disabled)")
            bf_result = self._placeholder_behavioral_fidelity(submission_content, ground_truth)

        # Step 6: Evaluate Semantic Quality (25%)
        logger.info("Evaluating Semantic Quality (25%) using LLM-as-judge...")
        # Note: evaluated_model not available here, but judge should be set differently in CLI
        sq_result = self.sq_evaluator.evaluate(
            submission_content,
            ground_truth,
            source_code,
            task.task_id,
            evaluated_model=None  # Could pass model name if available
        )

        # Step 7: Evaluate Traceability (10%)
        logger.info("Evaluating Traceability (10%)...")
        tr_result = self.tr_evaluator.evaluate(
            submission_content,
            ground_truth,
            source_code
        )

        # Step 8: Detect Critical Failures
        logger.info("Checking for critical failures...")
        critical_failures = self._detect_critical_failures(
            sc_result, bf_result, sq_result, tr_result, ground_truth
        )

        # Step 9: Calculate weighted LCB score
        if critical_failures:
            logger.warning(f"CRITICAL FAILURES DETECTED: {len(critical_failures)}")
            for cf in critical_failures:
                logger.warning(f"  - {cf}")
            lcb_score = 0.0
        else:
            lcb_score = self._calculate_lcb_score(
                sc_result["score"],
                bf_result["score"],
                sq_result["score"],
                tr_result["score"]
            )

        # Step 10: Determine Pass/Fail status (SWE-bench aligned)
        from legacycodebench.config import is_task_passed, get_pass_status
        
        result_for_pass_check = {
            "lcb_score": lcb_score,
            "behavioral_fidelity": bf_result["score"],
            "structural_completeness": sc_result["score"],
            "semantic_quality": sq_result["score"],
            "traceability": tr_result["score"],
            "critical_failures": critical_failures,
        }
        
        task_passed = is_task_passed(result_for_pass_check)
        pass_status = get_pass_status(result_for_pass_check)
        
        # Log results with pass/fail status
        pass_str = "✅ PASSED" if task_passed else "❌ FAILED"
        logger.info(f"FINAL LCB v2.0 SCORE: {lcb_score:.2%} | {pass_str}")
        if not task_passed:
            logger.info(f"  Failure reason: {pass_status['reason']}")
        logger.info("="*70)

        return {
            "score": lcb_score,
            "version": "2.0",
            "evaluation_method": "automated_ground_truth",
            
            # SWE-bench aligned pass/fail
            "passed": task_passed,
            "pass_status": pass_status,

            # Component scores
            "structural_completeness": sc_result["score"],
            "behavioral_fidelity": bf_result["score"],
            "semantic_quality": sq_result["score"],
            "traceability": tr_result["score"],

            # Detailed results
            "details": {
                "structural_completeness": sc_result,
                "behavioral_fidelity": bf_result,
                "semantic_quality": sq_result,
                "traceability": tr_result
            },

            # Critical failures
            "critical_failures": critical_failures,
            "has_critical_failures": len(critical_failures) > 0,

            # Metadata
            "ground_truth_confidence": ground_truth["metadata"]["confidence_score"],
            "ground_truth_automation": ground_truth["metadata"]["automation_level"],
            "task_id": task.task_id,
            "category": task.category
        }

    def _load_or_generate_ground_truth(self, task) -> Dict:
        """Load cached ground truth or generate from source code"""
        source_code_paths = task.get_input_files_absolute()

        if not source_code_paths:
            raise ValueError(f"No source files found for task {task.task_id}")

        main_file = source_code_paths[0]

        # Check cache
        cached_gt = self.ground_truth_gen.load_cached_ground_truth(
            main_file,
            self.ground_truth_cache_dir
        )

        if cached_gt:
            logger.info("Using cached ground truth")
            return cached_gt

        # Generate ground truth
        logger.info("Generating ground truth from source code...")
        ground_truth = self.ground_truth_gen.generate(
            source_code_paths,
            cache_dir=self.ground_truth_cache_dir
        )

        # Log summary
        summary = self.ground_truth_gen.generate_summary_report(ground_truth)
        logger.info(f"\n{summary}")

        return ground_truth

    def _load_source_code(self, paths: List[Path]) -> str:
        """Load and concatenate source code files"""
        code_parts = []

        for path in paths:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    code_parts.append(f"*> File: {path.name}\n{f.read()}\n")
            except Exception as e:
                logger.warning(f"Could not read {path}: {e}")

        return '\n'.join(code_parts)

    def _evaluate_behavioral_fidelity(self, submission_content: str,
                                     ground_truth: Dict,
                                     source_code_paths: List[Path],
                                     task_id: str) -> Dict:
        """
        Evaluate Behavioral Fidelity using execution-based testing.

        Algorithm:
        1. Generate test cases from ground truth
        2. Execute original COBOL with test cases
        3. Generate COBOL from documentation (constrained)
        4. Execute generated COBOL with same test cases
        5. Compare outputs
        6. Calculate fidelity score

        Returns:
            Dictionary with BF score and details
        """
        try:
            # Step 1: Generate test cases
            logger.info("  [1/5] Generating test cases from ground truth...")
            test_cases = self.test_generator.generate(ground_truth, task_id)

            if not test_cases:
                logger.warning("  No test cases generated")
                return self._placeholder_behavioral_fidelity(submission_content, ground_truth)

            logger.info(f"  Generated {len(test_cases)} test cases")

            # Step 2: Execute original COBOL
            logger.info("  [2/5] Executing original COBOL program...")
            original_source = self._load_source_code(source_code_paths)

            original_results = []
            for test_case in test_cases:
                result = self.cobol_executor.execute(
                    original_source,
                    test_case.inputs,
                    program_name=None  # Auto-detect from source
                )
                original_results.append(result)

            successful_original = sum(1 for r in original_results if r.success)
            logger.info(f"  Original execution: {successful_original}/{len(test_cases)} succeeded")

            # Step 3: Generate COBOL from documentation
            logger.info("  [3/5] Generating COBOL from documentation (constrained)...")
            generated = self.code_generator.generate(
                submission_content,
                ground_truth,
                program_name=None
            )

            # Check if documentation is complete enough
            if not generated.is_complete:
                logger.warning(
                    f"  Documentation too incomplete ({generated.gap_percentage:.1%} gaps)"
                )
                return {
                    "score": 0.0,
                    "tests_passed": 0,
                    "tests_failed": len(test_cases),
                    "total_tests": len(test_cases),
                    "gap_percentage": generated.gap_percentage,
                    "gaps": generated.gaps,
                    "incomplete_documentation": True,
                    "note": f"Documentation has {generated.gap_percentage:.1%} gaps (threshold: {self.code_generator.gap_threshold:.1%})"
                }

            # Step 4: Execute generated COBOL
            logger.info("  [4/5] Executing generated COBOL program...")
            generated_results = []
            for test_case in test_cases:
                result = self.cobol_executor.execute(
                    generated.code,
                    test_case.inputs,
                    program_name=None
                )
                generated_results.append(result)

            successful_generated = sum(1 for r in generated_results if r.success)
            logger.info(f"  Generated execution: {successful_generated}/{len(test_cases)} succeeded")

            # Step 5: Compare outputs
            logger.info("  [5/5] Comparing execution outputs...")
            comparison = self.behavior_comparator.compare(
                original_results,
                generated_results,
                test_cases
            )

            logger.info(
                f"  Behavioral Fidelity: {comparison.score:.2%} "
                f"({comparison.matches}/{comparison.total_tests} tests match)"
            )

            return {
                "score": comparison.score,
                "tests_passed": comparison.matches,
                "tests_failed": comparison.mismatches,
                "tests_error": comparison.errors,
                "total_tests": comparison.total_tests,
                "gap_percentage": generated.gap_percentage,
                "total_gaps": len(generated.gaps),
                "comparison_details": comparison.to_dict(),
                "placeholder": False
            }

        except Exception as e:
            logger.error(f"Behavioral Fidelity evaluation failed: {e}", exc_info=True)
            # Fall back to placeholder
            return {
                "score": 0.0,
                "tests_passed": 0,
                "tests_failed": 0,
                "total_tests": 0,
                "error": str(e),
                "note": "Execution-based evaluation failed, see error",
                "placeholder": True
            }

    def _placeholder_behavioral_fidelity(self, submission_content: str,
                                        ground_truth: Dict) -> Dict:
        """
        Placeholder for Behavioral Fidelity evaluation.

        Used when execution infrastructure is not available.
        """
        logger.warning("Behavioral Fidelity evaluation not available (using placeholder)")

        # For now, return neutral score
        return {
            "score": 0.75,  # Placeholder: neutral score
            "tests_passed": 0,
            "tests_failed": 0,
            "total_tests": 0,
            "note": "Placeholder - execution infrastructure not available",
            "placeholder": True
        }

    def _calculate_lcb_score(self, sc: float, bf: float, sq: float, tr: float) -> float:
        """
        Calculate weighted LCB v2.0 score.

        Per Section 5.2 of spec:
        LCB_Score = (0.30 × SC) + (0.35 × BF) + (0.25 × SQ) + (0.10 × TR)
        """
        return (
            self.weights["structural_completeness"] * sc +
            self.weights["behavioral_fidelity"] * bf +
            self.weights["semantic_quality"] * sq +
            self.weights["traceability"] * tr
        )

    def _detect_critical_failures(self, sc_result: Dict, bf_result: Dict,
                                  sq_result: Dict, tr_result: Dict,
                                  ground_truth: Dict) -> List[str]:
        """
        Detect critical failures per Section 5.3 of spec.

        Critical Failures (auto-fail):
        - CF-01: Missing primary calculation
        - CF-02: Hallucinated module
        - CF-03: Wrong data transformation
        - CF-04: Missing error handler
        - CF-05: Broken traceability
        - CF-06: False positive (execution passed but gaps in docs)
        """
        failures = []

        # CF-01: Missing primary calculation (top business rules not documented)
        business_rules = ground_truth.get("business_rules", {})
        if business_rules.get("total_rules", 0) > 0:
            # Check if major business rules are missing
            missing_rules = sc_result.get("missing_elements", {}).get("business_rules", [])

            if len(missing_rules) > business_rules["total_rules"] * 0.5:
                failures.append("CF-01: Missing primary calculations (>50% business rules not documented)")

        # CF-02: Hallucinated module
        # This would be detected by traceability - fabricated references
        invalid_refs = tr_result.get("invalid_references", [])
        fabricated_refs = [r for r in invalid_refs if "paragraph" in r.get("type", "") or "variable" in r.get("type", "")]

        if len(fabricated_refs) > 3:  # More than 3 fabricated references
            failures.append(f"CF-02: Hallucinated modules ({len(fabricated_refs)} fabricated references)")

        # CF-03: Wrong data transformation
        # Would be detected by behavioral fidelity (not yet implemented)
        if not bf_result.get("placeholder", False):
            if bf_result.get("score", 1.0) < 0.90:
                failures.append("CF-03: Wrong data transformation (execution mismatch)")

        # CF-04: Missing error handler
        error_handlers = ground_truth.get("error_handlers", [])
        if len(error_handlers) > 0:
            missing_handlers = sc_result.get("missing_elements", {}).get("error_handlers", [])

            if len(missing_handlers) == len(error_handlers):
                failures.append("CF-04: Missing error handlers (none documented)")

        # CF-05: Broken traceability
        # Only trigger if there are references but they're invalid
        # If no references at all, that's a quality issue but not a critical failure
        total_refs = tr_result.get("total_references", 0)
        if total_refs > 0 and tr_result["score"] < 0.20:  # Has references but < 20% valid
            failures.append(f"CF-05: Broken traceability ({tr_result['score']:.0%} valid references)")
        # Note: If total_references == 0, that's handled by traceability score (0.5 default)
        # but doesn't trigger critical failure (documentation may not need citations for simple code)

        # CF-06: False positive (would be detected by constrained generator)
        # Placeholder until execution infrastructure is ready
        if "gap_markers" in bf_result:
            failures.append("CF-06: False positive (documentation incomplete)")

        return failures
