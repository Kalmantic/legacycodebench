"""Documentation Evaluator v2.0

Main evaluator implementing the complete v2.0 framework.

Implements Section 5 of spec: Revised Scoring Framework
- Structural Completeness (25%)
- Behavioral Fidelity (35%) = IUE (20%) + BSM (15%)
- Semantic Quality (25%)
- Traceability (15%)
- Critical Failure Detection

v2.1: Replaced execution-only BF with IUE+BSM approach for 100% coverage.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
import subprocess
from datetime import datetime

from legacycodebench.static_analysis.ground_truth_generator import GroundTruthGenerator
from .structural_completeness import StructuralCompletenessEvaluator
from .semantic_quality import SemanticQualityEvaluator
from .traceability import TraceabilityEvaluator

# Import Behavioral Fidelity V2 (IUE + BSM)
from legacycodebench.evaluators_v2.behavioral_fidelity_v2 import BehavioralFidelityEvaluatorV2

# Legacy BF components (kept for backward compatibility)
try:
    from legacycodebench.execution.test_generator import TestGenerator
    from legacycodebench.execution.cobol_executor import COBOLExecutor
    from legacycodebench.execution.code_generator import ConstrainedCodeGenerator
    from legacycodebench.execution.behavior_comparator import BehaviorComparator
    LEGACY_BF_AVAILABLE = True
except ImportError:
    LEGACY_BF_AVAILABLE = False

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

        # Behavioral Fidelity V2 (IUE + BSM)
        # Always available - doesn't require Docker for BSM analysis
        self.enable_execution = enable_execution
        self.bf_evaluator_v2 = BehavioralFidelityEvaluatorV2(
            enable_execution=enable_execution
        )
        logger.info(f"Behavioral Fidelity V2 (IUE+BSM) initialized, "
                   f"execution={'enabled' if enable_execution else 'disabled'}")
        
        # Legacy BF components for Docker execution (v2.1.4)
        self.docker_execution_available = False
        if enable_execution and LEGACY_BF_AVAILABLE:
            try:
                self.test_generator = TestGenerator()
                self.cobol_executor = COBOLExecutor(docker_image=docker_image)
                self.code_generator = ConstrainedCodeGenerator()
                self.behavior_comparator = BehaviorComparator()
                self.docker_execution_available = True
                logger.info("Docker execution components initialized successfully")
            except Exception as e:
                logger.warning(f"Docker execution not available: {e}")
                logger.info("Will use IUE/BSM static analysis for Behavioral Fidelity")

        # Use same cache directory as CLI for consistency
        self.ground_truth_cache_dir = ground_truth_cache_dir or Path("cache/ground_truth")
        self.ground_truth_cache_dir.mkdir(parents=True, exist_ok=True)

        # Scoring weights per Section 5.2 of spec (v2.1 updated)
        # BF is split into IUE (20%) + BSM (15%) internally
        self.weights = {
            "structural_completeness": 0.25,   # Reduced from 30%
            "behavioral_fidelity": 0.35,       # IUE (20%) + BSM (15%)
            "semantic_quality": 0.25,          # Unchanged
            "traceability": 0.15               # Increased from 10%
        }

    def _get_reproducibility_metadata(self) -> Dict:
        """
        ADDED (Issue 7.1): Collect reproducibility metadata for results.

        Returns:
            Dict with git hash, Docker version, benchmark version, timestamp
        """
        metadata = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "benchmark_version": "2.0",
        }

        # Get git commit hash
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            metadata["git_commit"] = git_hash
        except:
            metadata["git_commit"] = "unknown"

        # Get Docker version if execution is enabled
        if self.enable_execution:
            try:
                docker_version = subprocess.check_output(
                    ["docker", "--version"],
                    stderr=subprocess.DEVNULL,
                    text=True
                ).strip()
                metadata["docker_version"] = docker_version
            except:
                metadata["docker_version"] = "unknown"
        else:
            metadata["docker_version"] = "not_used"

        return metadata

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

        # Step 4: Evaluate Structural Completeness (25%)
        logger.info("Evaluating Structural Completeness (25%)...")
        sc_result = self.sc_evaluator.evaluate(submission_content, ground_truth)

        # Step 5: Evaluate Behavioral Fidelity (35%)
        # v2.1.4: Try Docker execution first, fall back to IUE/BSM if unavailable
        if self.docker_execution_available:
            logger.info("Evaluating Behavioral Fidelity (35%) using Docker execution...")
            try:
                bf_result = self._evaluate_behavioral_fidelity(
                    submission_content, ground_truth, source_code_paths, task.task_id
                )
                if bf_result.get("total_tests", 0) > 0:
                    logger.info(f"Docker execution: {bf_result.get('tests_passed', 0)}/{bf_result.get('total_tests', 0)} tests passed")
                else:
                    # Docker available but no tests ran - fall back to IUE/BSM
                    logger.info("Docker execution produced no tests, falling back to IUE/BSM...")
                    bf_result = self.bf_evaluator_v2.evaluate(
                        source_code=source_code,
                        documentation=submission_content,
                        ground_truth=ground_truth,
                        task_id=task.task_id
                    )
            except Exception as e:
                logger.warning(f"Docker execution failed: {e}, falling back to IUE/BSM...")
                bf_result = self.bf_evaluator_v2.evaluate(
                    source_code=source_code,
                    documentation=submission_content,
                    ground_truth=ground_truth,
                    task_id=task.task_id
                )
        else:
            logger.info("Evaluating Behavioral Fidelity (35%) using IUE + BSM...")
            bf_result = self.bf_evaluator_v2.evaluate(
                source_code=source_code,
                documentation=submission_content,
                ground_truth=ground_truth,
                task_id=task.task_id
            )

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

        # Step 7: Evaluate Traceability (15%)
        logger.info("Evaluating Traceability (15%)...")
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
        pass_str = "[PASSED]" if task_passed else "[FAILED]"
        logger.info(f"FINAL LCB v2.0 SCORE: {lcb_score:.2%} | {pass_str}")
        if not task_passed:
            logger.info(f"  Failure reason: {pass_status['reason']}")
        logger.info("="*70)

        # ADDED (Issue 7.1): Collect reproducibility metadata
        reproducibility = self._get_reproducibility_metadata()

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
            "category": task.category,

            # ADDED (Issue 7.1): Reproducibility metadata
            "reproducibility": reproducibility
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

    def _load_source_code(self, paths: List[Path], for_execution: bool = False) -> str:
        """Load and concatenate source code files
        
        Args:
            paths: List of source file paths
            for_execution: If True, don't add comments (preserves COBOL format)
        """
        code_parts = []

        for path in paths:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if for_execution:
                        # Don't add comments - they break fixed-format COBOL
                        code_parts.append(content)
                    else:
                        # Add file markers for display/analysis
                        code_parts.append(f"*> File: {path.name}\n{content}\n")
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
            # Use for_execution=True to avoid adding comments that break COBOL format
            original_source = self._load_source_code(source_code_paths, for_execution=True)

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

        CRITICAL FIX (Issue 5.1/6.1): Returns 0.0 instead of 0.75
        to prevent inflating scores when execution is disabled.
        """
        logger.warning("Behavioral Fidelity evaluation not available (using placeholder)")

        # Return 0.0 when execution is disabled (not 0.75)
        # This prevents silently inflating scores by 26.25% (0.35 * 0.75)
        return {
            "score": 0.0,  # FIXED: Was 0.75, now 0.0 (Issue 5.1/6.1)
            "tests_passed": 0,
            "tests_failed": 0,
            "total_tests": 0,
            "note": "Placeholder - execution infrastructure not available (BF=0 by design)",
            "placeholder": True
        }

    def _calculate_lcb_score(self, sc: float, bf: float, sq: float, tr: float) -> float:
        """
        Calculate weighted LCB v2.0 score.

        Per Section 5.2 of spec:
        LCB_Score = (0.30 x SC) + (0.35 x BF) + (0.25 x SQ) + (0.10 x TR)
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

        # CF-01: Missing primary calculation (CRITICAL business rules not documented)
        # PRODUCTION GRADE v2.1.3: More lenient - keyword matching is imperfect
        # Only trigger if documentation is clearly missing business logic
        business_rules = ground_truth.get("business_rules", {})
        critical_rule_count = business_rules.get("critical_rules", 0)
        
        # Check SC score as proxy for documentation quality
        # If SC is reasonable (>30%), assume business rules are covered in some form
        sc_score = sc_result.get("score", 0)
        
        if critical_rule_count > 0 and sc_score < 0.30:
            # Low SC score indicates truly incomplete documentation
            # Get the list of critical rule IDs
            rules_by_priority = business_rules.get("rules_by_priority", {})
            critical_rule_ids = set(rules_by_priority.get("critical", []))
            
            # Check how many critical rules are missing
            missing_rules = sc_result.get("missing_elements", {}).get("business_rules", [])
            missing_critical = 0
            
            for missing in missing_rules:
                # Extract rule ID from the missing rule description
                if "BR-" in missing:
                    import re
                    match = re.search(r'BR-\d+', missing)
                    if match and match.group() in critical_rule_ids:
                        missing_critical += 1
            
            # Only fail if ALL critical rules are missing AND SC is very low
            if missing_critical == critical_rule_count:
                failures.append(f"CF-01: Missing primary calculations ({missing_critical}/{critical_rule_count} critical business rules not documented)")
        
        # FALLBACK: If no priority info, use old logic but be very lenient
        elif business_rules.get("total_rules", 0) > 0 and sc_score < 0.20:
            missing_rules = sc_result.get("missing_elements", {}).get("business_rules", [])
            if len(missing_rules) == business_rules["total_rules"]:  # ALL missing
                failures.append("CF-01: Missing primary calculations (no business rules documented)")

        # CF-02: Hallucinated module
        # This would be detected by traceability - fabricated references
        invalid_refs = tr_result.get("invalid_references", [])
        fabricated_refs = [r for r in invalid_refs if "paragraph" in r.get("type", "") or "variable" in r.get("type", "")]

        if len(fabricated_refs) > 3:  # More than 3 fabricated references
            failures.append(f"CF-02: Hallucinated modules ({len(fabricated_refs)} fabricated references)")

        # CF-03: Wrong data transformation
        # Only trigger if actual execution tests were run (not just IUE/BSM static analysis)
        # v2.1.4: Check for actual execution via total_tests > 0
        is_placeholder = bf_result.get("placeholder", False)
        total_tests = bf_result.get("total_tests", 0)
        actual_execution_ran = not is_placeholder and total_tests > 0
        
        if actual_execution_ran:
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
