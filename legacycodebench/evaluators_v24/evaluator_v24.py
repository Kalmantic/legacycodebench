"""
LegacyCodeBench V2.4 Evaluator Orchestrator

Main evaluator that coordinates multi-language evaluation using
the adapter abstraction layer.

Specification Reference: TDD_V2.4.md Section 4
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

from legacycodebench.models.enums import Language, VerificationMode, CompileFailureReason
from legacycodebench.models.results import (
    EvaluationResult,
    BFResult,
    TrackScore,
    CriticalFailure,
)
from legacycodebench.adapters import get_adapter, detect_language

logger = logging.getLogger(__name__)


class EvaluatorV24:
    """
    V2.4 Multi-Language Evaluator.
    
    Orchestrates evaluation across language adapters, reusing
    existing evaluator components:
    - V2.3.1 Structural/Documentation evaluators
    - V3 Behavioral Fidelity evaluator
    
    Key changes from V2.3.1:
    - Language detection from task_id
    - Language-specific BSM patterns
    - Language-specific critical failure config
    """

    VERSION = "v2.4"

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        enable_execution: bool = True,
    ):
        """
        Initialize the V2.4 evaluator.
        
        Args:
            dataset_path: Path to datasets for copybook/include resolution
            enable_execution: Whether to attempt execution (vs always static)
        """
        self.dataset_path = dataset_path
        self.enable_execution = enable_execution
        
        # Lazy-loaded component evaluators
        self._sc_evaluator = None
        self._dq_evaluator = None
        self._bf_evaluator = None

    def evaluate(
        self,
        task_id: str,
        documentation: str,
        source_code: str,
        ground_truth: Dict,
        task_metadata: Optional[Dict] = None,
    ) -> EvaluationResult:
        """
        Evaluate documentation for a task.
        
        Args:
            task_id: Task identifier (determines language)
            documentation: AI-generated documentation
            source_code: Original source code
            ground_truth: Ground truth data
            task_metadata: Optional task metadata
            
        Returns:
            EvaluationResult with all track scores and critical failures
        """
        start_time = time.time()
        
        # Step 1: Detect language and get adapter
        language = detect_language(task_id)
        adapter = get_adapter(language)
        
        logger.info(f"Evaluating {task_id} with {language.value} adapter")
        
        # Step 2: Run track evaluators
        sc_result = self._evaluate_structural(documentation, source_code, ground_truth, adapter)
        dq_result = self._evaluate_documentation(documentation, source_code, ground_truth)
        bf_result = self._evaluate_behavioral(
            documentation, source_code, ground_truth, task_id, adapter, task_metadata
        )
        
        # Step 3: Check critical failures (language-aware)
        critical_failures = self._check_critical_failures(
            documentation, source_code, ground_truth, 
            sc_result, dq_result, bf_result, adapter
        )
        
        # Step 4: Calculate final score
        # Weights: SC=30%, DQ=20%, BF=50%
        if critical_failures:
            # Any critical failure → LCB Score = 0
            lcb_score = 0.0
            passed = False
            pass_reason = f"Critical failure: {critical_failures[0].name}"
        else:
            lcb_score = (
                0.30 * sc_result.score +
                0.20 * dq_result.score +
                0.50 * bf_result.score
            )
            
            # Check thresholds
            passed = (
                sc_result.score >= 0.60 and
                dq_result.score >= 0.50 and
                bf_result.score >= 0.55
            )
            pass_reason = "All thresholds met" if passed else "Below threshold(s)"
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return EvaluationResult(
            task_id=task_id,
            language=language,
            structural_completeness=sc_result,
            documentation_quality=dq_result,
            behavioral_fidelity=bf_result,
            lcb_score=lcb_score,
            passed=passed,
            pass_reason=pass_reason,
            critical_failures=critical_failures,
            evaluation_time_ms=elapsed_ms,
            evaluator_version=self.VERSION,
        )

    def _evaluate_structural(
        self,
        documentation: str,
        source_code: str,
        ground_truth: Dict,
        adapter,
    ) -> TrackScore:
        """Evaluate Structural Completeness (Track 1)."""
        try:
            # Reuse V2.3.1 structural evaluator
            from legacycodebench.evaluators_v231.structural_v231 import StructuralEvaluatorV231
            
            if self._sc_evaluator is None:
                self._sc_evaluator = StructuralEvaluatorV231()
            
            # Get language-specific synonyms for TF-IDF expansion
            synonyms = adapter.get_synonyms()
            
            result = self._sc_evaluator.evaluate(
                documentation=documentation,
                ground_truth=ground_truth,
            )
            
            return TrackScore(
                score=result.get("score", 0.0),
                breakdown=result.get("breakdown", {}),
            )
        except Exception as e:
            logger.error(f"SC evaluation failed: {e}")
            return TrackScore(score=0.0, breakdown={"error": str(e)})

    def _evaluate_documentation(
        self,
        documentation: str,
        source_code: str,
        ground_truth: Dict,
    ) -> TrackScore:
        """Evaluate Documentation Quality (Track 2)."""
        try:
            # Reuse V2.3.1 documentation evaluator (language-agnostic)
            from legacycodebench.evaluators_v231.documentation_v231 import DocumentationEvaluatorV231
            
            if self._dq_evaluator is None:
                self._dq_evaluator = DocumentationEvaluatorV231()
            
            result = self._dq_evaluator.evaluate(
                documentation=documentation,
                source_code=source_code,
                ground_truth=ground_truth,
            )
            
            return TrackScore(
                score=result.get("score", 0.0),
                breakdown=result.get("breakdown", {}),
            )
        except Exception as e:
            logger.error(f"DQ evaluation failed: {e}")
            return TrackScore(score=0.0, breakdown={"error": str(e)})

    def _evaluate_behavioral(
        self,
        documentation: str,
        source_code: str,
        ground_truth: Dict,
        task_id: str,
        adapter,
        task_metadata: Optional[Dict],
    ) -> BFResult:
        """Evaluate Behavioral Fidelity (Track 3) using V3 evaluator."""
        try:
            # Use V3 behavioral evaluator
            from legacycodebench.evaluators_v3.behavioral_v3 import BehavioralEvaluatorV3
            
            # Get language-specific executor
            executor = adapter.get_executor() if self.enable_execution else None
            
            if self._bf_evaluator is None:
                self._bf_evaluator = BehavioralEvaluatorV3(
                    executor=executor,
                    dataset_path=self.dataset_path,
                    enable_execution=self.enable_execution,
                )
            
            # Get language-specific BSM patterns
            bsm_patterns = adapter.get_bsm_patterns()
            
            result = self._bf_evaluator.evaluate(
                documentation=documentation,
                source_code=source_code,
                ground_truth=ground_truth,
                task_metadata=task_metadata or {"task_id": task_id},
            )
            
            # Convert to BFResult dataclass
            if isinstance(result, dict):
                return BFResult(
                    score=result.get("score", 0.0),
                    verification_mode=VerificationMode(result.get("verification_mode", "static")),
                    mode_reason=result.get("mode_reason", ""),
                    compile_attempted=result.get("compile_attempted", True),
                    compile_succeeded=result.get("compile_succeeded", False),
                    compile_error=result.get("compile_error"),
                    failure_reason=CompileFailureReason(result.get("failure_reason", "none")),
                    fixable=result.get("fixable"),
                    execution_attempted=result.get("execution_attempted", False),
                    execution_succeeded=result.get("execution_succeeded", False),
                    runtime_error=result.get("runtime_error"),
                    tests_passed=result.get("tests_passed", 0),
                    tests_failed=result.get("tests_failed", 0),
                    tests_total=result.get("tests_total", 0),
                    claims_extracted=result.get("claims_extracted", 0),
                    claims_verified=result.get("claims_verified", 0),
                    claims_failed=result.get("claims_failed", 0),
                    claim_extraction_method=result.get("claim_extraction_method", "regex"),
                    bsm_matched=result.get("bsm_matched", 0),
                    bsm_total=result.get("bsm_total", 0),
                    claim_score=result.get("claim_score", 0.0),
                    bsm_score=result.get("bsm_score", 0.0),
                )
            else:
                # Already a BFResult-like object
                return BFResult(
                    score=getattr(result, 'score', 0.0),
                    verification_mode=VerificationMode.STATIC,
                    mode_reason=getattr(result, 'mode_reason', 'V3 evaluation'),
                    claims_extracted=getattr(result, 'claims_extracted', 0),
                    claims_verified=getattr(result, 'claims_verified', 0),
                    claims_failed=getattr(result, 'claims_failed', 0),
                    bsm_matched=getattr(result, 'bsm_matched', 0),
                    bsm_total=getattr(result, 'bsm_total', 0),
                    claim_score=getattr(result, 'claim_score', 0.0),
                    bsm_score=getattr(result, 'bsm_score', 0.0),
                )
                
        except Exception as e:
            logger.error(f"BF evaluation failed: {e}")
            return BFResult(
                score=0.0,
                verification_mode=VerificationMode.ERROR,
                mode_reason=f"Evaluation error: {str(e)}",
            )

    def _check_critical_failures(
        self,
        documentation: str,
        source_code: str,
        ground_truth: Dict,
        sc_result: TrackScore,
        dq_result: TrackScore,
        bf_result: BFResult,
        adapter,
    ) -> list:
        """
        Check for critical failures (language-aware).
        
        Uses adapter.get_critical_failure_config() to determine
        which CFs apply to this language.
        """
        failures = []
        cf_config = adapter.get_critical_failure_config()
        
        try:
            from legacycodebench.evaluators_v231.critical_failures_v231 import CriticalFailureDetector
            
            detector = CriticalFailureDetector()
            
            # Run detection with language-specific config
            detected = detector.detect(
                documentation=documentation,
                source_code=source_code,
                ground_truth=ground_truth,
                bf_result={
                    "score": bf_result.score,
                    "verification_mode": bf_result.verification_mode.value,
                    "claims_extracted": bf_result.claims_extracted,
                    "claims_failed": bf_result.claims_failed,
                    "bsm_matched": bf_result.bsm_matched,
                    "bsm_total": bf_result.bsm_total,
                },
            )
            
            # Filter by language config
            for cf in detected:
                cf_id = cf.get("cf_id", "")
                if cf_config.get(cf_id, True):
                    failures.append(CriticalFailure(
                        cf_id=cf_id,
                        name=cf.get("name", "Unknown"),
                        description=cf.get("description", ""),
                        evidence=cf.get("evidence", {}),
                    ))
                else:
                    logger.debug(f"Skipping {cf_id} - disabled for {adapter.language.value}")
                    
        except Exception as e:
            logger.error(f"Critical failure detection failed: {e}")
        
        return failures
