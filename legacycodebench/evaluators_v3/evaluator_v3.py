"""
V3.0 Evaluation Orchestrator

Main entry point for V3.0 evaluation, integrating the new BF V3 with compile-first
classification.

This is a thin wrapper around EvaluatorV231 that replaces the behavioral evaluator
with BehavioralEvaluatorV3 for improved transparency and honest scoring.

Changes from V2.3.1:
- BF uses compile-first classification (compiler tells us the truth)
- Full provenance tracking (mode, reason, compile_error in results)
- No silent fallbacks to heuristic scoring
- Static verification can actually FAIL bad documentation
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from ..evaluators_v231.evaluator_v231 import preprocess_documentation
from ..evaluators_v231.structural_v231 import StructuralEvaluatorV231
from ..evaluators_v231.documentation_v231 import DocumentationEvaluatorV231
from ..evaluators_v231.critical_failures_v231 import CriticalFailureDetectorV231
from ..evaluators_v231.scoring_v231 import ScoringEngineV231, EvaluationResultV231

from .behavioral_v3 import BehavioralEvaluatorV3
from .bf_result import BFResult, VerificationMode

logger = logging.getLogger(__name__)


@dataclass
class V3EvaluationContext:
    """Context for V3.0 evaluation."""
    task_id: str
    model: str
    documentation: str
    source_code: str
    ground_truth: Dict
    start_time: float = field(default_factory=time.time)


class EvaluatorV3:
    """
    V3.0 Evaluation Orchestrator
    
    This is the main entry point for V3.0 evaluation.
    
    Scoring Formula (same as V2.3.1):
        LCB = 0.30×SC + 0.20×DQ + 0.50×BF
    
    Key Difference from V2.3.1:
        - BF uses compile-first classification
        - Full provenance tracking in BF results
        - No silent fallbacks to heuristic
    
    Usage:
        evaluator = EvaluatorV3(executor=COBOLExecutor())
        result = evaluator.evaluate(
            task_id="LCB-T1-001",
            model="gpt-4o",
            documentation="...",
            source_code="...",
            ground_truth={...}
        )
    """
    
    def __init__(self, llm_client=None, executor=None, dataset_path=None):
        """
        Initialize V3.0 evaluator.
        
        Args:
            llm_client: Optional LLM client (for claim extraction fallback)
            executor: Optional COBOL executor for behavioral testing
            dataset_path: Path to dataset for copybook resolution
        """
        # Reuse V2.3.1 structural and documentation evaluators
        self.structural = StructuralEvaluatorV231()
        self.documentation = DocumentationEvaluatorV231()
        
        # Use NEW BF V3 evaluator
        self.behavioral = BehavioralEvaluatorV3(
            executor=executor,
            dataset_path=dataset_path,
            enable_execution=executor is not None,
        )
        
        self.cf_detector = CriticalFailureDetectorV231()
        self.scoring = ScoringEngineV231()
        
        self.llm = llm_client
        self.executor = executor
        
        logger.info("V3.0 Evaluator initialized (BF V3 with compile-first classification)")
    
    def evaluate(
        self,
        task_id: str,
        model: str,
        documentation: str,
        source_code: str,
        ground_truth: Dict,
        task_metadata: Dict = None,
        **kwargs
    ) -> EvaluationResultV231:
        """
        Perform complete V3.0 evaluation.
        
        Args:
            task_id: Task identifier
            model: Model name being evaluated
            documentation: AI-generated documentation
            source_code: Original COBOL source code
            ground_truth: Ground truth data
            task_metadata: Optional task metadata (tier, boosts, etc.)
            
        Returns:
            Complete EvaluationResultV231 (compatible with V2.3.1 format)
        """
        logger.info(f"Starting V3.0 evaluation for {task_id}")
        start_time = time.time()
        
        # Preprocess documentation
        documentation = preprocess_documentation(documentation)
        
        logger.debug(f"\n{'='*60}")
        logger.debug(f"V3.0 EVALUATION: {task_id}")
        logger.debug(f"{'='*60}")
        
        # =====================================================================
        # Track 1: Structural Completeness (30%) - Same as V2.3.1
        # =====================================================================
        logger.debug(f"\n{'-'*40}")
        logger.debug("TRACK 1: STRUCTURAL COMPLETENESS (30%)")
        logger.debug(f"{'-'*40}")
        
        sc_result = self.structural.evaluate(documentation, ground_truth)
        logger.info(f"SC Score: {sc_result.score:.2%}")
        
        # =====================================================================
        # Track 2: Documentation Quality (20%) - Same as V2.3.1
        # =====================================================================
        logger.debug(f"\n{'-'*40}")
        logger.debug("TRACK 2: DOCUMENTATION QUALITY (20%)")
        logger.debug(f"{'-'*40}")
        
        dq_result = self.documentation.evaluate(documentation, source_code)
        logger.info(f"DQ Score: {dq_result.score:.2%}")
        
        # =====================================================================
        # Track 3: Behavioral Fidelity (50%) - NEW V3 with compile-first
        # =====================================================================
        logger.debug(f"\n{'-'*40}")
        logger.debug("TRACK 3: BEHAVIORAL FIDELITY (50%) - V3.0")
        logger.debug(f"{'-'*40}")
        
        # Prepare task metadata
        metadata = task_metadata or {}
        metadata["task_id"] = task_id
        
        bf_result = self.behavioral.evaluate(
            documentation=documentation,
            source_code=source_code,
            ground_truth=ground_truth,
            task_metadata=metadata,
        )
        
        logger.info(f"BF Score: {bf_result.score:.2%} ({bf_result.verification_mode.value})")
        logger.info(f"BF Reason: {bf_result.mode_reason}")
        
        # Convert BFResult to V2.3.1 compatible BehavioralResult
        from ..evaluators_v231.behavioral_v231 import BehavioralResult
        
        bf_compat = BehavioralResult(
            score=bf_result.score,
            claim_score=bf_result.claim_score,
            bsm_score=bf_result.bsm_score,
            claim_count=bf_result.claims_total,
            silence_penalty=bf_result.silence_penalty,
            claims_verified=bf_result.claims_verified,
            claims_failed=bf_result.claims_failed,
            bsm_matched=bf_result.bsm_matched,
            bsm_total=bf_result.bsm_total,
            breakdown={
                "claim_score": bf_result.claim_score,
                "bsm_score": bf_result.bsm_score,
                "verification_mode": bf_result.verification_mode.value,
                "mode_reason": bf_result.mode_reason,
                "compile_error": bf_result.compile_error,
                "fixable": bf_result.fixable,
            },
            details={
                "bf_v3": bf_result.to_dict(),
            },
        )
        
        # =====================================================================
        # Track 4: Critical Failure Detection
        # =====================================================================
        logger.debug(f"\n{'-'*40}")
        logger.debug("TRACK 4: CRITICAL FAILURE DETECTION")
        logger.debug(f"{'-'*40}")
        
        # CF detector uses detect_all, passing SC score and BF result
        cfs = self.cf_detector.detect_all(
            documentation=documentation,
            ground_truth=ground_truth,
            sc_score=sc_result.score,
            bf_result=bf_compat.breakdown if bf_compat else None
        )
        
        # Add CF if silence penalty triggered
        if bf_result.silence_penalty:
            from ..evaluators_v231.critical_failures_v231 import CriticalFailure
            cfs.append(CriticalFailure(
                cf_id="CF-03",
                name="Silence Penalty",
                description="Silence penalty: documentation too vague for claim extraction",
                severity="HIGH",
            ))
        
        logger.info(f"Critical Failures: {len(cfs)}")
        
        # =====================================================================
        # Final Scoring
        # =====================================================================
        logger.debug(f"\n{'-'*40}")
        logger.debug("FINAL SCORING")
        logger.debug(f"{'-'*40}")
        
        # Build bf_breakdown with V3 provenance info
        bf_breakdown = bf_compat.breakdown if bf_compat else {}
        bf_breakdown["v3_provenance"] = {
            "evaluator_version": "v3.0",
            "verification_mode": bf_result.verification_mode.value,
            "mode_reason": bf_result.mode_reason,
            "execution_attempted": bf_result.execution_attempted,
            "execution_succeeded": bf_result.execution_succeeded,
            "compile_error": bf_result.compile_error[:200] if bf_result.compile_error else None,
            "claims_verified": bf_result.claims_verified,
            "claims_total": bf_result.claims_total,
        }
        
        result = self.scoring.calculate(
            sc_score=sc_result.score,
            dq_score=dq_result.score,
            bf_score=bf_compat.score if bf_compat else bf_result.score,
            critical_failures=cfs,
            sc_breakdown=sc_result.breakdown if hasattr(sc_result, 'breakdown') else {},
            dq_breakdown=dq_result.breakdown if hasattr(dq_result, 'breakdown') else {},
            bf_breakdown=bf_breakdown,
        )
        
        # Update version to indicate V3
        result.version = "3.0"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"V3.0 FINAL SCORE: {result.lcb_score:.1f}")
        logger.info(f"  SC: {result.sc_score:.1%} × 0.30 = {result.sc_score * 0.30:.2%}")
        logger.info(f"  DQ: {result.dq_score:.1%} × 0.20 = {result.dq_score * 0.20:.2%}")
        logger.info(f"  BF: {result.bf_score:.1%} × 0.50 = {result.bf_score * 0.50:.2%} [{bf_result.verification_mode.value}]")
        logger.info(f"  CFs: {len(cfs)}")
        logger.info(f"{'='*60}\n")
        
        return result
