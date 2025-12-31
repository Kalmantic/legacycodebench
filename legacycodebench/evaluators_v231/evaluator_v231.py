"""
V2.3.1 Evaluation Orchestrator

Main entry point for V2.3.1 evaluation.

Coordinates:
1. Structural Completeness (30%)
2. Documentation Quality (20%)
3. Behavioral Fidelity (50%)
4. Critical Failure Detection
5. Final Scoring

Innovations:
- Silence Penalty: < 3 claims -> BF = 0
- Zero Tolerance I/O: ANY hallucinated I/O -> CF-02
- Deterministic Synonyms: Smart matching without embeddings
- Boundary Testing: Min/Mid/Max for calculations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
import time

from .config_v231 import V231_CONFIG
from .structural_v231 import StructuralEvaluatorV231, StructuralResult
from .documentation_v231 import DocumentationEvaluatorV231, DocumentationResult
from .behavioral_v231 import BehavioralEvaluatorV231, BehavioralResult
from .critical_failures_v231 import CriticalFailureDetectorV231, CriticalFailure
from .scoring_v231 import ScoringEngineV231, EvaluationResultV231


logger = logging.getLogger(__name__)


@dataclass
class V231EvaluationContext:
    """Context for V2.3.1 evaluation."""
    task_id: str
    model: str
    documentation: str
    source_code: str
    ground_truth: Dict
    start_time: float = field(default_factory=time.time)


class EvaluatorV231:
    """
    V2.3.1 Evaluation Orchestrator
    
    This is the main entry point for V2.3.1 evaluation.
    
    Usage:
        evaluator = EvaluatorV231()
        result = evaluator.evaluate(
            task_id="LCB-T1-001",
            model="gpt-4o",
            documentation="...",
            source_code="...",
            ground_truth={...}
        )
    """
    
    def __init__(self, llm_client=None, executor=None):
        """
        Initialize V2.3.1 evaluator.
        
        Args:
            llm_client: Optional LLM client for claim extraction fallback
            executor: Optional COBOL executor for behavioral testing
        """
        self.structural = StructuralEvaluatorV231()
        self.documentation = DocumentationEvaluatorV231()
        self.behavioral = BehavioralEvaluatorV231(llm_client, executor)
        self.cf_detector = CriticalFailureDetectorV231()
        self.scoring = ScoringEngineV231()
        
        self.llm = llm_client
        self.executor = executor
        
        logger.info("V2.3.1 Evaluator initialized")
    
    def evaluate(
        self,
        task_id: str,
        model: str,
        documentation: str,
        source_code: str,
        ground_truth: Dict,
        **kwargs
    ) -> EvaluationResultV231:
        """
        Perform complete V2.3.1 evaluation.
        
        Args:
            task_id: Task identifier
            model: Model name being evaluated
            documentation: AI-generated documentation
            source_code: Original COBOL source code
            ground_truth: Ground truth data
            
        Returns:
            Complete EvaluationResultV231
        """
        logger.info(f"Starting V2.3.1 evaluation for {task_id}")
        start_time = time.time()
        
        logger.debug(f"\n{'='*60}")
        logger.debug(f"V2.3.1 EVALUATION: {task_id}")
        logger.debug(f"{'='*60}")
        
        # =====================================================================
        # Track 1: Structural Completeness (30%)
        # =====================================================================
        logger.debug(f"\n{'-'*40}")
        logger.debug("TRACK 1: STRUCTURAL COMPLETENESS (30%)")
        logger.debug(f"{'-'*40}")
        
        sc_result = self.structural.evaluate(documentation, ground_truth)
        sc_score = sc_result.score
        
        logger.debug(f"\n-> SC Score: {sc_score*100:.1f}%")
        for key, value in sc_result.breakdown.items():
            logger.debug(f"  * {key}: {value*100:.1f}%")
        
        # =====================================================================
        # Track 2: Documentation Quality (20%)
        # =====================================================================
        logger.debug(f"\n{'-'*40}")
        logger.debug("TRACK 2: DOCUMENTATION QUALITY (20%)")
        logger.debug(f"{'-'*40}")
        
        dq_result = self.documentation.evaluate(documentation, source_code)
        dq_score = dq_result.score
        
        logger.debug(f"\n-> DQ Score: {dq_score*100:.1f}%")
        for key, value in dq_result.breakdown.items():
            logger.debug(f"  * {key}: {value*100:.1f}%")
        
        # =====================================================================
        # Track 3: Behavioral Fidelity (50%)
        # =====================================================================
        logger.debug(f"\n{'-'*40}")
        logger.debug("TRACK 3: BEHAVIORAL FIDELITY (50%)")
        logger.debug(f"{'-'*40}")
        
        bf_result = self.behavioral.evaluate(documentation, source_code, ground_truth)
        bf_score = bf_result.score
        
        logger.debug(f"\n-> BF Score: {bf_score*100:.1f}%")
        if bf_result.silence_penalty:
            logger.debug("  [!] SILENCE PENALTY APPLIED")
        logger.debug(f"  * Claims: {bf_result.claim_count}")
        logger.debug(f"  * BSM: {bf_result.bsm_matched}/{bf_result.bsm_total}")
        
        # =====================================================================
        # Critical Failure Detection
        # =====================================================================
        logger.debug(f"\n{'-'*40}")
        logger.debug("CRITICAL FAILURE DETECTION")
        logger.debug(f"{'-'*40}")
        
        # Build bf_result dict for CF detector
        bf_result_dict = {
            "claim_score": bf_result.claim_score,
            "bsm_score": bf_result.bsm_score,
            "claims_verified": bf_result.claims_verified,
            "claims_failed": bf_result.claims_failed,
            "bsm_matched": bf_result.bsm_matched,
            "bsm_total": bf_result.bsm_total,
        }
        
        critical_failures = self.cf_detector.detect_all(
            documentation=documentation,
            ground_truth=ground_truth,
            sc_score=sc_score,
            bf_result=bf_result_dict,
        )
        
        if critical_failures:
            logger.debug(f"\n[!] {len(critical_failures)} CRITICAL FAILURE(S) DETECTED:")
            for cf in critical_failures:
                logger.debug(f"  * {cf.cf_id}: {cf.description}")
        else:
            logger.debug("\n[OK] No critical failures")
        
        # =====================================================================
        # Final Scoring
        # =====================================================================
        logger.debug(f"\n{'-'*40}")
        logger.debug("FINAL SCORING")
        logger.debug(f"{'-'*40}")
        
        result = self.scoring.calculate(
            sc_score=sc_score,
            dq_score=dq_score,
            bf_score=bf_score,
            critical_failures=critical_failures,
            sc_breakdown=sc_result.breakdown,
            dq_breakdown=dq_result.breakdown,
            bf_breakdown=bf_result.breakdown,
        )
        
        elapsed = time.time() - start_time
        
        # Print Summary
        logger.debug(f"\n{'='*60}")
        logger.debug(f"RESULT: {'PASSED' if result.passed else 'FAILED'}")
        logger.debug(f"{'='*60}")
        logger.debug(f"LCB Score: {result.lcb_score:.1f}%")
        logger.debug(f"")
        logger.debug(f"  SC: {sc_score*100:.0f}% (threshold: {V231_CONFIG['thresholds']['sc']*100:.0f}%) {'[OK]' if result.pass_status.sc_met else '[X]'}")
        logger.debug(f"  DQ: {dq_score*100:.0f}% (threshold: {V231_CONFIG['thresholds']['dq']*100:.0f}%) {'[OK]' if result.pass_status.dq_met else '[X]'}")
        logger.debug(f"  BF: {bf_score*100:.0f}% (threshold: {V231_CONFIG['thresholds']['bf']*100:.0f}%) {'[OK]' if result.pass_status.bf_met else '[X]'}")
        logger.debug(f"")
        logger.debug(f"Reason: {result.pass_status.reason}")
        logger.debug(f"Time: {elapsed:.2f}s")
        logger.debug(f"{'='*60}\n")
        
        logger.info(f"V2.3.1 evaluation complete: {result.lcb_score:.1f}% ({'PASSED' if result.passed else 'FAILED'})")
        
        return result


def evaluate_v231(
    task_id: str,
    model: str,
    documentation: str,
    source_code: str,
    ground_truth: Dict,
    llm_client=None,
    executor=None,
    **kwargs
) -> EvaluationResultV231:
    """
    Convenience function for V2.3.1 evaluation.
    
    Args:
        task_id: Task identifier
        model: Model name
        documentation: AI-generated documentation
        source_code: COBOL source code
        ground_truth: Ground truth data
        llm_client: Optional LLM client
        executor: Optional executor
        
    Returns:
        Complete evaluation result
    """
    evaluator = EvaluatorV231(llm_client, executor)
    return evaluator.evaluate(
        task_id=task_id,
        model=model,
        documentation=documentation,
        source_code=source_code,
        ground_truth=ground_truth,
        **kwargs
    )
