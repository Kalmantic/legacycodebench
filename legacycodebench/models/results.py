"""
Evaluation Result Models for LegacyCodeBench V2.4

Specification Reference: TDD_V2.4.md Section 2.4
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .enums import Language, VerificationMode, CompileFailureReason


@dataclass
class TrackScore:
    """
    Score for a single evaluation track (SC, DQ, or BF).
    """
    score: float                      # 0.0 - 1.0
    breakdown: Dict[str, float] = field(default_factory=dict)  # Component scores
    details: Dict[str, Any] = field(default_factory=dict)      # Additional info


@dataclass
class BFResult:
    """
    Behavioral Fidelity result with full provenance.
    
    Tracks every aspect of how the BF score was computed,
    enabling complete reproducibility and transparency.
    """
    score: float                      # Final BF score (0.0 - 1.0)
    
    # Verification method (required) - how claims were verified
    verification_mode: VerificationMode
    mode_reason: str                  # Human-readable explanation
    
    # Compilation tracking
    compile_attempted: bool = True
    compile_succeeded: bool = False
    compile_error: Optional[str] = None
    failure_reason: CompileFailureReason = CompileFailureReason.NONE
    fixable: Optional[bool] = None    # True if likely fixable (missing copybook)
    
    # Execution tracking (when mode == EXECUTED)
    execution_attempted: bool = False
    execution_succeeded: bool = False
    runtime_error: Optional[str] = None
    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0
    
    # Claim tracking (both modes)
    claims_extracted: int = 0
    claims_verified: int = 0
    claims_failed: int = 0
    claim_extraction_method: str = "regex"  # "regex" or "llm_fallback"
    
    # BSM tracking (both modes)
    bsm_matched: int = 0
    bsm_total: int = 0
    
    # Component scores
    claim_score: float = 0.0
    bsm_score: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "score": self.score,
            "verification_mode": self.verification_mode.value,
            "mode_reason": self.mode_reason,
            "compilation": {
                "attempted": self.compile_attempted,
                "succeeded": self.compile_succeeded,
                "error": self.compile_error,
                "failure_reason": self.failure_reason.value,
                "fixable": self.fixable,
            },
            "execution": {
                "attempted": self.execution_attempted,
                "succeeded": self.execution_succeeded,
                "runtime_error": self.runtime_error,
            },
            "tests": {
                "passed": self.tests_passed,
                "failed": self.tests_failed,
                "total": self.tests_total,
            },
            "claims": {
                "extracted": self.claims_extracted,
                "verified": self.claims_verified,
                "failed": self.claims_failed,
                "extraction_method": self.claim_extraction_method,
            },
            "bsm": {
                "matched": self.bsm_matched,
                "total": self.bsm_total,
            },
            "component_scores": {
                "claim_score": self.claim_score,
                "bsm_score": self.bsm_score,
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "BFResult":
        """Create BFResult from dictionary."""
        compilation = data.get("compilation", {})
        execution = data.get("execution", {})
        tests = data.get("tests", {})
        claims = data.get("claims", {})
        bsm = data.get("bsm", {})
        component_scores = data.get("component_scores", {})
        
        return cls(
            score=data.get("score", 0.0),
            verification_mode=VerificationMode(data.get("verification_mode", "static")),
            mode_reason=data.get("mode_reason", ""),
            compile_attempted=compilation.get("attempted", True),
            compile_succeeded=compilation.get("succeeded", False),
            compile_error=compilation.get("error"),
            failure_reason=CompileFailureReason(compilation.get("failure_reason", "none")),
            fixable=compilation.get("fixable"),
            execution_attempted=execution.get("attempted", False),
            execution_succeeded=execution.get("succeeded", False),
            runtime_error=execution.get("runtime_error"),
            tests_passed=tests.get("passed", 0),
            tests_failed=tests.get("failed", 0),
            tests_total=tests.get("total", 0),
            claims_extracted=claims.get("extracted", 0),
            claims_verified=claims.get("verified", 0),
            claims_failed=claims.get("failed", 0),
            claim_extraction_method=claims.get("extraction_method", "regex"),
            bsm_matched=bsm.get("matched", 0),
            bsm_total=bsm.get("total", 0),
            claim_score=component_scores.get("claim_score", 0.0),
            bsm_score=component_scores.get("bsm_score", 0.0),
        )


@dataclass
class CriticalFailure:
    """
    A critical failure that invalidates the entire task score.
    
    Critical failures represent documentation defects so severe
    that the documentation is unusable.
    """
    cf_id: str                        # CF-01, CF-02, etc.
    name: str                         # Human-readable name
    description: str                  # What went wrong
    evidence: Dict[str, Any] = field(default_factory=dict)  # Specific evidence


@dataclass
class EvaluationResult:
    """
    Complete evaluation result for a task.
    
    Contains all track scores, critical failures, and provenance.
    """
    task_id: str
    language: Language
    
    # Track scores
    structural_completeness: TrackScore
    documentation_quality: TrackScore
    behavioral_fidelity: BFResult
    
    # Final score (0.0 - 1.0)
    lcb_score: float
    passed: bool
    pass_reason: str                  # "All thresholds met" or failure reason
    
    # Critical failures
    critical_failures: List[CriticalFailure] = field(default_factory=list)
    
    # Audit trail
    llm_calls: List[Dict] = field(default_factory=list)  # For transparency
    evaluation_time_ms: int = 0
    evaluator_version: str = "v2.4"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "language": self.language.value,
            "lcb_score": self.lcb_score,
            "passed": self.passed,
            "pass_reason": self.pass_reason,
            "tracks": {
                "structural_completeness": {
                    "score": self.structural_completeness.score,
                    "breakdown": self.structural_completeness.breakdown,
                },
                "documentation_quality": {
                    "score": self.documentation_quality.score,
                    "breakdown": self.documentation_quality.breakdown,
                },
                "behavioral_fidelity": self.behavioral_fidelity.to_dict(),
            },
            "critical_failures": [
                {
                    "cf_id": cf.cf_id,
                    "name": cf.name,
                    "description": cf.description,
                    "evidence": cf.evidence,
                }
                for cf in self.critical_failures
            ],
            "audit": {
                "llm_calls": self.llm_calls,
                "evaluation_time_ms": self.evaluation_time_ms,
                "evaluator_version": self.evaluator_version,
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "EvaluationResult":
        """Create EvaluationResult from dictionary."""
        tracks = data.get("tracks", {})
        sc = tracks.get("structural_completeness", {})
        dq = tracks.get("documentation_quality", {})
        bf = tracks.get("behavioral_fidelity", {})
        audit = data.get("audit", {})
        
        return cls(
            task_id=data["task_id"],
            language=Language(data.get("language", "cobol")),
            structural_completeness=TrackScore(
                score=sc.get("score", 0.0),
                breakdown=sc.get("breakdown", {}),
            ),
            documentation_quality=TrackScore(
                score=dq.get("score", 0.0),
                breakdown=dq.get("breakdown", {}),
            ),
            behavioral_fidelity=BFResult.from_dict(bf) if bf else BFResult(
                score=0.0,
                verification_mode=VerificationMode.ERROR,
                mode_reason="No BF data",
            ),
            lcb_score=data.get("lcb_score", 0.0),
            passed=data.get("passed", False),
            pass_reason=data.get("pass_reason", "Unknown"),
            critical_failures=[
                CriticalFailure(
                    cf_id=cf["cf_id"],
                    name=cf["name"],
                    description=cf.get("description", ""),
                    evidence=cf.get("evidence", {}),
                )
                for cf in data.get("critical_failures", [])
            ],
            llm_calls=audit.get("llm_calls", []),
            evaluation_time_ms=audit.get("evaluation_time_ms", 0),
            evaluator_version=audit.get("evaluator_version", "v2.4"),
        )
