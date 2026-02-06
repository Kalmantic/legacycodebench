"""
BFResult: Complete Behavioral Fidelity result with full provenance.

This replaces the simple BehavioralResult from v2.3.1 with full transparency
about HOW the score was calculated.

Every BF score includes:
- mode: How was this verified? (executed/static/error)
- reason: Why this mode? (ibm_construct:CICS, copybook_missing:X.cpy, etc.)
- detailed breakdowns of claims, tests, and BSM
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VerificationMode(Enum):
    """
    How was the BF score calculated?
    
    EXECUTED: Program compiled and tests ran. Score from actual execution.
    STATIC: Program couldn't compile. Score from claim + BSM verification.
    ERROR: Something went wrong. Requires investigation.
    """
    EXECUTED = "executed"
    STATIC = "static"
    ERROR = "error"


@dataclass
class BFResult:
    """
    Complete BF evaluation result with full provenance.
    
    This is the core data structure for BF V3.0. Every field is designed
    for transparency - users should be able to trace any score back to
    exactly how it was calculated.
    
    Example (executed):
        BFResult(
            score=0.78,
            verification_mode=VerificationMode.EXECUTED,
            mode_reason="Compiled and executed successfully",
            tests_passed=12, tests_failed=3, tests_total=15,
            claim_score=0.80, bsm_score=0.80
        )
    
    Example (static):
        BFResult(
            score=0.71,
            verification_mode=VerificationMode.STATIC,
            mode_reason="ibm_construct:CICS",
            compile_error="unknown statement 'EXEC'",
            claims_verified=8, claims_failed=4, claims_total=12,
            claim_score=0.67, bsm_score=0.80
        )
    """
    
    # ==================== CORE SCORE ====================
    score: float  # 0.0 - 1.0, the final BF score
    
    # ==================== VERIFICATION METHOD ====================
    verification_mode: VerificationMode
    mode_reason: str  # Human-readable explanation
    
    # ==================== EXECUTION TRACKING ====================
    execution_attempted: bool = False
    execution_succeeded: bool = False
    compile_error: Optional[str] = None
    runtime_error: Optional[str] = None
    fixable: Optional[bool] = None  # True if issue might be fixable (e.g., missing copybook)
    
    # ==================== EXECUTION DETAILS (mode == EXECUTED) ====================
    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0
    test_details: List[Dict] = field(default_factory=list)
    
    # ==================== STATIC DETAILS (mode == STATIC) ====================
    claims_verified: int = 0
    claims_failed: int = 0
    claims_total: int = 0
    claim_details: List[Dict] = field(default_factory=list)
    
    # ==================== BSM DETAILS (both methods) ====================
    bsm_matched: int = 0
    bsm_total: int = 0
    bsm_details: List[Dict] = field(default_factory=list)
    
    # ==================== COMPONENT SCORES ====================
    claim_score: float = 0.0
    bsm_score: float = 0.0
    
    # ==================== INVESTIGATION FLAGS ====================
    requires_investigation: bool = False
    investigation_note: Optional[str] = None
    
    # ==================== SILENCE PENALTY ====================
    silence_penalty: bool = False  # True if < min_claims extracted

    # ==================== V2.4.2 HYBRID CLAIM SCORING ====================
    effective_claims: float = 0.0   # verified + partial * 0.5
    claim_target: int = 6           # Target for full score
    scoring_mode: str = "hybrid"    # "hybrid" or "legacy"
    claims_partial: int = 0         # Partial matches (not counted in verified)

    def __post_init__(self):
        """Validate the result after initialization."""
        # Ensure score is in valid range
        if not 0.0 <= self.score <= 1.0:
            logger.warning(f"BFResult score {self.score} out of range, clamping to [0, 1]")
            self.score = max(0.0, min(1.0, self.score))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize for JSON output.
        
        Returns a clean dict suitable for evaluation results JSON.
        Handles None values and enum conversion.
        """
        result = {
            "score": round(self.score, 4),
            "verification_mode": self.verification_mode.value,
            "mode_reason": self.mode_reason,
            
            "execution": {
                "attempted": self.execution_attempted,
                "succeeded": self.execution_succeeded,
                "compile_error": self.compile_error,
                "runtime_error": self.runtime_error,
                "fixable": self.fixable,
            },
            
            "claims": {
                "verified": self.claims_verified,
                "partial": self.claims_partial,
                "failed": self.claims_failed,
                "total": self.claims_total,
                "score": round(self.claim_score, 4),
                "effective": round(self.effective_claims, 2),
                "target": self.claim_target,
                "scoring_mode": self.scoring_mode,
            },
            
            "bsm": {
                "matched": self.bsm_matched,
                "total": self.bsm_total,
                "score": round(self.bsm_score, 4),
            },
            
            "silence_penalty": self.silence_penalty,
        }
        
        # Add test details only if executed
        if self.verification_mode == VerificationMode.EXECUTED:
            result["tests"] = {
                "passed": self.tests_passed,
                "failed": self.tests_failed,
                "total": self.tests_total,
            }
        
        # Add investigation info only if required
        if self.requires_investigation:
            result["investigation"] = {
                "required": True,
                "note": self.investigation_note,
            }
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BFResult':
        """
        Deserialize from JSON dict.
        
        Useful for loading cached results or test fixtures.
        """
        return cls(
            score=data.get("score", 0.0),
            verification_mode=VerificationMode(data.get("verification_mode", "error")),
            mode_reason=data.get("mode_reason", "Unknown"),
            
            execution_attempted=data.get("execution", {}).get("attempted", False),
            execution_succeeded=data.get("execution", {}).get("succeeded", False),
            compile_error=data.get("execution", {}).get("compile_error"),
            runtime_error=data.get("execution", {}).get("runtime_error"),
            fixable=data.get("execution", {}).get("fixable"),
            
            tests_passed=data.get("tests", {}).get("passed", 0),
            tests_failed=data.get("tests", {}).get("failed", 0),
            tests_total=data.get("tests", {}).get("total", 0),
            
            claims_verified=data.get("claims", {}).get("verified", 0),
            claims_partial=data.get("claims", {}).get("partial", 0),
            claims_failed=data.get("claims", {}).get("failed", 0),
            claims_total=data.get("claims", {}).get("total", 0),
            claim_score=data.get("claims", {}).get("score", 0.0),
            effective_claims=data.get("claims", {}).get("effective", 0.0),
            claim_target=data.get("claims", {}).get("target", 6),
            scoring_mode=data.get("claims", {}).get("scoring_mode", "hybrid"),
            
            bsm_matched=data.get("bsm", {}).get("matched", 0),
            bsm_total=data.get("bsm", {}).get("total", 0),
            bsm_score=data.get("bsm", {}).get("score", 0.0),
            
            requires_investigation=data.get("investigation", {}).get("required", False),
            investigation_note=data.get("investigation", {}).get("note"),
            
            silence_penalty=data.get("silence_penalty", False),
        )
    
    def summary(self) -> str:
        """
        One-line summary for logging.
        
        Returns:
            e.g., "BF: 78.0% (executed, 12/15 tests)"
            e.g., "BF: 71.0% (static:CICS, 8/12 claims)"
        """
        if self.verification_mode == VerificationMode.EXECUTED:
            return f"BF: {self.score*100:.1f}% (executed, {self.tests_passed}/{self.tests_total} tests)"
        elif self.verification_mode == VerificationMode.STATIC:
            reason_short = self.mode_reason.split(":")[0] if ":" in self.mode_reason else self.mode_reason
            return f"BF: {self.score*100:.1f}% (static:{reason_short}, {self.claims_verified}/{self.claims_total} claims)"
        else:
            return f"BF: {self.score*100:.1f}% (error: {self.mode_reason[:30]})"


# ==================== FACTORY FUNCTIONS ====================

def create_executed_result(
    score: float,
    tests_passed: int,
    tests_failed: int,
    claim_score: float,
    bsm_score: float,
    bsm_matched: int,
    bsm_total: int,
    test_details: List[Dict] = None,
    bsm_details: List[Dict] = None,
) -> BFResult:
    """
    Factory for execution-verified results.
    
    Use this when compilation and execution both succeeded.
    """
    return BFResult(
        score=score,
        verification_mode=VerificationMode.EXECUTED,
        mode_reason="Compiled and executed successfully",
        
        execution_attempted=True,
        execution_succeeded=True,
        
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        tests_total=tests_passed + tests_failed,
        test_details=test_details or [],
        
        claims_verified=tests_passed,  # For executed, claims verified = tests passed
        claims_failed=tests_failed,
        claims_total=tests_passed + tests_failed,
        claim_score=claim_score,
        
        bsm_matched=bsm_matched,
        bsm_total=bsm_total,
        bsm_score=bsm_score,
        bsm_details=bsm_details or [],
    )


def create_static_result(
    score: float,
    reason: str,
    claims_verified: int,
    claims_failed: int,
    claims_total: int,  # Added: actual total, not calculated
    claim_score: float,
    bsm_score: float,
    bsm_matched: int,
    bsm_total: int,
    compile_error: str = None,
    fixable: bool = None,
    claim_details: List[Dict] = None,
    bsm_details: List[Dict] = None,
    # V2.4.2: Hybrid claim scoring parameters
    claims_partial: int = 0,
    effective_claims: float = 0.0,
    claim_target: int = 6,
    scoring_mode: str = "hybrid",
) -> BFResult:
    """
    Factory for static-verified results.

    Use this when compilation failed but we can still verify via static analysis.

    V2.4.2: Added hybrid claim scoring parameters:
        claims_partial: Number of partial matches
        effective_claims: verified + partial * 0.5
        claim_target: Target for full score (default: 6)
        scoring_mode: "hybrid" or "legacy"
    """
    return BFResult(
        score=score,
        verification_mode=VerificationMode.STATIC,
        mode_reason=reason,

        execution_attempted=True if compile_error else False,
        execution_succeeded=False,
        compile_error=compile_error,
        fixable=fixable,

        claims_verified=claims_verified,
        claims_partial=claims_partial,
        claims_failed=claims_failed,
        claims_total=claims_total,  # Use actual total passed in
        claim_details=claim_details or [],
        claim_score=claim_score,

        # V2.4.2: Hybrid scoring
        effective_claims=effective_claims,
        claim_target=claim_target,
        scoring_mode=scoring_mode,

        bsm_matched=bsm_matched,
        bsm_total=bsm_total,
        bsm_score=bsm_score,
        bsm_details=bsm_details or [],
    )


def create_error_result(
    reason: str,
    error: str = None,
    investigation_note: str = None,
) -> BFResult:
    """
    Factory for error results.
    
    Use this when something went wrong and we can't provide a meaningful score.
    """
    return BFResult(
        score=0.0,
        verification_mode=VerificationMode.ERROR,
        mode_reason=reason,
        
        execution_attempted=True,
        execution_succeeded=False,
        compile_error=error,
        
        requires_investigation=True,
        investigation_note=investigation_note or reason,
    )
