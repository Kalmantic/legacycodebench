"""
Scoring Engine V2.3.1

Calculates final LCB Score from track scores.

Formula: LCB = 0.30×SC + 0.20×DQ + 0.50×BF

Pass criteria:
- SC ≥ 60%
- DQ ≥ 50%
- BF ≥ 55%
- No Critical Failures
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging

from .config_v231 import V231_CONFIG
from .critical_failures_v231 import CriticalFailure


logger = logging.getLogger(__name__)


@dataclass
class PassStatus:
    """Pass/fail determination with reason."""
    passed: bool
    reason: str
    sc_met: bool = True
    dq_met: bool = True
    bf_met: bool = True
    has_critical_failure: bool = False


@dataclass
class EvaluationResultV231:
    """Complete V2.3.1 evaluation result."""
    lcb_score: float
    passed: bool
    pass_status: PassStatus
    
    # Track scores
    sc_score: float = 0.0
    dq_score: float = 0.0
    bf_score: float = 0.0
    
    # Critical failures
    critical_failures: List[CriticalFailure] = field(default_factory=list)
    
    # Detailed breakdowns
    sc_breakdown: Dict = field(default_factory=dict)
    dq_breakdown: Dict = field(default_factory=dict)
    bf_breakdown: Dict = field(default_factory=dict)
    
    # Metadata
    version: str = "2.3.1"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "lcb_score": round(self.lcb_score, 2),
            "passed": self.passed,
            "pass_status": {
                "passed": self.pass_status.passed,
                "reason": self.pass_status.reason,
                "sc_met": self.pass_status.sc_met,
                "dq_met": self.pass_status.dq_met,
                "bf_met": self.pass_status.bf_met,
                "has_critical_failure": self.pass_status.has_critical_failure,
            },
            "scores": {
                "sc": round(self.sc_score * 100, 1),
                "dq": round(self.dq_score * 100, 1),
                "bf": round(self.bf_score * 100, 1),
                "overall": round(self.lcb_score, 1),
            },
            "critical_failures": [
                {"id": cf.cf_id, "name": cf.name, "description": cf.description}
                for cf in self.critical_failures
            ],
            "breakdown": {
                "sc": self.sc_breakdown,
                "dq": self.dq_breakdown,
                "bf": self.bf_breakdown,
            },
        }


class ScoringEngineV231:
    """
    V2.3.1 Scoring Engine
    
    Combines track scores into final LCB Score with pass/fail determination.
    """
    
    def __init__(self):
        self.weights = V231_CONFIG["weights"]
        self.thresholds = V231_CONFIG["thresholds"]
    
    def calculate(
        self,
        sc_score: float,
        dq_score: float,
        bf_score: float,
        critical_failures: List[CriticalFailure],
        sc_breakdown: Dict = None,
        dq_breakdown: Dict = None,
        bf_breakdown: Dict = None,
    ) -> EvaluationResultV231:
        """
        Calculate final LCB Score and pass status.
        
        Args:
            sc_score: Structural completeness score (0-1)
            dq_score: Documentation quality score (0-1)
            bf_score: Behavioral fidelity score (0-1)
            critical_failures: List of detected critical failures
            *_breakdown: Detailed breakdowns for each track
            
        Returns:
            Complete EvaluationResultV231
        """
        logger.debug("Stage 1: Critical Failure Check [...]")
        # Step 1: Check for critical failures
        if critical_failures:
            cf = critical_failures[0]
            logger.warning(f"Critical failure detected: {cf.cf_id}")
            logger.debug(f"Stage 1: Critical Failure {cf.cf_id} - Score = 0 [X]")
            
            return EvaluationResultV231(
                lcb_score=0.0,
                passed=False,
                pass_status=PassStatus(
                    passed=False,
                    reason=f"Critical failure: {cf.name}",
                    has_critical_failure=True,
                ),
                sc_score=sc_score,
                dq_score=dq_score,
                bf_score=bf_score,
                critical_failures=critical_failures,
                sc_breakdown=sc_breakdown or {},
                dq_breakdown=dq_breakdown or {},
                bf_breakdown=bf_breakdown or {},
            )
        logger.debug("Stage 1: No Critical Failures [OK]")
        
        logger.debug("Stage 2: Calculate LCB Score [...]")
        # Step 2: Calculate weighted score
        lcb_score = (
            self.weights["structural_completeness"] * sc_score +
            self.weights["documentation_quality"] * dq_score +
            self.weights["behavioral_fidelity"] * bf_score
        ) * 100
        logger.debug(f"Stage 2: LCB Score = {lcb_score:.1f} [OK]")
        
        logger.debug("Stage 3: Check Thresholds [...]")
        # Step 3: Check thresholds
        sc_met = sc_score >= self.thresholds["sc"]
        dq_met = dq_score >= self.thresholds["dq"]
        bf_met = bf_score >= self.thresholds["bf"]
        
        passed = sc_met and dq_met and bf_met
        
        if passed:
            reason = "All thresholds met"
            logger.debug("Stage 3: All Thresholds Met [OK]")
        else:
            failed_tracks = []
            if not sc_met:
                failed_tracks.append(f"SC ({sc_score*100:.0f}% < {self.thresholds['sc']*100:.0f}%)")
            if not dq_met:
                failed_tracks.append(f"DQ ({dq_score*100:.0f}% < {self.thresholds['dq']*100:.0f}%)")
            if not bf_met:
                failed_tracks.append(f"BF ({bf_score*100:.0f}% < {self.thresholds['bf']*100:.0f}%)")
            reason = f"Failed thresholds: {', '.join(failed_tracks)}"
            logger.debug(f"Stage 3: Thresholds Not Met [X] ({reason})")
        
        return EvaluationResultV231(
            lcb_score=lcb_score,
            passed=passed,
            pass_status=PassStatus(
                passed=passed,
                reason=reason,
                sc_met=sc_met,
                dq_met=dq_met,
                bf_met=bf_met,
                has_critical_failure=False,
            ),
            sc_score=sc_score,
            dq_score=dq_score,
            bf_score=bf_score,
            critical_failures=[],
            sc_breakdown=sc_breakdown or {},
            dq_breakdown=dq_breakdown or {},
            bf_breakdown=bf_breakdown or {},
        )
