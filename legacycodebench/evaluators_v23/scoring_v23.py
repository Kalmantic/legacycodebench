"""
Scoring Engine V2.3

Calculates the unified LCB Score using the 40/25/35 weighting.

Formula:
    LCB = (0.40 × Comprehension) + (0.25 × Documentation) + (0.35 × Behavioral)

Pass criteria:
    - Comprehension >= 70%
    - Documentation >= 60%
    - Behavioral >= 50%
    - No Critical Failures
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

from .config_v23 import V23_CONFIG
from .anti_gaming import AntiGamingResult


@dataclass
class LCBScoreV23:
    """Complete LCB Score result for V2.3."""
    # Main scores
    overall: float                      # 0-100 scale
    comprehension: float                # 0-1 scale
    documentation: float                # 0-1 scale
    behavioral: float                   # 0-1 scale
    
    # Pass/Fail
    passed: bool
    failure_reasons: List[str] = field(default_factory=list)
    
    # Critical failures
    critical_failures: List[str] = field(default_factory=list)
    
    # Anti-gaming
    anti_gaming_penalties: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    version: str = "2.3.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "scores": {
                "overall": round(self.overall, 2),
                "comprehension": round(self.comprehension * 100, 2),
                "documentation": round(self.documentation * 100, 2),
                "behavioral": round(self.behavioral * 100, 2)
            },
            "passed": self.passed,
            "failure_reasons": self.failure_reasons,
            "critical_failures": self.critical_failures,
            "anti_gaming": self.anti_gaming_penalties,
            "thresholds": {
                "comprehension": V23_CONFIG["thresholds"]["comprehension"] * 100,
                "documentation": V23_CONFIG["thresholds"]["documentation"] * 100,
                "behavioral": V23_CONFIG["thresholds"]["behavioral"] * 100
            },
            "weights": {
                "comprehension": V23_CONFIG["weights"]["comprehension"] * 100,
                "documentation": V23_CONFIG["weights"]["documentation"] * 100,
                "behavioral": V23_CONFIG["weights"]["behavioral"] * 100
            }
        }


class ScoringEngineV23:
    """
    V2.3 Scoring Engine
    
    Calculates unified LCB score with:
    - 40/25/35 weighting (Comprehension/Documentation/Behavioral)
    - Anti-gaming penalties
    - Pass/fail determination
    - Critical failure handling
    """
    
    def __init__(self):
        self.weights = V23_CONFIG["weights"]
        self.thresholds = V23_CONFIG["thresholds"]
        self.anti_gaming_config = V23_CONFIG["anti_gaming"]
    
    def calculate(
        self,
        comprehension_result,
        documentation_result,
        behavioral_result,
        cf_result,
        anti_gaming_result: Optional[AntiGamingResult] = None
    ) -> LCBScoreV23:
        """
        Calculate unified LCB score.
        
        Args:
            comprehension_result: ComprehensionResult or dict with 'overall'
            documentation_result: DocumentationResult or dict with 'overall'
            behavioral_result: BehavioralResult or dict with 'overall'
            cf_result: CriticalFailureResult or dict with 'any_triggered'
            anti_gaming_result: Optional AntiGamingResult
            
        Returns:
            LCBScoreV23 with complete scoring details
        """
        # Extract scores
        comp_score = self._get_score(comprehension_result)
        doc_score = self._get_score(documentation_result)
        beh_score = self._get_score(behavioral_result)
        
        # Apply anti-gaming penalties to comprehension
        penalties = {}
        if anti_gaming_result:
            comp_score, penalties = self._apply_anti_gaming(
                comp_score, anti_gaming_result
            )
        
        # Check for critical failures
        cf_triggered = self._check_cf(cf_result)
        cf_list = self._get_cf_list(cf_result)
        
        # Calculate weighted score
        if cf_triggered:
            # Critical failure: score = 0
            overall = 0.0
            passed = False
            failure_reasons = ["Critical failure detected"]
        else:
            # Normal scoring
            weighted = (
                self.weights["comprehension"] * comp_score +
                self.weights["documentation"] * doc_score +
                self.weights["behavioral"] * beh_score
            )
            overall = weighted * 100  # Convert to 0-100 scale
            
            # Check pass/fail
            passed, failure_reasons = self._check_pass(
                comp_score, doc_score, beh_score
            )
        
        return LCBScoreV23(
            overall=overall,
            comprehension=comp_score,
            documentation=doc_score,
            behavioral=beh_score,
            passed=passed,
            failure_reasons=failure_reasons,
            critical_failures=cf_list,
            anti_gaming_penalties=penalties
        )
    
    def _get_score(self, result) -> float:
        """Extract score from result object or dict."""
        if hasattr(result, 'overall'):
            return result.overall
        elif isinstance(result, dict):
            return result.get('overall', 0.0)
        else:
            return 0.0
    
    def _apply_anti_gaming(
        self,
        score: float,
        anti_gaming: AntiGamingResult
    ) -> tuple:
        """Apply anti-gaming penalties to comprehension score."""
        penalties = {}
        adjusted_score = score
        
        # Keyword stuffing penalty
        if anti_gaming.keyword_stuffing_score > self.anti_gaming_config["keyword_stuffing_threshold"]:
            penalty = anti_gaming.penalties_applied.get("keyword_stuffing", 0)
            adjusted_score *= (1 - penalty)
            penalties["keyword_stuffing"] = penalty
        
        # Parroting penalty
        if anti_gaming.parroting_score > self.anti_gaming_config["parroting_threshold"]:
            penalty = anti_gaming.penalties_applied.get("parroting", 0)
            adjusted_score *= (1 - penalty)
            penalties["parroting"] = penalty
        
        # Low abstraction penalty
        if anti_gaming.abstraction_score < self.anti_gaming_config["abstraction_minimum"]:
            penalty = self.anti_gaming_config["abstraction_penalty"]
            adjusted_score *= (1 - penalty)
            penalties["low_abstraction"] = penalty
        
        return max(0, min(1, adjusted_score)), penalties
    
    def _check_cf(self, cf_result) -> bool:
        """Check if any critical failure was triggered."""
        if hasattr(cf_result, 'any_triggered'):
            return cf_result.any_triggered
        elif isinstance(cf_result, dict):
            return cf_result.get('any_triggered', False)
        else:
            return False
    
    def _get_cf_list(self, cf_result) -> List[str]:
        """Get list of triggered critical failures."""
        if hasattr(cf_result, 'triggered_list'):
            return cf_result.triggered_list
        elif isinstance(cf_result, dict):
            return cf_result.get('triggered', [])
        else:
            return []
    
    def _check_pass(
        self,
        comp: float,
        doc: float,
        beh: float
    ) -> tuple:
        """Check if task passes based on thresholds."""
        failure_reasons = []
        
        if comp < self.thresholds["comprehension"]:
            failure_reasons.append(
                f"Comprehension {comp*100:.1f}% < {self.thresholds['comprehension']*100}%"
            )
        
        if doc < self.thresholds["documentation"]:
            failure_reasons.append(
                f"Documentation {doc*100:.1f}% < {self.thresholds['documentation']*100}%"
            )
        
        if beh < self.thresholds["behavioral"]:
            failure_reasons.append(
                f"Behavioral {beh*100:.1f}% < {self.thresholds['behavioral']*100}%"
            )
        
        passed = len(failure_reasons) == 0
        return passed, failure_reasons
    
    def format_result(self, score: LCBScoreV23, verbose: bool = False) -> str:
        """Format score result as string for CLI output."""
        status = "✓ PASSED" if score.passed else "✗ FAILED"
        
        lines = [
            f"─" * 60,
            f"LCB Score: {score.overall:.1f}/100  {status}",
            f"─" * 60,
            f"",
            f"Track Scores:",
            f"  Comprehension: {score.comprehension*100:5.1f}% (weight: 40%, threshold: 70%)",
            f"  Documentation: {score.documentation*100:5.1f}% (weight: 25%, threshold: 60%)",
            f"  Behavioral:    {score.behavioral*100:5.1f}% (weight: 35%, threshold: 50%)",
        ]
        
        if score.critical_failures:
            lines.append("")
            lines.append("Critical Failures:")
            for cf in score.critical_failures:
                lines.append(f"  ✗ {cf}")
        
        if score.failure_reasons and not score.critical_failures:
            lines.append("")
            lines.append("Failure Reasons:")
            for reason in score.failure_reasons:
                lines.append(f"  • {reason}")
        
        if score.anti_gaming_penalties and verbose:
            lines.append("")
            lines.append("Anti-Gaming Penalties:")
            for name, penalty in score.anti_gaming_penalties.items():
                lines.append(f"  • {name}: -{penalty*100:.1f}%")
        
        lines.append(f"─" * 60)
        
        return "\n".join(lines)
