"""
Critical Failure Detector V2.3

Detects all 6 critical failures that result in automatic task failure.

CF-01: Missing Core Calculations (0% CRITICAL rules)
CF-02: Hallucinated Logic (describes non-existent logic)
CF-03: Wrong Transformation (>50% outputs differ)
CF-04: Missing Error Handling (error handlers undocumented)
CF-05: BSM Specification Failure (>50% external calls wrong)
CF-06: Semantic Contradiction (documentation contradicts source)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .config_v23 import V23_CONFIG


@dataclass
class CriticalFailureResult:
    """Result of critical failure detection."""
    cf01_triggered: bool = False    # Missing Core Calculations
    cf02_triggered: bool = False    # Hallucinated Logic
    cf03_triggered: bool = False    # Wrong Transformation
    cf04_triggered: bool = False    # Missing Error Handling
    cf05_triggered: bool = False    # BSM Specification Failure
    cf06_triggered: bool = False    # Semantic Contradiction
    triggered_list: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)
    
    @property
    def any_triggered(self) -> bool:
        """Check if any critical failure was triggered."""
        return len(self.triggered_list) > 0
    
    @property
    def count(self) -> int:
        """Count of triggered critical failures."""
        return len(self.triggered_list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "any_triggered": self.any_triggered,
            "count": self.count,
            "triggered": self.triggered_list,
            "cf01": self.cf01_triggered,
            "cf02": self.cf02_triggered,
            "cf03": self.cf03_triggered,
            "cf04": self.cf04_triggered,
            "cf05": self.cf05_triggered,
            "cf06": self.cf06_triggered,
            "details": self.details
        }


class CriticalFailureDetectorV23:
    """
    Detects all 6 critical failures for V2.3.
    
    Any critical failure results in:
    - Task marked as FAILED
    - Task score set to 0
    """
    
    def __init__(self):
        self.cf_config = V23_CONFIG["critical_failures"]
    
    def detect(
        self,
        comprehension_result: Dict,
        documentation_result: Dict,
        behavioral_result: Dict,
        ground_truth: Dict,
        documentation: str = None
    ) -> CriticalFailureResult:
        """
        Detect all critical failures.
        
        Args:
            comprehension_result: Result from ComprehensionEvaluatorV23
            documentation_result: Result from DocumentationEvaluatorV23
            behavioral_result: Result from BehavioralEvaluatorV23
            ground_truth: Ground truth data for the task
            documentation: Optional - the documentation text
            
        Returns:
            CriticalFailureResult with all CF flags
        """
        result = CriticalFailureResult()
        
        # CF-01: Missing Core Calculations
        cf01, cf01_detail = self._check_cf01(comprehension_result, ground_truth)
        result.cf01_triggered = cf01
        if cf01:
            result.triggered_list.append("CF-01: Missing Core Calculations")
            result.details["cf01"] = cf01_detail
        
        # CF-02: Hallucinated Logic
        cf02, cf02_detail = self._check_cf02(comprehension_result)
        result.cf02_triggered = cf02
        if cf02:
            result.triggered_list.append("CF-02: Hallucinated Logic")
            result.details["cf02"] = cf02_detail
        
        # CF-03: Wrong Transformation
        cf03, cf03_detail = self._check_cf03(behavioral_result)
        result.cf03_triggered = cf03
        if cf03:
            result.triggered_list.append("CF-03: Wrong Transformation")
            result.details["cf03"] = cf03_detail
        
        # CF-04: Missing Error Handling
        cf04, cf04_detail = self._check_cf04(comprehension_result, ground_truth)
        result.cf04_triggered = cf04
        if cf04:
            result.triggered_list.append("CF-04: Missing Error Handling")
            result.details["cf04"] = cf04_detail
        
        # CF-05: BSM Specification Failure
        cf05, cf05_detail = self._check_cf05(behavioral_result)
        result.cf05_triggered = cf05
        if cf05:
            result.triggered_list.append("CF-05: BSM Specification Failure")
            result.details["cf05"] = cf05_detail
        
        # CF-06: Semantic Contradiction
        cf06, cf06_detail = self._check_cf06(comprehension_result)
        result.cf06_triggered = cf06
        if cf06:
            result.triggered_list.append("CF-06: Semantic Contradiction")
            result.details["cf06"] = cf06_detail
        
        return result
    
    def _check_cf01(
        self,
        comprehension_result: Dict,
        ground_truth: Dict
    ) -> tuple:
        """
        CF-01: Missing Core Calculations
        
        Triggers when 0% of CRITICAL business rules are documented.
        """
        # Get from comprehension result
        if hasattr(comprehension_result, 'cf01_triggered'):
            return comprehension_result.cf01_triggered, {
                "missing": getattr(comprehension_result, 'missing_critical_rules', [])
            }
        
        # Check from result dict
        if isinstance(comprehension_result, dict):
            details = comprehension_result.get("details", {})
            br_details = details.get("business_rules", {})
            
            critical_count = br_details.get("critical_count", 0)
            critical_match_rate = br_details.get("critical_match_rate", 1.0)
            
            # Trigger if critical rules exist but none matched
            triggered = critical_count > 0 and critical_match_rate == 0
            
            return triggered, {
                "critical_count": critical_count,
                "critical_match_rate": critical_match_rate,
                "missing": br_details.get("missing_critical", [])
            }
        
        return False, {}
    
    def _check_cf02(self, comprehension_result: Dict) -> tuple:
        """
        CF-02: Hallucinated Logic
        
        Triggers when documentation describes logic not present in source.
        """
        if hasattr(comprehension_result, 'cf02_triggered'):
            return comprehension_result.cf02_triggered, {
                "hallucinations": getattr(comprehension_result, 'hallucinations', [])
            }
        
        if isinstance(comprehension_result, dict):
            hallucinations = comprehension_result.get("hallucinations", [])
            triggered = len(hallucinations) > 0
            
            return triggered, {"hallucinations": hallucinations}
        
        return False, {}
    
    def _check_cf03(self, behavioral_result: Dict) -> tuple:
        """
        CF-03: Wrong Transformation
        
        Triggers when >50% of test outputs differ from expected.
        """
        if hasattr(behavioral_result, 'cf03_triggered'):
            return behavioral_result.cf03_triggered, {}
        
        if isinstance(behavioral_result, dict):
            # Check test failure rate
            details = behavioral_result.get("details", {})
            pure_details = details.get("pure", {})
            
            test_failure_rate = pure_details.get("test_failure_rate", 0)
            triggered = test_failure_rate > 0.5
            
            return triggered, {
                "test_failure_rate": test_failure_rate,
                "threshold": 0.5
            }
        
        return False, {}
    
    def _check_cf04(
        self,
        comprehension_result: Dict,
        ground_truth: Dict
    ) -> tuple:
        """
        CF-04: Missing Error Handling
        
        Triggers when error handlers exist in code but are not documented.
        """
        # Get error handlers from ground truth
        error_handlers = ground_truth.get("error_handlers", [])
        
        if not error_handlers:
            return False, {"message": "No error handlers in source"}
        
        # Check if documented
        if isinstance(comprehension_result, dict):
            details = comprehension_result.get("details", {})
            cf04_triggered = details.get("cf04_triggered", False)
            
            return cf04_triggered, {
                "error_handlers_count": len(error_handlers),
                "documented": not cf04_triggered
            }
        
        return False, {}
    
    def _check_cf05(self, behavioral_result: Dict) -> tuple:
        """
        CF-05: BSM Specification Failure
        
        Triggers when >50% of external calls are incorrectly specified.
        """
        if hasattr(behavioral_result, 'cf05_triggered'):
            return behavioral_result.cf05_triggered, {
                "bsm_score": getattr(behavioral_result, 'bsm_score', 1.0)
            }
        
        if isinstance(behavioral_result, dict):
            bsm_score = behavioral_result.get("bsm_score", 1.0)
            # Also check in details
            details = behavioral_result.get("details", {})
            mixed_details = details.get("mixed", {})
            cf05_from_details = mixed_details.get("cf05_triggered", False)
            
            triggered = bsm_score < 0.5 or cf05_from_details
            
            return triggered, {
                "bsm_score": bsm_score,
                "threshold": 0.5
            }
        
        return False, {}
    
    def _check_cf06(self, comprehension_result: Dict) -> tuple:
        """
        CF-06: Semantic Contradiction
        
        Triggers when documentation contradicts source logic.
        """
        if hasattr(comprehension_result, 'cf06_triggered'):
            return comprehension_result.cf06_triggered, {
                "contradictions": getattr(comprehension_result, 'contradictions', [])
            }
        
        if isinstance(comprehension_result, dict):
            contradictions = comprehension_result.get("contradictions", [])
            triggered = len(contradictions) > 0
            
            return triggered, {"contradictions": contradictions}
        
        return False, {}
    
    def get_failure_descriptions(self) -> Dict[str, str]:
        """Get descriptions for all critical failures."""
        return {
            cf_id: config["name"]
            for cf_id, config in self.cf_config.items()
        }
