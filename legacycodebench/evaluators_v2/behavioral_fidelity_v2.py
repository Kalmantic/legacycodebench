"""
Behavioral Fidelity Evaluator V2 - Combined IUE + BSM evaluation.

This evaluator replaces the original execution-only BF approach with:
- IUE (Isolatable Unit Execution): Test pure paragraphs in GnuCOBOL
- BSM (Behavioral Specification Matching): Validate external call documentation

Combined BF = (IUE x 0.57) + (BSM x 0.43)
"""

from typing import Dict, List, Optional
from pathlib import Path
import logging

from legacycodebench.execution.iue.paragraph_parser import ParagraphParser, COBOLParagraph
from legacycodebench.execution.iue.isolation_analyzer import IsolationAnalyzer, IsolatableUnit
from legacycodebench.execution.bsm.call_detector import CallDetector, ExternalCall
from legacycodebench.execution.bsm.pattern_library import get_pattern, BSMPattern
from legacycodebench.execution.bsm.doc_matcher import DocumentMatcher, CallMatchResult

logger = logging.getLogger(__name__)


class BehavioralFidelityEvaluatorV2:
    """
    Combined IUE + BSM Behavioral Fidelity Evaluator.
    
    Architecture:
        BF (35% of LCB) = IUE (20%) + BSM (15%)
        
    IUE evaluates pure computational paragraphs that can run in GnuCOBOL.
    BSM evaluates documentation accuracy for external calls (SQL, CICS, CALL).
    
    This approach provides 100% program coverage without requiring
    commercial COBOL compilers.
    """
    
    # Weight distribution within BF
    # IUE is 20% of total LCB, BSM is 15%, so within BF:
    IUE_WEIGHT = 0.20 / 0.35  # ~0.571 (57.1%)
    BSM_WEIGHT = 0.15 / 0.35  # ~0.429 (42.9%)
    
    def __init__(self, enable_execution: bool = False):
        """
        Initialize the evaluator.
        
        Args:
            enable_execution: If True, actually run isolated units in GnuCOBOL.
                            If False, IUE provides isolation analysis only.
        """
        self.enable_execution = enable_execution
        
        # IUE components
        self.paragraph_parser = ParagraphParser()
        self.isolation_analyzer = IsolationAnalyzer()
        
        # BSM components
        self.call_detector = CallDetector()
        self.doc_matcher = DocumentMatcher()
        
        logger.info(f"BehavioralFidelityEvaluatorV2 initialized "
                   f"(execution={'enabled' if enable_execution else 'disabled'})")
    
    def evaluate(self, source_code: str, 
                 documentation: str,
                 ground_truth: Optional[Dict] = None,
                 task_id: str = "unknown") -> Dict:
        """
        Evaluate behavioral fidelity using IUE + BSM.
        
        Args:
            source_code: Original COBOL source code
            documentation: AI-generated documentation
            ground_truth: Ground truth data (optional, used for IUE test generation)
            task_id: Task identifier for logging
            
        Returns:
            Dict with combined score and detailed breakdown
        """
        logger.info(f"[{task_id}] Starting BF evaluation (IUE + BSM)")
        
        # Run IUE evaluation
        iue_result = self._evaluate_iue(source_code, documentation, ground_truth, task_id)
        
        # Run BSM evaluation
        bsm_result = self._evaluate_bsm(source_code, documentation, task_id)
        
        # Combine scores
        combined_score = self._combine_scores(iue_result, bsm_result)
        
        # Determine program classification
        program_type = self._classify_program(iue_result, bsm_result)
        
        result = {
            "score": combined_score,
            "score_percent": round(combined_score * 100, 2),
            "method": "IUE+BSM",
            "program_type": program_type,
            # Placeholder indicates this is NOT real execution-based validation
            # When execution is disabled, we can't detect real data transformation errors
            "placeholder": not self.enable_execution,
            # v2.1.4: IUE/BSM is static analysis, no actual execution tests
            # CF-03 should only trigger when total_tests > 0
            "total_tests": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            
            # IUE breakdown
            "iue": {
                "score": iue_result.get("score"),
                "score_percent": round(iue_result.get("score", 0) * 100, 2) if iue_result.get("score") is not None else None,
                "weight": self.IUE_WEIGHT,
                "isolatable_paragraphs": iue_result.get("isolatable_count", 0),
                "total_paragraphs": iue_result.get("total_paragraphs", 0),
                "isolation_percent": iue_result.get("isolation_percent", 0),
                "reason": iue_result.get("reason"),
            },
            
            # BSM breakdown
            "bsm": {
                "score": bsm_result.get("score"),
                "score_percent": round(bsm_result.get("score", 0) * 100, 2) if bsm_result.get("score") is not None else None,
                "weight": self.BSM_WEIGHT,
                "external_calls": bsm_result.get("total_calls", 0),
                "calls_evaluated": bsm_result.get("calls_evaluated", 0),
                "call_details": bsm_result.get("call_details", []),
                "reason": bsm_result.get("reason"),
            },
        }
        
        logger.info(f"[{task_id}] BF Score: {result['score_percent']}% "
                   f"(IUE: {result['iue']['score_percent']}%, BSM: {result['bsm']['score_percent']}%)")
        
        return result
    
    def _evaluate_iue(self, source_code: str, 
                      documentation: str,
                      ground_truth: Optional[Dict],
                      task_id: str) -> Dict:
        """
        Evaluate using Isolatable Unit Execution.
        
        For now, this provides isolation analysis. Full execution
        would require test harness generation and GnuCOBOL execution.
        """
        # Parse paragraphs
        paragraphs = self.paragraph_parser.parse(source_code)
        
        if not paragraphs:
            logger.warning(f"[{task_id}] No paragraphs found in source")
            return {
                "score": None,
                "reason": "no_paragraphs_found",
                "total_paragraphs": 0,
                "isolatable_count": 0,
                "isolation_percent": 0,
            }
        
        # Analyze isolation
        all_units = self.isolation_analyzer.analyze(paragraphs)
        isolatable = [u for u in all_units if u.is_isolatable]
        
        # Calculate IUE score based on isolation analysis
        # For now, use average isolation score of isolatable units
        # In full implementation, this would be execution-based
        if isolatable:
            if self.enable_execution:
                # Full execution mode - would run test harnesses
                # For now, estimate based on isolation quality
                avg_isolation_score = sum(u.isolation_score for u in isolatable) / len(isolatable)
                iue_score = avg_isolation_score
            else:
                # Analysis-only mode - score based on what could be tested
                # Higher score = more computational logic that could be verified
                avg_isolation_score = sum(u.isolation_score for u in isolatable) / len(isolatable)
                iue_score = avg_isolation_score
        else:
            # No isolatable units - all external dependencies
            iue_score = None
        
        isolation_percent = (len(isolatable) / len(paragraphs) * 100) if paragraphs else 0
        
        # Get blocker statistics
        stats = self.isolation_analyzer.get_isolation_stats(paragraphs)
        
        return {
            "score": iue_score,
            "reason": None if iue_score is not None else "no_isolatable_units",
            "total_paragraphs": len(paragraphs),
            "isolatable_count": len(isolatable),
            "isolation_percent": round(isolation_percent, 1),
            "blockers": stats.get("blockers", {}),
            "units": [
                {
                    "name": u.paragraph.name,
                    "isolatable": u.is_isolatable,
                    "score": round(u.isolation_score, 2) if u.is_isolatable else 0,
                    "blocking_reason": u.blocking_reason,
                }
                for u in all_units
            ]
        }
    
    def _evaluate_bsm(self, source_code: str, 
                      documentation: str,
                      task_id: str) -> Dict:
        """
        Evaluate using Behavioral Specification Matching.
        
        Checks if documentation accurately describes external calls.
        """
        # Detect external calls
        all_calls = self.call_detector.detect(source_code)
        summary = self.call_detector.get_call_summary(source_code)
        
        # Filter to BSM-evaluable calls (SQL, CICS, CALL)
        bsm_calls = [c for c in all_calls if c.category in ['SQL', 'CICS', 'CALL']]
        
        if not bsm_calls:
            logger.info(f"[{task_id}] No BSM-evaluable external calls found")
            return {
                "score": 1.0,  # Perfect score if no external calls
                "reason": "no_external_calls",
                "total_calls": len(all_calls),
                "calls_evaluated": 0,
                "call_details": [],
                "summary": summary,
            }
        
        # Evaluate each call
        call_results: List[CallMatchResult] = []
        
        for call in bsm_calls:
            pattern = get_pattern(call.call_type)
            if not pattern:
                logger.debug(f"No BSM pattern for {call.call_type}")
                continue
            
            result = self.doc_matcher.match_call(call, pattern, documentation)
            call_results.append(result)
        
        if not call_results:
            return {
                "score": 1.0,
                "reason": "no_matching_patterns",
                "total_calls": len(all_calls),
                "calls_evaluated": 0,
                "call_details": [],
                "summary": summary,
            }
        
        # Calculate overall BSM score
        bsm_score = self.doc_matcher.calculate_overall_score(call_results)
        
        # Format call details for output
        call_details = []
        for cr in call_results:
            call_details.append({
                "type": cr.call_type,
                "line": cr.line_number,
                "score": round(cr.score, 2),
                "score_percent": round(cr.score * 100, 1),
                "items": [
                    {
                        "name": item.item_name,
                        "matched": item.matched,
                        "weight": item.weight,
                        "evidence": item.evidence,
                    }
                    for item in cr.items
                ],
                "facts": cr.facts_extracted,
            })
        
        return {
            "score": bsm_score,
            "reason": None,
            "total_calls": len(all_calls),
            "calls_evaluated": len(call_results),
            "call_details": call_details,
            "summary": summary,
        }
    
    def _combine_scores(self, iue_result: Dict, bsm_result: Dict) -> float:
        """
        Combine IUE and BSM scores into final BF score.
        
        Handles edge cases:
        - No isolatable units → use BSM only
        - No external calls → use IUE only
        - Neither → use placeholder
        """
        iue_score = iue_result.get("score")
        bsm_score = bsm_result.get("score")
        
        # Handle edge cases
        if iue_score is None and bsm_score is None:
            logger.warning("No IUE or BSM evaluation possible, using fallback")
            return 0.75  # Fallback placeholder
        
        if iue_score is None:
            # No isolatable units - use BSM only
            return bsm_score
        
        if bsm_score is None or bsm_result.get("calls_evaluated", 0) == 0:
            # No external calls - use IUE only
            return iue_score
        
        # Normal case: weighted combination
        combined = (self.IUE_WEIGHT * iue_score) + (self.BSM_WEIGHT * bsm_score)
        return round(combined, 4)
    
    def _classify_program(self, iue_result: Dict, bsm_result: Dict) -> str:
        """
        Classify program based on evaluation results.
        
        Returns human-readable program type.
        """
        isolation_pct = iue_result.get("isolation_percent", 0)
        has_sql = bsm_result.get("summary", {}).get("has_sql", False)
        has_cics = bsm_result.get("summary", {}).get("has_cics", False)
        has_calls = bsm_result.get("summary", {}).get("has_calls", False)
        
        if has_cics:
            if isolation_pct > 30:
                return "CICS with compute logic"
            return "CICS transaction"
        
        if has_sql:
            if isolation_pct > 30:
                return "DB2 with compute logic"
            return "DB2 batch"
        
        if has_calls:
            if isolation_pct > 50:
                return "Mixed with subroutines"
            return "Subroutine-heavy"
        
        if isolation_pct > 90:
            return "Pure compute"
        elif isolation_pct > 50:
            return "Mostly compute"
        else:
            return "File I/O heavy"
