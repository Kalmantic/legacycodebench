"""
Isolation Analyzer - Identify COBOL paragraphs that can be executed in isolation.

An isolatable unit is a paragraph that:
- Has NO external dependencies (EXEC SQL, EXEC CICS, CALL)
- Takes input from WORKING-STORAGE variables
- Produces output to WORKING-STORAGE variables or DISPLAY
- Contains pure computation or data transformation
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict
import logging

from .paragraph_parser import COBOLParagraph

logger = logging.getLogger(__name__)


@dataclass
class IsolatableUnit:
    """Represents a paragraph that can be tested in isolation."""
    paragraph: COBOLParagraph
    inputs: List[str]               # Variables needed as input
    outputs: List[str]              # Variables produced as output
    isolation_score: float          # 0-1, how "pure" this unit is
    blocking_reason: Optional[str]  # Why it can't be isolated (if any)
    is_isolatable: bool             # Final determination


class IsolationAnalyzer:
    """
    Analyze paragraphs to determine if they can be executed in isolation.
    
    Blocking patterns (prevent isolation):
    - EXEC SQL    → Database dependency
    - EXEC CICS   → Transaction dependency
    - EXEC DLI    → IMS dependency
    - CALL        → External program dependency
    - READ/WRITE  → File dependency
    """
    
    BLOCKING_PATTERNS = {
        r'EXEC\s+SQL':      'Database operation (EXEC SQL)',
        r'EXEC\s+CICS':     'Transaction operation (EXEC CICS)',
        r'EXEC\s+DLI':      'IMS database operation (EXEC DLI)',
        r'CALL\s+':         'External program call (CALL)',
        r'\bREAD\s+\w+':    'File read operation',
        r'\bWRITE\s+\w+':   'File write operation',
        r'\bOPEN\s+':       'File open operation',
        r'\bCLOSE\s+':      'File close operation',
        r'\bREWRITE\s+':    'File rewrite operation',
        r'\bDELETE\s+':     'File delete operation',
        r'\bSTART\s+':      'File start operation',
    }
    
    # Patterns that indicate computational logic (bonus for isolation)
    COMPUTATIONAL_PATTERNS = [
        r'\bCOMPUTE\s+',
        r'\bADD\s+',
        r'\bSUBTRACT\s+',
        r'\bMULTIPLY\s+',
        r'\bDIVIDE\s+',
        r'\bMOVE\s+',
        r'\bIF\s+',
        r'\bEVALUATE\s+',
    ]
    
    def analyze(self, paragraphs: List[COBOLParagraph]) -> List[IsolatableUnit]:
        """
        Analyze all paragraphs and return list of IsolatableUnit objects.
        
        Returns both isolatable and non-isolatable units, with blocking reasons.
        """
        units = []
        
        for para in paragraphs:
            blocking_reason = self._check_blocking_patterns(para.body)
            is_isolatable = blocking_reason is None
            
            unit = IsolatableUnit(
                paragraph=para,
                inputs=para.variables_read,
                outputs=para.variables_written,
                isolation_score=self._calculate_isolation_score(para) if is_isolatable else 0.0,
                blocking_reason=blocking_reason,
                is_isolatable=is_isolatable
            )
            units.append(unit)
        
        isolatable_count = sum(1 for u in units if u.is_isolatable)
        logger.info(f"Analyzed {len(paragraphs)} paragraphs: {isolatable_count} isolatable, "
                   f"{len(paragraphs) - isolatable_count} blocked")
        
        return units
    
    def get_isolatable_units(self, paragraphs: List[COBOLParagraph]) -> List[IsolatableUnit]:
        """Get only the isolatable units (convenience method)."""
        all_units = self.analyze(paragraphs)
        return [u for u in all_units if u.is_isolatable]
    
    def _check_blocking_patterns(self, body: str) -> Optional[str]:
        """
        Check if paragraph contains any blocking patterns.
        
        Returns blocking reason or None if isolatable.
        """
        for pattern, reason in self.BLOCKING_PATTERNS.items():
            if re.search(pattern, body, re.IGNORECASE):
                return reason
        return None  # No blocking patterns found
    
    def _calculate_isolation_score(self, para: COBOLParagraph) -> float:
        """
        Calculate how "pure" a paragraph is (0-1).
        
        Higher score = better for testing because:
        - More computational logic
        - Fewer dependencies on other paragraphs
        - Cleaner input/output interface
        """
        score = 0.5  # Base score
        
        # Bonus for computational patterns
        computational_count = 0
        for pattern in self.COMPUTATIONAL_PATTERNS:
            if re.search(pattern, para.body, re.IGNORECASE):
                computational_count += 1
        
        if computational_count >= 3:
            score += 0.3
        elif computational_count >= 1:
            score += 0.15
        
        # Penalty for many PERFORM calls (complex dependencies)
        if len(para.performs) > 5:
            score -= 0.3
        elif len(para.performs) > 2:
            score -= 0.15
        
        # Bonus for having clear outputs
        if len(para.variables_written) > 0:
            score += 0.1
        
        # Penalty for too many variables (complex state)
        total_vars = len(para.variables_read) + len(para.variables_written)
        if total_vars > 15:
            score -= 0.2
        elif total_vars > 10:
            score -= 0.1
        
        # Bonus for DISPLAY (can verify output)
        if 'DISPLAY' in para.body.upper():
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def get_isolation_stats(self, paragraphs: List[COBOLParagraph]) -> Dict:
        """Get statistics about isolation analysis."""
        units = self.analyze(paragraphs)
        isolatable = [u for u in units if u.is_isolatable]
        
        return {
            "total_paragraphs": len(paragraphs),
            "isolatable_count": len(isolatable),
            "isolatable_percent": len(isolatable) / len(paragraphs) * 100 if paragraphs else 0,
            "blocked_count": len(paragraphs) - len(isolatable),
            "blockers": self._count_blockers(units),
            "avg_isolation_score": sum(u.isolation_score for u in isolatable) / len(isolatable) if isolatable else 0,
        }
    
    def _count_blockers(self, units: List[IsolatableUnit]) -> Dict[str, int]:
        """Count how many paragraphs are blocked by each pattern."""
        counts = {}
        
        for unit in units:
            if unit.blocking_reason:
                counts[unit.blocking_reason] = counts.get(unit.blocking_reason, 0) + 1
        
        return counts
    
    def classify_program(self, paragraphs: List[COBOLParagraph]) -> str:
        """
        Classify a program based on its isolation analysis.
        
        Returns:
            "FULLY_EXECUTABLE" - All paragraphs isolatable, GnuCOBOL can run
            "PARTIALLY_EXECUTABLE" - Some paragraphs isolatable
            "NON_EXECUTABLE" - No paragraphs isolatable
        """
        stats = self.get_isolation_stats(paragraphs)
        
        if stats["isolatable_percent"] >= 95:
            return "FULLY_EXECUTABLE"
        elif stats["isolatable_count"] > 0:
            return "PARTIALLY_EXECUTABLE"
        else:
            return "NON_EXECUTABLE"
