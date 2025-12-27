"""
Call Detector - Detect external calls in COBOL source code.

Detects:
- EXEC SQL (SELECT, INSERT, UPDATE, DELETE, OPEN, FETCH, CLOSE)
- EXEC CICS (SEND, RECEIVE, READ, WRITE, LINK, XCTL)
- CALL statements (static and dynamic)
- File operations (READ, WRITE, OPEN, CLOSE)
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExternalCall:
    """Represents a detected external call in COBOL source."""
    call_type: str              # e.g., EXEC_SQL_SELECT, EXEC_CICS_SEND, CALL_STATIC
    content: str                # Statement content (without EXEC/END-EXEC)
    line_number: int            # Location in source (1-indexed)
    raw_match: str              # Original matched text
    category: str               # SQL, CICS, CALL, FILE


class CallDetector:
    """
    Detect external calls in COBOL source code.
    
    These are the calls that cannot be executed with GnuCOBOL
    and must be evaluated using BSM (Behavioral Specification Matching).
    """
    
    # Detection patterns for each call type
    PATTERNS = {
        # SQL patterns
        "EXEC_SQL_SELECT": (r'EXEC\s+SQL\s+SELECT(.+?)END-EXEC', 'SQL'),
        "EXEC_SQL_INSERT": (r'EXEC\s+SQL\s+INSERT(.+?)END-EXEC', 'SQL'),
        "EXEC_SQL_UPDATE": (r'EXEC\s+SQL\s+UPDATE(.+?)END-EXEC', 'SQL'),
        "EXEC_SQL_DELETE": (r'EXEC\s+SQL\s+DELETE(.+?)END-EXEC', 'SQL'),
        "EXEC_SQL_OPEN":   (r'EXEC\s+SQL\s+OPEN(.+?)END-EXEC', 'SQL'),
        "EXEC_SQL_FETCH":  (r'EXEC\s+SQL\s+FETCH(.+?)END-EXEC', 'SQL'),
        "EXEC_SQL_CLOSE":  (r'EXEC\s+SQL\s+CLOSE(.+?)END-EXEC', 'SQL'),
        "EXEC_SQL_DECLARE": (r'EXEC\s+SQL\s+DECLARE(.+?)END-EXEC', 'SQL'),
        
        # CICS patterns
        "EXEC_CICS_SEND":    (r'EXEC\s+CICS\s+SEND(.+?)END-EXEC', 'CICS'),
        "EXEC_CICS_RECEIVE": (r'EXEC\s+CICS\s+RECEIVE(.+?)END-EXEC', 'CICS'),
        "EXEC_CICS_READ":    (r'EXEC\s+CICS\s+READ(.+?)END-EXEC', 'CICS'),
        "EXEC_CICS_WRITE":   (r'EXEC\s+CICS\s+WRITE(.+?)END-EXEC', 'CICS'),
        "EXEC_CICS_REWRITE": (r'EXEC\s+CICS\s+REWRITE(.+?)END-EXEC', 'CICS'),
        "EXEC_CICS_DELETE":  (r'EXEC\s+CICS\s+DELETE(.+?)END-EXEC', 'CICS'),
        "EXEC_CICS_LINK":    (r'EXEC\s+CICS\s+LINK(.+?)END-EXEC', 'CICS'),
        "EXEC_CICS_XCTL":    (r'EXEC\s+CICS\s+XCTL(.+?)END-EXEC', 'CICS'),
        "EXEC_CICS_RETURN":  (r'EXEC\s+CICS\s+RETURN(.+?)END-EXEC', 'CICS'),
        "EXEC_CICS_HANDLE":  (r'EXEC\s+CICS\s+HANDLE(.+?)END-EXEC', 'CICS'),
        
        # CALL patterns
        "CALL_STATIC":  (r"CALL\s+['\"]([A-Za-z0-9-]+)['\"](.+?)(?:\.|END-CALL)", 'CALL'),
        "CALL_DYNAMIC": (r'CALL\s+([A-Z][A-Z0-9-]+)(?:\s+USING)?(.+?)(?:\.|END-CALL)', 'CALL'),
        
        # File patterns (these block IUE but don't have BSM patterns)
        "FILE_READ":  (r'\bREAD\s+([A-Z][A-Z0-9-]+)', 'FILE'),
        "FILE_WRITE": (r'\bWRITE\s+([A-Z][A-Z0-9-]+)', 'FILE'),
    }
    
    def detect(self, source_code: str) -> List[ExternalCall]:
        """
        Detect all external calls in source code.
        
        Args:
            source_code: Full COBOL source code
            
        Returns:
            List of ExternalCall objects
        """
        calls = []
        
        # Normalize source for multi-line matching
        normalized = self._normalize_source(source_code)
        
        for call_type, (pattern, category) in self.PATTERNS.items():
            for match in re.finditer(pattern, normalized, re.IGNORECASE | re.DOTALL):
                # Find line number in original source
                line_num = self._find_line_number(source_code, match.start())
                
                # Extract content (first group if available)
                content = match.group(1) if match.groups() else match.group(0)
                
                calls.append(ExternalCall(
                    call_type=call_type,
                    content=content.strip(),
                    line_number=line_num,
                    raw_match=match.group(0),
                    category=category
                ))
        
        logger.info(f"Detected {len(calls)} external calls")
        return calls
    
    def _normalize_source(self, source: str) -> str:
        """
        Normalize source for multi-line pattern matching.
        
        Removes sequence numbers, joins continuation lines.
        """
        lines = []
        for line in source.split('\n'):
            # Skip totally empty lines
            if not line.strip():
                lines.append('')
                continue
            
            # Remove sequence numbers (columns 1-6) if present
            if len(line) > 6 and line[:6].strip().isdigit():
                line = '      ' + line[6:]
            
            # Skip comment lines (indicator in column 7)
            if len(line) > 6 and line[6] in '*/-':
                continue
            
            lines.append(line)
        
        return '\n'.join(lines)
    
    def _find_line_number(self, source: str, char_pos: int) -> int:
        """Find line number for character position (1-indexed)."""
        return source[:char_pos].count('\n') + 1
    
    def get_call_summary(self, source_code: str) -> Dict:
        """
        Get summary of external calls in source.
        
        Returns:
            Dict with counts by type and category
        """
        calls = self.detect(source_code)
        
        summary = {
            "total_calls": len(calls),
            "by_type": {},
            "by_category": {
                "SQL": 0,
                "CICS": 0,
                "CALL": 0,
                "FILE": 0,
            },
            "has_sql": False,
            "has_cics": False,
            "has_calls": False,
            "has_files": False,
        }
        
        for call in calls:
            # Count by type
            summary["by_type"][call.call_type] = summary["by_type"].get(call.call_type, 0) + 1
            
            # Count by category
            summary["by_category"][call.category] += 1
            
            # Set flags
            if call.category == "SQL":
                summary["has_sql"] = True
            elif call.category == "CICS":
                summary["has_cics"] = True
            elif call.category == "CALL":
                summary["has_calls"] = True
            elif call.category == "FILE":
                summary["has_files"] = True
        
        return summary
    
    def get_bsm_evaluable_calls(self, source_code: str) -> List[ExternalCall]:
        """
        Get only the calls that can be evaluated with BSM.
        
        Excludes FILE operations which don't have BSM patterns.
        """
        all_calls = self.detect(source_code)
        return [c for c in all_calls if c.category in ['SQL', 'CICS', 'CALL']]
