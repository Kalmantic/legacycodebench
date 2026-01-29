"""
COBOL Language Adapter for LegacyCodeBench V2.4

Wraps existing COBOL-specific modules to provide a unified adapter interface.

Specification Reference: TDD_V2.4.md Section 3.2
"""

import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from legacycodebench.models.enums import Language
from .base import LanguageAdapter

logger = logging.getLogger(__name__)


class COBOLAdapter(LanguageAdapter):
    """
    COBOL language adapter.
    
    Provides COBOL-specific:
    - BSM patterns (16 patterns for SQL, CICS, CALL, FILE)
    - Synonyms for TF-IDF matching
    - Paragraph extraction
    - Blocking construct detection
    - Critical failure configuration
    """

    # =========================================================================
    # BSM PATTERNS (16 patterns)
    # =========================================================================
    BSM_PATTERNS = [
        # SQL patterns (6)
        {"id": "SQL_SELECT", "regex": r"EXEC\s+SQL\s+SELECT.+?FROM\s+(\w+)", "category": "SQL"},
        {"id": "SQL_INSERT", "regex": r"EXEC\s+SQL\s+INSERT\s+INTO\s+(\w+)", "category": "SQL"},
        {"id": "SQL_UPDATE", "regex": r"EXEC\s+SQL\s+UPDATE\s+(\w+)", "category": "SQL"},
        {"id": "SQL_DELETE", "regex": r"EXEC\s+SQL\s+DELETE\s+FROM\s+(\w+)", "category": "SQL"},
        {"id": "SQL_CURSOR_OPEN", "regex": r"EXEC\s+SQL\s+OPEN\s+(\w+)", "category": "SQL"},
        {"id": "SQL_CURSOR_FETCH", "regex": r"EXEC\s+SQL\s+FETCH\s+(\w+)", "category": "SQL"},

        # CICS patterns (5)
        {"id": "CICS_READ", "regex": r"EXEC\s+CICS\s+READ.+?FILE\s*\(\s*['\"]?(\w+)", "category": "CICS"},
        {"id": "CICS_WRITE", "regex": r"EXEC\s+CICS\s+WRITE.+?FILE\s*\(\s*['\"]?(\w+)", "category": "CICS"},
        {"id": "CICS_REWRITE", "regex": r"EXEC\s+CICS\s+REWRITE", "category": "CICS"},
        {"id": "CICS_SEND_MAP", "regex": r"EXEC\s+CICS\s+SEND\s+MAP\s*\(\s*['\"]?(\w+)", "category": "CICS"},
        {"id": "CICS_RECEIVE_MAP", "regex": r"EXEC\s+CICS\s+RECEIVE\s+MAP\s*\(\s*['\"]?(\w+)", "category": "CICS"},

        # CALL patterns (2)
        {"id": "CALL_STATIC", "regex": r"CALL\s+['\"](\w+)['\"]", "category": "CALL"},
        {"id": "CALL_DYNAMIC", "regex": r"CALL\s+([A-Z][A-Z0-9-]+)\s", "category": "CALL"},

        # File patterns (3)
        {"id": "FILE_READ", "regex": r"READ\s+(\w+)", "category": "FILE"},
        {"id": "FILE_WRITE", "regex": r"WRITE\s+(\w+)", "category": "FILE"},
        {"id": "FILE_REWRITE", "regex": r"REWRITE\s+(\w+)", "category": "FILE"},
    ]

    # =========================================================================
    # COBOL SYNONYMS (for TF-IDF expansion)
    # =========================================================================
    SYNONYMS = {
        # Computation
        "compute": ["calculate", "determine", "derive", "computes", "calculates", "calculated", "computed"],
        "add": ["sum", "accumulate", "total", "adds", "sums"],
        "subtract": ["deduct", "reduce", "minus", "subtracts", "deducts"],
        "multiply": ["times", "product", "multiplies"],
        "divide": ["split", "quotient", "divides"],
        
        # Control flow
        "perform": ["execute", "call", "invoke", "run", "performs", "executes"],
        "if": ["when", "check", "condition", "conditional"],
        "evaluate": ["switch", "case", "select", "evaluates"],
        
        # Data operations
        "move": ["assign", "set", "copy", "transfer", "moves", "assigns"],
        "read": ["input", "load", "fetch", "retrieve", "reads", "retrieves"],
        "write": ["output", "store", "save", "writes", "stores"],
        
        # File operations
        "open": ["initialize", "connect", "opens"],
        "close": ["terminate", "disconnect", "finalize", "closes"],
        
        # Error handling
        "error": ["exception", "failure", "invalid", "errors"],
        "handler": ["handle", "handling", "process", "trap"],
    }

    @property
    def language(self) -> Language:
        return Language.COBOL

    @property
    def file_extensions(self) -> List[str]:
        return ["cbl", "cob", "cobol"]

    def get_bsm_patterns(self) -> List[Dict]:
        """Return 16 BSM patterns for COBOL external calls."""
        return self.BSM_PATTERNS.copy()

    def get_synonyms(self) -> Dict[str, List[str]]:
        """Return COBOL synonym dictionary."""
        return self.SYNONYMS.copy()

    def extract_paragraphs(self, source: str) -> List[Dict]:
        """
        Extract COBOL paragraphs from PROCEDURE DIVISION.
        
        Paragraph names are:
        - On a line by itself (or with just a period)
        - Start with a letter or digit
        - End at the next paragraph name
        """
        paragraphs = []
        lines = source.split('\n')
        
        # Find PROCEDURE DIVISION
        proc_start = -1
        for i, line in enumerate(lines):
            if re.search(r'PROCEDURE\s+DIVISION', line, re.IGNORECASE):
                proc_start = i
                break
        
        if proc_start < 0:
            return paragraphs
        
        # Pattern for paragraph names
        # Must be at start of line (after possible spacing), end with period
        para_pattern = re.compile(r'^\s*([A-Z][A-Z0-9-]*)\s*\.\s*$', re.IGNORECASE)
        
        current_para = None
        current_start = -1
        current_content = []
        
        for i in range(proc_start, len(lines)):
            line = lines[i]
            match = para_pattern.match(line)
            
            if match:
                # Save previous paragraph
                if current_para:
                    paragraphs.append({
                        "name": current_para,
                        "start_line": current_start + 1,  # 1-indexed
                        "end_line": i,
                        "content": '\n'.join(current_content),
                        "type": self._classify_paragraph_content('\n'.join(current_content)),
                    })
                
                # Start new paragraph
                current_para = match.group(1).upper()
                current_start = i
                current_content = []
            elif current_para:
                current_content.append(line)
        
        # Don't forget the last paragraph
        if current_para:
            paragraphs.append({
                "name": current_para,
                "start_line": current_start + 1,
                "end_line": len(lines),
                "content": '\n'.join(current_content),
                "type": self._classify_paragraph_content('\n'.join(current_content)),
            })
        
        return paragraphs

    def _classify_paragraph_content(self, content: str) -> str:
        """Classify paragraph as PURE, MIXED, or INFRASTRUCTURE."""
        content_upper = content.upper()
        
        # INFRASTRUCTURE: File I/O operations
        infra_patterns = [
            r'^\s*(?:OPEN|CLOSE)\s+(?:INPUT|OUTPUT|I-O|EXTEND)',
            r'^\s*READ\s+[A-Z][A-Z0-9-]*\s*(?:INTO|AT\s+END|KEY)',
            r'^\s*WRITE\s+[A-Z][A-Z0-9-]*\s*(?:FROM|AFTER|BEFORE)',
        ]
        
        # MIXED: External calls
        external_patterns = [
            r'EXEC\s+SQL',
            r'EXEC\s+CICS',
            r'CALL\s+',
        ]
        
        # Logic: Computations
        logic_patterns = [
            r'\b(?:COMPUTE|IF|EVALUATE|MULTIPLY|DIVIDE|ADD|SUBTRACT)\b',
        ]
        
        has_external = any(re.search(p, content_upper) for p in external_patterns)
        has_logic = any(re.search(p, content_upper) for p in logic_patterns)
        has_infra = any(re.search(p, content_upper, re.MULTILINE) for p in infra_patterns)
        
        if has_infra and not has_logic:
            return "INFRASTRUCTURE"
        elif has_external:
            return "MIXED"
        else:
            return "PURE"

    def extract_data_structures(self, source: str) -> List[Dict]:
        """
        Extract COBOL data structure definitions from DATA DIVISION.
        """
        structures = []
        lines = source.split('\n')
        
        # Pattern for data items: level-number, name, optional PIC
        data_pattern = re.compile(
            r'^\s*(\d{1,2})\s+([A-Z][A-Z0-9-]*)\s*(?:PIC(?:TURE)?\s+(\S+))?',
            re.IGNORECASE
        )
        
        for i, line in enumerate(lines):
            match = data_pattern.match(line)
            if match:
                level = match.group(1)
                name = match.group(2).upper()
                pic = match.group(3) if match.group(3) else None
                
                structures.append({
                    "name": name,
                    "level": level,
                    "line_number": i + 1,
                    "pic_clause": pic,
                    "children": [],  # Would need more parsing to fill this
                })
        
        return structures

    def detect_blocking_constructs(self, source: str) -> List[str]:
        """
        Detect constructs that prevent GnuCOBOL compilation.
        """
        constructs = []
        source_upper = source.upper()
        
        if re.search(r'EXEC\s+CICS', source_upper):
            constructs.append("CICS")
        if re.search(r'EXEC\s+SQL', source_upper):
            constructs.append("DB2")
        if re.search(r'EXEC\s+DLI|CBLTDLI', source_upper):
            constructs.append("IMS")
        if re.search(r'MQOPEN|MQGET|MQPUT', source_upper):
            constructs.append("MQ")
        
        return constructs

    def get_critical_failure_config(self) -> Dict[str, bool]:
        """
        Return which critical failures apply to COBOL.
        
        All 5 CFs are enabled for COBOL.
        """
        return {
            "CF-01": True,   # Complete Silence
            "CF-02": True,   # Hallucinated Structure
            "CF-03": True,   # Behavioral Contradiction (executed tasks only)
            "CF-04": True,   # Missing Error Handling
            "CF-05": True,   # External Call Misspecification
        }

    def get_executor(self) -> Any:
        """
        Return the COBOL executor for compilation and execution.
        """
        try:
            from legacycodebench.execution.cobol_executor import COBOLExecutor
            return COBOLExecutor()
        except ImportError:
            logger.warning("COBOLExecutor not available, returning None")
            return None

    def is_comment(self, line: str) -> bool:
        """
        Check if a line is a COBOL comment.
        
        COBOL comments:
        - Fixed format: column 7 contains '*' or '/'
        - Free format: starts with '*>' (anywhere)
        """
        # Check for free-format comment
        if line.strip().startswith('*>'):
            return True
        
        # Check for fixed-format comment (column 7)
        if len(line) >= 7 and line[6] in ('*', '/'):
            return True
        
        return False

    def classify_paragraph(self, paragraph: Dict) -> str:
        """
        Classify a paragraph dict as PURE, MIXED, or INFRASTRUCTURE.
        """
        return paragraph.get("type", "PURE")

    def get_variable_pattern(self) -> str:
        """
        Return regex pattern for COBOL variable names.
        
        COBOL variables:
        - Start with a letter
        - Contain letters, digits, and hyphens
        - Case-insensitive (typically uppercase)
        """
        return r"[A-Za-z][A-Za-z0-9-]*"
