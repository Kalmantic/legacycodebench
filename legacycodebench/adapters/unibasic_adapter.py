"""
UniBasic Language Adapter for LegacyCodeBench V2.4

Provides UniBasic (Pick/MultiValue) language support.

Specification Reference: TDD_V2.4.md Section 3.3
"""

import re
import logging
from typing import List, Dict, Any, Optional

from legacycodebench.models.enums import Language
from .base import LanguageAdapter

logger = logging.getLogger(__name__)


class UniBasicAdapter(LanguageAdapter):
    """
    UniBasic language adapter.
    
    Provides UniBasic-specific:
    - BSM patterns (15 patterns for File I/O, Execution, Transaction, Conversion, Control)
    - Synonyms for TF-IDF matching
    - Subroutine extraction
    - Blocking construct detection (UniVerse/UniData-specific)
    - Critical failure configuration (CF-02/CF-03 disabled due to static-only verification)
    """

    # =========================================================================
    # BSM PATTERNS (15 patterns)
    # =========================================================================
    BSM_PATTERNS = [
        # File I/O patterns (8)
        {"id": "UB_READ", "regex": r"\bREAD\s+\w+\s+FROM\s+\w+", "category": "FILE_IO"},
        {"id": "UB_WRITE", "regex": r"\bWRITE\s+\w+\s+(?:ON|TO)\s+\w+", "category": "FILE_IO"},
        {"id": "UB_MATREAD", "regex": r"\bMATREAD\s+\w+\s+FROM\s+\w+", "category": "FILE_IO"},
        {"id": "UB_MATWRITE", "regex": r"\bMATWRITE\s+\w+\s+(?:ON|TO)\s+\w+", "category": "FILE_IO"},
        {"id": "UB_DELETE", "regex": r"\bDELETE\s+\w+", "category": "FILE_IO"},
        {"id": "UB_SELECT", "regex": r"\bSELECT\s+\w+", "category": "FILE_IO"},
        {"id": "UB_READV", "regex": r"\bREADV\s+\w+\s+FROM\s+\w+", "category": "FILE_IO"},
        {"id": "UB_WRITEV", "regex": r"\bWRITEV\s+\w+\s+(?:ON|TO)\s+\w+", "category": "FILE_IO"},

        # Execution patterns (2)
        {"id": "UB_EXECUTE", "regex": r"\bEXECUTE\s+", "category": "EXECUTION"},
        {"id": "UB_CALL", "regex": r"\bCALL\s+\w+", "category": "EXECUTION"},

        # Transaction patterns (2)
        {"id": "UB_TRANS_START", "regex": r"\bTRANSACTION\s+START", "category": "TRANSACTION"},
        {"id": "UB_TRANS_COMMIT", "regex": r"\bTRANSACTION\s+COMMIT", "category": "TRANSACTION"},

        # Conversion patterns (2)
        {"id": "UB_OCONV", "regex": r"\bOCONV\s*\(", "category": "CONVERSION"},
        {"id": "UB_ICONV", "regex": r"\bICONV\s*\(", "category": "CONVERSION"},

        # Control patterns (1)
        {"id": "UB_GOSUB", "regex": r"\bGOSUB\s+\w+", "category": "CONTROL"},
    ]

    # =========================================================================
    # UNIBASIC SYNONYMS (for TF-IDF expansion)
    # =========================================================================
    SYNONYMS = {
        # Data access
        "read": ["fetch", "retrieve", "get", "load", "reads", "fetches"],
        "write": ["store", "save", "put", "output", "writes", "stores"],
        "delete": ["remove", "erase", "clear", "deletes", "removes"],
        "select": ["query", "find", "search", "selects", "queries"],
        
        # Control flow
        "gosub": ["call", "invoke", "subroutine", "routine"],
        "return": ["exit", "end", "back", "returns"],
        "loop": ["iterate", "repeat", "cycle", "loops", "iterates"],
        
        # Data manipulation
        "convert": ["transform", "change", "format", "converts"],
        "extract": ["parse", "split", "get", "extracts", "parses"],
        
        # Error handling
        "locked": ["lock", "exclusive", "busy"],
        "else": ["otherwise", "alternative", "fallback"],
        
        # Transactions
        "transaction": ["commit", "rollback", "atomic"],
    }

    @property
    def language(self) -> Language:
        return Language.UNIBASIC

    @property
    def file_extensions(self) -> List[str]:
        return ["bp", "bas", "b"]

    def get_bsm_patterns(self) -> List[Dict]:
        """Return 15 BSM patterns for UniBasic external calls."""
        return self.BSM_PATTERNS.copy()

    def get_synonyms(self) -> Dict[str, List[str]]:
        """Return UniBasic synonym dictionary."""
        return self.SYNONYMS.copy()

    def extract_paragraphs(self, source: str) -> List[Dict]:
        """
        Extract UniBasic subroutines.
        
        UniBasic subroutines are defined as labels followed by code until RETURN.
        """
        subroutines = []
        lines = source.split('\n')
        
        # Pattern for subroutine labels (e.g., "PROCESS.DATA:" or "UPDATE.RECORD:")
        label_pattern = re.compile(r'^\s*([A-Z][A-Z0-9.]+):?\s*$', re.IGNORECASE)
        
        current_sub = None
        current_start = -1
        current_content = []
        
        for i, line in enumerate(lines):
            match = label_pattern.match(line)
            
            if match:
                # Save previous subroutine
                if current_sub:
                    subroutines.append({
                        "name": current_sub,
                        "start_line": current_start + 1,
                        "end_line": i,
                        "content": '\n'.join(current_content),
                        "type": self._classify_subroutine('\n'.join(current_content)),
                    })
                
                # Start new subroutine
                current_sub = match.group(1).upper()
                current_start = i
                current_content = []
            elif current_sub:
                current_content.append(line)
                
                # Check for RETURN to end subroutine
                if re.search(r'^\s*RETURN\s*$', line, re.IGNORECASE):
                    subroutines.append({
                        "name": current_sub,
                        "start_line": current_start + 1,
                        "end_line": i + 1,
                        "content": '\n'.join(current_content),
                        "type": self._classify_subroutine('\n'.join(current_content)),
                    })
                    current_sub = None
                    current_content = []
        
        # Handle last subroutine without RETURN
        if current_sub:
            subroutines.append({
                "name": current_sub,
                "start_line": current_start + 1,
                "end_line": len(lines),
                "content": '\n'.join(current_content),
                "type": self._classify_subroutine('\n'.join(current_content)),
            })
        
        return subroutines

    def _classify_subroutine(self, content: str) -> str:
        """Classify subroutine as PURE, MIXED, or INFRASTRUCTURE."""
        content_upper = content.upper()
        
        # INFRASTRUCTURE: File I/O only
        infra_patterns = [
            r'\bOPEN\s+',
            r'\bCLOSE\s+',
        ]
        
        # MIXED: External calls
        external_patterns = [
            r'\bCALL\s+',
            r'\bEXECUTE\s+',
        ]
        
        # Logic: Computations
        logic_patterns = [
            r'\bIF\s+',
            r'\bFOR\s+',
            r'\bLOOP\b',
        ]
        
        has_external = any(re.search(p, content_upper) for p in external_patterns)
        has_logic = any(re.search(p, content_upper) for p in logic_patterns)
        has_infra = any(re.search(p, content_upper) for p in infra_patterns)
        
        if has_infra and not has_logic:
            return "INFRASTRUCTURE"
        elif has_external:
            return "MIXED"
        else:
            return "PURE"

    def extract_data_structures(self, source: str) -> List[Dict]:
        """
        Extract UniBasic data structure definitions (DIM statements).
        """
        structures = []
        lines = source.split('\n')
        
        # Pattern for DIM statements
        dim_pattern = re.compile(r'^\s*DIM\s+(\w+)\s*\((.+?)\)', re.IGNORECASE)
        
        for i, line in enumerate(lines):
            match = dim_pattern.match(line)
            if match:
                name = match.group(1).upper()
                dimensions = match.group(2)
                
                structures.append({
                    "name": name,
                    "level": "DIM",
                    "line_number": i + 1,
                    "pic_clause": f"({dimensions})",
                    "children": [],
                })
        
        return structures

    def detect_blocking_constructs(self, source: str) -> List[str]:
        """
        Detect UniVerse/UniData-specific constructs that prevent ScarletDME execution.
        """
        constructs = []
        source_upper = source.upper()
        
        # UniVerse-specific
        if re.search(r'\bSQLEXECUTE\b', source_upper):
            constructs.append("UV_SQL")
        if re.search(r'\bCALLHTTP\b', source_upper):
            constructs.append("UV_HTTP")
        if re.search(r'\bUNIVERSE\.', source_upper):
            constructs.append("UV_API")
        
        # UniData-specific
        if re.search(r'\bU2REPLICATION\b', source_upper):
            constructs.append("UD_REPLICATION")
        
        return constructs

    def get_critical_failure_config(self) -> Dict[str, bool]:
        """
        Return which critical failures apply to UniBasic.
        
        CF-02 and CF-03 are DISABLED due to:
        - No execution verification available
        - Insufficient confidence for hallucination/contradiction detection
        """
        return {
            "CF-01": True,   # Complete Silence
            "CF-02": False,  # DISABLED: Insufficient confidence without execution
            "CF-03": False,  # DISABLED: Requires execution for behavioral contradiction
            "CF-04": True,   # Missing Error Handling
            "CF-05": True,   # External Call Misspecification
        }

    def get_executor(self) -> Any:
        """
        Return the UniBasic executor.
        
        Currently returns StubExecutor until ScarletDME integration is complete.
        """
        try:
            from legacycodebench.execution.stub_executor import StubExecutor
            return StubExecutor(self.language)
        except ImportError:
            logger.warning("StubExecutor not available, returning None")
            return None

    def is_comment(self, line: str) -> bool:
        """
        Check if a line is a UniBasic comment.
        
        UniBasic comments:
        - Start with '*' or '!' or 'REM'
        """
        stripped = line.strip()
        return (
            stripped.startswith('*') or
            stripped.startswith('!') or
            stripped.upper().startswith('REM ')
        )

    def classify_paragraph(self, paragraph: Dict) -> str:
        """Classify a subroutine dict as PURE, MIXED, or INFRASTRUCTURE."""
        return paragraph.get("type", "PURE")

    def get_variable_pattern(self) -> str:
        """
        Return regex pattern for UniBasic variable names.
        
        UniBasic variables:
        - Start with a letter
        - Contain letters, digits, and dots
        """
        return r"[A-Za-z][A-Za-z0-9.]*"
