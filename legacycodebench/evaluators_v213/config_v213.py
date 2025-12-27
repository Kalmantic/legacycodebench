"""
LegacyCodeBench V2.3.1 Configuration

Complete configuration for V2.3.1 evaluation including:
- Scoring weights (30/20/50)
- Pass thresholds
- COBOL synonyms
- Silence penalty settings
- Zero tolerance I/O settings
- Boundary testing settings
"""

from typing import Dict, List, Any


V213_CONFIG: Dict[str, Any] = {
    "version": "2.3.1",
    
    # =========================================================================
    # SCORING WEIGHTS (30/20/50)
    # =========================================================================
    "weights": {
        "structural_completeness": 0.30,
        "documentation_quality": 0.20,
        "behavioral_fidelity": 0.50,
    },
    
    # =========================================================================
    # PASS THRESHOLDS
    # =========================================================================
    "thresholds": {
        "sc": 0.60,  # Structural Completeness >= 60%
        "dq": 0.50,  # Documentation Quality >= 50%
        "bf": 0.55,  # Behavioral Fidelity >= 55%
    },
    
    # =========================================================================
    # STRUCTURAL COMPLETENESS SUB-WEIGHTS
    # =========================================================================
    "structural_weights": {
        "business_rules": 0.40,
        "data_structures": 0.25,
        "control_flow": 0.20,
        "external_calls": 0.15,
    },
    
    # =========================================================================
    # DOCUMENTATION QUALITY SUB-WEIGHTS
    # =========================================================================
    "documentation_weights": {
        "structure": 0.30,       # Required sections present
        "traceability": 0.30,   # Line citations valid
        "readability": 0.20,    # Flesch-Kincaid grade
        "abstraction": 0.20,    # WHY vs WHAT ratio
    },
    
    # =========================================================================
    # SILENCE PENALTY (Track 3)
    # =========================================================================
    "silence_penalty": {
        "min_claims": 1,         # PRODUCTION: min 1 claim to avoid over-penalizing
        "weight_low": 0.4,       # 3-4 claims: ClaimWeight = 0.4
        "weight_high": 0.6,      # 5+ claims: ClaimWeight = 0.6
        "enabled": True,
    },
    
    # =========================================================================
    # ZERO TOLERANCE I/O HALLUCINATION (CF-02)
    # =========================================================================
    "hallucination": {
        "io_tolerance": 0,           # Zero tolerance for I/O variables
        "internal_tolerance": 3,     # Allow up to 2 internal (>=3 triggers CF)
        "fuzzy_threshold": 0.80,     # 80% match for name variants
    },
    
    # =========================================================================
    # BOUNDARY VALUE TESTING
    # =========================================================================
    "boundary_testing": {
        "enabled": True,
        "tests_per_calc_claim": 3,   # min, mid, max
        "max_tests_total": 15,       # Cap total tests per task
    },
    
    # =========================================================================
    # COBOL SYNONYM DICTIONARY (Frozen for determinism)
    # =========================================================================
    "cobol_synonyms": {
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
    },
    
    # =========================================================================
    # LLM FALLBACK FOR CLAIM EXTRACTION
    # =========================================================================
    "llm_fallback": {
        "enabled": True,
        "trigger_claim_count": 3,    # If regex extracts < 3, use LLM
        "max_claims": 10,            # Cap LLM extracted claims
        "max_score_impact": 0.15,    # Max 15% score impact from LLM
        "temperature": 0,            # Deterministic
    },
    
    # =========================================================================
    # CLAIM EXTRACTION PATTERNS (Regex)
    # =========================================================================
    "claim_patterns": {
        "calculation": [
            # Support markdown formatting: `VAR` or (VAR) or (`VAR`)
            r"`?\(?([\w-]+)\)?`?\s+is\s+(?:calculated|computed|determined)\s+(?:by|as|from)\s+(.+)",
            r"(?:calculate|compute|determine)s?\s+`?\(?([\w-]+)\)?`?\s+(?:by|as|from)\s+(.+)",
            r"`?\(?([\w-]+)\)?`?\s+=\s+`?\(?([\w-]+)\)?`?\s*[\+\-\*\/]\s*`?\(?([\w-]+)\)?`?",
            # Handle descriptive text before variable: "The base cost (`P1`) is calculated"
            r"(?:the\s+)?[\w\s]+\(`?([\w-]+)`?\)\s+is\s+(?:calculated|computed|determined)\s+(?:by|as|from)\s+(.+)",
            # More flexible patterns for natural language
            r"`?\(?([\w-]+)\)?`?\s+is\s+(?:the\s+)?(?:sum|product|difference|result)\s+of\s+(.+)",
            r"(?:the\s+)?`?\(?([\w-]+)\)?`?\s+(?:equals?|=)\s+(.+)",
            r"`?\(?([\w-]+)\)?`?\s+is\s+(?:\d+%|\d+\s*percent)\s+of\s+(.+)",
        ],
        "conditional": [
            # Support markdown formatting
            r"when\s+`?\(?([\w-]+)\)?`?\s+(?:exceeds?|equals?|is greater|is less)\s+(.+?),\s*(.+)",
            r"if\s+`?\(?([\w-]+)\)?`?\s+(?:is|equals?|exceeds?)\s+(.+?),\s*(?:then\s+)?(.+)",
            # More flexible patterns
            r"(?:when|if)\s+(?:the\s+)?`?\(?([\w-]+)\)?`?\s+(?:is|are)\s+(.+?),\s*(.+)",
            r"for\s+`?\(?([\w-]+)\)?`?\s+(?:greater|less|over|under)\s+(.+?),\s*(.+)",
        ],
        "assignment": [
            # Support markdown formatting
            r"(?:result|output|value)\s+is\s+(?:stored|saved|placed)\s+in\s+`?\(?([\w-]+)\)?`?",
            r"`?\(?([\w-]+)\)?`?\s+(?:receives?|gets?|is set to)\s+(.+)",
            # More flexible patterns
            r"(?:stored|saved|placed|written)\s+(?:to|in|into)\s+(?:the\s+)?`?\(?([\w-]+)\)?`?",
            r"`?\(?([\w-]+)\)?`?\s+(?:contains?|holds?|stores?)\s+(.+)",
        ],
        "range": [
            # Support markdown formatting
            r"`?\(?([\w-]+)\)?`?\s+must\s+be\s+between\s+(\d+)\s+and\s+(\d+)",
            r"valid\s+(?:range|values?)\s+(?:for\s+)?`?\(?([\w-]+)\)?`?\s*:\s*(\d+)\s*-\s*(\d+)",
        ],
        "error": [
            r"if\s+(.+?)\s+fails?,\s*(.+)",
            r"on\s+(?:error|exception|failure),?\s+(.+)",
            # NEW: More natural error patterns
            r"(?:when|if)\s+(?:an?\s+)?(?:error|exception|failure)\s+occurs?,?\s*(.+)",
            r"(?:invalid|error|failed)\s+(.+?)\s+(?:triggers?|causes?|results? in)\s+(.+)",
        ],
    },
    
    # =========================================================================
    # PARAGRAPH CLASSIFICATION (§5.1-5.2)
    # =========================================================================
    "classification": {
        # Paragraph names matching these are INFRASTRUCTURE
        "infrastructure_patterns": [
            r"^\d{4}-(OPEN|CLOSE|READ|WRITE|INIT|HOUSE)",
            r"^0000[-_]MAIN",
            r"^9999[-_]",
        ],
        
        # Code patterns indicating external calls (MIXED paragraphs)
        "external_patterns": [
            r"EXEC\s+(SQL|CICS)",
            r"CALL\s+",
        ],
        
        # Code patterns indicating business logic (PURE/MIXED paragraphs)
        "logic_patterns": [
            r"\b(COMPUTE|IF|EVALUATE|MULTIPLY|DIVIDE)\b",
        ]
    },
    
    # =========================================================================
    # BSM PATTERNS (16 patterns)
    # =========================================================================
    "bsm_patterns": {
        # SQL (6)
        "SQL_SELECT": {
            "code_pattern": r"EXEC\s+SQL\s+SELECT\s+(.+?)\s+INTO\s+(.+?)\s+FROM\s+(\w+)",
            "doc_keywords": ["select", "retrieve", "fetch", "query"],
            "requirements": ["table_name", "columns"],
        },
        "SQL_INSERT": {
            "code_pattern": r"EXEC\s+SQL\s+INSERT\s+INTO\s+(\w+)",
            "doc_keywords": ["insert", "add", "create"],
            "requirements": ["table_name"],
        },
        "SQL_UPDATE": {
            "code_pattern": r"EXEC\s+SQL\s+UPDATE\s+(\w+)",
            "doc_keywords": ["update", "modify"],
            "requirements": ["table_name"],
        },
        "SQL_DELETE": {
            "code_pattern": r"EXEC\s+SQL\s+DELETE\s+FROM\s+(\w+)",
            "doc_keywords": ["delete", "remove"],
            "requirements": ["table_name"],
        },
        "SQL_CURSOR_OPEN": {
            "code_pattern": r"EXEC\s+SQL\s+OPEN\s+(\w+)",
            "doc_keywords": ["open", "cursor"],
            "requirements": ["cursor_name"],
        },
        "SQL_CURSOR_FETCH": {
            "code_pattern": r"EXEC\s+SQL\s+FETCH\s+(\w+)",
            "doc_keywords": ["fetch", "cursor"],
            "requirements": ["cursor_name"],
        },
        
        # CICS (5)
        "CICS_READ": {
            "code_pattern": r"EXEC\s+CICS\s+READ\s+FILE\s*\(\s*['\"]?(\w+)",
            "doc_keywords": ["read", "retrieve", "cics"],
            "requirements": ["file_name"],
        },
        "CICS_WRITE": {
            "code_pattern": r"EXEC\s+CICS\s+WRITE\s+FILE\s*\(\s*['\"]?(\w+)",
            "doc_keywords": ["write", "store", "cics"],
            "requirements": ["file_name"],
        },
        "CICS_REWRITE": {
            "code_pattern": r"EXEC\s+CICS\s+REWRITE",
            "doc_keywords": ["rewrite", "update", "cics"],
            "requirements": [],
        },
        "CICS_SEND_MAP": {
            "code_pattern": r"EXEC\s+CICS\s+SEND\s+MAP\s*\(\s*['\"]?(\w+)",
            "doc_keywords": ["send", "display", "map"],
            "requirements": ["map_name"],
        },
        "CICS_RECEIVE_MAP": {
            "code_pattern": r"EXEC\s+CICS\s+RECEIVE\s+MAP\s*\(\s*['\"]?(\w+)",
            "doc_keywords": ["receive", "input", "map"],
            "requirements": ["map_name"],
        },
        
        # CALL (2)
        "CALL_STATIC": {
            "code_pattern": r"CALL\s+['\"](\w+)['\"]",
            "doc_keywords": ["call", "invoke", "program"],
            "requirements": ["program_name"],
        },
        "CALL_DYNAMIC": {
            "code_pattern": r"CALL\s+(WS-[\w-]+)",
            "doc_keywords": ["call", "dynamic", "variable"],
            "requirements": ["program_variable"],
        },
        
        # File (3)
        "FILE_READ": {
            "code_pattern": r"READ\s+(\w+)",
            "doc_keywords": ["read", "input", "file"],
            "requirements": ["file_name"],
        },
        "FILE_WRITE": {
            "code_pattern": r"WRITE\s+(\w+)",
            "doc_keywords": ["write", "output", "record"],
            "requirements": ["record_name"],
        },
        "FILE_REWRITE": {
            "code_pattern": r"REWRITE\s+(\w+)",
            "doc_keywords": ["rewrite", "update", "record"],
            "requirements": ["record_name"],
        },
    },
    
    # =========================================================================
    # CRITICAL FAILURES
    # =========================================================================
    "critical_failures": {
        "CF01": {
            "name": "Missing Core Calculations",
            "trigger": "0% CRITICAL business rules documented",
            "threshold": 0.0,
        },
        "CF02": {
            "name": "Hallucinated I/O Variable",
            "trigger": "ANY I/O variable that doesn't exist (COBOL-style with hyphens)",
            "threshold": 0,  # Zero tolerance for I/O hallucinations
            # NOTE: CF02b removed in v2.3.1 - internal vars are often valid abstractions
        },
        "CF03": {
            "name": "Behavioral Contradiction",
            "trigger": ">=50% of claims fail execution verification",
            "threshold": 0.50,
        },
        "CF04": {
            "name": "Missing Error Handlers",
            "trigger": "Error handlers exist but undocumented",
            "threshold": 0.0,
        },
        "CF05": {
            "name": "BSM Pattern Failures",
            "trigger": ">70% external calls incorrectly documented",
            "threshold": 0.70,  # Relaxed from 0.50 to reduce false positives on T2+ programs
        },
    },
    
    # =========================================================================
    # EXECUTION SETTINGS
    # =========================================================================
    "execution": {
        "timeout_seconds": 30,
        "memory_limit_mb": 512,
        "numeric_tolerance": 0.001,
        "docker_image": "legacycodebench/gnucobol:3.2",
    },
    
    # =========================================================================
    # MATCHING SETTINGS
    # =========================================================================
    "matching": {
        "tfidf_weight": 0.60,     # 60% TF-IDF
        "keyword_weight": 0.40,   # 40% keyword
        "similarity_threshold": 0.30,  # Minimum similarity
    },
    
    # =========================================================================
    # READABILITY SETTINGS
    # =========================================================================
    "readability": {
        "target_grade_min": 8,
        "target_grade_max": 12,
        "penalty_per_grade": 0.05,  # 5% penalty per grade outside range
    },
}


def get_weight(track: str) -> float:
    """Get weight for a track."""
    return V213_CONFIG["weights"].get(track, 0.0)


def get_threshold(track: str) -> float:
    """Get pass threshold for a track."""
    return V213_CONFIG["thresholds"].get(track, 0.0)


def get_synonym(word: str) -> str:
    """Get base synonym for a word."""
    word_lower = word.lower()
    for base, alternatives in V213_CONFIG["cobol_synonyms"].items():
        if word_lower == base or word_lower in alternatives:
            return base
    return word_lower


def get_bsm_pattern(pattern_name: str) -> dict:
    """Get BSM pattern configuration."""
    return V213_CONFIG["bsm_patterns"].get(pattern_name, {})
