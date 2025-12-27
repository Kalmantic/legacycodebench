"""
LegacyCodeBench V2.3 Configuration

Scoring weights, thresholds, and configuration for V2.3 evaluation.
"""

V23_CONFIG = {
    "version": "2.3.0",
    
    # =========================================================================
    # SCORING WEIGHTS (40/25/35)
    # =========================================================================
    "weights": {
        "comprehension": 0.40,   # Understanding business logic
        "documentation": 0.25,   # Quality of documentation
        "behavioral": 0.35       # Execution-based verification
    },
    
    # =========================================================================
    # PASS THRESHOLDS
    # =========================================================================
    "thresholds": {
        "comprehension": 0.70,   # Must score >= 70%
        "documentation": 0.60,   # Must score >= 60%
        "behavioral": 0.50       # Must score >= 50%
    },
    
    # =========================================================================
    # COMPREHENSION SUB-WEIGHTS
    # =========================================================================
    "comprehension_weights": {
        "business_rules": 0.40,   # Coverage of business rules
        "data_flow": 0.25,        # Understanding data structures
        "abstraction": 0.20,      # WHY vs WHAT explanations
        "dependencies": 0.15      # External dependencies documented
    },
    
    # =========================================================================
    # DOCUMENTATION SUB-WEIGHTS
    # =========================================================================
    "documentation_weights": {
        "structural": 0.40,       # Completeness of structure
        "semantic": 0.35,         # Quality and accuracy
        "traceability": 0.25      # Line number citations
    },
    
    # =========================================================================
    # BEHAVIORAL SUB-WEIGHTS (for MIXED paragraphs)
    # =========================================================================
    "behavioral_weights": {
        "logic_regeneration": 0.60,  # Template-based regeneration
        "bsm_validation": 0.40       # BSM pattern matching
    },
    
    # =========================================================================
    # ANTI-GAMING THRESHOLDS
    # =========================================================================
    "anti_gaming": {
        "keyword_stuffing_threshold": 0.30,  # Penalty starts at 30%
        "keyword_stuffing_max_penalty": 0.20, # Max 20% score reduction
        "parroting_threshold": 0.50,          # Penalty starts at 50%
        "parroting_max_penalty": 0.30,        # Max 30% score reduction
        "abstraction_minimum": 0.40,          # Min 40% abstraction
        "abstraction_penalty": 0.20           # 20% penalty if below
    },
    
    # =========================================================================
    # PARAGRAPH CLASSIFICATION
    # =========================================================================
    "classification": {
        # Paragraph names matching these are INFRASTRUCTURE
        "infrastructure_patterns": [
            r"^\d{4}-OPEN[-_]",
            r"^\d{4}-CLOSE[-_]",
            r"^\d{4}-READ[-_]",
            r"^\d{4}-WRITE[-_]",
            r"^\d{4}-INIT",
            r"^\d{4}-INITIALIZE",
            r"^\d{4}-HOUSEKEEP",
            r"^\d{4}-HOUSE[-_]KEEP",
            r"^\d{4}-EOJ",
            r"^\d{4}-END[-_]OF[-_]JOB",
            r"^\d{4}-TERMINATION",
            r"^\d{4}-MAINLINE",
            r"^\d{4}-MAIN[-_]PROCESS",
            r"^0000[-_]MAIN",
            r"^9999[-_]",
        ],
        
        # Code patterns indicating external calls (blocking for PURE)
        "external_patterns": [
            r"EXEC\s+SQL",
            r"EXEC\s+CICS",
            r"EXEC\s+DLI",
            r"CALL\s+['\"]",
            r"CALL\s+WS-",
        ],
        
        # Code patterns indicating business logic
        "logic_patterns": [
            r"\bCOMPUTE\b",
            r"\bMULTIPLY\b",
            r"\bDIVIDE\b",
            r"\bADD\b(?!\s+TO\s+ADDRESS)",
            r"\bSUBTRACT\b",
            r"\bIF\b",
            r"\bEVALUATE\b",
        ]
    },
    
    # =========================================================================
    # BSM PATTERNS (16 patterns)
    # =========================================================================
    "bsm_patterns": {
        "SQL_SELECT": {
            "code_pattern": r"EXEC\s+SQL\s+SELECT\s+(.+?)\s+INTO\s+:(.+?)\s+FROM\s+(\w+)",
            "requirements": ["table_name", "columns", "target_variables", "where_conditions"]
        },
        "SQL_INSERT": {
            "code_pattern": r"EXEC\s+SQL\s+INSERT\s+INTO\s+(\w+)",
            "requirements": ["table_name", "columns", "source_variables"]
        },
        "SQL_UPDATE": {
            "code_pattern": r"EXEC\s+SQL\s+UPDATE\s+(\w+)",
            "requirements": ["table_name", "columns_updated", "where_conditions"]
        },
        "SQL_DELETE": {
            "code_pattern": r"EXEC\s+SQL\s+DELETE\s+FROM\s+(\w+)",
            "requirements": ["table_name", "where_conditions"]
        },
        "SQL_CURSOR_DECLARE": {
            "code_pattern": r"EXEC\s+SQL\s+DECLARE\s+(\w+)\s+CURSOR",
            "requirements": ["cursor_name", "select_statement"]
        },
        "SQL_CURSOR_OPEN": {
            "code_pattern": r"EXEC\s+SQL\s+OPEN\s+(\w+)",
            "requirements": ["cursor_name"]
        },
        "SQL_CURSOR_FETCH": {
            "code_pattern": r"EXEC\s+SQL\s+FETCH\s+(\w+)",
            "requirements": ["cursor_name", "into_variables"]
        },
        "SQL_CURSOR_CLOSE": {
            "code_pattern": r"EXEC\s+SQL\s+CLOSE\s+(\w+)",
            "requirements": ["cursor_name"]
        },
        "CICS_READ": {
            "code_pattern": r"EXEC\s+CICS\s+READ\s+FILE\s*\(\s*['\"]?(\w+)",
            "requirements": ["file_name", "ridfld", "into_variable"]
        },
        "CICS_WRITE": {
            "code_pattern": r"EXEC\s+CICS\s+WRITE\s+FILE\s*\(\s*['\"]?(\w+)",
            "requirements": ["file_name", "from_variable", "ridfld"]
        },
        "CICS_REWRITE": {
            "code_pattern": r"EXEC\s+CICS\s+REWRITE\s+FILE\s*\(\s*['\"]?(\w+)",
            "requirements": ["file_name", "from_variable"]
        },
        "CICS_DELETE": {
            "code_pattern": r"EXEC\s+CICS\s+DELETE\s+FILE\s*\(\s*['\"]?(\w+)",
            "requirements": ["file_name", "ridfld"]
        },
        "CICS_SEND": {
            "code_pattern": r"EXEC\s+CICS\s+SEND\s+(MAP|TEXT)",
            "requirements": ["map_name", "from_variable"]
        },
        "CICS_RECEIVE": {
            "code_pattern": r"EXEC\s+CICS\s+RECEIVE\s+MAP",
            "requirements": ["map_name", "into_variable"]
        },
        "CALL_STATIC": {
            "code_pattern": r"CALL\s+['\"](\w+)['\"]",
            "requirements": ["program_name", "parameters", "return_values"]
        },
        "CALL_DYNAMIC": {
            "code_pattern": r"CALL\s+(WS-\w+)",
            "requirements": ["program_variable", "parameters", "dispatch_reason"]
        }
    },
    
    # =========================================================================
    # CRITICAL FAILURES
    # =========================================================================
    "critical_failures": {
        "CF01": {
            "name": "Missing Core Calculations",
            "trigger": "0% of CRITICAL business rules documented",
            "applies_to": "comprehension"
        },
        "CF02": {
            "name": "Hallucinated Logic",
            "trigger": "Documentation describes logic not in source",
            "applies_to": "comprehension"
        },
        "CF03": {
            "name": "Wrong Transformation",
            "trigger": ">50% of executed test outputs differ",
            "applies_to": "behavioral"
        },
        "CF04": {
            "name": "Missing Error Handling",
            "trigger": "Error handlers exist but undocumented",
            "applies_to": "comprehension"
        },
        "CF05": {
            "name": "BSM Specification Failure",
            "trigger": ">50% of external calls incorrectly specified",
            "applies_to": "behavioral"
        },
        "CF06": {
            "name": "Semantic Contradiction",
            "trigger": "Documentation contradicts source logic",
            "applies_to": "comprehension"
        }
    },
    
    # =========================================================================
    # EXECUTION SETTINGS
    # =========================================================================
    "execution": {
        "timeout_seconds": 30,
        "memory_limit_mb": 512,
        "max_test_cases": 10,
        "numeric_tolerance": 0.001,
        "docker_network": "none"
    },
    
    # =========================================================================
    # LLM SETTINGS
    # =========================================================================
    "llm": {
        "temperature": 0,           # Deterministic output
        "max_retries": 3,
        "judge_model": "gpt-4o",    # For quality assessment
        "generation_model": None    # Use default from config
    }
}


def get_weight(track: str) -> float:
    """Get weight for a track."""
    return V23_CONFIG["weights"].get(track, 0.0)


def get_threshold(track: str) -> float:
    """Get pass threshold for a track."""
    return V23_CONFIG["thresholds"].get(track, 0.5)


def get_bsm_pattern(pattern_name: str) -> dict:
    """Get BSM pattern configuration."""
    return V23_CONFIG["bsm_patterns"].get(pattern_name, {})


def get_classification_patterns() -> dict:
    """Get paragraph classification patterns."""
    return V23_CONFIG["classification"]
