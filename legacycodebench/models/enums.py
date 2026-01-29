"""
Core Enumerations for LegacyCodeBench V2.4

Specification Reference: TDD_V2.4.md Section 2.1
"""

from enum import Enum


class Language(Enum):
    """Supported legacy languages."""
    COBOL = "cobol"
    UNIBASIC = "unibasic"


class VerificationMode(Enum):
    """
    How behavioral claims were verified.
    
    - EXECUTED: Program compiled and ran, claims verified via execution
    - STATIC: Compilation failed, used static code analysis
    - ERROR: Unexpected failure requiring investigation
    """
    EXECUTED = "executed"
    STATIC = "static"
    ERROR = "error"


class CompileFailureReason(Enum):
    """
    Reason for compilation failure.
    
    Used to determine if failure is fixable and for provenance tracking.
    """
    NONE = "none"                          # Compilation succeeded
    COPYBOOK_MISSING = "copybook_missing"  # COBOL: missing .cpy file
    INCLUDE_MISSING = "include_missing"    # UniBasic: missing $INCLUDE
    IBM_CONSTRUCT = "ibm_construct"        # EXEC CICS, EXEC SQL, etc.
    VENDOR_API = "vendor_api"              # UniVerse/UniData-specific API
    SYNTAX_ERROR = "syntax_error"          # Other compilation error


class RulePriority(Enum):
    """
    Business rule priority levels.
    
    Weights for TF-IDF matching:
    - CRITICAL: 3x weight, threshold 0.50
    - IMPORTANT: 2x weight, threshold 0.40
    - TRIVIAL: 1x weight, threshold 0.30
    """
    CRITICAL = "critical"
    IMPORTANT = "important"
    TRIVIAL = "trivial"


class ClaimType(Enum):
    """
    Types of behavioral claims that can be extracted from documentation.
    
    Used for claim extraction and verification routing.
    """
    CALCULATION = "calculation"   # X = Y * Z
    CONDITIONAL = "conditional"   # IF X THEN Y
    ASSIGNMENT = "assignment"     # X := Y
    RANGE = "range"               # X BETWEEN Y AND Z
    ERROR = "error"               # ON ERROR DO Y
