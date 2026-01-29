"""
Compile Error Classification

Analyzes GnuCOBOL compiler errors to determine:
1. What type of error occurred (missing copybook, IBM construct, etc.)
2. Whether the issue is fixable
3. What IBM middleware is being used (CICS, DB2, IMS, MQ)

This is the "compile-first classification" logic from BF V3.0.
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CompileClassification:
    """
    Result of analyzing a compilation error.
    
    Attributes:
        error_type: Type of error (copybook_missing, ibm_construct, compile_error)
        reason: Human-readable reason string for BFResult
        fixable: Whether this might be fixable
        ibm_constructs: List of IBM middleware detected (e.g., ["CICS", "DB2"])
        missing_files: List of missing copybooks detected
    """
    error_type: str  # "copybook_missing" | "ibm_construct" | "compile_error"
    reason: str  # For BFResult.mode_reason
    fixable: Optional[bool]  # True=fixable, False=not fixable, None=unknown
    ibm_constructs: List[str]  # e.g., ["CICS", "DB2"]
    missing_files: List[str]  # e.g., ["CIPAUSMY.cpy"]


def classify_compile_error(error: str) -> CompileClassification:
    """
    Classify a GnuCOBOL compilation error.
    
    This is the core of compile-first classification. The compiler tells us
    exactly what's wrong, and we classify it into actionable categories.
    
    Args:
        error: Full compiler error output
        
    Returns:
        CompileClassification with error type, reason, and details
        
    Examples:
        >>> classify_compile_error("CIPAUSMY: No such file or directory")
        CompileClassification(error_type="copybook_missing", reason="copybook_missing:CIPAUSMY", ...)
        
        >>> classify_compile_error("unknown statement 'EXEC'")
        CompileClassification(error_type="ibm_construct", reason="ibm_construct:EXEC", ...)
    """
    logger.debug(f"Classifying compile error: {error[:200]}...")
    
    # Extract missing files (copybooks)
    missing_files = extract_missing_files(error)
    
    # Extract IBM constructs
    ibm_constructs = get_ibm_constructs(error)
    
    # Classify based on what we found
    # Priority:
    # 1. IBM-specific missing files (SQLCA, DFHAID) → ibm_construct
    # 2. IBM constructs in error (EXEC CICS/SQL/DLI) → ibm_construct  
    # 3. Regular missing copybooks → copybook_missing
    # 4. Generic compile error
    
    # Check if missing files are IBM-specific (e.g., SQLCA for DB2)
    if missing_files and is_ibm_specific_missing_file(missing_files):
        # IBM-specific copybook - indicates IBM middleware usage
        # Detect which middleware based on the copybook name
        inferred_constructs = []
        for mf in missing_files:
            mf_upper = mf.upper()
            if mf_upper in ("SQLCA", "SQLDA", "SQLCODE"):
                inferred_constructs.append("DB2")
            elif mf_upper.startswith("DFH"):
                inferred_constructs.append("CICS")
            elif mf_upper in ("DLIUIB", "PCB"):
                inferred_constructs.append("IMS")
            elif mf_upper.startswith("CMQ"):
                inferred_constructs.append("MQ")
        
        all_constructs = list(set(ibm_constructs + inferred_constructs))
        
        return CompileClassification(
            error_type="ibm_construct",
            reason=f"ibm_construct:{','.join(all_constructs) if all_constructs else 'INFERRED'}",
            fixable=False,
            ibm_constructs=all_constructs,
            missing_files=missing_files,
        )
    
    if ibm_constructs:
        # IBM middleware - not fixable with GnuCOBOL
        return CompileClassification(
            error_type="ibm_construct",
            reason=f"ibm_construct:{','.join(ibm_constructs)}",
            fixable=False,
            ibm_constructs=ibm_constructs,
            missing_files=missing_files,
        )
    
    # Regular missing copybooks (not IBM-specific)
    if missing_files:
        return CompileClassification(
            error_type="copybook_missing",
            reason=f"copybook_missing:{','.join(missing_files[:3])}",  # Limit to 3
            fixable=True,
            ibm_constructs=[],
            missing_files=missing_files,
        )
    
    # Generic compile error (syntax errors, etc.)
    return CompileClassification(
        error_type="compile_error",
        reason="compile_error",
        fixable=None,  # Unknown - might be cascading, might be real
        ibm_constructs=[],
        missing_files=[],
    )


def extract_missing_files(error: str) -> List[str]:
    """
    Extract missing file names from compiler error.
    
    GnuCOBOL reports missing copybooks like:
        "CIPAUSMY: No such file or directory"
        "CSUTLDWY.cpy: No such file or directory"
    
    Args:
        error: Compiler error output
        
    Returns:
        List of missing file names (without path)
    """
    missing = []
    
    # Pattern: FILENAME: No such file or directory
    # Also handles: FILENAME.cpy: No such file
    pattern = r'([A-Z0-9_\-]+(?:\.cpy)?)\s*:\s*No such file'
    
    for match in re.finditer(pattern, error, re.IGNORECASE):
        filename = match.group(1).upper()
        # Remove .cpy extension for consistency
        if filename.endswith('.CPY'):
            filename = filename[:-4]
        if filename not in missing:
            missing.append(filename)
    
    return missing


def get_ibm_constructs(error: str) -> List[str]:
    """
    Detect ALL IBM-specific constructs in compiler error.
    
    Returns list of constructs found (e.g., ["CICS", "DB2"]).
    
    This handles programs that use multiple IBM middleware:
    - CICS + DB2 (common for online transactions with database)
    - CICS + MQ (message queue integration)
    - IMS + DB2 (batch with database)
    
    Args:
        error: Compiler error output
        
    Returns:
        List of IBM middleware names detected
    """
    constructs = []
    error_upper = error.upper()
    
    # CICS detection
    cics_patterns = [
        "EXEC CICS",
        "DFHCOMMAREA",
        "DFHRESP",
        "DFHVALUE",
        "EIBCALEN",
        "EIBTRNID",
        "DFHAID",
        "DFHBMSCA",
    ]
    if any(p in error_upper for p in cics_patterns):
        constructs.append("CICS")
    
    # DB2/SQL detection
    sql_patterns = [
        "EXEC SQL",
        "SQLCA",
        "SQLCODE",
        "INCLUDE SQLCA",
    ]
    if any(p in error_upper for p in sql_patterns):
        constructs.append("DB2")
    
    # IMS detection
    ims_patterns = [
        "EXEC DLI",
        "CBLTDLI",
        "AIBTDLI",
        "PCB MASK",
    ]
    if any(p in error_upper for p in ims_patterns):
        constructs.append("IMS")
    
    # MQ detection
    mq_patterns = [
        "MQOPEN",
        "MQGET",
        "MQPUT",
        "MQCLOSE",
        "MQCONN",
    ]
    if any(p in error_upper for p in mq_patterns):
        constructs.append("MQ")
    
    # Fallback: generic EXEC statement not recognized
    if not constructs and "unknown statement 'exec'" in error.lower():
        constructs.append("EXEC")
    
    return constructs


def is_ibm_error_before_missing_file(error: str) -> bool:
    """
    Check if IBM construct error appears before missing file error.
    
    This helps determine the ROOT cause. If the compiler complains about
    EXEC before complaining about missing files, the IBM construct is
    the primary issue.
    
    Args:
        error: Compiler error output
        
    Returns:
        True if IBM error appears first, False otherwise
    """
    error_lower = error.lower()
    
    # Find position of first IBM error
    ibm_pos = len(error)
    for pattern in ["unknown statement 'exec'", "exec cics", "exec sql", "exec dli"]:
        pos = error_lower.find(pattern)
        if pos != -1 and pos < ibm_pos:
            ibm_pos = pos
    
    # Find position of first missing file error
    file_pos = error_lower.find("no such file")
    
    # IBM error came first
    return ibm_pos < file_pos if file_pos != -1 else ibm_pos != len(error)


def is_ibm_specific_missing_file(missing_files: List[str]) -> bool:
    """
    Check if missing files are IBM-specific (e.g., SQLCA, DFHAID).
    
    These files indicate IBM middleware usage even if EXEC hasn't been
    encountered yet.
    
    Args:
        missing_files: List of missing file names
        
    Returns:
        True if any missing file is IBM-specific
    """
    # IBM-specific copybooks that indicate middleware usage
    ibm_copybooks = {
        # DB2/SQL
        "SQLCA", "SQLDA", "SQLCODE",
        # CICS
        "DFHAID", "DFHBMSCA", "DFHCOMMAREA", "DFHRESP", "DFHVALUE",
        "DFHEIBLK", "DFHEIVAR",
        # IMS
        "DLIUIB", "PCB",
        # MQ
        "CMQV", "CMQC",
    }
    
    for missing in missing_files:
        if missing.upper() in ibm_copybooks:
            return True
    
    return False


def is_missing_copybook_error(error: str) -> bool:
    """
    Quick check if error is primarily due to missing copybook.
    
    Args:
        error: Compiler error output
        
    Returns:
        True if missing copybook is the primary issue
    """
    classification = classify_compile_error(error)
    return classification.error_type == "copybook_missing"


def is_ibm_construct_error(error: str) -> bool:
    """
    Quick check if error is due to IBM-specific constructs.
    
    Args:
        error: Compiler error output
        
    Returns:
        True if IBM middleware is detected
    """
    classification = classify_compile_error(error)
    return classification.error_type == "ibm_construct"


def extract_missing_file(error: str) -> Optional[str]:
    """
    Extract the first missing file name from error.
    
    Convenience function for simple cases.
    
    Args:
        error: Compiler error output
        
    Returns:
        First missing file name, or None if not found
    """
    missing = extract_missing_files(error)
    return missing[0] if missing else None
