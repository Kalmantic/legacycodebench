"""
BSM Pattern Library - Checklist patterns for external call documentation.

Each pattern defines:
- What type of call it applies to
- What elements must be documented
- Weight for each element
- Regex hints for extracting facts from code
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ChecklistItem:
    """A single item in a BSM checklist."""
    name: str                   # e.g., "table_name"
    weight: int                 # Points (0-100 total per pattern)
    description: str            # Human-readable description
    extraction_hint: Optional[str] = None  # Regex hint for extracting from code


@dataclass
class BSMPattern:
    """BSM pattern for a specific call type."""
    call_type: str                              # e.g., "EXEC_SQL_SELECT"
    description: str                            # Human-readable description
    checklist: List[ChecklistItem]              # Items to check in documentation
    extraction_regexes: Dict[str, str] = field(default_factory=dict)  # Regex to extract facts


# =============================================================================
# Pattern Definitions
# =============================================================================

BSM_PATTERNS: Dict[str, BSMPattern] = {
    
    # -------------------------------------------------------------------------
    # SQL Patterns
    # -------------------------------------------------------------------------
    
    "EXEC_SQL_SELECT": BSMPattern(
        call_type="EXEC_SQL_SELECT",
        description="SQL SELECT query",
        checklist=[
            ChecklistItem("table_name", 25, 
                         "Documentation must mention the table being queried"),
            ChecklistItem("columns", 20,
                         "Documentation must describe columns/data retrieved"),
            ChecklistItem("filter_logic", 20,
                         "Documentation must explain WHERE conditions or key"),
            ChecklistItem("host_variables", 15,
                         "Documentation must mention COBOL variables receiving data"),
            ChecklistItem("error_handling", 20,
                         "Documentation must mention SQLCODE or error handling"),
        ],
        extraction_regexes={
            "table": r"FROM\s+([A-Z][A-Z0-9_-]+)",
            "columns": r"SELECT\s+(.+?)\s+FROM",
            "where": r"WHERE\s+(.+?)(?:END-EXEC|$)",
            "into": r"INTO\s+(.+?)(?:FROM|$)",
        }
    ),
    
    "EXEC_SQL_INSERT": BSMPattern(
        call_type="EXEC_SQL_INSERT",
        description="SQL INSERT statement",
        checklist=[
            ChecklistItem("table_name", 30,
                         "Documentation must mention target table"),
            ChecklistItem("data_source", 25,
                         "Documentation must describe what data is inserted"),
            ChecklistItem("validation", 20,
                         "Documentation must mention pre-insert validation"),
            ChecklistItem("error_handling", 25,
                         "Documentation must mention SQLCODE or commit handling"),
        ],
        extraction_regexes={
            "table": r"INTO\s+([A-Z][A-Z0-9_-]+)",
            "values": r"VALUES\s*\((.+?)\)",
        }
    ),
    
    "EXEC_SQL_UPDATE": BSMPattern(
        call_type="EXEC_SQL_UPDATE",
        description="SQL UPDATE statement",
        checklist=[
            ChecklistItem("table_name", 25,
                         "Documentation must mention target table"),
            ChecklistItem("columns_updated", 25,
                         "Documentation must describe what columns are changed"),
            ChecklistItem("filter_logic", 25,
                         "Documentation must explain WHERE conditions"),
            ChecklistItem("error_handling", 25,
                         "Documentation must mention SQLCODE or commit handling"),
        ],
        extraction_regexes={
            "table": r"UPDATE\s+([A-Z][A-Z0-9_-]+)",
            "set": r"SET\s+(.+?)\s+WHERE",
            "where": r"WHERE\s+(.+?)(?:END-EXEC|$)",
        }
    ),
    
    "EXEC_SQL_DELETE": BSMPattern(
        call_type="EXEC_SQL_DELETE",
        description="SQL DELETE statement",
        checklist=[
            ChecklistItem("table_name", 30,
                         "Documentation must mention target table"),
            ChecklistItem("filter_logic", 35,
                         "Documentation must explain what records are deleted"),
            ChecklistItem("error_handling", 35,
                         "Documentation must mention SQLCODE or commit handling"),
        ],
        extraction_regexes={
            "table": r"FROM\s+([A-Z][A-Z0-9_-]+)",
            "where": r"WHERE\s+(.+?)(?:END-EXEC|$)",
        }
    ),
    
    # Cursor operations
    "EXEC_SQL_OPEN": BSMPattern(
        call_type="EXEC_SQL_OPEN",
        description="SQL cursor open",
        checklist=[
            ChecklistItem("cursor_name", 40,
                         "Documentation must mention cursor name"),
            ChecklistItem("purpose", 30,
                         "Documentation must explain why cursor is opened"),
            ChecklistItem("error_handling", 30,
                         "Documentation must mention SQLCODE handling"),
        ],
        extraction_regexes={
            "cursor": r"OPEN\s+([A-Z][A-Z0-9_-]+)",
        }
    ),
    
    "EXEC_SQL_FETCH": BSMPattern(
        call_type="EXEC_SQL_FETCH",
        description="SQL cursor fetch",
        checklist=[
            ChecklistItem("cursor_name", 30,
                         "Documentation must mention cursor name"),
            ChecklistItem("data_retrieved", 30,
                         "Documentation must describe fetched data"),
            ChecklistItem("loop_logic", 20,
                         "Documentation must explain fetch loop"),
            ChecklistItem("error_handling", 20,
                         "Documentation must mention SQLCODE or end-of-data"),
        ],
        extraction_regexes={
            "cursor": r"FETCH\s+([A-Z][A-Z0-9_-]+)",
            "into": r"INTO\s+(.+?)(?:END-EXEC|$)",
        }
    ),
    
    # -------------------------------------------------------------------------
    # CICS Patterns
    # -------------------------------------------------------------------------
    
    "EXEC_CICS_SEND": BSMPattern(
        call_type="EXEC_CICS_SEND",
        description="CICS screen send (display)",
        checklist=[
            ChecklistItem("map_name", 25,
                         "Documentation must mention BMS map name"),
            ChecklistItem("mapset", 15,
                         "Documentation must mention mapset"),
            ChecklistItem("screen_fields", 25,
                         "Documentation must describe displayed fields"),
            ChecklistItem("data_binding", 20,
                         "Documentation must explain field-to-variable mapping"),
            ChecklistItem("resp_handling", 15,
                         "Documentation must mention RESP code handling"),
        ],
        extraction_regexes={
            "map": r"MAP\s*\(\s*['\"]?([A-Z][A-Z0-9]+)",
            "mapset": r"MAPSET\s*\(\s*['\"]?([A-Z][A-Z0-9]+)",
        }
    ),
    
    "EXEC_CICS_RECEIVE": BSMPattern(
        call_type="EXEC_CICS_RECEIVE",
        description="CICS screen receive (input)",
        checklist=[
            ChecklistItem("map_name", 25,
                         "Documentation must mention BMS map name"),
            ChecklistItem("input_fields", 30,
                         "Documentation must describe user input fields"),
            ChecklistItem("validation", 25,
                         "Documentation must explain input validation"),
            ChecklistItem("resp_handling", 20,
                         "Documentation must mention RESP code handling"),
        ],
        extraction_regexes={
            "map": r"MAP\s*\(\s*['\"]?([A-Z][A-Z0-9]+)",
            "into": r"INTO\s*\(\s*([A-Z][A-Z0-9-]+)",
        }
    ),
    
    "EXEC_CICS_READ": BSMPattern(
        call_type="EXEC_CICS_READ",
        description="CICS file/VSAM read",
        checklist=[
            ChecklistItem("file_name", 30,
                         "Documentation must mention file/dataset name"),
            ChecklistItem("key_field", 25,
                         "Documentation must explain record key"),
            ChecklistItem("data_retrieved", 25,
                         "Documentation must describe retrieved data"),
            ChecklistItem("resp_handling", 20,
                         "Documentation must mention RESP handling"),
        ],
        extraction_regexes={
            "file": r"FILE\s*\(\s*['\"]?([A-Z][A-Z0-9]+)",
            "ridfld": r"RIDFLD\s*\(\s*([A-Z][A-Z0-9-]+)",
        }
    ),
    
    "EXEC_CICS_WRITE": BSMPattern(
        call_type="EXEC_CICS_WRITE",
        description="CICS file/VSAM write",
        checklist=[
            ChecklistItem("file_name", 30,
                         "Documentation must mention file/dataset name"),
            ChecklistItem("data_written", 30,
                         "Documentation must describe data being written"),
            ChecklistItem("key_field", 20,
                         "Documentation must explain record key"),
            ChecklistItem("resp_handling", 20,
                         "Documentation must mention RESP handling"),
        ],
        extraction_regexes={
            "file": r"FILE\s*\(\s*['\"]?([A-Z][A-Z0-9]+)",
            "from": r"FROM\s*\(\s*([A-Z][A-Z0-9-]+)",
        }
    ),
    
    "EXEC_CICS_LINK": BSMPattern(
        call_type="EXEC_CICS_LINK",
        description="CICS program link (call with return)",
        checklist=[
            ChecklistItem("program_name", 30,
                         "Documentation must name the linked program"),
            ChecklistItem("purpose", 25,
                         "Documentation must explain why program is called"),
            ChecklistItem("commarea", 25,
                         "Documentation must describe COMMAREA data"),
            ChecklistItem("resp_handling", 20,
                         "Documentation must mention RESP handling"),
        ],
        extraction_regexes={
            "program": r"PROGRAM\s*\(\s*['\"]?([A-Z][A-Z0-9]+)",
            "commarea": r"COMMAREA\s*\(\s*([A-Z][A-Z0-9-]+)",
        }
    ),
    
    "EXEC_CICS_XCTL": BSMPattern(
        call_type="EXEC_CICS_XCTL",
        description="CICS program transfer (no return)",
        checklist=[
            ChecklistItem("program_name", 35,
                         "Documentation must name the target program"),
            ChecklistItem("purpose", 25,
                         "Documentation must explain transfer reason"),
            ChecklistItem("commarea", 25,
                         "Documentation must describe COMMAREA data"),
            ChecklistItem("resp_handling", 15,
                         "Documentation must mention RESP handling"),
        ],
        extraction_regexes={
            "program": r"PROGRAM\s*\(\s*['\"]?([A-Z][A-Z0-9]+)",
            "commarea": r"COMMAREA\s*\(\s*([A-Z][A-Z0-9-]+)",
        }
    ),
    
    # -------------------------------------------------------------------------
    # CALL Patterns
    # -------------------------------------------------------------------------
    
    "CALL_STATIC": BSMPattern(
        call_type="CALL_STATIC",
        description="Static program CALL",
        checklist=[
            ChecklistItem("program_name", 25,
                         "Documentation must name the called program"),
            ChecklistItem("purpose", 20,
                         "Documentation must explain why program is called"),
            ChecklistItem("parameters", 25,
                         "Documentation must describe USING parameters"),
            ChecklistItem("return_values", 15,
                         "Documentation must explain return values"),
            ChecklistItem("error_handling", 15,
                         "Documentation must mention call failure handling"),
        ],
        extraction_regexes={
            "program": r"CALL\s+['\"]([A-Z][A-Z0-9-]+)['\"]",
            "params": r"USING\s+(.+?)(?:\.|END-CALL|$)",
        }
    ),
    
    "CALL_DYNAMIC": BSMPattern(
        call_type="CALL_DYNAMIC",
        description="Dynamic program CALL",
        checklist=[
            ChecklistItem("program_variable", 25,
                         "Documentation must mention program name variable"),
            ChecklistItem("purpose", 20,
                         "Documentation must explain call purpose"),
            ChecklistItem("parameters", 25,
                         "Documentation must describe USING parameters"),
            ChecklistItem("return_values", 15,
                         "Documentation must explain return values"),
            ChecklistItem("error_handling", 15,
                         "Documentation must mention call failure handling"),
        ],
        extraction_regexes={
            "program_var": r"CALL\s+([A-Z][A-Z0-9-]+)",
            "params": r"USING\s+(.+?)(?:\.|END-CALL|$)",
        }
    ),
    
    # -------------------------------------------------------------------------
    # File Patterns
    # -------------------------------------------------------------------------
    
    "FILE_READ": BSMPattern(
        call_type="FILE_READ",
        description="File READ operation",
        checklist=[
            ChecklistItem("file_name", 25,
                         "Documentation must mention file/dataset name"),
            ChecklistItem("record_format", 25,
                         "Documentation must describe record structure"),
            ChecklistItem("key_field", 20,
                         "Documentation must explain key for indexed files"),
            ChecklistItem("eof_handling", 15,
                         "Documentation must describe end-of-file handling"),
            ChecklistItem("error_handling", 15,
                         "Documentation must mention file status checking"),
        ],
        extraction_regexes={
            "file": r"READ\s+([A-Z][A-Z0-9-]+)",
            "key": r"KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]+)",
        }
    ),
    
    "FILE_WRITE": BSMPattern(
        call_type="FILE_WRITE",
        description="File WRITE operation",
        checklist=[
            ChecklistItem("file_name", 25,
                         "Documentation must mention file/dataset name"),
            ChecklistItem("record_format", 25,
                         "Documentation must describe record structure"),
            ChecklistItem("data_written", 25,
                         "Documentation must explain what data is written"),
            ChecklistItem("error_handling", 25,
                         "Documentation must mention file status checking"),
        ],
        extraction_regexes={
            "file": r"WRITE\s+([A-Z][A-Z0-9-]+)",
            "from": r"FROM\s+([A-Z][A-Z0-9-]+)",
        }
    ),
}


def get_pattern(call_type: str) -> Optional[BSMPattern]:
    """Get BSM pattern for a call type."""
    return BSM_PATTERNS.get(call_type)


def get_all_patterns() -> Dict[str, BSMPattern]:
    """Get all BSM patterns."""
    return BSM_PATTERNS.copy()


def get_patterns_by_category(category: str) -> Dict[str, BSMPattern]:
    """Get BSM patterns for a category (SQL, CICS, CALL, FILE)."""
    return {k: v for k, v in BSM_PATTERNS.items() if category.upper() in k}
