"""
Document Matcher - Match AI documentation against BSM checklists.

This is the core of BSM evaluation:
1. Extract facts from COBOL source (table names, map names, etc.)
2. Check if documentation mentions each required element
3. Score based on checklist coverage
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

from .pattern_library import BSMPattern, ChecklistItem
from .call_detector import ExternalCall

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of matching a single checklist item."""
    item_name: str
    weight: int
    matched: bool
    evidence: Optional[str]     # Text that matched (for audit)
    description: str            # From checklist item


@dataclass
class CallMatchResult:
    """Result of matching all checklist items for one call."""
    call_type: str
    line_number: int
    score: float                # 0-1 score for this call
    items: List[MatchResult]    # Individual item results
    facts_extracted: Dict       # Facts extracted from code


class FactExtractor:
    """Extract facts from COBOL source code based on pattern regexes."""
    
    def extract(self, call: ExternalCall, pattern: BSMPattern) -> Dict[str, any]:
        """
        Extract facts from an external call using pattern's regexes.
        
        Args:
            call: The detected external call
            pattern: BSM pattern with extraction regexes
            
        Returns:
            Dict of extracted facts (e.g., {"table": "ACCOUNTS", "columns": ["BAL", "RATE"]})
        """
        facts = {}
        content = call.content + " " + call.raw_match  # Search in both
        
        for fact_name, regex in pattern.extraction_regexes.items():
            matches = re.findall(regex, content, re.IGNORECASE)
            if matches:
                if len(matches) == 1:
                    facts[fact_name] = matches[0].strip() if isinstance(matches[0], str) else matches[0]
                else:
                    facts[fact_name] = [m.strip() if isinstance(m, str) else m for m in matches]
        
        return facts


class DocumentMatcher:
    """
    Match AI documentation against BSM checklist items.
    
    Each checklist item is checked deterministically:
    - Look for specific terms/patterns in documentation
    - No LLM judgment - pure pattern matching
    """
    
    # Keywords that indicate error handling is documented
    ERROR_KEYWORDS = [
        'sqlcode', 'resp', 'response', 'error', 'exception', 
        'fail', 'failure', 'status', 'handle', 'check', 'validate',
        'invalid', 'not found', 'notfnd', 'duprec', 'rollback'
    ]
    
    # Keywords for validation
    VALIDATION_KEYWORDS = [
        'validate', 'check', 'verify', 'ensure', 'confirm',
        'must be', 'should be', 'required', 'mandatory'
    ]
    
    def __init__(self):
        self.fact_extractor = FactExtractor()
    
    def match_call(self, call: ExternalCall, 
                   pattern: BSMPattern,
                   documentation: str) -> CallMatchResult:
        """
        Match documentation against all checklist items for a call.
        
        Args:
            call: The external call to evaluate
            pattern: BSM pattern for this call type
            documentation: AI-generated documentation
            
        Returns:
            CallMatchResult with score and item details
        """
        # Extract facts from code
        facts = self.fact_extractor.extract(call, pattern)
        
        # Match each checklist item
        item_results = []
        for item in pattern.checklist:
            matched, evidence = self._check_item(item, facts, documentation)
            item_results.append(MatchResult(
                item_name=item.name,
                weight=item.weight,
                matched=matched,
                evidence=evidence,
                description=item.description
            ))
        
        # Calculate score
        total_weight = sum(r.weight for r in item_results)
        earned_weight = sum(r.weight for r in item_results if r.matched)
        score = earned_weight / total_weight if total_weight > 0 else 0.0
        
        return CallMatchResult(
            call_type=call.call_type,
            line_number=call.line_number,
            score=score,
            items=item_results,
            facts_extracted=facts
        )
    
    def _check_item(self, item: ChecklistItem, 
                    facts: Dict, 
                    documentation: str) -> tuple:
        """
        Check if a single checklist item is covered in documentation.
        
        Returns:
            (matched: bool, evidence: Optional[str])
        """
        doc_lower = documentation.lower()
        
        # =====================================================================
        # Table/File/Map name checks
        # =====================================================================
        
        if item.name in ["table_name", "file_name", "map_name"]:
            # Get the actual name from facts
            name = facts.get("table") or facts.get("file") or facts.get("map")
            if name and isinstance(name, str):
                if name.lower() in doc_lower:
                    return True, f"Found '{name}'"
            return False, None
        
        if item.name == "mapset":
            mapset = facts.get("mapset")
            if mapset and isinstance(mapset, str):
                if mapset.lower() in doc_lower:
                    return True, f"Found mapset '{mapset}'"
            return False, None
        
        if item.name == "cursor_name":
            cursor = facts.get("cursor")
            if cursor and isinstance(cursor, str):
                if cursor.lower() in doc_lower:
                    return True, f"Found cursor '{cursor}'"
            return False, None
        
        if item.name == "program_name":
            program = facts.get("program")
            if program and isinstance(program, str):
                if program.lower() in doc_lower:
                    return True, f"Found program '{program}'"
            return False, None
        
        if item.name == "program_variable":
            prog_var = facts.get("program_var")
            if prog_var and isinstance(prog_var, str):
                if prog_var.lower() in doc_lower:
                    return True, f"Found program variable '{prog_var}'"
            return False, None
        
        # =====================================================================
        # Columns/Fields checks
        # =====================================================================
        
        if item.name in ["columns", "columns_updated"]:
            columns_str = facts.get("columns") or facts.get("set")
            if columns_str:
                # Parse column names from SQL
                if isinstance(columns_str, str):
                    col_names = re.findall(r'([A-Z][A-Z0-9_-]+)', columns_str, re.IGNORECASE)
                    if col_names:
                        mentioned = sum(1 for c in col_names if c.lower() in doc_lower)
                        if mentioned >= len(col_names) * 0.3:  # At least 30% mentioned
                            return True, f"Found {mentioned}/{len(col_names)} columns"
            return False, None
        
        # =====================================================================
        # Filter/Key logic checks
        # =====================================================================
        
        if item.name in ["filter_logic", "key_field"]:
            # Check for WHERE-related keywords
            filter_keywords = ['where', 'filter', 'condition', 'criteria', 'key', 'match', 'equal']
            
            # Also check for actual key field from code
            key = facts.get("where") or facts.get("key") or facts.get("ridfld")
            if key and isinstance(key, str):
                key_vars = re.findall(r'([A-Z][A-Z0-9_-]+)', key, re.IGNORECASE)
                if any(v.lower() in doc_lower for v in key_vars):
                    return True, f"Found key variable reference"
            
            if any(kw in doc_lower for kw in filter_keywords):
                return True, "Found filter/key keywords"
            return False, None
        
        # =====================================================================
        # Host variables / data binding checks
        # =====================================================================
        
        if item.name in ["host_variables", "data_binding"]:
            # Check for COBOL variable naming patterns
            if re.search(r'ws-\w+|working.storage|cobol.variable', doc_lower):
                return True, "Found COBOL variable references"
            
            # Check for INTO clause variables
            into_vars = facts.get("into")
            if into_vars and isinstance(into_vars, str):
                var_names = re.findall(r':?([A-Z][A-Z0-9-]+)', into_vars, re.IGNORECASE)
                if any(v.lower() in doc_lower for v in var_names):
                    return True, "Found host variable reference"
            return False, None
        
        # =====================================================================
        # Error handling checks
        # =====================================================================
        
        if item.name in ["error_handling", "resp_handling"]:
            if any(kw in doc_lower for kw in self.ERROR_KEYWORDS):
                return True, "Found error handling keywords"
            return False, None
        
        # =====================================================================
        # Validation checks
        # =====================================================================
        
        if item.name == "validation":
            if any(kw in doc_lower for kw in self.VALIDATION_KEYWORDS):
                return True, "Found validation keywords"
            return False, None
        
        # =====================================================================
        # Input/Output field checks
        # =====================================================================
        
        if item.name in ["screen_fields", "input_fields"]:
            screen_keywords = ['field', 'screen', 'display', 'input', 'output', 'enter', 'show']
            if sum(1 for kw in screen_keywords if kw in doc_lower) >= 2:
                return True, "Found screen field descriptions"
            return False, None
        
        if item.name in ["data_retrieved", "data_written", "data_source"]:
            data_keywords = ['data', 'record', 'information', 'value', 'content', 'retrieve', 'fetch', 'store', 'insert']
            if sum(1 for kw in data_keywords if kw in doc_lower) >= 2:
                return True, "Found data description"
            return False, None
        
        # =====================================================================
        # Purpose checks
        # =====================================================================
        
        if item.name == "purpose":
            purpose_keywords = ['purpose', 'used for', 'performs', 'responsible', 'handles', 'processes', 'function']
            if any(kw in doc_lower for kw in purpose_keywords):
                return True, "Found purpose description"
            return False, None
        
        # =====================================================================
        # Parameter checks
        # =====================================================================
        
        if item.name == "parameters":
            param_keywords = ['parameter', 'argument', 'pass', 'using', 'input', 'output', 'receive', 'return']
            if any(kw in doc_lower for kw in param_keywords):
                return True, "Found parameter descriptions"
            
            # Check for actual parameter variables
            params = facts.get("params")
            if params and isinstance(params, str):
                param_vars = re.findall(r'([A-Z][A-Z0-9-]+)', params, re.IGNORECASE)
                if any(v.lower() in doc_lower for v in param_vars):
                    return True, "Found parameter variable reference"
            return False, None
        
        if item.name == "return_values":
            return_keywords = ['return', 'result', 'output', 'response', 'code', 'status']
            if any(kw in doc_lower for kw in return_keywords):
                return True, "Found return value description"
            return False, None
        
        # =====================================================================
        # COMMAREA checks
        # =====================================================================
        
        if item.name == "commarea":
            if 'commarea' in doc_lower or 'communication area' in doc_lower:
                return True, "Found COMMAREA reference"
            
            commarea = facts.get("commarea")
            if commarea and isinstance(commarea, str):
                if commarea.lower() in doc_lower:
                    return True, f"Found COMMAREA '{commarea}'"
            return False, None
        
        # =====================================================================
        # Loop/EOF handling checks
        # =====================================================================
        
        if item.name == "loop_logic":
            loop_keywords = ['loop', 'iterate', 'each', 'until', 'while', 'next', 'continue']
            if any(kw in doc_lower for kw in loop_keywords):
                return True, "Found loop description"
            return False, None
        
        if item.name == "eof_handling":
            eof_keywords = ['end of file', 'eof', 'end-of-file', 'at end', 'no more', 'last record']
            if any(kw in doc_lower for kw in eof_keywords):
                return True, "Found EOF handling description"
            return False, None
        
        # =====================================================================
        # Record format checks
        # =====================================================================
        
        if item.name == "record_format":
            record_keywords = ['record', 'layout', 'structure', 'format', 'field', 'length', 'position']
            if sum(1 for kw in record_keywords if kw in doc_lower) >= 2:
                return True, "Found record format description"
            return False, None
        
        # =====================================================================
        # Default: item not specifically handled
        # =====================================================================
        
        # Fall back to searching for the item name itself
        search_term = item.name.replace('_', ' ')
        if search_term in doc_lower:
            return True, f"Found '{search_term}'"
        
        return False, None
    
    def calculate_overall_score(self, results: List[CallMatchResult]) -> float:
        """Calculate overall BSM score from multiple call results."""
        if not results:
            return 1.0  # No external calls = perfect BSM score
        
        return sum(r.score for r in results) / len(results)
