"""
Code Pattern Matchers for V3.1 Code-Based Static Verification

These matchers search COBOL source code for specific patterns:
- ComputationMatcher: COMPUTE, ADD, SUBTRACT, MULTIPLY, DIVIDE, MOVE
- ConditionalMatcher: IF, EVALUATE statements
- CallPerformMatcher: CALL, PERFORM statements

Used by CodeBasedVerifier to verify behavioral claims against actual code.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Set, Dict
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes for Match Results
# ============================================================================

@dataclass
class ComputationMatch:
    """Result of finding a computation in source code."""
    target: str           # Variable being computed
    operation: str        # Type: compute, add, subtract, multiply, divide, move
    expression: str       # The expression/operands
    line: int            # Line number in source
    statement: str        # Full matched statement


@dataclass
class ConditionalMatch:
    """Result of finding a conditional in source code."""
    variable: str         # Variable being checked
    condition_type: str   # Type: if_equals, if_greater, if_less, evaluate
    value: Optional[str]  # Value being compared to (if any)
    line: int            # Line number in source
    then_block: str       # Code in the THEN branch


@dataclass
class PerformMatch:
    """Result of finding a PERFORM statement."""
    target: str           # Paragraph being performed
    line: int            # Line number in source


@dataclass
class CallMatch:
    """Result of finding a CALL statement."""
    target: str           # Program being called
    line: int            # Line number in source


# ============================================================================
# Computation Matcher
# ============================================================================

class ComputationMatcher:
    """
    Find computation statements in COBOL source.
    
    Patterns matched:
      COMPUTE target = expression
      ADD x TO target
      ADD x y GIVING target
      SUBTRACT x FROM target
      MULTIPLY x BY target
      DIVIDE x INTO target
      MOVE x TO target
    """
    
    # COBOL keywords to filter out from operands
    COBOL_KEYWORDS = {
        'OF', 'IN', 'BY', 'TO', 'FROM', 'GIVING', 'INTO', 'ROUNDED',
        'SIZE', 'ERROR', 'ON', 'NOT', 'END-COMPUTE', 'END-ADD',
        'END-SUBTRACT', 'END-MULTIPLY', 'END-DIVIDE', 'REMAINDER',
        'CORRESPONDING', 'CORR', 'ALL', 'ZEROES', 'ZEROS', 'SPACES',
        'QUOTES', 'HIGH-VALUES', 'LOW-VALUES', 'TRUE', 'FALSE',
    }
    
    def find_computations(self, source: str, target_var: str) -> List[ComputationMatch]:
        """
        Find all computations targeting the specified variable.
        
        Args:
            source: COBOL source code
            target_var: Variable name to search for
            
        Returns:
            List of ComputationMatch objects
        """
        matches = []
        
        # Escape hyphens for regex
        var_pattern = re.escape(target_var)
        
        # Pattern: COMPUTE target = expression
        compute_pattern = rf'COMPUTE\s+{var_pattern}\s*=\s*(.+?)(?:\.|\s+END-COMPUTE)'
        for match in re.finditer(compute_pattern, source, re.IGNORECASE | re.DOTALL):
            matches.append(ComputationMatch(
                target=target_var,
                operation='compute',
                expression=match.group(1).strip(),
                line=self._get_line_number(source, match.start()),
                statement=self._clean_statement(match.group(0)),
            ))
        
        # Pattern: ADD x TO target
        add_to_pattern = rf'ADD\s+(.+?)\s+TO\s+{var_pattern}(?:\s|\.)'
        for match in re.finditer(add_to_pattern, source, re.IGNORECASE):
            matches.append(ComputationMatch(
                target=target_var,
                operation='add',
                expression=match.group(1).strip(),
                line=self._get_line_number(source, match.start()),
                statement=self._clean_statement(match.group(0)),
            ))
        
        # Pattern: ADD x GIVING target
        add_giving_pattern = rf'ADD\s+(.+?)\s+GIVING\s+{var_pattern}(?:\s|\.)'
        for match in re.finditer(add_giving_pattern, source, re.IGNORECASE):
            matches.append(ComputationMatch(
                target=target_var,
                operation='add',
                expression=match.group(1).strip(),
                line=self._get_line_number(source, match.start()),
                statement=self._clean_statement(match.group(0)),
            ))
        
        # Pattern: SUBTRACT x FROM target
        subtract_pattern = rf'SUBTRACT\s+(.+?)\s+FROM\s+{var_pattern}(?:\s|\.)'
        for match in re.finditer(subtract_pattern, source, re.IGNORECASE):
            matches.append(ComputationMatch(
                target=target_var,
                operation='subtract',
                expression=match.group(1).strip(),
                line=self._get_line_number(source, match.start()),
                statement=self._clean_statement(match.group(0)),
            ))
        
        # Pattern: MULTIPLY x BY target
        multiply_pattern = rf'MULTIPLY\s+(.+?)\s+BY\s+{var_pattern}(?:\s|\.)'
        for match in re.finditer(multiply_pattern, source, re.IGNORECASE):
            matches.append(ComputationMatch(
                target=target_var,
                operation='multiply',
                expression=match.group(1).strip(),
                line=self._get_line_number(source, match.start()),
                statement=self._clean_statement(match.group(0)),
            ))
        
        # Pattern: DIVIDE x INTO target
        divide_pattern = rf'DIVIDE\s+(.+?)\s+INTO\s+{var_pattern}(?:\s|\.)'
        for match in re.finditer(divide_pattern, source, re.IGNORECASE):
            matches.append(ComputationMatch(
                target=target_var,
                operation='divide',
                expression=match.group(1).strip(),
                line=self._get_line_number(source, match.start()),
                statement=self._clean_statement(match.group(0)),
            ))
        
        # Pattern: MOVE x TO target
        move_pattern = rf'MOVE\s+(.+?)\s+TO\s+{var_pattern}(?:\s|\.)'
        for match in re.finditer(move_pattern, source, re.IGNORECASE):
            matches.append(ComputationMatch(
                target=target_var,
                operation='move',
                expression=match.group(1).strip(),
                line=self._get_line_number(source, match.start()),
                statement=self._clean_statement(match.group(0)),
            ))
        
        return matches
    
    def extract_operands(self, computation: ComputationMatch) -> List[str]:
        """
        Extract variable names from computation expression.
        
        Args:
            computation: ComputationMatch object
            
        Returns:
            List of variable names used in the expression
        """
        expr = computation.expression
        
        # Match COBOL variable names (letters, digits, hyphens)
        # Must start with letter, can contain letters, digits, hyphens
        var_pattern = r'\b([A-Z][A-Z0-9\-]*(?:[A-Z0-9])?)(?:\s|[^\w\-]|$)'
        
        candidates = re.findall(var_pattern, expr, re.IGNORECASE)
        
        # Filter out keywords and literals
        operands = []
        for v in candidates:
            v_upper = v.upper()
            if v_upper not in self.COBOL_KEYWORDS:
                # Skip numeric literals
                if not v.isdigit():
                    # Skip quoted strings
                    if not v.startswith("'") and not v.startswith('"'):
                        operands.append(v_upper)
        
        return operands
    
    def _get_line_number(self, source: str, position: int) -> int:
        """Get line number for a position in source."""
        return source[:position].count('\n') + 1
    
    def _clean_statement(self, statement: str) -> str:
        """Clean up matched statement for display."""
        # Remove excessive whitespace
        cleaned = ' '.join(statement.split())
        # Truncate if too long
        if len(cleaned) > 80:
            cleaned = cleaned[:77] + '...'
        return cleaned


# ============================================================================
# Conditional Matcher
# ============================================================================

class ConditionalMatcher:
    """
    Find conditional statements in COBOL source.
    
    Patterns matched:
      IF condition-var = value ...
      IF condition-var > value ...
      IF condition-var EQUAL TO value ...
      IF condition-var NOT = value ...
      EVALUATE condition-var ...
    """
    
    def find_conditionals(self, source: str, condition_var: str) -> List[ConditionalMatch]:
        """
        Find all conditionals checking the specified variable.
        
        Args:
            source: COBOL source code
            condition_var: Variable name to search for
            
        Returns:
            List of ConditionalMatch objects
        """
        matches = []
        var_pattern = re.escape(condition_var)
        
        # Pattern: IF var = value
        if_equals_pattern = rf'IF\s+{var_pattern}\s*(?:=|EQUAL(?:\s+TO)?)\s*([^\s\n]+)'
        for match in re.finditer(if_equals_pattern, source, re.IGNORECASE):
            then_block = self._extract_then_block(source, match.end())
            matches.append(ConditionalMatch(
                variable=condition_var,
                condition_type='if_equals',
                value=match.group(1).strip(),
                line=self._get_line_number(source, match.start()),
                then_block=then_block,
            ))
        
        # Pattern: IF var > value
        if_greater_pattern = rf'IF\s+{var_pattern}\s*(?:>|GREATER(?:\s+THAN)?)\s*([^\s\n]+)'
        for match in re.finditer(if_greater_pattern, source, re.IGNORECASE):
            then_block = self._extract_then_block(source, match.end())
            matches.append(ConditionalMatch(
                variable=condition_var,
                condition_type='if_greater',
                value=match.group(1).strip(),
                line=self._get_line_number(source, match.start()),
                then_block=then_block,
            ))
        
        # Pattern: IF var < value
        if_less_pattern = rf'IF\s+{var_pattern}\s*(?:<|LESS(?:\s+THAN)?)\s*([^\s\n]+)'
        for match in re.finditer(if_less_pattern, source, re.IGNORECASE):
            then_block = self._extract_then_block(source, match.end())
            matches.append(ConditionalMatch(
                variable=condition_var,
                condition_type='if_less',
                value=match.group(1).strip(),
                line=self._get_line_number(source, match.start()),
                then_block=then_block,
            ))
        
        # Pattern: IF var NOT = value
        if_not_pattern = rf'IF\s+{var_pattern}\s*(?:NOT\s*=|NOT\s+EQUAL)\s*([^\s\n]+)'
        for match in re.finditer(if_not_pattern, source, re.IGNORECASE):
            then_block = self._extract_then_block(source, match.end())
            matches.append(ConditionalMatch(
                variable=condition_var,
                condition_type='if_not_equals',
                value=match.group(1).strip(),
                line=self._get_line_number(source, match.start()),
                then_block=then_block,
            ))
        
        # Pattern: EVALUATE var
        evaluate_pattern = rf'EVALUATE\s+{var_pattern}(?:\s|$)'
        for match in re.finditer(evaluate_pattern, source, re.IGNORECASE):
            then_block = self._extract_evaluate_block(source, match.end())
            matches.append(ConditionalMatch(
                variable=condition_var,
                condition_type='evaluate',
                value=None,
                line=self._get_line_number(source, match.start()),
                then_block=then_block,
            ))
        
        # Pattern: IF var (just checking variable, no specific comparison)
        # This catches things like "IF WS-EOF-FLAG"
        if_simple_pattern = rf'IF\s+{var_pattern}(?:\s|$)'
        for match in re.finditer(if_simple_pattern, source, re.IGNORECASE):
            # Skip if we already matched this position
            already_matched = any(
                abs(m.line - self._get_line_number(source, match.start())) < 2
                for m in matches
            )
            if not already_matched:
                then_block = self._extract_then_block(source, match.end())
                matches.append(ConditionalMatch(
                    variable=condition_var,
                    condition_type='if_flag',
                    value=None,
                    line=self._get_line_number(source, match.start()),
                    then_block=then_block,
                ))
        
        return matches
    
    def extract_actions(self, conditional: ConditionalMatch) -> List[str]:
        """
        Extract actions from THEN block (shallow - immediate block only).
        
        Args:
            conditional: ConditionalMatch object
            
        Returns:
            List of action types found (e.g., ['abend', 'display'])
        """
        return self._extract_actions_from_block(conditional.then_block)
    
    def extract_actions_deep(
        self, 
        conditional: ConditionalMatch, 
        full_source: str,
        max_depth: int = 2
    ) -> List[str]:
        """
        Extract actions from THEN block, following PERFORM chains.
        
        This is the V3.2 enhancement - instead of just seeing "perform"
        as an action, we follow the PERFORM to see what it actually does.
        
        Args:
            conditional: ConditionalMatch object
            full_source: Complete COBOL source for paragraph lookup
            max_depth: How many levels of PERFORM to follow (default 2)
            
        Returns:
            List of action types found including from PERFORMed paragraphs
        """
        actions = set()
        visited_paragraphs = set()
        
        # Get direct actions from THEN block
        direct_actions = self._extract_actions_from_block(conditional.then_block)
        actions.update(direct_actions)
        
        # Follow PERFORM chains
        perform_targets = self._extract_perform_targets(conditional.then_block)
        
        for para_name in perform_targets:
            self._follow_perform_chain(
                para_name, full_source, actions, visited_paragraphs, 
                current_depth=1, max_depth=max_depth
            )
        
        return list(actions)
    
    def _extract_actions_from_block(self, block: str) -> List[str]:
        """Extract action types from a code block."""
        actions = []
        block_upper = block.upper()
        
        # Check for common actions
        if re.search(r'\bSTOP\s+RUN\b', block_upper):
            actions.append('stop_run')
        if re.search(r'\bGOBACK\b', block_upper):
            actions.append('goback')
        if re.search(r'\bABEND\b', block_upper):
            actions.append('abend')
        if re.search(r'\bDISPLAY\b', block_upper):
            actions.append('display')
        if re.search(r'\bPERFORM\s+', block_upper):
            actions.append('perform')
        if re.search(r'\bCALL\s+', block_upper):
            actions.append('call')
        if re.search(r'\bMOVE\s+', block_upper):
            actions.append('move')
        if re.search(r'\bADD\s+|COMPUTE\s+|SUBTRACT\s+|MULTIPLY\s+|DIVIDE\s+', block_upper):
            actions.append('compute')
        if re.search(r'\bWRITE\s+', block_upper):
            actions.append('write')
        if re.search(r'\bREAD\s+', block_upper):
            actions.append('read')
        if re.search(r'\bSET\s+', block_upper):
            actions.append('set')
        
        return actions
    
    def _extract_perform_targets(self, block: str) -> List[str]:
        """Extract paragraph names from PERFORM statements."""
        pattern = r'PERFORM\s+([A-Z0-9][A-Z0-9\-]*)'
        matches = re.findall(pattern, block, re.IGNORECASE)
        return [m.upper() for m in matches]
    
    def _follow_perform_chain(
        self,
        para_name: str,
        full_source: str,
        actions: set,
        visited: set,
        current_depth: int,
        max_depth: int
    ):
        """
        Recursively follow a PERFORM chain to find all actions.
        
        Args:
            para_name: Paragraph name to find
            full_source: Complete COBOL source
            actions: Set to add found actions to
            visited: Set of already-visited paragraphs (prevents loops)
            current_depth: Current recursion depth
            max_depth: Maximum depth to follow
        """
        # Prevent infinite loops and excessive depth
        if para_name in visited or current_depth > max_depth:
            return
        
        visited.add(para_name)
        
        # Get paragraph body
        para_body = _paragraph_extractor.get_paragraph_body(full_source, para_name)
        
        if not para_body:
            logger.debug(f"Paragraph {para_name} not found in source")
            return
        
        # Extract direct actions from this paragraph
        para_actions = self._extract_actions_from_block(para_body)
        actions.update(para_actions)
        
        # Recursively follow nested PERFORMs
        nested_performs = self._extract_perform_targets(para_body)
        for nested_para in nested_performs:
            if nested_para != para_name:  # Prevent self-recursion
                self._follow_perform_chain(
                    nested_para, full_source, actions, visited,
                    current_depth + 1, max_depth
                )
    
    def _extract_then_block(self, source: str, start: int) -> str:
        """Extract the THEN block from an IF statement."""
        # Find END-IF or ELSE or next paragraph
        remaining = source[start:]
        
        # Look for END-IF
        end_if_match = re.search(r'\bEND-IF\b', remaining, re.IGNORECASE)
        else_match = re.search(r'\bELSE\b', remaining, re.IGNORECASE)
        
        # Find next paragraph (line starting with paragraph name followed by period)
        next_para = re.search(r'\n\s*[A-Z0-9][A-Z0-9\-]*\.\s*\n', remaining)
        
        # Determine end position
        end_pos = len(remaining)
        
        if end_if_match:
            end_pos = min(end_pos, end_if_match.start())
        if else_match:
            end_pos = min(end_pos, else_match.start())
        if next_para:
            end_pos = min(end_pos, next_para.start())
        
        # Limit to reasonable size
        end_pos = min(end_pos, 500)
        
        return remaining[:end_pos]
    
    def _extract_evaluate_block(self, source: str, start: int) -> str:
        """Extract the EVALUATE block."""
        remaining = source[start:]
        
        # Find END-EVALUATE
        end_eval = re.search(r'\bEND-EVALUATE\b', remaining, re.IGNORECASE)
        
        if end_eval:
            return remaining[:end_eval.start()]
        
        # Fallback: next paragraph
        next_para = re.search(r'\n\s*[A-Z0-9][A-Z0-9\-]*\.\s*\n', remaining)
        if next_para:
            return remaining[:next_para.start()]
        
        return remaining[:500]
    
    def _get_line_number(self, source: str, position: int) -> int:
        """Get line number for a position in source."""
        return source[:position].count('\n') + 1


# ============================================================================
# Call/Perform Matcher
# ============================================================================

class ParagraphExtractor:
    """
    Extract paragraph bodies from COBOL source code.
    
    Used for PERFORM chain following - when we see PERFORM 9100-DISPLAY-ERROR,
    we need to find what 9100-DISPLAY-ERROR actually does.
    """
    
    def __init__(self):
        """Initialize with empty cache."""
        self._cache: Dict[int, Dict[str, str]] = {}  # source_hash -> {para_name: body}
    
    def get_paragraph_body(self, source: str, para_name: str) -> Optional[str]:
        """
        Get the body of a named paragraph.
        
        Args:
            source: COBOL source code
            para_name: Paragraph name (e.g., "9100-DISPLAY-ERROR")
            
        Returns:
            Paragraph body or None if not found
        """
        source_hash = hash(source)
        
        # Build cache on first access for this source
        if source_hash not in self._cache:
            self._cache[source_hash] = self._build_paragraph_index(source)
        
        return self._cache[source_hash].get(para_name.upper())
    
    def _build_paragraph_index(self, source: str) -> Dict[str, str]:
        """
        Build index of all paragraphs in source.
        
        Returns:
            Dict mapping paragraph name to body
        """
        index = {}
        lines = source.split('\n')
        
        # Pattern: paragraph name in area A (columns 8-11 in fixed format, 
        # or just starting early in free format), followed by period
        # Examples: "       9100-DISPLAY-ERROR.", "       MAIN-LOGIC."
        para_pattern = re.compile(r'^\s{0,7}([A-Z0-9][A-Z0-9\-]*)\.\s*$', re.IGNORECASE)
        
        current_para = None
        current_body_lines = []
        
        for line in lines:
            match = para_pattern.match(line)
            
            if match:
                # Save previous paragraph
                if current_para:
                    index[current_para] = '\n'.join(current_body_lines)
                
                # Start new paragraph
                current_para = match.group(1).upper()
                current_body_lines = []
            elif current_para:
                current_body_lines.append(line)
        
        # Save last paragraph
        if current_para:
            index[current_para] = '\n'.join(current_body_lines)
        
        return index
    
    def get_all_paragraphs(self, source: str) -> List[str]:
        """Get list of all paragraph names in source."""
        source_hash = hash(source)
        if source_hash not in self._cache:
            self._cache[source_hash] = self._build_paragraph_index(source)
        return list(self._cache[source_hash].keys())


# Global paragraph extractor (cached)
_paragraph_extractor = ParagraphExtractor()


class CallPerformMatcher:
    """
    Find CALL and PERFORM statements in COBOL source.
    """
    
    def find_performs(self, source: str, target_para: str = None) -> List[PerformMatch]:
        """
        Find PERFORM statements, optionally filtered by target.
        
        Args:
            source: COBOL source code
            target_para: Optional paragraph name to filter by
            
        Returns:
            List of PerformMatch objects
        """
        pattern = r'PERFORM\s+([A-Z0-9][A-Z0-9\-]*)'
        matches = []
        
        for match in re.finditer(pattern, source, re.IGNORECASE):
            para_name = match.group(1).upper()
            if target_para is None or para_name == target_para.upper():
                matches.append(PerformMatch(
                    target=para_name,
                    line=source[:match.start()].count('\n') + 1,
                ))
        
        return matches
    
    def find_calls(self, source: str, target_prog: str = None) -> List[CallMatch]:
        """
        Find CALL statements, optionally filtered by target.
        
        Args:
            source: COBOL source code
            target_prog: Optional program name to filter by
            
        Returns:
            List of CallMatch objects
        """
        # Pattern handles both quoted and unquoted program names
        pattern = r'CALL\s+[\'"]?([A-Z0-9][A-Z0-9\-]*)[\'"]?'
        matches = []
        
        for match in re.finditer(pattern, source, re.IGNORECASE):
            prog_name = match.group(1).upper()
            if target_prog is None or prog_name == target_prog.upper():
                matches.append(CallMatch(
                    target=prog_name,
                    line=source[:match.start()].count('\n') + 1,
                ))
        
        return matches


# ============================================================================
# Variable Finder
# ============================================================================

class VariableFinder:
    """
    Find where variables are used in COBOL source.
    """
    
    def find_usages(self, source: str, variable: str) -> List[int]:
        """
        Find all lines where a variable is used.
        
        Args:
            source: COBOL source code
            variable: Variable name to search for
            
        Returns:
            List of line numbers
        """
        pattern = rf'\b{re.escape(variable)}\b'
        lines = []
        
        for match in re.finditer(pattern, source, re.IGNORECASE):
            line = source[:match.start()].count('\n') + 1
            if line not in lines:
                lines.append(line)
        
        return lines
    
    def variable_exists(self, source: str, variable: str) -> bool:
        """Check if a variable is used anywhere in the source."""
        pattern = rf'\b{re.escape(variable)}\b'
        return bool(re.search(pattern, source, re.IGNORECASE))
