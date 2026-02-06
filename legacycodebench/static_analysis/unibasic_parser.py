"""
UniBasic Parser for Static Analysis

Extracts subroutines, data structures, and control flow from UniBasic source code.

Specification Reference: TDD_V2.4.md Section 5
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class UniBasicSubroutine:
    """Represents a UniBasic subroutine/label."""
    name: str                         # Subroutine name
    start_line: int                   # Start line number
    end_line: int                     # End line number (RETURN or next label)
    content: str                      # Subroutine content
    statements: List[str] = field(default_factory=list)  # Individual statements


@dataclass
class UniBasicVariable:
    """Represents a UniBasic variable/array."""
    name: str                         # Variable name
    var_type: str                     # SCALAR, ARRAY, DIM
    line_number: int                  # Where defined
    dimensions: Optional[str] = None  # For DIM statements


@dataclass
class UniBasicExternalCall:
    """Represents an external call (CALL, EXECUTE)."""
    call_type: str                    # CALL, EXECUTE
    target: str                       # Called program/command
    line_number: int                  # Line of call


@dataclass
class ParsedUniBasic:
    """Complete parsed UniBasic structure."""
    file_path: Path
    subroutines: List[UniBasicSubroutine]
    variables: List[UniBasicVariable]
    external_calls: List[UniBasicExternalCall]
    raw_lines: List[str]
    total_lines: int


class UniBasicParser:
    """
    UniBasic parser for ground truth extraction.
    
    Handles Pick/MultiValue BASIC syntax including:
    - Subroutine labels (NAME:)
    - DIM statements for arrays
    - GOSUB/RETURN control flow
    - CALL/EXECUTE for external programs
    """

    # Pattern for subroutine labels (e.g., "PROCESS.DATA:", "10:")
    LABEL_PATTERN = re.compile(r'^\s*([A-Za-z0-9_.]+)\s*:\s*$')
    
    # Pattern for DIM statements
    DIM_PATTERN = re.compile(r'^\s*DIM\s+(\w+)\s*\(\s*(.+?)\s*\)', re.IGNORECASE)
    
    # Pattern for CALL statements
    CALL_PATTERN = re.compile(r'^\s*CALL\s+(\w+)', re.IGNORECASE)
    
    # Pattern for EXECUTE statements
    EXECUTE_PATTERN = re.compile(r'^\s*EXECUTE\s+(.+)', re.IGNORECASE)
    
    # Pattern for GOSUB
    GOSUB_PATTERN = re.compile(r'^\s*GOSUB\s+(\w+)', re.IGNORECASE)
    
    # Pattern for RETURN
    RETURN_PATTERN = re.compile(r'^\s*RETURN\s*$', re.IGNORECASE)
    
    # Comment patterns
    COMMENT_PATTERNS = [
        re.compile(r'^\s*\*'),           # * at start
        re.compile(r'^\s*!'),            # ! at start
        re.compile(r'^\s*REM\s', re.IGNORECASE),  # REM statement
    ]

    def __init__(self):
        self.parsed_files: Dict[str, ParsedUniBasic] = {}

    def parse_file(self, file_path: Path) -> ParsedUniBasic:
        """
        Parse a UniBasic file into structured representation.
        
        Args:
            file_path: Path to UniBasic source file
            
        Returns:
            ParsedUniBasic structure
        """
        logger.info(f"Parsing UniBasic file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_lines = f.readlines()
        
        # Parse components
        subroutines = self._parse_subroutines(raw_lines)
        variables = self._parse_variables(raw_lines)
        external_calls = self._parse_external_calls(raw_lines)
        
        result = ParsedUniBasic(
            file_path=file_path,
            subroutines=subroutines,
            variables=variables,
            external_calls=external_calls,
            raw_lines=raw_lines,
            total_lines=len(raw_lines),
        )
        
        self.parsed_files[str(file_path)] = result
        return result

    def parse_string(self, source: str, file_path: str = "inline") -> ParsedUniBasic:
        """Parse UniBasic from a string."""
        raw_lines = source.split('\n')
        
        subroutines = self._parse_subroutines(raw_lines)
        variables = self._parse_variables(raw_lines)
        external_calls = self._parse_external_calls(raw_lines)
        
        return ParsedUniBasic(
            file_path=Path(file_path),
            subroutines=subroutines,
            variables=variables,
            external_calls=external_calls,
            raw_lines=raw_lines,
            total_lines=len(raw_lines),
        )

    def _parse_subroutines(self, lines: List[str]) -> List[UniBasicSubroutine]:
        """Extract subroutines from source lines."""
        subroutines = []
        
        current_label = None
        current_start = -1
        current_content = []
        
        for i, line in enumerate(lines):
            # Skip comments
            if self._is_comment(line):
                continue
            
            # Check for label
            match = self.LABEL_PATTERN.match(line)
            
            if match:
                # Save previous subroutine
                if current_label:
                    subroutines.append(UniBasicSubroutine(
                        name=current_label,
                        start_line=current_start + 1,
                        end_line=i,
                        content='\n'.join(current_content),
                        statements=self._extract_statements(current_content),
                    ))
                
                # Start new subroutine
                current_label = match.group(1).upper()
                current_start = i
                current_content = []
            elif current_label:
                current_content.append(line)
                
                # Check for RETURN
                if self.RETURN_PATTERN.match(line):
                    subroutines.append(UniBasicSubroutine(
                        name=current_label,
                        start_line=current_start + 1,
                        end_line=i + 1,
                        content='\n'.join(current_content),
                        statements=self._extract_statements(current_content),
                    ))
                    current_label = None
                    current_content = []
        
        # Handle last subroutine
        if current_label:
            subroutines.append(UniBasicSubroutine(
                name=current_label,
                start_line=current_start + 1,
                end_line=len(lines),
                content='\n'.join(current_content),
                statements=self._extract_statements(current_content),
            ))
        
        return subroutines

    def _parse_variables(self, lines: List[str]) -> List[UniBasicVariable]:
        """Extract variable definitions from source lines."""
        variables = []
        seen_names = set()
        
        for i, line in enumerate(lines):
            # Skip comments
            if self._is_comment(line):
                continue
            
            # Check for DIM
            match = self.DIM_PATTERN.match(line)
            if match:
                name = match.group(1).upper()
                dims = match.group(2)
                
                if name not in seen_names:
                    variables.append(UniBasicVariable(
                        name=name,
                        var_type="DIM",
                        line_number=i + 1,
                        dimensions=dims,
                    ))
                    seen_names.add(name)
        
        return variables

    def _parse_external_calls(self, lines: List[str]) -> List[UniBasicExternalCall]:
        """Extract external calls from source lines."""
        calls = []
        
        for i, line in enumerate(lines):
            # Skip comments
            if self._is_comment(line):
                continue
            
            # Check for CALL
            match = self.CALL_PATTERN.match(line)
            if match:
                calls.append(UniBasicExternalCall(
                    call_type="CALL",
                    target=match.group(1).upper(),
                    line_number=i + 1,
                ))
                continue
            
            # Check for EXECUTE
            match = self.EXECUTE_PATTERN.match(line)
            if match:
                target = match.group(1).strip()
                # Extract command name from string if present
                if target.startswith('"') or target.startswith("'"):
                    target = target.strip('"\'').split()[0]
                
                calls.append(UniBasicExternalCall(
                    call_type="EXECUTE",
                    target=target,
                    line_number=i + 1,
                ))
        
        return calls

    def _is_comment(self, line: str) -> bool:
        """Check if a line is a comment."""
        for pattern in self.COMMENT_PATTERNS:
            if pattern.match(line):
                return True
        return False

    def _extract_statements(self, lines: List[str]) -> List[str]:
        """Extract individual statements from source lines."""
        statements = []
        
        for line in lines:
            line = line.strip()
            if not line or self._is_comment(line):
                continue
            
            # Handle multiple statements on one line (separated by ;)
            for stmt in line.split(';'):
                stmt = stmt.strip()
                if stmt:
                    statements.append(stmt)
        
        return statements

    def get_gosub_targets(self, source: str) -> List[str]:
        """Get all GOSUB targets from source."""
        targets = []
        for line in source.split('\n'):
            match = self.GOSUB_PATTERN.match(line)
            if match:
                targets.append(match.group(1).upper())
        return targets

    def get_error_handlers(self, source: str) -> List[Dict]:
        """Get error handling constructs from source."""
        handlers = []
        lines = source.split('\n')
        
        for i, line in enumerate(lines):
            line_upper = line.upper()
            
            # ON ERROR GOTO/GOSUB
            if 'ON ERROR' in line_upper:
                match = re.search(r'ON\s+ERROR\s+(GOTO|GOSUB)\s+(\w+)', line_upper)
                if match:
                    handlers.append({
                        "handler_type": f"ON_ERROR_{match.group(1)}",
                        "line_number": i + 1,
                        "target": match.group(2),
                    })
            
            # LOCKED clause
            if 'LOCKED' in line_upper:
                handlers.append({
                    "handler_type": "LOCKED",
                    "line_number": i + 1,
                    "target": "",
                })
        
        return handlers
