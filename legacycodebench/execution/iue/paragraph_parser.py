"""
Paragraph Parser - Parse COBOL PROCEDURE DIVISION into paragraphs.

A COBOL paragraph is similar to a function. It has a name and contains
executable statements. This parser extracts paragraphs from source code
for isolated testing.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class COBOLParagraph:
    """Represents a parsed COBOL paragraph."""
    name: str                           # Paragraph name (e.g., "CALCULATE-INTEREST")
    body: str                           # Full paragraph content
    line_start: int                     # Line number where paragraph starts (1-indexed)
    line_end: int                       # Line number where paragraph ends
    section: Optional[str] = None       # Parent section if any
    performs: List[str] = field(default_factory=list)    # Paragraphs this one PERFORMs
    variables_read: List[str] = field(default_factory=list)   # Variables read (input)
    variables_written: List[str] = field(default_factory=list) # Variables written (output)


class ParagraphParser:
    """
    Parse COBOL PROCEDURE DIVISION into individual paragraphs.
    
    COBOL paragraph structure:
        PARAGRAPH-NAME.
            COBOL statements
            ...
        
        NEXT-PARAGRAPH.
            ...
    
    Paragraphs in fixed-format COBOL start at column 8 (after 6-char sequence
    number area and 1-char indicator area).
    """
    
    # Regex to match paragraph names
    # Fixed format: columns 8-11 contain paragraph name, ends with period
    PARAGRAPH_PATTERN = re.compile(
        r'^.{6}\s([A-Z0-9][A-Z0-9-]*)\.\s*$',
        re.IGNORECASE
    )
    
    # Alternative pattern for free-format COBOL
    FREE_FORMAT_PATTERN = re.compile(
        r'^\s*([A-Z0-9][A-Z0-9-]*)\.\s*$',
        re.IGNORECASE
    )
    
    # COBOL reserved words/terminators that should NOT be detected as paragraphs
    RESERVED_WORDS = {
        # Statement terminators
        'END-IF', 'END-EXEC', 'END-PERFORM', 'END-EVALUATE', 'END-CALL',
        'END-READ', 'END-WRITE', 'END-REWRITE', 'END-DELETE', 'END-START',
        'END-RETURN', 'END-SEARCH', 'END-STRING', 'END-UNSTRING',
        'END-MULTIPLY', 'END-DIVIDE', 'END-ADD', 'END-SUBTRACT',
        'END-COMPUTE', 'END-ACCEPT', 'END-DISPLAY',
        # Common reserved words that might appear on their own line
        'STOP', 'GOBACK', 'CONTINUE', 'EXIT', 'ELSE', 'WHEN', 'OTHER',
        'NOT', 'AND', 'OR', 'TRUE', 'FALSE', 'ZERO', 'ZEROS', 'ZEROES',
        'SPACE', 'SPACES', 'HIGH-VALUE', 'HIGH-VALUES', 'LOW-VALUE', 'LOW-VALUES',
    }
    
    def parse(self, source_code: str) -> List[COBOLParagraph]:
        """
        Parse source into list of paragraphs.
        
        Args:
            source_code: Full COBOL source code
            
        Returns:
            List of COBOLParagraph objects
        """
        paragraphs = []
        lines = source_code.split('\n')
        
        # Find PROCEDURE DIVISION
        proc_start = self._find_procedure_division(lines)
        if proc_start == -1:
            logger.warning("PROCEDURE DIVISION not found")
            return []
        
        # Find all paragraph boundaries
        boundaries = self._find_paragraph_boundaries(lines, proc_start)
        
        if not boundaries:
            logger.warning("No paragraphs found in PROCEDURE DIVISION")
            return []
        
        logger.info(f"Found {len(boundaries)} paragraphs")
        
        # Extract each paragraph
        for i, (name, start_line) in enumerate(boundaries):
            # End is either next paragraph or end of file
            end_line = boundaries[i + 1][1] - 1 if i + 1 < len(boundaries) else len(lines)
            
            body = '\n'.join(lines[start_line:end_line])
            
            paragraph = COBOLParagraph(
                name=name,
                body=body,
                line_start=start_line + 1,  # 1-indexed
                line_end=end_line,
                section=None,
                performs=self._extract_performs(body),
                variables_read=self._extract_variables_read(body),
                variables_written=self._extract_variables_written(body)
            )
            paragraphs.append(paragraph)
        
        return paragraphs
    
    def _find_procedure_division(self, lines: List[str]) -> int:
        """Find line number of PROCEDURE DIVISION."""
        for i, line in enumerate(lines):
            if 'PROCEDURE DIVISION' in line.upper():
                return i
        return -1
    
    def _find_paragraph_boundaries(self, lines: List[str], start: int) -> List[tuple]:
        """
        Find (name, line_number) for each paragraph.
        
        Uses both fixed-format and free-format detection.
        """
        boundaries = []
        
        for i in range(start + 1, len(lines)):
            line = lines[i]
            
            # Skip empty lines
            if not line.strip():
                continue
            
            # Skip comments (indicator in column 7)
            if len(line) > 6 and line[6] == '*':
                continue
            
            # Try fixed format match
            match = self.PARAGRAPH_PATTERN.match(line)
            if match:
                name = match.group(1).upper()
                # Skip division/section markers and reserved words
                if not any(kw in name for kw in ['DIVISION', 'SECTION']) and name not in self.RESERVED_WORDS:
                    boundaries.append((name, i))
                continue
            
            # Try free format match
            match = self.FREE_FORMAT_PATTERN.match(line)
            if match:
                name = match.group(1).upper()
                if not any(kw in name for kw in ['DIVISION', 'SECTION']) and name not in self.RESERVED_WORDS:
                    boundaries.append((name, i))
        
        return boundaries
    
    def _extract_performs(self, body: str) -> List[str]:
        """Extract PERFORM targets from paragraph body."""
        pattern = re.compile(r'PERFORM\s+([A-Z0-9-]+)', re.IGNORECASE)
        matches = pattern.findall(body)
        # Filter out PERFORM VARYING, PERFORM UNTIL, etc.
        return [m.upper() for m in matches if m.upper() not in 
                ['VARYING', 'UNTIL', 'TIMES', 'THRU', 'THROUGH', 'WITH', 'TEST']]
    
    def _extract_variables_read(self, body: str) -> List[str]:
        """
        Extract variables that are read (inputs).
        
        Looks for patterns like:
        - MOVE X TO Y (reads X)
        - IF X > 5 (reads X)
        - COMPUTE Y = X + Z (reads X, Z)
        - ADD X TO Y (reads X)
        """
        variables = set()
        
        # MOVE source TO target
        for match in re.finditer(r'MOVE\s+(\w+)\s+TO', body, re.IGNORECASE):
            var = match.group(1).upper()
            if not var.isdigit() and not var.startswith("'"):
                variables.add(var)
        
        # IF conditions
        for match in re.finditer(r'IF\s+(\w+)\s*[<>=]', body, re.IGNORECASE):
            var = match.group(1).upper()
            if not var.isdigit():
                variables.add(var)
        
        # COMPUTE right side
        for match in re.finditer(r'COMPUTE\s+\w+\s*=\s*(.+?)(?:\.|$)', body, re.IGNORECASE | re.DOTALL):
            expr = match.group(1)
            for var in re.findall(r'([A-Z][A-Z0-9-]*)', expr, re.IGNORECASE):
                if not var.isdigit():
                    variables.add(var.upper())
        
        # ADD source TO target
        for match in re.finditer(r'ADD\s+(\w+)\s+TO', body, re.IGNORECASE):
            var = match.group(1).upper()
            if not var.isdigit():
                variables.add(var)
        
        return list(variables)
    
    def _extract_variables_written(self, body: str) -> List[str]:
        """
        Extract variables that are written (outputs).
        
        Looks for patterns like:
        - MOVE X TO Y (writes Y)
        - COMPUTE Y = X (writes Y)
        - ADD X TO Y (writes Y)
        """
        variables = set()
        
        # MOVE source TO target(s)
        for match in re.finditer(r'MOVE\s+.+?\s+TO\s+(.+?)(?:\.|$)', body, re.IGNORECASE):
            targets = match.group(1)
            for var in re.findall(r'([A-Z][A-Z0-9-]+)', targets, re.IGNORECASE):
                variables.add(var.upper())
        
        # COMPUTE target = expression
        for match in re.finditer(r'COMPUTE\s+([A-Z][A-Z0-9-]*)\s*=', body, re.IGNORECASE):
            variables.add(match.group(1).upper())
        
        # ADD X TO target
        for match in re.finditer(r'ADD\s+.+?\s+TO\s+([A-Z][A-Z0-9-]+)', body, re.IGNORECASE):
            variables.add(match.group(1).upper())
        
        # SUBTRACT X FROM target
        for match in re.finditer(r'SUBTRACT\s+.+?\s+FROM\s+([A-Z][A-Z0-9-]+)', body, re.IGNORECASE):
            variables.add(match.group(1).upper())
        
        return list(variables)
    
    def get_paragraph_stats(self, paragraphs: List[COBOLParagraph]) -> dict:
        """Get statistics about parsed paragraphs."""
        return {
            "total_paragraphs": len(paragraphs),
            "total_lines": sum(p.line_end - p.line_start + 1 for p in paragraphs),
            "avg_lines_per_paragraph": sum(p.line_end - p.line_start + 1 for p in paragraphs) / len(paragraphs) if paragraphs else 0,
            "paragraphs_with_performs": sum(1 for p in paragraphs if p.performs),
            "total_perform_calls": sum(len(p.performs) for p in paragraphs),
        }
