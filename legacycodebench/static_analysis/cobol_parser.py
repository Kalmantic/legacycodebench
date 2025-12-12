"""COBOL Parser for Static Analysis

Implements lexical and syntactic analysis per Section 2.1 of spec.
Automation Level: 100% for standard COBOL constructs.

This is a pragmatic parser focused on extractability:
- Handles standard COBOL syntax (not all edge cases)
- Extracts key structures: DATA DIVISION, PROCEDURE DIVISION
- Builds simplified AST for analysis
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class COBOLDivision:
    """Represents a COBOL division"""
    name: str
    start_line: int
    end_line: int
    content: str


@dataclass
class COBOLSection:
    """Represents a section within a division"""
    name: str
    division: str
    start_line: int
    end_line: int
    content: str


@dataclass
class COBOLParagraph:
    """Represents a paragraph in PROCEDURE DIVISION"""
    name: str
    start_line: int
    end_line: int
    content: str
    statements: List[str] = field(default_factory=list)


@dataclass
class ParsedCOBOL:
    """Complete parsed COBOL structure"""
    file_path: Path
    divisions: Dict[str, COBOLDivision]
    sections: List[COBOLSection]
    paragraphs: List[COBOLParagraph]
    raw_lines: List[str]
    total_lines: int


class COBOLParser:
    """
    COBOL parser for ground truth extraction.

    Based on Section 2.1: Lexical and Syntactic Analysis
    - 100% automation for keywords, identifiers, structure
    - Handles DATA DIVISION and PROCEDURE DIVISION
    - Supports both FIXED FORMAT and FREE FORMAT COBOL
    """

    def __init__(self):
        self.division_pattern = re.compile(
            r'^\s*(IDENTIFICATION|ENVIRONMENT|DATA|PROCEDURE)\s+DIVISION\s*\.?',
            re.IGNORECASE
        )
        self.section_pattern = re.compile(
            r'^\s*([A-Za-z][A-Za-z0-9-]*)\s+SECTION\s*\.?',
            re.IGNORECASE
        )
        self.paragraph_pattern = re.compile(
            r'^\s*([A-Za-z][A-Za-z0-9-]*)\s*\.',
            re.IGNORECASE
        )
        self.is_free_format = False

    def parse_file(self, file_path: Path) -> ParsedCOBOL:
        """
        Parse a COBOL file into structured representation.

        Args:
            file_path: Path to COBOL source file

        Returns:
            ParsedCOBOL structure with divisions, sections, paragraphs
        """
        logger.info(f"Parsing COBOL file: {file_path}")

        # Read file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_lines = f.readlines()

        # Detect format (free vs fixed)
        self.is_free_format = self._detect_format(raw_lines)
        logger.info(f"Detected format: {'FREE' if self.is_free_format else 'FIXED'}")

        # Clean lines (remove sequence numbers, comments)
        cleaned_lines = self._clean_lines(raw_lines)

        # Parse divisions
        divisions = self._parse_divisions(cleaned_lines)

        # Parse sections
        sections = self._parse_sections(cleaned_lines, divisions)

        # Parse paragraphs (only in PROCEDURE DIVISION)
        paragraphs = self._parse_paragraphs(cleaned_lines, divisions)

        return ParsedCOBOL(
            file_path=file_path,
            divisions=divisions,
            sections=sections,
            paragraphs=paragraphs,
            raw_lines=cleaned_lines,
            total_lines=len(cleaned_lines)
        )

    def _detect_format(self, raw_lines: List[str]) -> bool:
        """
        Detect if COBOL file is free format or fixed format.
        
        Returns True if free format, False if fixed format.
        """
        for line in raw_lines[:20]:  # Check first 20 lines
            line_upper = line.upper().strip()
            # Check for free format directive
            if '>>SOURCE FORMAT' in line_upper and 'FREE' in line_upper:
                return True
            if '$SET SOURCEFORMAT' in line_upper and 'FREE' in line_upper:
                return True
        
        # Heuristic: if lines start with COBOL keywords at column 1, likely free format
        for line in raw_lines[:50]:
            stripped = line.strip().upper()
            if stripped.startswith(('IDENTIFICATION', 'ID ', 'PROGRAM-ID', 'DATA ', 'PROCEDURE')):
                # Check if it's at column 7-8 (fixed) or earlier (free)
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces < 6:
                    return True
        
        return False

    def _clean_lines(self, raw_lines: List[str]) -> List[str]:
        """
        Clean COBOL lines:
        - For fixed format: Remove sequence numbers (columns 1-6)
        - For free format: Keep lines as-is
        - Remove comments (*> for free format, column 7 * for fixed)
        - Keep line structure for line number references
        """
        cleaned = []

        for line in raw_lines:
            if self.is_free_format:
                # Free format: no column restrictions
                clean_line = line.rstrip()
                
                # Skip directive lines
                if clean_line.strip().upper().startswith('>>'):
                    cleaned.append('')
                    continue
                
                # Skip comment lines (*> style)
                stripped = clean_line.strip()
                if stripped.startswith('*>') or stripped.startswith('*'):
                    cleaned.append('')
                    continue
                
                # Remove inline comments (*> ... )
                if '*>' in clean_line:
                    clean_line = clean_line[:clean_line.index('*>')].rstrip()
                
                cleaned.append(clean_line)
            else:
                # Fixed format: columns 1-6 are sequence numbers
                if len(line) > 6 and line[6] in ['*', '/', 'D', 'd']:
                    # Comment line or debug line
                    cleaned.append('')
                    continue

                # Remove sequence number area (columns 1-6)
                if len(line) > 7:
                    clean_line = line[6:]
                else:
                    clean_line = line

                # Check for comment indicator in column 7 (now column 1)
                if clean_line and clean_line[0] == '*':
                    cleaned.append('')
                    continue

                cleaned.append(clean_line.rstrip())

        return cleaned

    def _parse_divisions(self, lines: List[str]) -> Dict[str, COBOLDivision]:
        """Parse COBOL divisions"""
        divisions = {}
        current_division = None
        division_start = 0

        for i, line in enumerate(lines):
            match = self.division_pattern.match(line)

            if match:
                # Save previous division
                if current_division:
                    divisions[current_division] = COBOLDivision(
                        name=current_division,
                        start_line=division_start,
                        end_line=i - 1,
                        content='\n'.join(lines[division_start:i])
                    )

                # Start new division
                current_division = match.group(1).upper()
                division_start = i

        # Save last division
        if current_division:
            divisions[current_division] = COBOLDivision(
                name=current_division,
                start_line=division_start,
                end_line=len(lines) - 1,
                content='\n'.join(lines[division_start:])
            )

        logger.info(f"Found {len(divisions)} divisions: {list(divisions.keys())}")
        return divisions

    def _parse_sections(self, lines: List[str],
                       divisions: Dict[str, COBOLDivision]) -> List[COBOLSection]:
        """Parse sections within divisions"""
        sections = []

        for div_name, division in divisions.items():
            div_lines = lines[division.start_line:division.end_line + 1]

            current_section = None
            section_start = 0

            for i, line in enumerate(div_lines):
                match = self.section_pattern.match(line)

                if match:
                    # Save previous section
                    if current_section:
                        sections.append(COBOLSection(
                            name=current_section,
                            division=div_name,
                            start_line=division.start_line + section_start,
                            end_line=division.start_line + i - 1,
                            content='\n'.join(div_lines[section_start:i])
                        ))

                    # Start new section
                    current_section = match.group(1).upper()
                    section_start = i

            # Save last section
            if current_section:
                sections.append(COBOLSection(
                    name=current_section,
                    division=div_name,
                    start_line=division.start_line + section_start,
                    end_line=division.end_line,
                    content='\n'.join(div_lines[section_start:])
                ))

        logger.info(f"Found {len(sections)} sections")
        return sections

    def _parse_paragraphs(self, lines: List[str],
                         divisions: Dict[str, COBOLDivision]) -> List[COBOLParagraph]:
        """Parse paragraphs in PROCEDURE DIVISION"""
        paragraphs = []

        # Only parse paragraphs in PROCEDURE DIVISION
        if 'PROCEDURE' not in divisions:
            return paragraphs

        proc_div = divisions['PROCEDURE']
        proc_lines = lines[proc_div.start_line:proc_div.end_line + 1]

        current_paragraph = None
        paragraph_start = 0

        for i, line in enumerate(proc_lines):
            # Skip empty lines and division header
            if not line.strip() or 'PROCEDURE DIVISION' in line.upper():
                continue

            # Check if this is a paragraph (starts at beginning, ends with period)
            match = self.paragraph_pattern.match(line)

            if match and not self._is_statement(line):
                # This is likely a paragraph name

                # Save previous paragraph
                if current_paragraph:
                    para_content = '\n'.join(proc_lines[paragraph_start:i])
                    paragraphs.append(COBOLParagraph(
                        name=current_paragraph,
                        start_line=proc_div.start_line + paragraph_start,
                        end_line=proc_div.start_line + i - 1,
                        content=para_content,
                        statements=self._extract_statements(para_content)
                    ))

                # Start new paragraph
                current_paragraph = match.group(1).upper()
                paragraph_start = i

        # Save last paragraph
        if current_paragraph:
            para_content = '\n'.join(proc_lines[paragraph_start:])
            paragraphs.append(COBOLParagraph(
                name=current_paragraph,
                start_line=proc_div.start_line + paragraph_start,
                end_line=proc_div.end_line,
                content=para_content,
                statements=self._extract_statements(para_content)
            ))

        logger.info(f"Found {len(paragraphs)} paragraphs")
        return paragraphs

    def _is_statement(self, line: str) -> bool:
        """Check if line is a COBOL statement (not a paragraph name)"""
        # Common COBOL verbs
        verbs = [
            'MOVE', 'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'COMPUTE',
            'IF', 'ELSE', 'EVALUATE', 'WHEN', 'PERFORM', 'GO', 'GOTO',
            'CALL', 'OPEN', 'CLOSE', 'READ', 'WRITE', 'REWRITE',
            'ACCEPT', 'DISPLAY', 'STOP', 'EXIT', 'CONTINUE',
            'SEARCH', 'SET', 'INITIALIZE', 'INSPECT', 'STRING', 'UNSTRING'
        ]

        line_upper = line.strip().upper()
        return any(line_upper.startswith(verb) for verb in verbs)

    def _extract_statements(self, paragraph_content: str) -> List[str]:
        """Extract individual statements from paragraph"""
        statements = []

        # Split by periods (statement terminators)
        # This is simplified - real COBOL has complex period rules
        parts = paragraph_content.split('.')

        for part in parts:
            stmt = part.strip()
            if stmt and not self.paragraph_pattern.match(stmt):
                statements.append(stmt)

        return statements

    def get_data_division_content(self, parsed: ParsedCOBOL) -> str:
        """Get DATA DIVISION content for structure extraction"""
        if 'DATA' in parsed.divisions:
            return parsed.divisions['DATA'].content
        return ""

    def get_procedure_division_content(self, parsed: ParsedCOBOL) -> str:
        """Get PROCEDURE DIVISION content for control flow analysis"""
        if 'PROCEDURE' in parsed.divisions:
            return parsed.divisions['PROCEDURE'].content
        return ""
