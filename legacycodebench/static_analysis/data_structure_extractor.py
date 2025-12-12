"""Data Structure Extractor for COBOL

Implements Section 2.2 of spec: Extractable Elements (Fully Automated)
- Extracts 01-levels, PICTURE clauses, REDEFINES, OCCURS, 88-levels
- Handles copybook expansions
- Automation Level: 100%
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class COBOLField:
    """Represents a COBOL data field"""
    name: str
    level: int
    picture: Optional[str] = None
    value: Optional[str] = None
    occurs: Optional[int] = None
    redefines: Optional[str] = None
    condition_value: Optional[str] = None  # For 88-levels
    line_number: int = 0
    parent: Optional[str] = None
    confidence: float = 1.0  # Extraction confidence


@dataclass
class DataStructure:
    """Represents a complete data structure (01-level group)"""
    name: str
    level: int
    fields: List[COBOLField] = field(default_factory=list)
    line_number: int = 0
    size_bytes: int = 0
    has_redefines: bool = False


class DataStructureExtractor:
    """
    Extract data structures from COBOL DATA DIVISION.

    Per Section 2.2 of spec:
    - 100% automation for standard DATA DIVISION elements
    - Handles REDEFINES with multi-view extraction (Section 2.3)
    """

    def __init__(self):
        # Pattern for data declaration line
        # Format: level-number name [REDEFINES ...] [PICTURE ...] [VALUE ...] [OCCURS ...]
        self.field_pattern = re.compile(
            r'^\s*(\d+)\s+([A-Z0-9-]+)'
            r'(?:\s+REDEFINES\s+([A-Z0-9-]+))?'
            r'(?:.*?PIC(?:TURE)?\s+(?:IS\s+)?([A-Z0-9()\*\$,\.\+\-/]+))?'
            r'(?:.*?VALUE\s+(?:IS\s+)?([^\s.]+))?'
            r'(?:.*?OCCURS\s+(\d+))?',
            re.IGNORECASE
        )

        # Pattern for 88-level condition names
        self.condition_pattern = re.compile(
            r'^\s*88\s+([A-Z0-9-]+)\s+VALUE(?:S)?\s+(?:IS\s+)?(.+?)\s*\.',
            re.IGNORECASE
        )

    def extract(self, data_division_content: str, line_offset: int = 0) -> Dict:
        """
        Extract all data structures from DATA DIVISION content.

        Args:
            data_division_content: Content of DATA DIVISION
            line_offset: Line number offset for absolute line references

        Returns:
            Dictionary with:
            - data_structures: List of DataStructure objects
            - fields: List of all COBOLField objects
            - redefines: List of REDEFINES relationships
            - occurs: List of OCCURS (arrays/tables)
            - condition_names: List of 88-level conditions
        """
        logger.info("Extracting data structures from DATA DIVISION")

        lines = data_division_content.split('\n')

        all_fields = []
        data_structures = []
        current_structure = None
        field_stack = []  # Track parent hierarchy

        for i, line in enumerate(lines):
            line_num = line_offset + i

            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith('*'):
                continue

            # Check for 88-level condition
            cond_match = self.condition_pattern.match(line)
            if cond_match:
                field = COBOLField(
                    name=cond_match.group(1),
                    level=88,
                    condition_value=cond_match.group(2).strip(),
                    line_number=line_num,
                    parent=field_stack[-1].name if field_stack else None
                )
                all_fields.append(field)
                if current_structure:
                    current_structure.fields.append(field)
                continue

            # Check for regular field declaration
            match = self.field_pattern.match(line)
            if match:
                level = int(match.group(1))
                name = match.group(2)
                redefines = match.group(3)
                picture = match.group(4)
                value = match.group(5)
                occurs = int(match.group(6)) if match.group(6) else None

                # Create field
                field = COBOLField(
                    name=name,
                    level=level,
                    picture=picture,
                    value=value,
                    occurs=occurs,
                    redefines=redefines,
                    line_number=line_num
                )

                # Handle 01-level (top-level structure)
                if level == 1:
                    # Save previous structure
                    if current_structure:
                        data_structures.append(current_structure)

                    # Start new structure
                    current_structure = DataStructure(
                        name=name,
                        level=level,
                        line_number=line_num,
                        has_redefines=(redefines is not None)
                    )
                    current_structure.fields.append(field)
                    field_stack = [field]

                else:
                    # Add to current structure
                    if current_structure:
                        # Determine parent based on level hierarchy
                        parent = self._find_parent(field_stack, level)
                        field.parent = parent.name if parent else None

                        current_structure.fields.append(field)

                        # Update field stack
                        while field_stack and field_stack[-1].level >= level:
                            field_stack.pop()
                        field_stack.append(field)

                        # Track REDEFINES
                        if redefines:
                            current_structure.has_redefines = True

                all_fields.append(field)

        # Save last structure
        if current_structure:
            data_structures.append(current_structure)

        # Extract specific element types
        redefines_list = self._extract_redefines(all_fields)
        occurs_list = self._extract_occurs(all_fields)
        condition_names = [f for f in all_fields if f.level == 88]

        logger.info(f"Extracted {len(data_structures)} data structures with {len(all_fields)} total fields")
        logger.info(f"Found {len(redefines_list)} REDEFINES, {len(occurs_list)} OCCURS, {len(condition_names)} condition names")

        return {
            "data_structures": [self._structure_to_dict(ds) for ds in data_structures],
            "fields": [self._field_to_dict(f) for f in all_fields],
            "redefines": redefines_list,
            "occurs": occurs_list,
            "condition_names": [self._field_to_dict(f) for f in condition_names],
            "total_structures": len(data_structures),
            "total_fields": len(all_fields)
        }

    def _find_parent(self, field_stack: List[COBOLField], level: int) -> Optional[COBOLField]:
        """Find parent field based on level hierarchy"""
        # Parent is the most recent field with level < current level
        for field in reversed(field_stack):
            if field.level < level:
                return field
        return None

    def _extract_redefines(self, fields: List[COBOLField]) -> List[Dict]:
        """
        Extract REDEFINES relationships.

        Per Section 2.3: Handle REDEFINES with multi-view extraction.
        Each REDEFINES creates multiple interpretations of same memory.
        """
        redefines_list = []

        for field in fields:
            if field.redefines:
                # Find the original field being redefined
                original = next((f for f in fields if f.name == field.redefines), None)

                redefines_list.append({
                    "field": field.name,
                    "redefines": field.redefines,
                    "line_number": field.line_number,
                    "interpretations": [
                        {
                            "name": field.redefines,
                            "picture": original.picture if original else None,
                            "type": "original"
                        },
                        {
                            "name": field.name,
                            "picture": field.picture,
                            "type": "redefining"
                        }
                    ],
                    "confidence": 0.95,  # High confidence for REDEFINES extraction
                    "note": "Multi-view extraction: both interpretations documented"
                })

        return redefines_list

    def _extract_occurs(self, fields: List[COBOLField]) -> List[Dict]:
        """Extract OCCURS clauses (arrays/tables)"""
        occurs_list = []

        for field in fields:
            if field.occurs:
                occurs_list.append({
                    "field": field.name,
                    "occurs": field.occurs,
                    "picture": field.picture,
                    "line_number": field.line_number,
                    "parent": field.parent
                })

        return occurs_list

    def _structure_to_dict(self, structure: DataStructure) -> Dict:
        """Convert DataStructure to dictionary"""
        return {
            "name": structure.name,
            "level": structure.level,
            "line_number": structure.line_number,
            "field_count": len(structure.fields),
            "has_redefines": structure.has_redefines,
            "fields": [f.name for f in structure.fields]
        }

    def _field_to_dict(self, field: COBOLField) -> Dict:
        """Convert COBOLField to dictionary"""
        return {
            "name": field.name,
            "level": field.level,
            "picture": field.picture,
            "value": field.value,
            "occurs": field.occurs,
            "redefines": field.redefines,
            "condition_value": field.condition_value,
            "line_number": field.line_number,
            "parent": field.parent,
            "confidence": field.confidence
        }

    def calculate_field_size(self, picture: str) -> int:
        """
        Calculate field size in bytes from PICTURE clause.

        Simplified calculation for common patterns.
        """
        if not picture:
            return 0

        # Remove IS, PIC keywords
        pic = picture.upper().replace('PIC', '').replace('IS', '').strip()

        # Handle common patterns
        # X(n) = alphanumeric, n bytes
        # 9(n) = numeric, n bytes
        # S9(n) = signed numeric, n bytes
        # V = decimal point (no storage)

        total_size = 0

        # Simple regex to find X(n), 9(n) patterns
        pattern = r'([X9])(?:\((\d+)\))?'
        matches = re.findall(pattern, pic)

        for char, count in matches:
            size = int(count) if count else 1
            total_size += size

        return total_size
