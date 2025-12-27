"""Traceability Evaluator (10% weight)

Implements Section 5.5 of spec: Traceability Validation Automation
Measures: Does documentation cite specific source code locations? Are citations valid?

Formula: TR = (Claims_With_Valid_References / Total_Claims) x 100
"""

import re
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


class TraceabilityEvaluator:
    """
    Evaluate traceability of documentation to source code.

    Per Section 5.5 of spec:
    - Extract references (line numbers, paragraphs, variables, code excerpts)
    - Validate each reference against source code
    - 100% automation
    """

    def __init__(self):
        # Reference patterns per Section 5.5.1
        # FIXED: More precise patterns to avoid matching natural language
        self.reference_patterns = {
            "line_number": r'lines?\s*(\d+)',
            "line_range": r'lines?\s*(\d+)\s*[-–to]+\s*(\d+)',
            "paragraph": r'paragraph\s+([A-Z0-9-]+)',
            "section": r'section\s+([A-Z0-9-]+)',
            # FIXED: Variable pattern now requires:
            # 1. Backtick-quoted names OR
            # 2. COBOL-style names (uppercase with hyphens, min 3 chars)
            # This avoids matching natural language like "data from", "data structures"
            "variable": r'(?:field|variable|data\s+item|record)\s+`([A-Z][A-Z0-9-]+)`|(?:field|variable)\s+([A-Z][A-Z0-9-]{2,})',
            "code_excerpt": r'```(?:cobol)?\n(.*?)\n```'
        }
        
        # Common English words to exclude from variable detection
        self.excluded_words = {
            'from', 'about', 'to', 'for', 'with', 'the', 'and', 'or', 
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'structures', 'structure', 'integrity', 'processing',
            'validation', 'handling', 'format', 'section', 'type'
        }

    def evaluate(self, submission_content: str, ground_truth: Dict,
                source_code: str) -> Dict:
        """
        Evaluate traceability of documentation.

        Args:
            submission_content: AI-generated documentation
            ground_truth: Ground truth with extracted elements
            source_code: Original COBOL source code

        Returns:
            Dictionary with:
            - score: Traceability score (0.0 to 1.0)
            - total_references: Number of references found
            - valid_references: Number of valid references
            - invalid_references: List of invalid references
        """
        logger.info("Evaluating traceability (reference validation)")

        # Extract all references from documentation
        references = self._extract_references(submission_content)

        if not references:
            # No references = lack of specificity
            logger.warning("No source code references found in documentation")
            return {
                "score": 0.5,  # Partial credit (per spec Section 5.5.2)
                "total_references": 0,
                "valid_references": 0,
                "invalid_references": [],
                "warning": "No source code references found"
            }

        # Validate each reference
        valid_count = 0
        invalid_refs = []

        for ref in references:
            is_valid = self._validate_reference(ref, ground_truth, source_code)

            if is_valid:
                valid_count += 1
            else:
                invalid_refs.append(ref)

        # Calculate score
        score = valid_count / len(references)

        logger.info(f"Traceability Score: {score:.2%} ({valid_count}/{len(references)} valid)")

        return {
            "score": score,
            "total_references": len(references),
            "valid_references": valid_count,
            "invalid_references": invalid_refs,
            "reference_types": self._count_by_type(references)
        }

    def _extract_references(self, content: str) -> List[Dict]:
        """
        Extract all source code references from documentation.

        Per Section 5.5.1: Reference Extraction Methods
        FIXED: More precise extraction to avoid false positives from natural language
        """
        references = []

        for ref_type, pattern in self.reference_patterns.items():
            if ref_type == "code_excerpt":
                # Handle code blocks specially
                matches = re.finditer(pattern, content, re.DOTALL | re.MULTILINE)
            elif ref_type == "variable":
                # Variable pattern should NOT use IGNORECASE to avoid matching
                # natural language like "data from", "data structures"
                matches = re.finditer(pattern, content)
            else:
                matches = re.finditer(pattern, content, re.IGNORECASE)

            for match in matches:
                if ref_type == "line_number":
                    references.append({
                        "type": "line_number",
                        "line": int(match.group(1)),
                        "original_text": match.group(0)
                    })

                elif ref_type == "line_range":
                    references.append({
                        "type": "line_range",
                        "start": int(match.group(1)),
                        "end": int(match.group(2)),
                        "original_text": match.group(0)
                    })

                elif ref_type in ["paragraph", "section"]:
                    name = match.group(1)
                    # Filter out common English words
                    if name.lower() not in self.excluded_words:
                        references.append({
                            "type": ref_type,
                            "name": name,
                            "original_text": match.group(0)
                        })

                elif ref_type == "variable":
                    # Variable pattern has two alternative groups
                    name = match.group(1) or match.group(2)
                    if name and name.lower() not in self.excluded_words:
                        # Additional validation: COBOL variables typically have hyphens
                        # or are at least 3 chars and uppercase
                        if len(name) >= 3:
                            references.append({
                                "type": ref_type,
                                "name": name,
                                "original_text": match.group(0)
                            })

                elif ref_type == "code_excerpt":
                    references.append({
                        "type": "code_excerpt",
                        "code": match.group(1),
                        "original_text": match.group(0)
                    })

        logger.info(f"Extracted {len(references)} references")
        return references

    def _validate_reference(self, ref: Dict, ground_truth: Dict,
                          source_code: str) -> bool:
        """
        Validate a single reference.

        Per Section 5.5.2: Reference Validity Scoring
        Returns: True if valid, False if broken/fabricated
        """
        ref_type = ref["type"]

        if ref_type == "line_number":
            return self._validate_line_number(ref["line"], source_code)

        elif ref_type == "line_range":
            return self._validate_line_range(ref["start"], ref["end"], source_code)

        elif ref_type == "paragraph":
            return self._validate_paragraph(ref["name"], ground_truth)

        elif ref_type == "section":
            return self._validate_section(ref["name"], ground_truth)

        elif ref_type == "variable":
            return self._validate_variable(ref["name"], ground_truth)

        elif ref_type == "code_excerpt":
            return self._validate_code_excerpt(ref["code"], source_code)

        return False

    def _validate_line_number(self, line_num: int, source_code: str) -> bool:
        """Check if line number exists in source"""
        lines = source_code.split('\n')
        return 1 <= line_num <= len(lines)

    def _validate_line_range(self, start: int, end: int, source_code: str) -> bool:
        """Check if line range is valid"""
        if start > end:
            return False

        lines = source_code.split('\n')
        return 1 <= start <= len(lines) and 1 <= end <= len(lines)

    def _validate_paragraph(self, name: str, ground_truth: Dict) -> bool:
        """Check if paragraph exists in control flow"""
        paragraphs = ground_truth.get("control_flow", {}).get("paragraphs", [])

        return any(
            p.get("name", "").upper() == name.upper()
            for p in paragraphs
        )

    def _validate_section(self, name: str, ground_truth: Dict) -> bool:
        """Check if section exists (less common in modern COBOL)"""
        # Sections are tracked in control flow
        # For now, just check if it appears in any division/section names
        divisions = ground_truth.get("metadata", {}).get("divisions", [])
        return name.upper() in [d.upper() for d in divisions]

    def _validate_variable(self, name: str, ground_truth: Dict) -> bool:
        """Check if variable exists in data structures"""
        fields = ground_truth.get("data_structures", {}).get("fields", [])

        return any(
            f.get("name", "").upper() == name.upper()
            for f in fields
        )

    def _validate_code_excerpt(self, excerpt: str, source_code: str,
                              similarity_threshold: float = 0.90) -> bool:
        """
        Validate code excerpt using fuzzy matching.

        Per Section 5.5.1: Fuzzy match with Levenshtein ≤ 10% (similarity ≥ 90%)
        """
        # Normalize whitespace and case
        excerpt_normalized = ' '.join(excerpt.split()).upper()
        source_normalized = ' '.join(source_code.split()).upper()

        # Check exact substring match first
        if excerpt_normalized in source_normalized:
            return True

        # Fuzzy match using SequenceMatcher
        # Check if excerpt appears with ≥90% similarity anywhere in source
        excerpt_len = len(excerpt_normalized)

        # Sliding window approach
        for i in range(len(source_normalized) - excerpt_len + 1):
            window = source_normalized[i:i + excerpt_len]
            similarity = SequenceMatcher(None, excerpt_normalized, window).ratio()

            if similarity >= similarity_threshold:
                return True

        # Also try matching individual lines
        excerpt_lines = [l.strip() for l in excerpt.split('\n') if l.strip()]
        source_lines = [l.strip().upper() for l in source_code.split('\n')]

        if len(excerpt_lines) == 0:
            return False

        # Check if at least 80% of excerpt lines appear in source
        matching_lines = 0
        for excerpt_line in excerpt_lines:
            excerpt_line_norm = ' '.join(excerpt_line.split()).upper()

            for source_line in source_lines:
                source_line_norm = ' '.join(source_line.split())

                if excerpt_line_norm in source_line_norm or \
                   SequenceMatcher(None, excerpt_line_norm, source_line_norm).ratio() >= 0.85:
                    matching_lines += 1
                    break

        line_match_ratio = matching_lines / len(excerpt_lines)
        return line_match_ratio >= 0.80

    def _count_by_type(self, references: List[Dict]) -> Dict[str, int]:
        """Count references by type"""
        counts = {}

        for ref in references:
            ref_type = ref["type"]
            counts[ref_type] = counts.get(ref_type, 0) + 1

        return counts
