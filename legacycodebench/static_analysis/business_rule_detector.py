"""Business Rule Detector for COBOL

Implements Section 2.5: Business Rule Inference Engine
- Pattern-based classification of business rules
- Threshold checks, date calculations, validation rules
- Automation Level: 80-95% (Medium-High confidence)
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BusinessRule:
    """Represents a detected business rule"""
    rule_id: str
    rule_type: str
    description: str
    pattern: str
    confidence: float
    line_number: int
    paragraph: str
    source_statement: str
    keywords: List[str]


class BusinessRuleDetector:
    """
    Detect business rules using pattern-based classification.

    Per Section 2.5 of spec:
    - Threshold checks: High confidence (95%)
    - Date calculations: High confidence (95%)
    - Validation rules: High confidence (95%)
    - Complex formulas: Medium confidence (70%)
    """

    def __init__(self):
        # Pattern definitions from Section 2.5
        self.patterns = {
            "threshold_check": {
                "regex": r'IF\s+([A-Z0-9-]+)\s*([<>=]+)\s*([0-9]+)',
                "confidence": 0.95,
                "description": "Threshold comparison"
            },
            "date_calculation": {
                "regex": r'(DATE|DAY|MONTH|YEAR|AGE|CURRENT-DATE)',
                "confidence": 0.95,
                "description": "Date/time operation"
            },
            "validation_rule": {
                "regex": r'IF\s+([A-Z0-9-]+)\s+(NOT\s+)?=\s*[\'"]?([A-Z0-9]+)[\'"]?',
                "confidence": 0.95,
                "description": "Value validation"
            },
            "accumulation": {
                "regex": r'(ADD|COMPUTE)\s+(.+?)\s+TO\s+([A-Z0-9-]+)',
                "confidence": 0.90,
                "description": "Running total/accumulation"
            },
            "conditional_assignment": {
                "regex": r'IF\s+(.+?)\s+MOVE\s+(.+?)\s+TO\s+([A-Z0-9-]+)',
                "confidence": 0.80,
                "description": "Conditional value assignment"
            },
            "complex_formula": {
                "regex": r'COMPUTE\s+([A-Z0-9-]+)\s*=\s*(.+)',
                "confidence": 0.70,
                "description": "Mathematical calculation"
            },
            "range_check": {
                "regex": r'IF\s+([A-Z0-9-]+)\s+>=\s*([0-9]+)\s+AND\s+([A-Z0-9-]+)\s+<=\s*([0-9]+)',
                "confidence": 0.95,
                "description": "Value range validation"
            },
            "status_check": {
                "regex": r'IF\s+(FILE-STATUS|SQL-STATUS|RETURN-CODE|STATUS)',
                "confidence": 0.90,
                "description": "Status/error code check"
            }
        }

    def detect(self, parsed_cobol, line_offset: int = 0) -> Dict:
        """
        Detect business rules in PROCEDURE DIVISION.

        Args:
            parsed_cobol: ParsedCOBOL object
            line_offset: Line number offset

        Returns:
            Dictionary with detected business rules
        """
        logger.info("Detecting business rules using pattern matching")

        rules = []
        rule_counter = 1

        for paragraph in parsed_cobol.paragraphs:
            para_rules = self._detect_in_paragraph(paragraph, rule_counter)
            rules.extend(para_rules)
            rule_counter += len(para_rules)

        # Group by type
        rules_by_type = self._group_by_type(rules)

        # Extract keywords for semantic matching
        keywords = self._extract_keywords(rules)

        logger.info(f"Detected {len(rules)} business rules")
        logger.info(f"Rule types: {list(rules_by_type.keys())}")

        return {
            "business_rules": [self._rule_to_dict(r) for r in rules],
            "rules_by_type": rules_by_type,
            "total_rules": len(rules),
            "high_confidence_rules": len([r for r in rules if r.confidence >= 0.90]),
            "medium_confidence_rules": len([r for r in rules if 0.70 <= r.confidence < 0.90]),
            "keywords": keywords,
            "avg_confidence": sum(r.confidence for r in rules) / len(rules) if rules else 0.0
        }

    def _detect_in_paragraph(self, paragraph, start_id: int) -> List[BusinessRule]:
        """Detect business rules in a single paragraph"""
        rules = []
        rule_id = start_id

        for i, stmt in enumerate(paragraph.statements):
            # Try each pattern
            for pattern_name, pattern_def in self.patterns.items():
                match = re.search(pattern_def["regex"], stmt, re.IGNORECASE)

                if match:
                    # Extract rule details
                    description = self._generate_description(
                        pattern_name,
                        pattern_def["description"],
                        match,
                        stmt
                    )

                    keywords = self._extract_statement_keywords(stmt)

                    rule = BusinessRule(
                        rule_id=f"BR-{rule_id:03d}",
                        rule_type=pattern_name,
                        description=description,
                        pattern=pattern_def["regex"],
                        confidence=pattern_def["confidence"],
                        line_number=paragraph.start_line + i,
                        paragraph=paragraph.name,
                        source_statement=stmt.strip(),
                        keywords=keywords
                    )

                    rules.append(rule)
                    rule_id += 1

                    # Don't match multiple patterns on same statement
                    break

        return rules

    def _generate_description(self, pattern_type: str, base_desc: str,
                             match: re.Match, statement: str) -> str:
        """Generate human-readable description of the rule"""

        if pattern_type == "threshold_check":
            field = match.group(1)
            operator = match.group(2)
            value = match.group(3)
            return f"{base_desc}: {field} {operator} {value}"

        elif pattern_type == "validation_rule":
            field = match.group(1)
            negation = match.group(2) or ""
            value = match.group(3)
            return f"{base_desc}: {field} {negation}= {value}"

        elif pattern_type == "accumulation":
            operation = match.group(1)
            value = match.group(2)
            target = match.group(3)
            return f"{base_desc}: {operation} {value} TO {target}"

        elif pattern_type == "complex_formula":
            variable = match.group(1)
            expression = match.group(2)
            # Truncate long expressions
            if len(expression) > 50:
                expression = expression[:47] + "..."
            return f"{base_desc}: {variable} = {expression}"

        else:
            return base_desc

    def _extract_statement_keywords(self, statement: str) -> List[str]:
        """Extract keywords from statement for semantic matching"""
        # Remove COBOL keywords
        cobol_keywords = {
            'IF', 'THEN', 'ELSE', 'END-IF', 'MOVE', 'TO', 'FROM',
            'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'COMPUTE',
            'PERFORM', 'UNTIL', 'VARYING', 'GO', 'GOTO'
        }

        words = re.findall(r'[A-Z][A-Z0-9-]*', statement.upper())

        # Filter out COBOL keywords
        keywords = [w for w in words if w not in cobol_keywords]

        # Return unique keywords
        return list(set(keywords))

    def _group_by_type(self, rules: List[BusinessRule]) -> Dict[str, List[str]]:
        """Group rules by type"""
        grouped = {}

        for rule in rules:
            if rule.rule_type not in grouped:
                grouped[rule.rule_type] = []
            grouped[rule.rule_type].append(rule.rule_id)

        return grouped

    def _extract_keywords(self, rules: List[BusinessRule]) -> List[str]:
        """Extract all unique keywords across all rules"""
        all_keywords = set()

        for rule in rules:
            all_keywords.update(rule.keywords)

        return sorted(list(all_keywords))

    def _rule_to_dict(self, rule: BusinessRule) -> Dict:
        """Convert BusinessRule to dictionary"""
        return {
            "rule_id": rule.rule_id,
            "rule_type": rule.rule_type,
            "description": rule.description,
            "confidence": rule.confidence,
            "line_number": rule.line_number,
            "paragraph": rule.paragraph,
            "source_statement": rule.source_statement,
            "keywords": rule.keywords
        }

    def detect_error_handlers(self, parsed_cobol) -> List[Dict]:
        """
        Detect error handling patterns.

        Per Section 2.2 of spec: ON SIZE ERROR, INVALID KEY, AT END, FILE STATUS checks
        """
        error_handlers = []

        error_patterns = [
            (r'ON\s+SIZE\s+ERROR', 'size_error'),
            (r'INVALID\s+KEY', 'invalid_key'),
            (r'AT\s+END', 'end_of_file'),
            (r'NOT\s+AT\s+END', 'not_at_end'),
            (r'FILE-STATUS', 'file_status_check'),
            (r'IF\s+.*-STATUS\s*(?:=|NOT)', 'status_check')
        ]

        for paragraph in parsed_cobol.paragraphs:
            for i, stmt in enumerate(paragraph.statements):
                for pattern, handler_type in error_patterns:
                    if re.search(pattern, stmt, re.IGNORECASE):
                        error_handlers.append({
                            "type": handler_type,
                            "paragraph": paragraph.name,
                            "line_number": paragraph.start_line + i,
                            "statement": stmt.strip()
                        })

        logger.info(f"Detected {len(error_handlers)} error handlers")
        return error_handlers
