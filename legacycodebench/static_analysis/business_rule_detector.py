"""Business Rule Detector for COBOL

Implements Section 2.5: Business Rule Inference Engine
- Pattern-based classification of business rules
- Threshold checks, date calculations, validation rules
- Automation Level: 80-95% (Medium-High confidence)

PRODUCTION GRADE REDESIGN (v2.1):
- Added priority classification (CRITICAL, IMPORTANT, TRIVIAL)
- Filtered out trivial date/time MOVE operations
- Focus on actual business logic, not boilerplate
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RulePriority(Enum):
    """Business rule priority for evaluation weighting"""
    CRITICAL = "critical"     # Core business calculations, validations
    IMPORTANT = "important"   # Secondary business logic
    TRIVIAL = "trivial"       # Boilerplate (date moves, counters)


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
    priority: RulePriority = field(default=RulePriority.IMPORTANT)


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
        # PRODUCTION GRADE: Added priority classification
        self.patterns = {
            # CRITICAL: Core business calculations - these MUST be documented
            "complex_formula": {
                "regex": r'COMPUTE\s+([A-Z0-9-]+)\s*=\s*(.+)',
                "confidence": 0.95,
                "description": "Mathematical calculation",
                "priority": RulePriority.CRITICAL
            },
            "threshold_check": {
                "regex": r'IF\s+([A-Z0-9-]+)\s*([<>=]+)\s*([0-9]+)',
                "confidence": 0.95,
                "description": "Threshold comparison",
                "priority": RulePriority.CRITICAL
            },
            "range_check": {
                "regex": r'IF\s+([A-Z0-9-]+)\s+>=\s*([0-9]+)\s+AND\s+([A-Z0-9-]+)\s+<=\s*([0-9]+)',
                "confidence": 0.95,
                "description": "Value range validation",
                "priority": RulePriority.CRITICAL
            },
            
            # IMPORTANT: Business validations and conditional logic
            "validation_rule": {
                "regex": r'IF\s+([A-Z0-9-]+)\s+(NOT\s+)?=\s*[\'"]?([A-Z0-9]+)[\'"]?',
                "confidence": 0.95,
                "description": "Value validation",
                "priority": RulePriority.IMPORTANT
            },
            "conditional_assignment": {
                "regex": r'IF\s+(.+?)\s+MOVE\s+(.+?)\s+TO\s+([A-Z0-9-]+)',
                "confidence": 0.80,
                "description": "Conditional value assignment",
                "priority": RulePriority.IMPORTANT
            },
            "status_check": {
                "regex": r'IF\s+(FILE-STATUS|SQL-STATUS|RETURN-CODE|STATUS)',
                "confidence": 0.90,
                "description": "Status/error code check",
                "priority": RulePriority.IMPORTANT
            },
            
            # TRIVIAL: Boilerplate operations - don't penalize for missing these
            "accumulation": {
                "regex": r'(ADD|COMPUTE)\s+(.+?)\s+TO\s+([A-Z0-9-]+)',
                "confidence": 0.90,
                "description": "Running total/accumulation",
                "priority": RulePriority.TRIVIAL  # Counters are boilerplate
            },
            "date_formatting": {
                # Only match ACTUAL date calculations, not simple MOVE statements
                # Changed from matching any DATE/MONTH/YEAR to requiring COMPUTE/ADD with dates
                "regex": r'(COMPUTE|ADD|SUBTRACT)\s+.*(DATE|DAY|MONTH|YEAR|AGE)',
                "confidence": 0.90,
                "description": "Date calculation",
                "priority": RulePriority.IMPORTANT
            },
        }
        
        # Patterns to EXCLUDE (trivial boilerplate, not real business rules)
        self.exclude_patterns = [
            r'^MOVE\s+.*(DATE|DAY|MONTH|YEAR)\s+TO',  # Simple date moves
            r'^MOVE\s+SPACES\s+TO',                    # Initialize with spaces
            r'^MOVE\s+ZEROS?\s+TO',                    # Initialize with zeros
            r'^MOVE\s+LOW-VALUES\s+TO',                # Initialize with low-values
            r'^INITIALIZE\s+',                          # Simple initialization
            r'^DISPLAY\s+',                             # Display statements
            r'^ADD\s+\+?1\s+TO\s+.*-(CNT|COUNT|CTR)',  # Simple counter increments
        ]

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
        
        # PRODUCTION GRADE: Group by priority for evaluation
        rules_by_priority = self._group_by_priority(rules)

        # Extract keywords for semantic matching
        keywords = self._extract_keywords(rules)
        
        # Count by priority
        critical_rules = [r for r in rules if r.priority == RulePriority.CRITICAL]
        important_rules = [r for r in rules if r.priority == RulePriority.IMPORTANT]
        trivial_rules = [r for r in rules if r.priority == RulePriority.TRIVIAL]

        logger.info(f"Detected {len(rules)} business rules")
        logger.info(f"  CRITICAL: {len(critical_rules)}, IMPORTANT: {len(important_rules)}, TRIVIAL: {len(trivial_rules)}")
        logger.info(f"Rule types: {list(rules_by_type.keys())}")

        return {
            "business_rules": [self._rule_to_dict(r) for r in rules],
            "rules_by_type": rules_by_type,
            "rules_by_priority": rules_by_priority,
            "total_rules": len(rules),
            "critical_rules": len(critical_rules),
            "important_rules": len(important_rules),
            "trivial_rules": len(trivial_rules),
            "high_confidence_rules": len([r for r in rules if r.confidence >= 0.90]),
            "medium_confidence_rules": len([r for r in rules if 0.70 <= r.confidence < 0.90]),
            "keywords": keywords,
            "avg_confidence": sum(r.confidence for r in rules) / len(rules) if rules else 0.0
        }

    def _is_excluded(self, statement: str) -> bool:
        """Check if statement matches exclusion patterns (trivial boilerplate)"""
        for pattern in self.exclude_patterns:
            if re.search(pattern, statement, re.IGNORECASE):
                return True
        return False

    def _detect_in_paragraph(self, paragraph, start_id: int) -> List[BusinessRule]:
        """Detect business rules in a single paragraph"""
        rules = []
        rule_id = start_id

        for i, stmt in enumerate(paragraph.statements):
            # Clean statement (collapse internal newlines to spaces for regex matching)
            clean_stmt = re.sub(r'\s+', ' ', stmt).strip()
            
            # PRODUCTION GRADE: Skip excluded boilerplate statements
            if self._is_excluded(clean_stmt):
                continue
            
            # Try each pattern
            for pattern_name, pattern_def in self.patterns.items():
                match = re.search(pattern_def["regex"], clean_stmt, re.IGNORECASE)

                if match:
                    # Extract rule details
                    description = self._generate_description(
                        pattern_name,
                        pattern_def["description"],
                        match,
                        clean_stmt
                    )

                    keywords = self._extract_statement_keywords(clean_stmt)
                    priority = pattern_def.get("priority", RulePriority.IMPORTANT)

                    rule = BusinessRule(
                        rule_id=f"BR-{rule_id:03d}",
                        rule_type=pattern_name,
                        description=description,
                        pattern=pattern_def["regex"],
                        confidence=pattern_def["confidence"],
                        line_number=paragraph.start_line + i,
                        paragraph=paragraph.name,
                        source_statement=stmt.strip(),
                        keywords=keywords,
                        priority=priority
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
        # PRODUCTION GRADE: Comprehensive exclusion list to prevent false positives
        # These are COBOL reserved words and generic terms that match too easily
        cobol_keywords = {
            # Basic COBOL verbs and statements
            'IF', 'THEN', 'ELSE', 'END-IF', 'MOVE', 'TO', 'FROM',
            'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'COMPUTE',
            'PERFORM', 'UNTIL', 'VARYING', 'GO', 'GOTO',
            'DISPLAY', 'ACCEPT', 'STOP', 'RUN', 'EXIT',
            'OPEN', 'CLOSE', 'READ', 'WRITE', 'REWRITE', 'DELETE',
            'CALL', 'USING', 'RETURNING', 'GIVING',
            # Evaluate
            'EVALUATE', 'WHEN', 'OTHER', 'END-EVALUATE', 'TRUE', 'FALSE',
            # Conditions
            'AND', 'OR', 'NOT', 'IS', 'ARE', 'EQUAL', 'GREATER', 'LESS',
            'THAN', 'ZERO', 'ZEROS', 'ZEROES', 'SPACE', 'SPACES',
            # Status/Error handling
            'INVALID', 'VALID', 'ERROR', 'SIZE', 'ON', 'AT', 'END',
            'STATUS', 'FILE', 'KEY', 'RECORD', 'INTO', 'LENGTH',
            # Data division
            'PIC', 'PICTURE', 'VALUE', 'FILLER', 'REDEFINES', 'OCCURS',
            'BINARY', 'COMP', 'PACKED', 'DECIMAL',
            # Arithmetic
            'ROUNDED', 'REMAINDER', 'GIVING',
            # Generic single-letter or short
            'A', 'B', 'C', 'X', 'V', 'S', 'Z', 'N', '9',
            # Very generic
            'DATA', 'TYPE', 'FLAG', 'NEW', 'OLD', 'SET', 'RESET',
            'WARNING', 'INFO', 'DEBUG', 'LINE', 'ADVANCE',
        }

        words = re.findall(r'[A-Z][A-Z0-9-]*', statement.upper())

        # Filter out COBOL keywords and short words
        keywords = [
            w for w in words 
            if w not in cobol_keywords 
            and len(w) >= 4  # Require minimum length
            and not w.startswith('END-')  # Skip END-* constructs
        ]

        # Return unique keywords, limit to most important ones
        # Prefer longer/more specific keywords
        unique_keywords = list(set(keywords))
        unique_keywords.sort(key=lambda x: -len(x))  # Longer first
        return unique_keywords[:10]  # Limit to top 10 most specific

    def _group_by_type(self, rules: List[BusinessRule]) -> Dict[str, List[str]]:
        """Group rules by type"""
        grouped = {}

        for rule in rules:
            if rule.rule_type not in grouped:
                grouped[rule.rule_type] = []
            grouped[rule.rule_type].append(rule.rule_id)

        return grouped

    def _group_by_priority(self, rules: List[BusinessRule]) -> Dict[str, List[str]]:
        """Group rules by priority for evaluation weighting"""
        grouped = {
            "critical": [],
            "important": [],
            "trivial": []
        }

        for rule in rules:
            priority_key = rule.priority.value
            grouped[priority_key].append(rule.rule_id)

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
            "priority": rule.priority.value,  # PRODUCTION GRADE: Include priority
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
