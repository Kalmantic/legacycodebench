"""Structural Completeness Evaluator (30% weight)

Implements Section 5.1 of spec: Structural Completeness (SC)
Measures: Did the AI document all elements that exist in the code?

Formula: SC = (Elements_Documented / Elements_Extracted) x 100
"""

import re
from typing import Dict, List, Set
import logging

logger = logging.getLogger(__name__)


class StructuralCompletenessEvaluator:
    """
    Evaluate structural completeness of documentation.

    Compares documentation against ground truth extracted by static analysis.
    NO comparison to reference docs - validates against actual code elements.
    """

    def __init__(self):
        # Element category weights per spec
        self.element_weights = {
            "data_structures": 0.25,
            "business_rules": 0.30,
            "control_flow": 0.20,
            "file_operations": 0.10,
            "external_calls": 0.08,
            "error_handlers": 0.07,
        }

    def _normalize_markdown(self, content: str) -> str:
        """
        Strip markdown formatting for regex matching (Issue 5.4).
        
        Removes bold (**), italic (*), and code (`) formatting that could
        prevent element names from matching (e.g., **CALC-INTEREST** vs CALC-INTEREST).
        """
        # Remove bold, italic, and code formatting
        content = re.sub(r'\*\*|__', '', content)  # Bold
        content = re.sub(r'\*|_', '', content)      # Italic
        content = re.sub(r'`', '', content)         # Code
        return content

    def evaluate(self, submission_content: str, ground_truth: Dict) -> Dict:
        """
        Evaluate structural completeness.

        Args:
            submission_content: AI-generated documentation (markdown)
            ground_truth: Automated ground truth from static analysis

        Returns:
            Dictionary with:
            - score: Overall SC score (0.0 to 1.0)
            - by_category: Scores for each element category
            - missing_elements: List of elements not documented
        """
        logger.info("Evaluating structural completeness")

        # FIXED (Issue 5.4): Normalize markdown before matching
        normalized_content = self._normalize_markdown(submission_content)
        content_upper = normalized_content.upper()

        results = {}
        missing_elements = {}

        # Evaluate each category
        results["data_structures"] = self._eval_data_structures(
            content_upper, ground_truth.get("data_structures", {}), missing_elements
        )

        results["business_rules"] = self._eval_business_rules(
            content_upper, ground_truth.get("business_rules", {}), missing_elements
        )

        results["control_flow"] = self._eval_control_flow(
            content_upper, ground_truth.get("control_flow", {}), missing_elements
        )

        results["file_operations"] = self._eval_file_operations(
            content_upper, ground_truth.get("dependencies", {}), missing_elements
        )

        results["external_calls"] = self._eval_external_calls(
            content_upper, ground_truth.get("dependencies", {}), missing_elements
        )

        results["error_handlers"] = self._eval_error_handlers(
            content_upper, ground_truth.get("error_handlers", []), missing_elements
        )

        # Calculate weighted overall score
        overall_score = sum(
            results[category] * weight
            for category, weight in self.element_weights.items()
        )

        logger.info(f"Structural Completeness Score: {overall_score:.2%}")

        return {
            "score": overall_score,
            "by_category": results,
            "missing_elements": missing_elements,
            "total_elements": sum(len(v) for v in missing_elements.values()),
            "documented_percentage": overall_score
        }

    def _eval_data_structures(self, content: str, data_structures: Dict,
                              missing: Dict) -> float:
        """Evaluate data structure coverage"""
        if not data_structures or data_structures.get("total_structures", 0) == 0:
            return 1.0

        structures = data_structures.get("data_structures", [])
        fields = data_structures.get("fields", [])

        documented_structures = 0
        documented_fields = 0
        missing_list = []

        # Check 01-level structures
        for structure in structures:
            if self._element_mentioned(structure["name"], content):
                documented_structures += 1
            else:
                missing_list.append(f"Data structure: {structure['name']}")

        # Check significant fields (level < 50, has PICTURE)
        significant_fields = [
            f for f in fields
            if f.get("level", 99) < 50 and f.get("picture") and f.get("level") != 88
        ]

        for field in significant_fields:
            if self._element_mentioned(field["name"], content):
                documented_fields += 1
            else:
                missing_list.append(f"Field: {field['name']}")

        # Check REDEFINES (important for understanding)
        redefines = data_structures.get("redefines", [])
        documented_redefines = sum(
            1 for r in redefines
            if self._element_mentioned(r["field"], content)
        )

        if missing_list:
            missing["data_structures"] = missing_list

        # Score calculation
        total_elements = len(structures) + len(significant_fields) + len(redefines)
        if total_elements == 0:
            return 1.0

        documented_elements = documented_structures + documented_fields + documented_redefines
        return documented_elements / total_elements

    def _eval_business_rules(self, content: str, business_rules: Dict,
                            missing: Dict) -> float:
        """Evaluate business rule coverage"""
        rules = business_rules.get("business_rules", [])

        if not rules:
            return 1.0

        documented_rules = 0
        missing_list = []

        for rule in rules:
            # Check if rule is mentioned (by keywords or description)
            if self._rule_mentioned(rule, content):
                documented_rules += 1
            else:
                missing_list.append(f"Rule {rule['rule_id']}: {rule['description']}")

        if missing_list:
            missing["business_rules"] = missing_list

        return documented_rules / len(rules) if rules else 1.0

    def _eval_control_flow(self, content: str, control_flow: Dict,
                          missing: Dict) -> float:
        """Evaluate control flow coverage"""
        paragraphs = control_flow.get("paragraphs", [])

        if not paragraphs:
            return 1.0

        documented_paragraphs = 0
        missing_list = []

        # Check significant paragraphs (with statements)
        significant_paras = [p for p in paragraphs if p.get("statement_count", 0) > 0]

        for para in significant_paras:
            if self._element_mentioned(para["name"], content):
                documented_paragraphs += 1
            else:
                missing_list.append(f"Paragraph: {para['name']}")

        # Check PERFORM targets
        performs = control_flow.get("perform_targets", [])
        documented_performs = sum(
            1 for p in performs
            if self._element_mentioned(p["target"], content)
        )

        if missing_list:
            missing["control_flow"] = missing_list

        total = len(significant_paras) + len(performs)
        if total == 0:
            return 1.0

        documented = documented_paragraphs + documented_performs
        return documented / total

    def _eval_file_operations(self, content: str, dependencies: Dict,
                              missing: Dict) -> float:
        """Evaluate file operation coverage"""
        files_info = dependencies.get("files", {})
        files = files_info.get("files", [])

        if not files:
            return 1.0

        documented_files = 0
        missing_list = []

        for file_info in files:
            if self._element_mentioned(file_info["name"], content):
                documented_files += 1
            else:
                missing_list.append(f"File: {file_info['name']}")

        if missing_list:
            missing["file_operations"] = missing_list

        return documented_files / len(files) if files else 1.0

    def _eval_external_calls(self, content: str, dependencies: Dict,
                            missing: Dict) -> float:
        """Evaluate external call coverage"""
        calls = dependencies.get("calls", [])

        if not calls:
            return 1.0

        documented_calls = 0
        missing_list = []

        for call in calls:
            if self._element_mentioned(call["callee"], content):
                documented_calls += 1
            else:
                missing_list.append(f"CALL to: {call['callee']}")

        if missing_list:
            missing["external_calls"] = missing_list

        return documented_calls / len(calls) if calls else 1.0

    def _eval_error_handlers(self, content: str, error_handlers: List,
                            missing: Dict) -> float:
        """Evaluate error handler coverage"""
        if not error_handlers:
            return 1.0

        documented_handlers = 0
        missing_list = []

        # Group by type to avoid duplicates
        handler_types = set(h["type"] for h in error_handlers)

        for handler_type in handler_types:
            # Check if this type of error handling is mentioned
            if self._error_handler_mentioned(handler_type, content):
                documented_handlers += 1
            else:
                missing_list.append(f"Error handler: {handler_type}")

        if missing_list:
            missing["error_handlers"] = missing_list

        return documented_handlers / len(handler_types) if handler_types else 1.0

    def _element_mentioned(self, element_name: str, content: str) -> bool:
        """
        Check if element is mentioned in documentation.

        Uses flexible matching:
        - Exact name match
        - Name with hyphens replaced by spaces/underscores
        - Partial word boundary match
        """
        if not element_name:
            return False

        element_upper = element_name.upper()

        # Exact match
        if element_upper in content:
            return True

        # Try variations (hyphens to spaces/underscores)
        variations = [
            element_upper.replace('-', ' '),
            element_upper.replace('-', '_'),
            element_upper.replace('-', '')
        ]

        for variation in variations:
            if variation in content:
                return True

        # Partial match (for long names)
        # If element has hyphens, try matching individual parts
        if '-' in element_name:
            parts = element_name.split('-')
            # At least 2 significant parts should be mentioned
            if len(parts) >= 2:
                significant_parts = [p for p in parts if len(p) > 2]
                matches = sum(1 for p in significant_parts if p.upper() in content)
                if matches >= min(2, len(significant_parts)):
                    return True

        return False

    def _rule_mentioned(self, rule: Dict, content: str) -> bool:
        """Check if business rule is mentioned"""
        # Check if rule description keywords appear
        keywords = rule.get("keywords", [])

        if not keywords:
            return False

        # Require at least 60% of keywords to match
        matches = sum(1 for kw in keywords if kw.upper() in content)
        threshold = len(keywords) * 0.6

        return matches >= threshold

    def _error_handler_mentioned(self, handler_type: str, content: str) -> bool:
        """Check if error handler type is mentioned"""
        # Map technical names to common documentation terms
        # PRODUCTION GRADE v2.1.3: Added general error handling terms
        handler_keywords = {
            "size_error": ["SIZE ERROR", "OVERFLOW", "ARITHMETIC ERROR", "NUMERIC OVERFLOW"],
            "invalid_key": ["INVALID KEY", "RECORD NOT FOUND", "KEY ERROR", "INVALID RECORD", "INVALID DATA"],
            "end_of_file": ["END OF FILE", "EOF", "AT END", "END-OF-FILE"],
            "not_at_end": ["NOT AT END", "RECORD FOUND", "MORE RECORDS"],
            "file_status_check": ["FILE STATUS", "FILE ERROR", "I/O ERROR", "IO ERROR"],
            "status_check": ["STATUS", "RETURN CODE", "ERROR CODE", "ERROR HANDLING", "ERROR CONDITION"]
        }
        
        # Also check for general error handling documentation
        general_error_terms = [
            "ERROR HANDLING", "ERROR HANDLER", "EXCEPTION", "ERROR CONDITION",
            "HANDLES ERROR", "HANDLE ERROR", "CATCHING ERROR", "TRAP ERROR",
            "ERROR RECOVERY", "ERROR PROCESSING", "INVALID", "VALIDATION ERROR",
            "DISPLAYS WARNING", "WARNING MESSAGE", "ERROR MESSAGE"
        ]

        keywords = handler_keywords.get(handler_type, [handler_type.upper()])
        
        # Check specific keywords first
        if any(kw in content for kw in keywords):
            return True
        
        # Fallback: check if any general error handling is documented
        # This handles cases where documentation uses natural language
        return any(term in content for term in general_error_terms)
