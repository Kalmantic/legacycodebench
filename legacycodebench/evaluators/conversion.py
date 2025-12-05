"""Conversion task evaluator for COBOL to modern language conversion"""

import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json

from legacycodebench.config import CONVERSION_WEIGHTS, CONVERSION_TARGETS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversionEvaluator:
    """Evaluate COBOL to modern language conversion submissions"""

    def __init__(self):
        self.weights = CONVERSION_WEIGHTS
        self.targets = CONVERSION_TARGETS

    def evaluate(self, submission_path: Path, task) -> Dict:
        """Evaluate a conversion submission"""
        if not submission_path.exists():
            return {
                "score": 0.0,
                "compilation": 0.0,
                "functional": 0.0,
                "code_quality": 0.0,
                "completeness": 0.0,
                "errors": ["Submission file not found"],
            }

        try:
            with open(submission_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {
                "score": 0.0,
                "errors": [f"Failed to read submission: {e}"],
            }

        target_language = task.target_language or "java"

        # Evaluate each metric
        compilation_score, compilation_details = self._check_compilation(
            content, target_language, submission_path
        )
        functional_score, functional_details = self._check_functional_equivalence(
            content, target_language, task
        )
        quality_score, quality_details = self._check_code_quality(
            content, target_language
        )
        completeness_score, completeness_details = self._check_completeness(
            content, target_language, task
        )

        # Calculate weighted score
        score = (
            self.weights["compilation"] * compilation_score +
            self.weights["functional"] * functional_score +
            self.weights["code_quality"] * quality_score +
            self.weights["completeness"] * completeness_score
        )

        return {
            "score": round(score, 4),
            "compilation": round(compilation_score, 4),
            "functional": round(functional_score, 4),
            "code_quality": round(quality_score, 4),
            "completeness": round(completeness_score, 4),
            "evaluation_method": "heuristic",
            "target_language": target_language,
            "details": {
                "compilation": compilation_details,
                "functional": functional_details,
                "quality": quality_details,
                "completeness": completeness_details,
            }
        }

    def _check_compilation(self, content: str, target_language: str,
                          submission_path: Path) -> Tuple[float, Dict]:
        """Check if the converted code compiles/parses successfully"""
        details = {"compilable": False, "errors": [], "warnings": []}

        if target_language == "python":
            # Python: check syntax
            try:
                compile(content, submission_path.name, 'exec')
                details["compilable"] = True
                return 1.0, details
            except SyntaxError as e:
                details["errors"].append(f"Syntax error: {e}")
                return 0.0, details

        elif target_language == "java":
            # Java: check for basic structure
            score = 0.0

            # Check for class definition
            if re.search(r'public\s+class\s+\w+', content):
                score += 0.3
                details["has_class"] = True

            # Check for main method or proper methods
            if re.search(r'(public\s+static\s+void\s+main|public\s+\w+\s+\w+\s*\()', content):
                score += 0.3
                details["has_methods"] = True

            # Check for proper imports
            if re.search(r'import\s+[\w.]+;', content):
                score += 0.2
                details["has_imports"] = True

            # Check for balanced braces
            if content.count('{') == content.count('}'):
                score += 0.2
                details["balanced_braces"] = True

            details["compilable"] = score >= 0.8
            return score, details

        elif target_language == "csharp":
            # C#: check for basic structure
            score = 0.0

            # Check for namespace/class
            if re.search(r'(namespace|class)\s+\w+', content):
                score += 0.4
                details["has_class"] = True

            # Check for methods
            if re.search(r'(public|private|static)\s+\w+\s+\w+\s*\(', content):
                score += 0.3
                details["has_methods"] = True

            # Check for using statements
            if re.search(r'using\s+[\w.]+;', content):
                score += 0.3
                details["has_usings"] = True

            details["compilable"] = score >= 0.7
            return score, details

        # Unknown language - basic check
        return 0.5, {"compilable": None, "note": "Unknown target language"}

    def _check_functional_equivalence(self, content: str, target_language: str,
                                      task) -> Tuple[float, Dict]:
        """Check functional equivalence (without actual execution)"""
        details = {"checks": []}
        score = 0.0
        max_checks = 5

        # Check 1: Variable/data structure preservation
        # Look for data structure definitions
        if target_language == "python":
            data_patterns = [r'class\s+\w+', r'def\s+\w+', r'\w+\s*=\s*\{', r'\w+\s*=\s*\[']
        elif target_language == "java":
            data_patterns = [r'private\s+\w+', r'public\s+\w+', r'int\s+\w+', r'String\s+\w+']
        else:
            data_patterns = [r'private\s+\w+', r'public\s+\w+']

        data_structures_found = sum(1 for p in data_patterns if re.search(p, content))
        if data_structures_found >= 2:
            score += 1.0
            details["checks"].append("Data structures defined")

        # Check 2: Control flow preservation
        control_patterns = [r'\bif\b', r'\bfor\b', r'\bwhile\b', r'\bswitch\b', r'\btry\b']
        control_found = sum(1 for p in control_patterns if re.search(p, content))
        if control_found >= 2:
            score += 1.0
            details["checks"].append("Control flow structures present")

        # Check 3: File I/O operations (common in COBOL)
        io_patterns = [r'open\s*\(', r'read\s*\(', r'write\s*\(', r'close\s*\(',
                      r'FileReader', r'FileWriter', r'BufferedReader', r'StreamReader']
        io_found = sum(1 for p in io_patterns if re.search(p, content, re.IGNORECASE))
        if io_found >= 1:
            score += 1.0
            details["checks"].append("File I/O operations present")
        else:
            # Not all programs have file I/O, give partial credit
            score += 0.5

        # Check 4: Arithmetic/computation operations
        arith_patterns = [r'[\+\-\*\/]', r'\bMath\.', r'\bcalculate', r'\bcompute']
        arith_found = sum(1 for p in arith_patterns if re.search(p, content, re.IGNORECASE))
        if arith_found >= 1:
            score += 1.0
            details["checks"].append("Arithmetic operations present")

        # Check 5: Error handling
        error_patterns = [r'\btry\b', r'\bcatch\b', r'\bexcept\b', r'\braise\b', r'\bthrow\b']
        error_found = sum(1 for p in error_patterns if re.search(p, content))
        if error_found >= 1:
            score += 1.0
            details["checks"].append("Error handling present")

        return score / max_checks, details

    def _check_code_quality(self, content: str, target_language: str) -> Tuple[float, Dict]:
        """Check code quality metrics"""
        details = {"issues": [], "good_practices": []}
        score = 1.0  # Start with full score, deduct for issues

        # Check 1: No hardcoded magic numbers (except 0, 1)
        magic_numbers = re.findall(r'(?<!["\'])\b\d{2,}\b(?!["\'])', content)
        if len(magic_numbers) > 10:
            score -= 0.2
            details["issues"].append(f"Many magic numbers: {len(magic_numbers)}")
        else:
            details["good_practices"].append("Limited magic numbers")

        # Check 2: Proper naming conventions
        if target_language == "python":
            # Python: snake_case for functions/variables
            good_names = len(re.findall(r'\bdef\s+[a-z][a-z_0-9]*\s*\(', content))
            bad_names = len(re.findall(r'\bdef\s+[A-Z][a-zA-Z0-9]*\s*\(', content))
        elif target_language in ["java", "csharp"]:
            # Java/C#: camelCase for methods, PascalCase for classes
            good_names = len(re.findall(r'\bclass\s+[A-Z][a-zA-Z0-9]*', content))
            bad_names = len(re.findall(r'\bclass\s+[a-z][a-zA-Z0-9]*', content))
        else:
            good_names, bad_names = 1, 0

        if bad_names > good_names:
            score -= 0.15
            details["issues"].append("Naming convention issues")
        else:
            details["good_practices"].append("Good naming conventions")

        # Check 3: Code organization (functions/methods exist)
        if target_language == "python":
            func_count = len(re.findall(r'\bdef\s+\w+', content))
        else:
            func_count = len(re.findall(r'(public|private|protected)\s+\w+\s+\w+\s*\(', content))

        if func_count < 2:
            score -= 0.15
            details["issues"].append("Code not well-organized into functions")
        else:
            details["good_practices"].append(f"Well-organized ({func_count} methods)")

        # Check 4: Comments present
        comment_patterns = [r'//.*', r'/\*.*?\*/', r'#.*', r'""".*?"""', r"'''.*?'''"]
        comments_found = sum(len(re.findall(p, content, re.DOTALL)) for p in comment_patterns)
        if comments_found < 3:
            score -= 0.1
            details["issues"].append("Few comments")
        else:
            details["good_practices"].append(f"Good documentation ({comments_found} comments)")

        # Check 5: Reasonable line length
        long_lines = sum(1 for line in content.split('\n') if len(line) > 120)
        if long_lines > 10:
            score -= 0.1
            details["issues"].append(f"{long_lines} lines exceed 120 chars")

        return max(0.0, score), details

    def _check_completeness(self, content: str, target_language: str,
                           task) -> Tuple[float, Dict]:
        """Check if conversion is complete"""
        details = {"completeness_checks": []}
        score = 0.0
        total_checks = 4

        # Check 1: Minimum code length (converted code should be substantial)
        lines = [l for l in content.split('\n') if l.strip() and not l.strip().startswith(('#', '//', '/*'))]
        if len(lines) >= 50:
            score += 1.0
            details["completeness_checks"].append(f"Substantial code ({len(lines)} lines)")
        elif len(lines) >= 20:
            score += 0.5
            details["completeness_checks"].append(f"Partial code ({len(lines)} lines)")
        else:
            details["completeness_checks"].append(f"Minimal code ({len(lines)} lines)")

        # Check 2: Main entry point exists
        if target_language == "python":
            if re.search(r'if\s+__name__\s*==\s*["\']__main__["\']', content) or \
               re.search(r'\bdef\s+main\s*\(', content):
                score += 1.0
                details["completeness_checks"].append("Has main entry point")
        elif target_language == "java":
            if re.search(r'public\s+static\s+void\s+main', content):
                score += 1.0
                details["completeness_checks"].append("Has main method")
        elif target_language == "csharp":
            if re.search(r'static\s+void\s+Main', content):
                score += 1.0
                details["completeness_checks"].append("Has Main method")
        else:
            score += 0.5

        # Check 3: No TODO/FIXME placeholders
        placeholders = len(re.findall(r'\b(TODO|FIXME|XXX|HACK)\b', content, re.IGNORECASE))
        if placeholders == 0:
            score += 1.0
            details["completeness_checks"].append("No TODO placeholders")
        elif placeholders <= 2:
            score += 0.5
            details["completeness_checks"].append(f"Few TODOs ({placeholders})")
        else:
            details["completeness_checks"].append(f"Many TODOs ({placeholders})")

        # Check 4: No obvious stub methods
        stub_patterns = [r'pass\s*$', r'throw\s+new\s+NotImplementedException',
                        r'return\s+null\s*;?\s*$', r'return\s+0\s*;?\s*$']
        stubs = sum(len(re.findall(p, content, re.MULTILINE)) for p in stub_patterns)
        if stubs <= 1:
            score += 1.0
            details["completeness_checks"].append("No stub methods")
        elif stubs <= 3:
            score += 0.5
            details["completeness_checks"].append(f"Few stubs ({stubs})")
        else:
            details["completeness_checks"].append(f"Many stubs ({stubs})")

        return score / total_checks, details


def evaluate_conversion(submission_path: Path, task) -> Dict:
    """Convenience function to evaluate a conversion submission"""
    evaluator = ConversionEvaluator()
    return evaluator.evaluate(submission_path, task)
