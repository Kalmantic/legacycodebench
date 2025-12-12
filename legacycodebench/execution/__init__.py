"""
Execution-based evaluation for Behavioral Fidelity

This module implements execution-based validation for v2.0 evaluation.
It generates test cases, executes COBOL programs, generates code from
documentation, and compares behavior.

Components:
- TestGenerator: Generate test cases from ground truth
- COBOLExecutor: Execute COBOL programs in Docker sandbox
- ConstrainedCodeGenerator: Generate code from documentation (strict mode)
- BehaviorComparator: Compare outputs and detect false positives

Weight in v2.0 evaluation: 35% (Behavioral Fidelity)
"""

from legacycodebench.execution.test_generator import TestGenerator, TestCase
from legacycodebench.execution.cobol_executor import COBOLExecutor, ExecutionResult
from legacycodebench.execution.code_generator import ConstrainedCodeGenerator, GeneratedCode
from legacycodebench.execution.behavior_comparator import BehaviorComparator, ComparisonResult

__all__ = [
    "TestGenerator",
    "TestCase",
    "COBOLExecutor",
    "ExecutionResult",
    "ConstrainedCodeGenerator",
    "GeneratedCode",
    "BehaviorComparator",
    "ComparisonResult",
]
