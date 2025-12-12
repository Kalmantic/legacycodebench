"""v2.0 Evaluators for LegacyCodeBench

Implements the 4-pillar evaluation framework from Section 5 of spec:
1. Structural Completeness (30%) - Element coverage
2. Behavioral Fidelity (35%) - Execution-based validation
3. Semantic Quality (25%) - LLM-as-judge
4. Traceability (10%) - Reference validation

This replaces v1.0's ROUGE/BLEU reference-based evaluation with
validation against source code ground truth.
"""

from .structural_completeness import StructuralCompletenessEvaluator
from .semantic_quality import SemanticQualityEvaluator
from .traceability import TraceabilityEvaluator
from .documentation_v2 import DocumentationEvaluatorV2

__all__ = [
    "StructuralCompletenessEvaluator",
    "SemanticQualityEvaluator",
    "TraceabilityEvaluator",
    "DocumentationEvaluatorV2",
]
