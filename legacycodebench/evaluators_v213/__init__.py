"""
LegacyCodeBench V2.3.1 Evaluation System

This module implements the V2.3.1 evaluation architecture with 4 key innovations:
1. Silence Penalty - Claims < 3 → BF score = 0
2. Zero Tolerance I/O - Any hallucinated I/O → CF-02
3. Deterministic Synonyms - Frozen COBOL synonym dictionary
4. Boundary Value Testing - Min/Mid/Max tests per calculation claim

Scoring: LCB = 0.30×SC + 0.20×DQ + 0.50×BF
Thresholds: SC≥60%, DQ≥50%, BF≥55%
"""

from .config_v213 import V213_CONFIG
from .synonyms import COBOL_SYNONYMS, expand_synonyms
from .structural_v213 import StructuralEvaluatorV213
from .documentation_v213 import DocumentationEvaluatorV213
from .claim_extractor import ClaimExtractor, Claim, ClaimType
from .test_generator import TestGenerator, TestCase
from .behavioral_v213 import BehavioralEvaluatorV213
from .critical_failures_v213 import CriticalFailureDetectorV213, CriticalFailure
from .scoring_v213 import ScoringEngineV213, EvaluationResultV213
from .evaluator_v213 import EvaluatorV213, evaluate_v213

__version__ = "2.3.1"

__all__ = [
    # Config
    "V213_CONFIG",
    
    # Synonyms
    "COBOL_SYNONYMS",
    "expand_synonyms",
    
    # Evaluators
    "StructuralEvaluatorV213",
    "DocumentationEvaluatorV213",
    "BehavioralEvaluatorV213",
    
    # Claims & Tests
    "ClaimExtractor",
    "Claim",
    "ClaimType",
    "TestGenerator",
    "TestCase",
    
    # Critical Failures
    "CriticalFailureDetectorV213",
    "CriticalFailure",
    
    # Scoring
    "ScoringEngineV213",
    "EvaluationResultV213",
    
    # Main Interface
    "EvaluatorV213",
    "evaluate_v213",
]
