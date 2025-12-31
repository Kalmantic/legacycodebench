"""
LegacyCodeBench V2.3.1 Evaluation System

This module implements the V2.3.1 evaluation architecture with 4 key innovations:
1. Silence Penalty - Claims < 1 (min_claims=1) → BF score = 0
2. Zero Tolerance I/O - Hyphen-pattern hallucinations → CF-02
3. Deterministic Synonyms - Frozen COBOL synonym dictionary
4. Boundary Value Testing - Min/Mid/Max tests per calculation claim

Scoring: LCB = 0.30×SC + 0.20×DQ + 0.50×BF
Thresholds: SC≥60%, DQ≥50%, BF≥55%
"""

from .config_v231 import V231_CONFIG
from .synonyms import COBOL_SYNONYMS, expand_synonyms
from .structural_v231 import StructuralEvaluatorV231
from .documentation_v231 import DocumentationEvaluatorV231
from .claim_extractor import ClaimExtractor, Claim, ClaimType
from .test_generator import TestGenerator, TestCase
from .behavioral_v231 import BehavioralEvaluatorV231
from .critical_failures_v231 import CriticalFailureDetectorV231, CriticalFailure
from .scoring_v231 import ScoringEngineV231, EvaluationResultV231
from .evaluator_v231 import EvaluatorV231, evaluate_v231

__version__ = "2.3.1"

__all__ = [
    # Config
    "V231_CONFIG",
    
    # Synonyms
    "COBOL_SYNONYMS",
    "expand_synonyms",
    
    # Evaluators
    "StructuralEvaluatorV231",
    "DocumentationEvaluatorV231",
    "BehavioralEvaluatorV231",
    
    # Claims & Tests
    "ClaimExtractor",
    "Claim",
    "ClaimType",
    "TestGenerator",
    "TestCase",
    
    # Critical Failures
    "CriticalFailureDetectorV231",
    "CriticalFailure",
    
    # Scoring
    "ScoringEngineV231",
    "EvaluationResultV231",
    
    # Main Interface
    "EvaluatorV231",
    "evaluate_v231",
]
