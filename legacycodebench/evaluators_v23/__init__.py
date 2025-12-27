# V2.3 Evaluators Package
"""
LegacyCodeBench V2.3 Evaluators

This package implements the hybrid Template + BSM evaluation architecture
with three-way paragraph classification (PURE/MIXED/INFRASTRUCTURE) and
comprehensive anti-gaming mechanisms.

Key differences from V2:
- 3-way paragraph classification (PURE/MIXED/INFRASTRUCTURE)
- MIXED paragraph handling (logic extraction + BSM)
- Anti-gaming mechanisms (5 types)
- 6 critical failures (CF-01 through CF-06)
- Scoring: 40% Comprehension, 25% Documentation, 35% Behavioral
"""

from .config_v23 import V23_CONFIG
from .paragraph_classifier import ParagraphClassifier, ParagraphType, ClassifiedParagraph
from .anti_gaming import AntiGamingAnalyzer, AntiGamingResult
from .comprehension_v23 import ComprehensionEvaluatorV23, ComprehensionResult
from .documentation_v23 import DocumentationEvaluatorV23, DocumentationResult
from .behavioral_v23 import BehavioralEvaluatorV23, BehavioralResult
from .critical_failures_v23 import CriticalFailureDetectorV23, CriticalFailureResult
from .scoring_v23 import ScoringEngineV23, LCBScoreV23
from .evaluator_v23 import EvaluatorV23, EvaluationResultV23, evaluate_v23

__all__ = [
    # Config
    'V23_CONFIG',
    
    # Classification
    'ParagraphClassifier',
    'ParagraphType',
    'ClassifiedParagraph',
    
    # Anti-Gaming
    'AntiGamingAnalyzer',
    'AntiGamingResult',
    
    # Evaluators
    'ComprehensionEvaluatorV23',
    'ComprehensionResult',
    'DocumentationEvaluatorV23',
    'DocumentationResult',
    'BehavioralEvaluatorV23',
    'BehavioralResult',
    
    # Critical Failures
    'CriticalFailureDetectorV23',
    'CriticalFailureResult',
    
    # Scoring
    'ScoringEngineV23',
    'LCBScoreV23',
    
    # Main Evaluator
    'EvaluatorV23',
    'EvaluationResultV23',
    'evaluate_v23',
]
