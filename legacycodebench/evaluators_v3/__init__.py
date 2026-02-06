"""
Behavioral Fidelity Evaluator V3.1

Key Changes from V3.0:
1. Code-based static verification (replaces TF-IDF text matching)
2. Claims verified against actual source code structure
3. Pattern matchers for COMPUTE, IF, MOVE, CALL statements

Key Changes from V2.3.1:
1. Compile-first classification (no pattern guessing)
2. Full provenance tracking in every result
3. No silent fallbacks to heuristic scoring
4. Static verification that can actually FAIL bad documentation

Design: docs/v2.3.2/BF_V3_DESIGN.md
"""

from .bf_result import BFResult, VerificationMode
from .behavioral_v3 import BehavioralEvaluatorV3
from .static_verifier import StaticVerifier
from .copybook_resolver import CopybookResolver
from .evaluator_v3 import EvaluatorV3
from .code_verifier import CodeBasedVerifier  # V3.1
from .code_matchers import (  # V3.1
    ComputationMatcher,
    ConditionalMatcher,
    CallPerformMatcher,
    VariableFinder,
)

__all__ = [
    'BFResult',
    'VerificationMode', 
    'BehavioralEvaluatorV3',
    'StaticVerifier',
    'CopybookResolver',
    'EvaluatorV3',
    'CodeBasedVerifier',
    'ComputationMatcher',
    'ConditionalMatcher',
    'CallPerformMatcher',
    'VariableFinder',
]
