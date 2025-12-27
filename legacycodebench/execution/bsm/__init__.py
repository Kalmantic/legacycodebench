# BSM - Behavioral Specification Matching
"""
Components for validating documentation accuracy against external call patterns.
"""

from .call_detector import CallDetector, ExternalCall
from .pattern_library import BSM_PATTERNS, get_pattern, BSMPattern, ChecklistItem
from .doc_matcher import DocumentMatcher, MatchResult

__all__ = [
    'CallDetector',
    'ExternalCall',
    'BSM_PATTERNS',
    'get_pattern',
    'BSMPattern',
    'ChecklistItem',
    'DocumentMatcher',
    'MatchResult',
]
