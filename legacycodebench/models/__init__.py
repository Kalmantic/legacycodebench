"""
LegacyCodeBench V2.4 Data Models

Centralized data models as specified in TDD_V2.4.md Section 2.
"""

from .enums import (
    Language,
    VerificationMode,
    CompileFailureReason,
    RulePriority,
    ClaimType,
)
from .task import Task
from .ground_truth import GroundTruth, BusinessRule, DataStructure, ExternalCall
from .results import EvaluationResult, BFResult, TrackScore, CriticalFailure

__all__ = [
    # Enums
    "Language",
    "VerificationMode",
    "CompileFailureReason",
    "RulePriority",
    "ClaimType",
    # Task
    "Task",
    # Ground Truth
    "GroundTruth",
    "BusinessRule",
    "DataStructure",
    "ExternalCall",
    # Results
    "EvaluationResult",
    "BFResult",
    "TrackScore",
    "CriticalFailure",
]
