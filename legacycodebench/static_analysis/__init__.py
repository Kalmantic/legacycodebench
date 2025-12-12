"""Static Analysis Module for Automated Ground Truth Generation

This module implements Section 2 of LegacyCodeBench v2.0:
- Automated ground truth extraction from COBOL source code
- 95-100% automation level for most elements
- Confidence scoring for extracted elements
"""

from .ground_truth_generator import GroundTruthGenerator
from .cobol_parser import COBOLParser
from .data_structure_extractor import DataStructureExtractor
from .control_flow_analyzer import ControlFlowAnalyzer
from .business_rule_detector import BusinessRuleDetector
from .dependency_analyzer import DependencyAnalyzer

__all__ = [
    "GroundTruthGenerator",
    "COBOLParser",
    "DataStructureExtractor",
    "ControlFlowAnalyzer",
    "BusinessRuleDetector",
    "DependencyAnalyzer",
]
