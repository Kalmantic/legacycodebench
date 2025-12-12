"""Ground Truth Generator

Orchestrates all static analysis components to generate automated ground truth.

Implements Section 2: Automated Ground Truth Generation
- 95-100% automation for ground truth extraction
- Confidence scoring per Section 2.6
- Handles REDEFINES, external calls, business rules
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

from .cobol_parser import COBOLParser
from .data_structure_extractor import DataStructureExtractor
from .control_flow_analyzer import ControlFlowAnalyzer
from .business_rule_detector import BusinessRuleDetector
from .dependency_analyzer import DependencyAnalyzer

logger = logging.getLogger(__name__)


class GroundTruthGenerator:
    """
    Generate automated ground truth from COBOL source code.

    This is the core of v2.0's automation strategy:
    - No manual expert annotation required
    - 95%+ automation level
    - Confidence scoring for each element
    - Multi-view extraction for REDEFINES
    """

    def __init__(self):
        self.parser = COBOLParser()
        self.data_extractor = DataStructureExtractor()
        self.control_flow = ControlFlowAnalyzer()
        self.business_rules = BusinessRuleDetector()
        self.dependencies = DependencyAnalyzer()

    def generate(self, source_files: List[Path],
                cache_dir: Optional[Path] = None) -> Dict:
        """
        Generate complete ground truth from COBOL source files.

        Args:
            source_files: List of COBOL source file paths (main + copybooks)
            cache_dir: Optional directory to cache ground truth

        Returns:
            Complete ground truth dictionary with all extracted elements
        """
        logger.info(f"Generating ground truth for {len(source_files)} file(s)")

        # Parse main file (first in list)
        main_file = source_files[0]
        logger.info(f"Parsing main file: {main_file}")

        parsed = self.parser.parse_file(main_file)

        # Extract all components
        logger.info("Extracting data structures...")
        data_div = self.parser.get_data_division_content(parsed)
        data_structures = self.data_extractor.extract(
            data_div,
            line_offset=parsed.divisions.get('DATA', None).start_line if 'DATA' in parsed.divisions else 0
        )

        logger.info("Analyzing control flow...")
        control_flow = self.control_flow.analyze(parsed)

        logger.info("Detecting business rules...")
        business_rules = self.business_rules.detect(parsed)

        logger.info("Detecting error handlers...")
        error_handlers = self.business_rules.detect_error_handlers(parsed)

        logger.info("Analyzing dependencies...")
        deps = self.dependencies.analyze(parsed, main_file)

        # Process copybooks if provided
        copybook_data = []
        if len(source_files) > 1:
            logger.info(f"Processing {len(source_files) - 1} copybook(s)...")
            for copybook_file in source_files[1:]:
                copybook_gt = self._process_copybook(copybook_file)
                copybook_data.append(copybook_gt)

        # Calculate confidence scores
        confidence = self._calculate_confidence_scores(
            data_structures,
            control_flow,
            business_rules,
            error_handlers,
            deps
        )

        # Build complete ground truth
        ground_truth = {
            "metadata": {
                "source_file": str(main_file),
                "total_lines": parsed.total_lines,
                "divisions": list(parsed.divisions.keys()),
                "generation_method": "automated_static_analysis",
                "automation_level": confidence["overall_automation"],
                "confidence_score": confidence["overall_confidence"]
            },
            "data_structures": data_structures,
            "control_flow": control_flow,
            "business_rules": business_rules,
            "error_handlers": error_handlers,
            "dependencies": deps,
            "copybooks": copybook_data,
            "confidence_breakdown": confidence,
            "element_count": {
                "data_structures": data_structures["total_structures"],
                "fields": data_structures["total_fields"],
                "paragraphs": control_flow["total_paragraphs"],
                "business_rules": business_rules["total_rules"],
                "error_handlers": len(error_handlers),
                "external_calls": len(deps["calls"]),
                "copybooks": len(deps["copybooks"]),
                "files": deps["files"]["total_files"]
            }
        }

        # Cache if requested
        if cache_dir:
            self._cache_ground_truth(ground_truth, main_file, cache_dir)

        logger.info(f"Ground truth generated successfully")
        logger.info(f"  Confidence: {confidence['overall_confidence']:.2%}")
        logger.info(f"  Automation: {confidence['overall_automation']:.2%}")
        logger.info(f"  Total elements: {sum(ground_truth['element_count'].values())}")

        return ground_truth

    def _process_copybook(self, copybook_file: Path) -> Dict:
        """Process a copybook file"""
        logger.info(f"  Processing copybook: {copybook_file.name}")

        parsed = self.parser.parse_file(copybook_file)

        # Extract data structures (main purpose of copybooks)
        data_div = self.parser.get_data_division_content(parsed)
        data_structures = self.data_extractor.extract(
            data_div,
            line_offset=parsed.divisions.get('DATA', None).start_line if 'DATA' in parsed.divisions else 0
        )

        return {
            "file": str(copybook_file),
            "name": copybook_file.stem,
            "data_structures": data_structures,
            "total_fields": data_structures["total_fields"]
        }

    def _calculate_confidence_scores(self, data_structures: Dict,
                                     control_flow: Dict,
                                     business_rules: Dict,
                                     error_handlers: List[Dict],
                                     deps: Dict) -> Dict:
        """
        Calculate confidence scores per Section 2.6 of spec.

        Confidence Levels:
        - Definitive (95-100%): Directly parsed from syntax
        - High (80-94%): Pattern-matched with clear indicators
        - Medium (60-79%): Inferred from context, multiple interpretations
        - Low (<60%): Ambiguous patterns, requires semantic understanding
        """

        # Data structures: Definitive (100%)
        data_confidence = 1.0

        # Control flow: Definitive (100%)
        control_confidence = 1.0

        # Business rules: High to Medium (depends on pattern)
        # Average confidence from detected rules
        if business_rules["total_rules"] > 0:
            business_confidence = business_rules["avg_confidence"]
        else:
            business_confidence = 1.0  # No rules = no uncertainty

        # Error handlers: Definitive (100%)
        error_confidence = 1.0

        # Dependencies: Mostly definitive, except dynamic calls
        dynamic_calls = sum(1 for c in deps["calls"] if c.get("is_dynamic", False))
        total_calls = len(deps["calls"])
        if total_calls > 0:
            deps_confidence = 1.0 - (dynamic_calls / total_calls * 0.2)  # 20% penalty for dynamic
        else:
            deps_confidence = 1.0

        # Calculate weighted overall confidence
        weights = {
            "data_structures": 0.25,
            "control_flow": 0.25,
            "business_rules": 0.30,
            "error_handlers": 0.10,
            "dependencies": 0.10
        }

        overall_confidence = (
            weights["data_structures"] * data_confidence +
            weights["control_flow"] * control_confidence +
            weights["business_rules"] * business_confidence +
            weights["error_handlers"] * error_confidence +
            weights["dependencies"] * deps_confidence
        )

        # Automation level (how much was automated vs needs review)
        # Elements with confidence >= 0.95 are fully automated
        definitive_count = sum([
            data_structures["total_structures"],
            control_flow["total_paragraphs"],
            len(error_handlers)
        ])

        # Elements with confidence 0.80-0.95 need automated cross-check
        high_count = business_rules["high_confidence_rules"]

        # Elements with confidence 0.60-0.80 need LLM verification
        medium_count = business_rules["medium_confidence_rules"]

        # Elements with confidence < 0.60 need human review
        low_count = 0  # Currently none, but could add for complex patterns

        total_elements = definitive_count + high_count + medium_count + low_count

        if total_elements > 0:
            automation_level = (definitive_count + high_count * 0.9 + medium_count * 0.7) / total_elements
        else:
            automation_level = 1.0

        return {
            "overall_confidence": overall_confidence,
            "overall_automation": automation_level,
            "by_category": {
                "data_structures": {
                    "confidence": data_confidence,
                    "level": "definitive"
                },
                "control_flow": {
                    "confidence": control_confidence,
                    "level": "definitive"
                },
                "business_rules": {
                    "confidence": business_confidence,
                    "level": self._get_confidence_level(business_confidence)
                },
                "error_handlers": {
                    "confidence": error_confidence,
                    "level": "definitive"
                },
                "dependencies": {
                    "confidence": deps_confidence,
                    "level": self._get_confidence_level(deps_confidence)
                }
            },
            "element_counts_by_confidence": {
                "definitive": definitive_count,
                "high": high_count,
                "medium": medium_count,
                "low": low_count
            },
            "requires_human_review": low_count > 0,
            "review_percentage": (medium_count + low_count) / total_elements if total_elements > 0 else 0
        }

    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level label"""
        if confidence >= 0.95:
            return "definitive"
        elif confidence >= 0.80:
            return "high"
        elif confidence >= 0.60:
            return "medium"
        else:
            return "low"

    def _cache_ground_truth(self, ground_truth: Dict, source_file: Path,
                           cache_dir: Path):
        """Cache ground truth to file"""
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Use source file name as cache key
        cache_file = cache_dir / f"{source_file.stem}_ground_truth.json"

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(ground_truth, f, indent=2)

        logger.info(f"Ground truth cached to: {cache_file}")

    def load_cached_ground_truth(self, source_file: Path,
                                cache_dir: Path) -> Optional[Dict]:
        """Load cached ground truth if available"""
        cache_file = cache_dir / f"{source_file.stem}_ground_truth.json"

        if cache_file.exists():
            logger.info(f"Loading cached ground truth from: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        return None

    def generate_summary_report(self, ground_truth: Dict) -> str:
        """Generate human-readable summary of ground truth"""
        report_lines = [
            "=" * 70,
            "GROUND TRUTH SUMMARY",
            "=" * 70,
            "",
            f"Source File: {ground_truth['metadata']['source_file']}",
            f"Total Lines: {ground_truth['metadata']['total_lines']}",
            f"Confidence Score: {ground_truth['metadata']['confidence_score']:.2%}",
            f"Automation Level: {ground_truth['metadata']['automation_level']:.2%}",
            "",
            "ELEMENT COUNTS:",
            f"  Data Structures: {ground_truth['element_count']['data_structures']}",
            f"  Fields: {ground_truth['element_count']['fields']}",
            f"  Paragraphs: {ground_truth['element_count']['paragraphs']}",
            f"  Business Rules: {ground_truth['element_count']['business_rules']}",
            f"  Error Handlers: {ground_truth['element_count']['error_handlers']}",
            f"  External Calls: {ground_truth['element_count']['external_calls']}",
            f"  Copybooks: {ground_truth['element_count']['copybooks']}",
            f"  Files: {ground_truth['element_count']['files']}",
            "",
            "CONFIDENCE BREAKDOWN:",
        ]

        for category, info in ground_truth['confidence_breakdown']['by_category'].items():
            report_lines.append(
                f"  {category:20s}: {info['confidence']:5.1%} ({info['level']})"
            )

        report_lines.extend([
            "",
            "AUTOMATION STATUS:",
            f"  Definitive elements: {ground_truth['confidence_breakdown']['element_counts_by_confidence']['definitive']}",
            f"  High confidence: {ground_truth['confidence_breakdown']['element_counts_by_confidence']['high']}",
            f"  Medium confidence: {ground_truth['confidence_breakdown']['element_counts_by_confidence']['medium']}",
            f"  Low confidence: {ground_truth['confidence_breakdown']['element_counts_by_confidence']['low']}",
            f"  Requires human review: {ground_truth['confidence_breakdown']['requires_human_review']}",
            "",
            "=" * 70
        ])

        return "\n".join(report_lines)
