"""
UniBasic Ground Truth Generator

Generates ground truth for UniBasic programs using the parser.

Specification Reference: TDD_V2.4.md Section 5
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json

from legacycodebench.models.enums import Language, RulePriority
from legacycodebench.models.ground_truth import (
    GroundTruth,
    BusinessRule,
    DataStructure,
    Paragraph,
    ExternalCall,
    ErrorHandler,
)
from .unibasic_parser import UniBasicParser, ParsedUniBasic

logger = logging.getLogger(__name__)


class UniBasicGroundTruthGenerator:
    """
    Generates ground truth for UniBasic programs.
    
    Uses UniBasicParser to extract:
    - Subroutines → Paragraphs
    - DIM statements → DataStructures
    - CALL/EXECUTE → ExternalCalls
    - ON ERROR/LOCKED → ErrorHandlers
    - Control flow → BusinessRules
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the generator.
        
        Args:
            cache_dir: Directory for caching generated ground truth
        """
        self.parser = UniBasicParser()
        self.cache_dir = cache_dir

    def generate(
        self,
        source_files: List[Path],
        task_id: str = "UNKNOWN",
    ) -> GroundTruth:
        """
        Generate ground truth for UniBasic source files.
        
        Args:
            source_files: List of source file paths
            task_id: Task identifier
            
        Returns:
            GroundTruth object
        """
        if not source_files:
            return self._empty_ground_truth(task_id)
        
        # Parse all source files
        parsed: List[ParsedUniBasic] = []
        all_source = ""
        
        for file_path in source_files:
            if file_path.exists():
                p = self.parser.parse_file(file_path)
                parsed.append(p)
                all_source += "\n".join(p.raw_lines)
        
        if not parsed:
            return self._empty_ground_truth(task_id)
        
        # Compute source hash
        source_hash = hashlib.sha256(all_source.encode()).hexdigest()
        
        # Check cache
        if self.cache_dir:
            cached = self._load_cache(task_id, source_hash)
            if cached:
                logger.info(f"Using cached ground truth for {task_id}")
                return cached
        
        # Generate components
        paragraphs = self._extract_paragraphs(parsed)
        data_structures = self._extract_data_structures(parsed)
        external_calls = self._extract_external_calls(parsed)
        error_handlers = self._extract_error_handlers(all_source)
        business_rules = self._extract_business_rules(parsed, all_source)
        
        # Calculate metrics
        loc = sum(p.total_lines for p in parsed)
        complexity = len(paragraphs) + len(business_rules)
        
        gt = GroundTruth(
            task_id=task_id,
            source_hash=source_hash,
            language=Language.UNIBASIC,
            business_rules=business_rules,
            data_structures=data_structures,
            paragraphs=paragraphs,
            external_calls=external_calls,
            error_handlers=error_handlers,
            loc=loc,
            cyclomatic_complexity=complexity,
        )
        
        # Cache result
        if self.cache_dir:
            self._save_cache(task_id, source_hash, gt)
        
        return gt

    def _extract_paragraphs(self, parsed_list: List[ParsedUniBasic]) -> List[Paragraph]:
        """Extract paragraphs from parsed UniBasic."""
        paragraphs = []
        
        for parsed in parsed_list:
            for sub in parsed.subroutines:
                para_type = self._classify_subroutine(sub.content)
                
                paragraphs.append(Paragraph(
                    name=sub.name,
                    paragraph_type=para_type,
                    start_line=sub.start_line,
                    end_line=sub.end_line,
                    content=sub.content,
                ))
        
        return paragraphs

    def _classify_subroutine(self, content: str) -> str:
        """Classify subroutine as PURE, MIXED, or INFRASTRUCTURE."""
        content_upper = content.upper()
        
        # Check for external calls
        if 'CALL ' in content_upper or 'EXECUTE ' in content_upper:
            return "MIXED"
        
        # Check for file I/O only
        io_patterns = ['OPEN ', 'CLOSE ', 'READ ', 'WRITE ', 'MATREAD', 'MATWRITE']
        has_io = any(p in content_upper for p in io_patterns)
        
        logic_patterns = ['IF ', 'FOR ', 'LOOP', 'BEGIN CASE', 'GOSUB ']
        has_logic = any(p in content_upper for p in logic_patterns)
        
        if has_io and not has_logic:
            return "INFRASTRUCTURE"
        
        return "PURE"

    def _extract_data_structures(self, parsed_list: List[ParsedUniBasic]) -> List[DataStructure]:
        """Extract data structures from parsed UniBasic."""
        structures = []
        
        for parsed in parsed_list:
            for var in parsed.variables:
                structures.append(DataStructure(
                    name=var.name,
                    level="DIM" if var.var_type == "DIM" else "SCALAR",
                    line_number=var.line_number,
                    pic_clause=f"({var.dimensions})" if var.dimensions else None,
                    children=[],
                ))
        
        return structures

    def _extract_external_calls(self, parsed_list: List[ParsedUniBasic]) -> List[ExternalCall]:
        """Extract external calls from parsed UniBasic."""
        calls = []
        
        for parsed in parsed_list:
            for call in parsed.external_calls:
                calls.append(ExternalCall(
                    call_type=call.call_type,
                    target=call.target,
                    operation=call.call_type,
                    line_number=call.line_number,
                ))
        
        return calls

    def _extract_error_handlers(self, source: str) -> List[ErrorHandler]:
        """Extract error handlers from source."""
        handlers_raw = self.parser.get_error_handlers(source)
        
        handlers = []
        for h in handlers_raw:
            handlers.append(ErrorHandler(
                handler_type=h["handler_type"],
                line_number=h["line_number"],
                paragraph=h.get("target", ""),
                action="",
            ))
        
        return handlers

    def _extract_business_rules(
        self,
        parsed_list: List[ParsedUniBasic],
        source: str
    ) -> List[BusinessRule]:
        """
        Extract business rules from UniBasic source.
        
        Rules are inferred from:
        - IF statements with business logic
        - Calculations and assignments
        - GOSUB calls (indicate logic flow)
        """
        rules = []
        rule_id = 0
        
        lines = source.split('\n')
        
        for i, line in enumerate(lines):
            line_strip = line.strip()
            line_upper = line_strip.upper()
            
            # Skip comments and empty lines
            if not line_strip or line_strip.startswith('*') or line_strip.startswith('!'):
                continue
            
            # IF statements
            if line_upper.startswith('IF ') and ' THEN' in line_upper:
                rule_id += 1
                rules.append(BusinessRule(
                    rule_id=f"BR-{rule_id:03d}",
                    description=f"Conditional logic: {line_strip[:60]}...",
                    priority=RulePriority.IMPORTANT,
                    line_number=i + 1,
                    paragraph=self._find_containing_paragraph(i, parsed_list),
                    keywords=self._extract_keywords(line_strip),
                    source_excerpt=line_strip[:100],
                ))
            
            # BEGIN CASE
            elif 'BEGIN CASE' in line_upper:
                rule_id += 1
                rules.append(BusinessRule(
                    rule_id=f"BR-{rule_id:03d}",
                    description="Case/switch logic block",
                    priority=RulePriority.IMPORTANT,
                    line_number=i + 1,
                    paragraph=self._find_containing_paragraph(i, parsed_list),
                    keywords=["case", "switch", "conditional"],
                    source_excerpt=line_strip[:100],
                ))
            
            # Calculations (= with expression)
            elif '=' in line_strip and not line_upper.startswith('IF '):
                parts = line_strip.split('=')
                if len(parts) == 2:
                    right = parts[1].strip()
                    # Check if it's a calculation (contains operators or function calls)
                    if any(op in right for op in ['+', '-', '*', '/', '(']) and not right.startswith('"'):
                        rule_id += 1
                        rules.append(BusinessRule(
                            rule_id=f"BR-{rule_id:03d}",
                            description=f"Calculation: {parts[0].strip()}",
                            priority=RulePriority.CRITICAL if 'TOTAL' in line_upper or 'SUM' in line_upper else RulePriority.IMPORTANT,
                            line_number=i + 1,
                            paragraph=self._find_containing_paragraph(i, parsed_list),
                            keywords=self._extract_keywords(line_strip),
                            source_excerpt=line_strip[:100],
                        ))
        
        return rules

    def _find_containing_paragraph(
        self,
        line_index: int,
        parsed_list: List[ParsedUniBasic]
    ) -> str:
        """Find the paragraph containing a line."""
        for parsed in parsed_list:
            for sub in parsed.subroutines:
                if sub.start_line <= line_index + 1 <= sub.end_line:
                    return sub.name
        return ""

    def _extract_keywords(self, line: str) -> List[str]:
        """Extract keywords from a line for TF-IDF matching."""
        import re
        # Extract variable-like tokens
        tokens = re.findall(r'\b[A-Za-z][A-Za-z0-9_.]*\b', line)
        # Filter out common UniBasic keywords
        keywords_to_skip = {
            'IF', 'THEN', 'ELSE', 'END', 'BEGIN', 'CASE', 'FOR', 'TO', 'NEXT',
            'LOOP', 'WHILE', 'UNTIL', 'REPEAT', 'RETURN', 'GOSUB', 'GOTO',
            'AND', 'OR', 'NOT', 'EQ', 'NE', 'LT', 'GT', 'LE', 'GE',
        }
        return [t for t in tokens if t.upper() not in keywords_to_skip]

    def _empty_ground_truth(self, task_id: str) -> GroundTruth:
        """Return an empty ground truth."""
        return GroundTruth(
            task_id=task_id,
            source_hash="",
            language=Language.UNIBASIC,
        )

    def _load_cache(self, task_id: str, source_hash: str) -> Optional[GroundTruth]:
        """Load cached ground truth if available and valid."""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{task_id}_unibasic.json"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            if data.get("source_hash") != source_hash:
                logger.debug(f"Cache miss for {task_id} - source hash changed")
                return None
            
            return GroundTruth.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load cache for {task_id}: {e}")
            return None

    def _save_cache(self, task_id: str, source_hash: str, gt: GroundTruth) -> None:
        """Save ground truth to cache."""
        if not self.cache_dir:
            return
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{task_id}_unibasic.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(gt.to_dict(), f, indent=2)
            logger.debug(f"Cached ground truth for {task_id}")
        except Exception as e:
            logger.warning(f"Failed to cache ground truth for {task_id}: {e}")
