"""
Paragraph Classifier for V2.3.1

Three-way classification of COBOL paragraphs:
- PURE: Contains only computational logic (claim verification via execution)
- MIXED: Contains both logic AND external calls (claims + BSM)
- INFRASTRUCTURE: File I/O, initialization (BSM-only)
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from .config_v213 import V213_CONFIG


class ParagraphType(Enum):
    """Classification of paragraph purpose."""
    PURE = "pure"                       # Only logic, can regenerate + execute
    MIXED = "mixed"                     # Logic + external, extract + BSM
    INFRASTRUCTURE = "infrastructure"   # File I/O, preserve unchanged


@dataclass
class CodeBlock:
    """A block of code within a paragraph."""
    content: str
    start_offset: int
    end_offset: int
    block_type: str  # "logic", "external", "other"


@dataclass
class ClassifiedParagraph:
    """Result of paragraph classification."""
    name: str
    paragraph_type: ParagraphType
    content: str
    start_line: int
    end_line: int
    logic_blocks: List[CodeBlock] = field(default_factory=list)
    external_blocks: List[CodeBlock] = field(default_factory=list)
    blocking_patterns: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    @property
    def can_regenerate(self) -> bool:
        """Can this paragraph be regenerated from documentation?"""
        return self.paragraph_type in (ParagraphType.PURE, ParagraphType.MIXED)
    
    @property
    def requires_bsm(self) -> bool:
        """Does this paragraph need BSM validation?"""
        return self.paragraph_type == ParagraphType.MIXED
    
    @property
    def line_count(self) -> int:
        """Number of lines in paragraph."""
        return self.end_line - self.start_line + 1


class ParagraphClassifier:
    """
    Three-way paragraph classification for V2.3.
    
    Classification algorithm:
    1. Check if paragraph name matches infrastructure patterns → INFRASTRUCTURE
    2. Check for external call patterns (EXEC SQL/CICS, CALL)
    3. Check for business logic patterns (COMPUTE, IF, EVALUATE)
    4. If both external AND logic → MIXED
    5. If only logic → PURE
    6. Otherwise → INFRASTRUCTURE
    """
    
    def __init__(self):
        config = V213_CONFIG["classification"]
        self.infrastructure_patterns = [
            re.compile(p, re.IGNORECASE) for p in config["infrastructure_patterns"]
        ]
        self.external_patterns = [
            re.compile(p, re.IGNORECASE) for p in config["external_patterns"]
        ]
        self.logic_patterns = [
            re.compile(p, re.IGNORECASE) for p in config["logic_patterns"]
        ]
    
    def classify(self, paragraph: Dict) -> ClassifiedParagraph:
        """
        Classify a single paragraph.
        
        Args:
            paragraph: Dict with 'name', 'content', 'start_line', 'end_line'
            
        Returns:
            ClassifiedParagraph with type and extracted blocks
        """
        name = paragraph.get("name", "")
        content = paragraph.get("content", "")
        start_line = paragraph.get("start_line", 0)
        end_line = paragraph.get("end_line", 0)
        
        # Step 1: Check infrastructure by name
        if self._is_infrastructure_name(name):
            return ClassifiedParagraph(
                name=name,
                paragraph_type=ParagraphType.INFRASTRUCTURE,
                content=content,
                start_line=start_line,
                end_line=end_line,
                confidence=0.95
            )
        
        # Step 2: Detect external calls and business logic
        has_external, external_patterns = self._has_external_calls(content)
        has_logic = self._has_business_logic(content)
        
        # Step 3: Classify based on content
        if has_external and has_logic:
            # MIXED: Extract logic and external blocks
            logic_blocks, external_blocks = self._extract_blocks(content)
            return ClassifiedParagraph(
                name=name,
                paragraph_type=ParagraphType.MIXED,
                content=content,
                start_line=start_line,
                end_line=end_line,
                logic_blocks=logic_blocks,
                external_blocks=external_blocks,
                blocking_patterns=external_patterns,
                confidence=0.85
            )
        elif has_logic:
            # PURE: All business logic
            return ClassifiedParagraph(
                name=name,
                paragraph_type=ParagraphType.PURE,
                content=content,
                start_line=start_line,
                end_line=end_line,
                logic_blocks=[CodeBlock(content, 0, len(content), "logic")],
                confidence=0.90
            )
        else:
            # INFRASTRUCTURE: No business logic
            return ClassifiedParagraph(
                name=name,
                paragraph_type=ParagraphType.INFRASTRUCTURE,
                content=content,
                start_line=start_line,
                end_line=end_line,
                confidence=0.80
            )
    
    def classify_all(self, paragraphs: List[Dict]) -> Dict[str, List[ClassifiedParagraph]]:
        """
        Classify all paragraphs and group by type.
        
        Returns:
            Dict with 'pure', 'mixed', 'infrastructure' lists
        """
        result = {
            "pure": [],
            "mixed": [],
            "infrastructure": []
        }
        
        for para in paragraphs:
            classified = self.classify(para)
            result[classified.paragraph_type.value].append(classified)
        
        return result
    
    def _is_infrastructure_name(self, name: str) -> bool:
        """Check if paragraph name indicates infrastructure."""
        name_upper = name.upper()
        return any(pattern.match(name_upper) for pattern in self.infrastructure_patterns)
    
    def _has_external_calls(self, content: str) -> Tuple[bool, List[str]]:
        """Check if content contains external call patterns."""
        found_patterns = []
        for pattern in self.external_patterns:
            if pattern.search(content):
                found_patterns.append(pattern.pattern)
        return len(found_patterns) > 0, found_patterns
    
    def _has_business_logic(self, content: str) -> bool:
        """Check if content contains business logic patterns."""
        return any(pattern.search(content) for pattern in self.logic_patterns)
    
    def _extract_blocks(self, content: str) -> Tuple[List[CodeBlock], List[CodeBlock]]:
        """
        Extract logic and external blocks from MIXED paragraph.
        
        Strategy:
        1. Find all EXEC...END-EXEC blocks → external
        2. Everything between/around them → logic (if contains logic patterns)
        """
        logic_blocks = []
        external_blocks = []
        
        # Pattern to match EXEC blocks
        exec_pattern = re.compile(
            r"(EXEC\s+(?:SQL|CICS|DLI).*?END-EXEC\.?)",
            re.IGNORECASE | re.DOTALL
        )
        
        # Find all EXEC blocks
        last_end = 0
        for match in exec_pattern.finditer(content):
            # Check for logic before this EXEC block
            before_text = content[last_end:match.start()]
            if before_text.strip() and self._has_business_logic(before_text):
                logic_blocks.append(CodeBlock(
                    content=before_text.strip(),
                    start_offset=last_end,
                    end_offset=match.start(),
                    block_type="logic"
                ))
            
            # Add the EXEC block as external
            external_blocks.append(CodeBlock(
                content=match.group(1).strip(),
                start_offset=match.start(),
                end_offset=match.end(),
                block_type="external"
            ))
            
            last_end = match.end()
        
        # Check for logic after the last EXEC block
        after_text = content[last_end:]
        if after_text.strip() and self._has_business_logic(after_text):
            logic_blocks.append(CodeBlock(
                content=after_text.strip(),
                start_offset=last_end,
                end_offset=len(content),
                block_type="logic"
            ))
        
        return logic_blocks, external_blocks
    
    def get_statistics(self, paragraphs: List[Dict]) -> Dict:
        """Get classification statistics for a set of paragraphs."""
        classified = self.classify_all(paragraphs)
        
        total = len(paragraphs)
        pure_count = len(classified["pure"])
        mixed_count = len(classified["mixed"])
        infra_count = len(classified["infrastructure"])
        
        return {
            "total": total,
            "pure": pure_count,
            "mixed": mixed_count,
            "infrastructure": infra_count,
            "pure_percent": pure_count / total * 100 if total > 0 else 0,
            "mixed_percent": mixed_count / total * 100 if total > 0 else 0,
            "infrastructure_percent": infra_count / total * 100 if total > 0 else 0,
            "testable_percent": (pure_count + mixed_count) / total * 100 if total > 0 else 0
        }
