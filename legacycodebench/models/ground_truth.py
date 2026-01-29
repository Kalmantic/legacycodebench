"""
Ground Truth Models for LegacyCodeBench V2.4

Specification Reference: TDD_V2.4.md Section 2.3
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .enums import Language, RulePriority


@dataclass
class BusinessRule:
    """
    A business rule extracted from source code.
    
    Business rules represent the core logic/behavior of a program.
    They are weighted by priority for scoring.
    """
    rule_id: str                      # e.g., "BR-001"
    description: str                  # Human-readable description
    priority: RulePriority            # CRITICAL, IMPORTANT, TRIVIAL
    line_number: int                  # Source line where rule is implemented
    paragraph: str                    # COBOL paragraph or UniBasic subroutine
    keywords: List[str] = field(default_factory=list)  # For TF-IDF matching
    source_excerpt: str = ""          # Relevant source code snippet


@dataclass
class DataStructure:
    """
    A data structure definition from source code.
    
    For COBOL: Working-Storage, Linkage Section, File Section items
    For UniBasic: DIM statements, dynamic arrays
    """
    name: str                         # Variable/structure name
    level: str                        # COBOL: "01", "05" | UniBasic: "FIELD", "DIM"
    line_number: int                  # Source line of definition
    pic_clause: Optional[str] = None  # COBOL only: PIC X(10), PIC 9(5)V99
    children: List[str] = field(default_factory=list)  # Nested field names


@dataclass
class ExternalCall:
    """
    An external call (database, file, inter-program) from source code.
    
    Used for BSM (Behavioral Similarity Matching) validation.
    """
    call_type: str                    # SQL, CICS, CALL, FILE, EXECUTE
    target: str                       # Table name, program name, file name
    operation: str                    # SELECT, READ, WRITE, etc.
    line_number: int                  # Source line of call
    paragraph: str = ""               # Containing paragraph/subroutine


@dataclass
class ErrorHandler:
    """An error handling construct from source code."""
    handler_type: str                 # ON_ERROR, ON_SIZE_ERROR, AT_END
    line_number: int                  # Source line
    paragraph: str = ""               # Containing paragraph
    action: str = ""                  # What the handler does


@dataclass
class Paragraph:
    """
    A paragraph (COBOL) or subroutine (UniBasic) from source code.
    """
    name: str                         # Paragraph/subroutine name
    paragraph_type: str               # PURE, MIXED, INFRASTRUCTURE
    start_line: int                   # First line of paragraph
    end_line: int                     # Last line of paragraph
    content: str = ""                 # Source code content


@dataclass
class GroundTruth:
    """
    Complete ground truth for a benchmark task.
    
    Generated from static analysis of source code.
    Used for evaluation scoring.
    """
    task_id: str
    source_hash: str                  # SHA-256 of source file for integrity
    language: Language
    
    # Core components
    business_rules: List[BusinessRule] = field(default_factory=list)
    data_structures: List[DataStructure] = field(default_factory=list)
    paragraphs: List[Paragraph] = field(default_factory=list)
    external_calls: List[ExternalCall] = field(default_factory=list)
    error_handlers: List[ErrorHandler] = field(default_factory=list)
    
    # Statistics
    loc: int = 0                      # Lines of code
    cyclomatic_complexity: int = 0    # Complexity metric
    
    def get_critical_rules(self) -> List[BusinessRule]:
        """Get all CRITICAL priority business rules."""
        return [r for r in self.business_rules if r.priority == RulePriority.CRITICAL]
    
    def get_important_rules(self) -> List[BusinessRule]:
        """Get all IMPORTANT priority business rules."""
        return [r for r in self.business_rules if r.priority == RulePriority.IMPORTANT]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "source_hash": self.source_hash,
            "language": self.language.value,
            "business_rules": [
                {
                    "rule_id": r.rule_id,
                    "description": r.description,
                    "priority": r.priority.value,
                    "line_number": r.line_number,
                    "paragraph": r.paragraph,
                    "keywords": r.keywords,
                    "source_excerpt": r.source_excerpt,
                }
                for r in self.business_rules
            ],
            "data_structures": [
                {
                    "name": d.name,
                    "level": d.level,
                    "line_number": d.line_number,
                    "pic_clause": d.pic_clause,
                    "children": d.children,
                }
                for d in self.data_structures
            ],
            "paragraphs": [
                {
                    "name": p.name,
                    "type": p.paragraph_type,
                    "start_line": p.start_line,
                    "end_line": p.end_line,
                }
                for p in self.paragraphs
            ],
            "external_calls": [
                {
                    "call_type": e.call_type,
                    "target": e.target,
                    "operation": e.operation,
                    "line_number": e.line_number,
                    "paragraph": e.paragraph,
                }
                for e in self.external_calls
            ],
            "error_handlers": [
                {
                    "handler_type": h.handler_type,
                    "line_number": h.line_number,
                    "paragraph": h.paragraph,
                    "action": h.action,
                }
                for h in self.error_handlers
            ],
            "loc": self.loc,
            "cyclomatic_complexity": self.cyclomatic_complexity,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "GroundTruth":
        """Create GroundTruth from dictionary (JSON deserialization)."""
        return cls(
            task_id=data["task_id"],
            source_hash=data.get("source_hash", ""),
            language=Language(data.get("language", "cobol")),
            business_rules=[
                BusinessRule(
                    rule_id=r["rule_id"],
                    description=r["description"],
                    priority=RulePriority(r.get("priority", "important")),
                    line_number=r["line_number"],
                    paragraph=r.get("paragraph", ""),
                    keywords=r.get("keywords", []),
                    source_excerpt=r.get("source_excerpt", ""),
                )
                for r in data.get("business_rules", [])
            ],
            data_structures=[
                DataStructure(
                    name=d["name"],
                    level=d.get("level", "01"),
                    line_number=d["line_number"],
                    pic_clause=d.get("pic_clause"),
                    children=d.get("children", []),
                )
                for d in data.get("data_structures", [])
            ],
            paragraphs=[
                Paragraph(
                    name=p["name"],
                    paragraph_type=p.get("type", "PURE"),
                    start_line=p.get("start_line", 0),
                    end_line=p.get("end_line", 0),
                    content=p.get("content", ""),
                )
                for p in data.get("paragraphs", [])
            ],
            external_calls=[
                ExternalCall(
                    call_type=e["call_type"],
                    target=e["target"],
                    operation=e["operation"],
                    line_number=e["line_number"],
                    paragraph=e.get("paragraph", ""),
                )
                for e in data.get("external_calls", [])
            ],
            error_handlers=[
                ErrorHandler(
                    handler_type=h.get("handler_type", ""),
                    line_number=h.get("line_number", 0),
                    paragraph=h.get("paragraph", ""),
                    action=h.get("action", ""),
                )
                for h in data.get("error_handlers", [])
            ],
            loc=data.get("loc", 0),
            cyclomatic_complexity=data.get("cyclomatic_complexity", 0),
        )
