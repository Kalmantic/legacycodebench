"""Control Flow Analyzer for COBOL

Implements Section 2.2: Control Flow extraction
- PERFORM targets, GO TO destinations, loop structures
- Paragraph sequences and call graphs
- Automation Level: 100% for standard control flow
"""

import re
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ControlFlowNode:
    """Represents a control flow node (paragraph or statement)"""
    name: str
    type: str  # 'paragraph', 'perform', 'goto', 'if', 'evaluate'
    line_number: int
    targets: List[str] = field(default_factory=list)  # For PERFORM, GO TO
    conditions: List[str] = field(default_factory=list)  # For IF, EVALUATE


class ControlFlowAnalyzer:
    """
    Analyze control flow in COBOL PROCEDURE DIVISION.

    Per Section 2.2 of spec: Extract PERFORM targets, GO TO destinations,
    loop structures, paragraph sequences.
    """

    def __init__(self):
        # PERFORM patterns
        self.perform_pattern = re.compile(
            r'PERFORM\s+([A-Z0-9-]+)(?:\s+THROUGH\s+([A-Z0-9-]+))?',
            re.IGNORECASE
        )

        # GO TO pattern
        self.goto_pattern = re.compile(
            r'GO\s*TO\s+([A-Z0-9-]+)',
            re.IGNORECASE
        )

        # IF pattern
        self.if_pattern = re.compile(
            r'IF\s+(.+?)\s+THEN',
            re.IGNORECASE
        )

        # EVALUATE pattern
        self.evaluate_pattern = re.compile(
            r'EVALUATE\s+(.+?)(?:\s+WHEN|$)',
            re.IGNORECASE
        )

        # CALL pattern (external calls)
        self.call_pattern = re.compile(
            r'CALL\s+[\'"]?([A-Z0-9-]+)[\'"]?',
            re.IGNORECASE
        )

    def analyze(self, parsed_cobol, line_offset: int = 0) -> Dict:
        """
        Analyze control flow in PROCEDURE DIVISION.

        Args:
            parsed_cobol: ParsedCOBOL object from COBOLParser
            line_offset: Line number offset

        Returns:
            Dictionary with control flow information
        """
        logger.info("Analyzing control flow in PROCEDURE DIVISION")

        paragraphs = parsed_cobol.paragraphs

        # Extract control flow elements
        perform_targets = self._extract_performs(paragraphs, line_offset)
        goto_targets = self._extract_gotos(paragraphs, line_offset)
        loops = self._extract_loops(paragraphs)
        conditions = self._extract_conditions(paragraphs)
        call_graph = self._extract_calls(paragraphs)

        # Build control flow graph
        cfg = self._build_control_flow_graph(paragraphs, perform_targets, goto_targets)

        # Detect anti-patterns (for difficulty assessment)
        anti_patterns = self._detect_anti_patterns(goto_targets, paragraphs)

        logger.info(f"Found {len(perform_targets)} PERFORM statements")
        logger.info(f"Found {len(goto_targets)} GO TO statements")
        logger.info(f"Found {len(loops)} loop structures")
        logger.info(f"Found {len(conditions)} conditional statements")

        return {
            "paragraphs": [self._paragraph_to_dict(p) for p in paragraphs],
            "perform_targets": perform_targets,
            "goto_targets": goto_targets,
            "loops": loops,
            "conditions": conditions,
            "call_graph": call_graph,
            "control_flow_graph": cfg,
            "anti_patterns": anti_patterns,
            "total_paragraphs": len(paragraphs),
            "complexity_score": self._calculate_complexity(perform_targets, goto_targets, conditions)
        }

    def _extract_performs(self, paragraphs: List, line_offset: int) -> List[Dict]:
        """Extract PERFORM statements and their targets"""
        performs = []

        for paragraph in paragraphs:
            for i, stmt in enumerate(paragraph.statements):
                matches = self.perform_pattern.finditer(stmt)

                for match in matches:
                    target = match.group(1)
                    through = match.group(2)

                    performs.append({
                        "source": paragraph.name,
                        "target": target,
                        "through": through,
                        "line_number": paragraph.start_line + i,
                        "type": "perform_through" if through else "perform",
                        "statement": stmt.strip()
                    })

        return performs

    def _extract_gotos(self, paragraphs: List, line_offset: int) -> List[Dict]:
        """Extract GO TO statements (anti-pattern indicator)"""
        gotos = []

        for paragraph in paragraphs:
            for i, stmt in enumerate(paragraph.statements):
                matches = self.goto_pattern.finditer(stmt)

                for match in matches:
                    target = match.group(1)

                    gotos.append({
                        "source": paragraph.name,
                        "target": target,
                        "line_number": paragraph.start_line + i,
                        "type": "goto",
                        "statement": stmt.strip()
                    })

        return gotos

    def _extract_loops(self, paragraphs: List) -> List[Dict]:
        """
        Extract loop structures.

        COBOL loops are typically:
        - PERFORM VARYING
        - PERFORM UNTIL
        - PERFORM n TIMES
        """
        loops = []

        loop_patterns = [
            (r'PERFORM\s+VARYING\s+(.+)', 'varying'),
            (r'PERFORM\s+UNTIL\s+(.+)', 'until'),
            (r'PERFORM\s+(\d+)\s+TIMES', 'times')
        ]

        for paragraph in paragraphs:
            for stmt in paragraph.statements:
                for pattern, loop_type in loop_patterns:
                    match = re.search(pattern, stmt, re.IGNORECASE)
                    if match:
                        loops.append({
                            "paragraph": paragraph.name,
                            "type": loop_type,
                            "condition": match.group(1).strip(),
                            "line_number": paragraph.start_line,
                            "statement": stmt.strip()
                        })

        return loops

    def _extract_conditions(self, paragraphs: List) -> List[Dict]:
        """Extract IF and EVALUATE conditions (business rules)"""
        conditions = []

        for paragraph in paragraphs:
            for i, stmt in enumerate(paragraph.statements):
                # IF statements
                if_match = self.if_pattern.search(stmt)
                if if_match:
                    conditions.append({
                        "paragraph": paragraph.name,
                        "type": "if",
                        "condition": if_match.group(1).strip(),
                        "line_number": paragraph.start_line + i,
                        "statement": stmt.strip()
                    })

                # EVALUATE statements
                eval_match = self.evaluate_pattern.search(stmt)
                if eval_match:
                    conditions.append({
                        "paragraph": paragraph.name,
                        "type": "evaluate",
                        "expression": eval_match.group(1).strip(),
                        "line_number": paragraph.start_line + i,
                        "statement": stmt.strip()
                    })

        return conditions

    def _extract_calls(self, paragraphs: List) -> List[Dict]:
        """Extract CALL statements (external dependencies)"""
        calls = []

        for paragraph in paragraphs:
            for i, stmt in enumerate(paragraph.statements):
                matches = self.call_pattern.finditer(stmt)

                for match in matches:
                    program = match.group(1)

                    calls.append({
                        "caller": paragraph.name,
                        "callee": program,
                        "line_number": paragraph.start_line + i,
                        "type": "call",
                        "statement": stmt.strip()
                    })

        return calls

    def _build_control_flow_graph(self, paragraphs: List,
                                  performs: List[Dict],
                                  gotos: List[Dict]) -> Dict:
        """
        Build control flow graph showing execution paths.

        Returns adjacency list representation.
        """
        graph = {}

        # Initialize with all paragraphs
        for para in paragraphs:
            graph[para.name] = {
                "successors": [],
                "predecessors": [],
                "type": "paragraph"
            }

        # Add PERFORM edges
        for perf in performs:
            source = perf["source"]
            target = perf["target"]

            if source in graph and target in graph:
                graph[source]["successors"].append({
                    "target": target,
                    "type": "perform"
                })
                graph[target]["predecessors"].append({
                    "source": source,
                    "type": "perform"
                })

        # Add GO TO edges
        for goto in gotos:
            source = goto["source"]
            target = goto["target"]

            if source in graph and target in graph:
                graph[source]["successors"].append({
                    "target": target,
                    "type": "goto"
                })
                graph[target]["predecessors"].append({
                    "source": source,
                    "type": "goto"
                })

        return graph

    def _detect_anti_patterns(self, gotos: List[Dict], paragraphs: List) -> Dict:
        """
        Detect anti-patterns that increase difficulty.

        Per Section 6.3 of spec: Anti-patterns include GO TO spaghetti,
        dead code, ambiguous names.
        """
        anti_patterns = {
            "goto_count": len(gotos),
            "goto_density": len(gotos) / len(paragraphs) if paragraphs else 0,
            "has_goto_spaghetti": len(gotos) > len(paragraphs) * 0.15,  # >15% density
            "forward_gotos": sum(1 for g in gotos if self._is_forward_goto(g, paragraphs)),
            "backward_gotos": sum(1 for g in gotos if not self._is_forward_goto(g, paragraphs)),
        }

        return anti_patterns

    def _is_forward_goto(self, goto: Dict, paragraphs: List) -> bool:
        """Check if GO TO is forward (target after source)"""
        source_para = next((p for p in paragraphs if p.name == goto["source"]), None)
        target_para = next((p for p in paragraphs if p.name == goto["target"]), None)

        if source_para and target_para:
            return target_para.start_line > source_para.start_line

        return False

    def _calculate_complexity(self, performs: List[Dict],
                             gotos: List[Dict],
                             conditions: List[Dict]) -> int:
        """
        Calculate cyclomatic complexity estimate.

        Simplified: nodes + edges + conditions
        """
        # Rough estimate: number of decision points
        decision_points = len(conditions) + len(performs) + len(gotos)

        return decision_points

    def _paragraph_to_dict(self, paragraph) -> Dict:
        """Convert paragraph to dictionary"""
        return {
            "name": paragraph.name,
            "start_line": paragraph.start_line,
            "end_line": paragraph.end_line,
            "statement_count": len(paragraph.statements)
        }
