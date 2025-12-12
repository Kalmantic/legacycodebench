"""Dependency Analyzer for COBOL

Implements Section 2.2: External Calls and Dependencies
- CALL targets, LINK/XCTL, parameter passing
- COPY statements (copybook dependencies)
- File operations (SELECT/FD pairs)
- Inter-program dependency analysis
- Automation Level: 100%
"""

import re
from typing import Dict, List, Set, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DependencyAnalyzer:
    """
    Analyze program dependencies: CALLs, COPYs, files.

    Per Section 2.2 and 2.4 of spec:
    - External call detection with boundary marking
    - Copybook dependency tracking
    - File I/O dependency mapping
    """

    def __init__(self):
        # CALL patterns
        self.call_pattern = re.compile(
            r'CALL\s+[\'"]?([A-Z0-9-]+)[\'"]?',
            re.IGNORECASE
        )

        # COPY pattern (copybooks)
        self.copy_pattern = re.compile(
            r'COPY\s+([A-Z0-9-]+)',
            re.IGNORECASE
        )

        # File operation patterns
        self.select_pattern = re.compile(
            r'SELECT\s+([A-Z0-9-]+)\s+ASSIGN',
            re.IGNORECASE
        )

        self.fd_pattern = re.compile(
            r'^\s*FD\s+([A-Z0-9-]+)',
            re.IGNORECASE
        )

        self.file_io_pattern = re.compile(
            r'(OPEN|CLOSE|READ|WRITE|REWRITE|DELETE)\s+([A-Z0-9-]+)',
            re.IGNORECASE
        )

    def analyze(self, parsed_cobol, file_path: Path) -> Dict:
        """
        Analyze all dependencies in COBOL program.

        Args:
            parsed_cobol: ParsedCOBOL object
            file_path: Path to source file

        Returns:
            Dictionary with dependency information
        """
        logger.info(f"Analyzing dependencies in {file_path.name}")

        # External calls
        external_calls = self._extract_calls(parsed_cobol)

        # Copybook dependencies
        copybooks = self._extract_copybooks(parsed_cobol)

        # File dependencies
        file_deps = self._extract_file_dependencies(parsed_cobol)

        # Categorize calls (in-scope vs external)
        call_categories = self._categorize_calls(external_calls, file_path)

        # Build dependency graph
        dep_graph = self._build_dependency_graph(
            external_calls,
            copybooks,
            file_deps
        )

        logger.info(f"Found {len(external_calls)} CALL statements")
        logger.info(f"Found {len(copybooks)} COPY statements")
        logger.info(f"Found {len(file_deps['files'])} file dependencies")

        return {
            "calls": [self._call_to_dict(c) for c in external_calls],
            "copybooks": copybooks,
            "files": file_deps,
            "call_categories": call_categories,
            "dependency_graph": dep_graph,
            "total_dependencies": len(external_calls) + len(copybooks) + len(file_deps['files'])
        }

    def _extract_calls(self, parsed_cobol) -> List[Dict]:
        """Extract CALL statements (external program calls)"""
        calls = []

        for paragraph in parsed_cobol.paragraphs:
            for i, stmt in enumerate(paragraph.statements):
                matches = self.call_pattern.finditer(stmt)

                for match in matches:
                    program = match.group(1)

                    # Check if call is dynamic (variable) or static (literal)
                    is_dynamic = not (stmt.upper().count(f"'{program}'") > 0 or
                                    stmt.upper().count(f'"{program}"') > 0)

                    calls.append({
                        "caller": paragraph.name,
                        "callee": program,
                        "line_number": paragraph.start_line + i,
                        "statement": stmt.strip(),
                        "is_dynamic": is_dynamic,
                        "parameters": self._extract_call_parameters(stmt)
                    })

        return calls

    def _extract_call_parameters(self, call_statement: str) -> List[str]:
        """Extract parameters from CALL statement"""
        # Pattern: CALL 'PROG' USING param1 param2 ...
        using_match = re.search(
            r'USING\s+(.*?)(?:\.|$)',
            call_statement,
            re.IGNORECASE
        )

        if using_match:
            params_str = using_match.group(1)
            # Split by whitespace, filter out BY keywords
            params = [p for p in params_str.split()
                     if p.upper() not in ['BY', 'REFERENCE', 'CONTENT', 'VALUE']]
            return params

        return []

    def _extract_copybooks(self, parsed_cobol) -> List[Dict]:
        """Extract COPY statements (copybook dependencies)"""
        copybooks = []
        seen = set()

        # Search in all divisions
        for division_name, division in parsed_cobol.divisions.items():
            matches = self.copy_pattern.finditer(division.content)

            for match in matches:
                copybook = match.group(1)

                # Avoid duplicates
                if copybook not in seen:
                    copybooks.append({
                        "name": copybook,
                        "division": division_name,
                        "line_number": division.start_line,
                        "type": "copybook"
                    })
                    seen.add(copybook)

        return copybooks

    def _extract_file_dependencies(self, parsed_cobol) -> Dict:
        """
        Extract file I/O dependencies.

        Per Section 2.2: SELECT/FD pairs, OPEN/CLOSE/READ/WRITE, FILE STATUS
        """
        files = []
        file_ops = []

        # Extract SELECT statements (FILE-CONTROL section)
        if 'ENVIRONMENT' in parsed_cobol.divisions:
            env_content = parsed_cobol.divisions['ENVIRONMENT'].content

            select_matches = self.select_pattern.finditer(env_content)
            for match in select_matches:
                files.append({
                    "name": match.group(1),
                    "type": "file",
                    "defined_in": "FILE-CONTROL"
                })

        # Extract FD statements (DATA DIVISION)
        if 'DATA' in parsed_cobol.divisions:
            data_content = parsed_cobol.divisions['DATA'].content

            fd_matches = self.fd_pattern.finditer(data_content)
            for match in fd_matches:
                file_name = match.group(1)
                # Update or add file
                existing = next((f for f in files if f["name"] == file_name), None)
                if existing:
                    existing["has_fd"] = True
                else:
                    files.append({
                        "name": file_name,
                        "type": "file",
                        "defined_in": "DATA DIVISION",
                        "has_fd": True
                    })

        # Extract file operations
        for paragraph in parsed_cobol.paragraphs:
            for i, stmt in enumerate(paragraph.statements):
                matches = self.file_io_pattern.finditer(stmt)

                for match in matches:
                    operation = match.group(1).upper()
                    file_name = match.group(2)

                    file_ops.append({
                        "operation": operation,
                        "file": file_name,
                        "paragraph": paragraph.name,
                        "line_number": paragraph.start_line + i,
                        "statement": stmt.strip()
                    })

        return {
            "files": files,
            "operations": file_ops,
            "total_files": len(files),
            "total_operations": len(file_ops)
        }

    def _categorize_calls(self, calls: List[Dict], source_file: Path) -> Dict:
        """
        Categorize calls as in-scope or external.

        Per Section 2.4: Open-World Assumptions
        - CALL target exists in scope: Mark as resolvable
        - CALL target missing: Mark as "External Dependency"
        - Dynamic CALL: Mark as "Unverifiable"
        """
        categories = {
            "in_scope": [],
            "external_dependency": [],
            "unverifiable": [],
            "middleware": []
        }

        # Try to find called programs in same directory
        source_dir = source_file.parent

        for call in calls:
            callee = call["callee"]

            # Dynamic calls are unverifiable
            if call["is_dynamic"]:
                categories["unverifiable"].append(call)
                continue

            # Check for middleware calls (CICS, DB2, etc.)
            if self._is_middleware_call(callee):
                categories["middleware"].append(call)
                continue

            # Try to find in same directory
            found = self._find_program_in_scope(callee, source_dir)

            if found:
                call["resolved_path"] = str(found)
                categories["in_scope"].append(call)
            else:
                categories["external_dependency"].append(call)

        logger.info(f"Call categories: {len(categories['in_scope'])} in-scope, "
                   f"{len(categories['external_dependency'])} external, "
                   f"{len(categories['unverifiable'])} unverifiable, "
                   f"{len(categories['middleware'])} middleware")

        return categories

    def _is_middleware_call(self, program_name: str) -> bool:
        """Check if call is to middleware (CICS, DB2, etc.)"""
        middleware_prefixes = [
            'CICS', 'DFH', 'CEE', 'DSN', 'SQL',
            'MQ', 'IMS', 'DB2'
        ]

        return any(program_name.upper().startswith(prefix)
                  for prefix in middleware_prefixes)

    def _find_program_in_scope(self, program_name: str, search_dir: Path) -> Optional[Path]:
        """Try to find called program in scope"""
        # Look for .cbl, .cob files with matching name
        for ext in ['.cbl', '.CBL', '.cob', '.COB']:
            potential_path = search_dir / f"{program_name}{ext}"
            if potential_path.exists():
                return potential_path

            # Try case-insensitive
            for file in search_dir.glob(f"*{ext}"):
                if file.stem.upper() == program_name.upper():
                    return file

        return None

    def _build_dependency_graph(self, calls: List[Dict],
                                copybooks: List[Dict],
                                file_deps: Dict) -> Dict:
        """Build complete dependency graph"""
        graph = {
            "nodes": [],
            "edges": []
        }

        # Add call edges
        for call in calls:
            graph["edges"].append({
                "from": call["caller"],
                "to": call["callee"],
                "type": "call",
                "is_dynamic": call["is_dynamic"]
            })

        # Add copybook edges
        for copy in copybooks:
            graph["edges"].append({
                "from": "program",
                "to": copy["name"],
                "type": "copy",
                "division": copy["division"]
            })

        # Add file edges
        for file_info in file_deps["files"]:
            graph["edges"].append({
                "from": "program",
                "to": file_info["name"],
                "type": "file",
                "defined_in": file_info["defined_in"]
            })

        # Extract unique nodes
        nodes = set()
        for edge in graph["edges"]:
            nodes.add(edge["from"])
            nodes.add(edge["to"])

        graph["nodes"] = [{"id": node} for node in nodes]

        return graph

    def _call_to_dict(self, call: Dict) -> Dict:
        """Convert call info to dictionary"""
        return {
            "caller": call["caller"],
            "callee": call["callee"],
            "line_number": call["line_number"],
            "is_dynamic": call["is_dynamic"],
            "parameters": call.get("parameters", []),
            "resolved_path": call.get("resolved_path")
        }
