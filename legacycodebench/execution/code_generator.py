"""Constrained Code Generator for Behavioral Fidelity Evaluation

Generates COBOL code from documentation with strict gap detection.

Critical for preventing false positives (CF-06):
- Uses [GAP: description] markers for missing information
- Validates all code elements are grounded in documentation
- Rejects if >30% of LOC are gaps
- Ensures documentation completeness before execution testing

Weight in v2.0 evaluation: 35% (Behavioral Fidelity)
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GeneratedCode:
    """Results from constrained code generation"""
    code: str
    gaps: List[Dict[str, str]]  # List of gap markers
    gap_percentage: float  # Percentage of LOC that are gaps
    is_complete: bool  # True if gap_percentage < 30%
    error_message: Optional[str] = None
    metadata: Dict = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "code": self.code,
            "gaps": self.gaps,
            "gap_percentage": self.gap_percentage,
            "is_complete": self.is_complete,
            "error_message": self.error_message,
            "metadata": self.metadata or {}
        }


class ConstrainedCodeGenerator:
    """
    Generate COBOL code from documentation with strict constraints.

    Key Principle: NEVER hallucinate or infer missing information.
    If documentation doesn't specify something, use [GAP: ...] marker.

    This prevents false positives where vague documentation
    accidentally passes execution tests.

    Algorithm:
    1. Parse documentation to extract code elements
    2. Generate COBOL structure (divisions, sections, paragraphs)
    3. For each element, check if documentation provides details
    4. If details missing, insert [GAP: what's missing]
    5. Calculate gap percentage
    6. Reject if >30% gaps (documentation too incomplete)

    Uses LLM in strict mode with gap-aware prompts.
    """

    def __init__(self, llm_model: str = "gpt-4o", gap_threshold: float = 0.30):
        """
        Initialize constrained code generator.

        Args:
            llm_model: LLM model to use for generation
            gap_threshold: Maximum allowed gap percentage (default 30%)
        """
        self.llm_model = llm_model
        self.gap_threshold = gap_threshold

        # Initialize LLM interface (lazy import to avoid circular dependency)
        self.llm = None

    def generate(self, documentation: str, ground_truth: Dict,
                program_name: Optional[str] = None) -> GeneratedCode:
        """
        Generate COBOL code from documentation with gap detection.

        Args:
            documentation: AI-generated documentation to test
            ground_truth: Ground truth for validation
            program_name: Optional program name

        Returns:
            GeneratedCode with code, gaps, and completeness flag
        """
        logger.info("Generating COBOL code from documentation (constrained mode)")

        try:
            # Extract program structure from ground truth
            expected_structure = self._extract_expected_structure(ground_truth)

            # Generate code using LLM with strict gap-aware prompt
            generated_code = self._generate_with_llm(
                documentation,
                expected_structure,
                program_name
            )

            # Detect and analyze gaps
            gaps = self._detect_gaps(generated_code)
            gap_percentage = self._calculate_gap_percentage(generated_code, gaps)

            # Check if documentation is complete enough
            is_complete = gap_percentage < self.gap_threshold

            if not is_complete:
                logger.warning(
                    f"Documentation incomplete: {gap_percentage:.1%} gaps "
                    f"(threshold: {self.gap_threshold:.1%})"
                )

            return GeneratedCode(
                code=generated_code,
                gaps=gaps,
                gap_percentage=gap_percentage,
                is_complete=is_complete,
                metadata={
                    "total_gaps": len(gaps),
                    "gap_threshold": self.gap_threshold,
                    "expected_elements": len(expected_structure.get("paragraphs", []))
                }
            )

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return GeneratedCode(
                code="",
                gaps=[],
                gap_percentage=1.0,
                is_complete=False,
                error_message=str(e)
            )

    def _extract_expected_structure(self, ground_truth: Dict) -> Dict:
        """
        Extract expected program structure from ground truth.

        This guides the generator on what elements should exist.

        Args:
            ground_truth: Ground truth from static analysis

        Returns:
            Dictionary with expected structure elements
        """
        structure = {
            "program_name": ground_truth.get("metadata", {}).get("source_file", "PROGRAM"),
            "data_structures": [],
            "paragraphs": [],
            "business_rules": [],
            "files": []
        }

        # Extract data structures (variable names, no implementation details)
        data_structures = ground_truth.get("data_structures", {})
        for ds in data_structures.get("data_structures", []):
            structure["data_structures"].append({
                "name": ds["name"],
                "level": ds["level"]
            })

        # Extract paragraph names (no logic details)
        control_flow = ground_truth.get("control_flow", {})
        for para in control_flow.get("paragraphs", []):
            structure["paragraphs"].append({
                "name": para["name"],
                "line_number": para.get("line_number", 0)
            })

        # Extract business rules (conditions only, no actions)
        business_rules = ground_truth.get("business_rules", {})
        for rule in business_rules.get("rules", []):
            structure["business_rules"].append({
                "type": rule.get("type", ""),
                "description": rule.get("description", "")
            })

        # Extract file operations
        deps = ground_truth.get("dependencies", {})
        structure["files"] = deps.get("files", {}).get("file_list", [])

        return structure

    def _generate_with_llm(self, documentation: str,
                          expected_structure: Dict,
                          program_name: Optional[str]) -> str:
        """
        Generate COBOL code using LLM with gap-aware prompt.

        Uses strict system prompt that enforces gap markers.

        Args:
            documentation: Documentation to generate from
            expected_structure: Expected program structure
            program_name: Optional program name

        Returns:
            Generated COBOL code with gap markers
        """
        # Try LLM API first, fall back to template-based if not available
        try:
            # Lazy-load LLM interface
            if self.llm is None:
                try:
                    import os
                    # Check if we have API keys
                    has_openai = bool(os.getenv("OPENAI_API_KEY"))
                    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

                    if has_openai or has_anthropic:
                        # We can use real LLM
                        logger.info("LLM API keys found, using real code generation")
                        generated_code = self._call_llm_api(documentation, expected_structure, program_name)
                        return generated_code
                    else:
                        logger.info("No LLM API keys found, using template-based generation")
                        raise ImportError("No API keys")
                except ImportError:
                    logger.info("AI integration not available, using template-based generation")
                    raise

        except Exception as e:
            logger.warning(f"LLM generation failed: {e}, falling back to template-based")

        # Fallback: template-based generation
        generated_code = self._generate_template_based(
            documentation,
            expected_structure,
            program_name
        )

        return generated_code

    def _call_llm_api(self, documentation: str,
                     expected_structure: Dict,
                      program_name: Optional[str]) -> str:
        """Call LLM API for code generation"""
        import os
        import json

        # Build gap-aware system prompt
        system_prompt = self._build_gap_aware_prompt(expected_structure)

        # Build user prompt with structure hints
        user_prompt = f"""Generate COBOL code from the following documentation.

CRITICAL RULES:
1. Use ONLY information explicitly stated in the documentation
2. For ANY missing information, use [GAP: description of what's missing]
3. Do NOT infer, assume, or hallucinate missing details
4. Include gap markers for:
   - Missing data types or PICTURE clauses
   - Missing business logic or calculations
   - Missing error handling
   - Missing file operations
   - Vague or unclear specifications

DOCUMENTATION:
{documentation}

EXPECTED STRUCTURE HINTS (from original code):
- Program name: {expected_structure.get('program_name', 'PROGRAM')}
- Data structures: {len(expected_structure['data_structures'])} (names: {', '.join([ds['name'] for ds in expected_structure['data_structures'][:5]])})
- Paragraphs: {len(expected_structure['paragraphs'])} (names: {', '.join([p['name'] for p in expected_structure['paragraphs'][:5]])})
- Business rules: {len(expected_structure['business_rules'])}
- Files: {len(expected_structure.get('files', []))}

Generate complete COBOL program matching this structure. Use gap markers where documentation lacks details."""

        # Try OpenAI first
        if os.getenv("OPENAI_API_KEY"):
            try:
                import openai
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                logger.info(f"Calling OpenAI {self.llm_model} for constrained code generation...")

                response = client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0,  # Deterministic
                    max_tokens=8000
                )

                generated_code = response.choices[0].message.content
                logger.info(f"Generated {len(generated_code)} chars of COBOL code")
                return self._extract_cobol_from_response(generated_code)

            except Exception as e:
                logger.error(f"OpenAI call failed: {e}")

        # Try Anthropic if OpenAI failed
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

                logger.info(f"Calling Anthropic Claude for constrained code generation...")

                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=8000,
                    temperature=0,
                    messages=[
                        {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
                    ]
                )

                generated_code = response.content[0].text
                logger.info(f"Generated {len(generated_code)} chars of COBOL code")
                return self._extract_cobol_from_response(generated_code)

            except Exception as e:
                logger.error(f"Anthropic call failed: {e}")

        # If all LLM calls failed, raise to trigger fallback
        raise RuntimeError("All LLM API calls failed")

    def _extract_cobol_from_response(self, response: str) -> str:
        """Extract COBOL code from LLM response (may have markdown blocks)"""
        # Remove markdown code blocks if present
        if "```cobol" in response.lower():
            # Extract from ```cobol ... ```
            import re
            match = re.search(r'```cobol\n(.*?)\n```', response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1)
        elif "```" in response:
            # Extract from ``` ... ```
            import re
            match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
            if match:
                return match.group(1)

        # Return as-is if no code blocks
        return response

    def _generate_template_based(self, documentation: str,
                                 expected_structure: Dict,
                                 program_name: Optional[str]) -> str:
        """
        Generate code using template-based approach (fallback).

        This is a simplified version. Full implementation would use LLM.

        Args:
            documentation: Documentation text
            expected_structure: Expected structure
            program_name: Program name

        Returns:
            COBOL code with gap markers
        """
        prog_name = program_name or "GENERATED"

        # Start with basic structure
        code_lines = [
            "       IDENTIFICATION DIVISION.",
            f"       PROGRAM-ID. {prog_name}.",
            "      *> Generated from documentation with gap detection",
            "",
            "       ENVIRONMENT DIVISION.",
            "       CONFIGURATION SECTION.",
            "       INPUT-OUTPUT SECTION.",
            "       FILE-CONTROL."
        ]

        # Add file definitions (with gaps if not in docs)
        for file_info in expected_structure.get("files", []):
            file_name = file_info.get("name", "UNKNOWN")
            if self._is_documented(documentation, file_name):
                code_lines.append(f"           SELECT {file_name} ASSIGN TO '{file_name}.dat'.")
            else:
                code_lines.append(f"      *> [GAP: File definition for {file_name} not documented]")

        code_lines.extend([
            "",
            "       DATA DIVISION.",
            "       FILE SECTION.",
            ""
        ])

        # Add file descriptions (with gaps)
        for file_info in expected_structure.get("files", []):
            file_name = file_info.get("name", "UNKNOWN")
            code_lines.append(f"      *> [GAP: FD definition for {file_name} - record layout not documented]")

        code_lines.extend([
            "",
            "       WORKING-STORAGE SECTION.",
            ""
        ])

        # Add data structures (with gaps for details)
        for ds in expected_structure.get("data_structures", []):
            ds_name = ds["name"]
            if self._is_documented(documentation, ds_name):
                # Check if PICTURE clause is documented
                if self._has_picture_documented(documentation, ds_name):
                    code_lines.append(f"       01  {ds_name}.")
                    code_lines.append(f"      *> [GAP: Field details for {ds_name} not fully documented]")
                else:
                    code_lines.append(f"      *> [GAP: Data structure {ds_name} - type and layout not documented]")
            else:
                code_lines.append(f"      *> [GAP: Data structure {ds_name} not documented]")

        code_lines.extend([
            "",
            "       PROCEDURE DIVISION.",
            ""
        ])

        # Add paragraphs (with gaps for logic)
        for para in expected_structure.get("paragraphs", []):
            para_name = para["name"]

            if para_name == "MAIN-PROCEDURE" or para_name.endswith("-MAIN"):
                code_lines.extend([
                    f"       {para_name}.",
                ])

                # Add business logic (with gaps)
                for rule in expected_structure.get("business_rules", []):
                    rule_desc = rule.get("description", "")
                    if self._is_documented(documentation, rule_desc):
                        code_lines.append(f"      *> [GAP: Implementation of '{rule_desc}' not documented]")
                    else:
                        code_lines.append(f"      *> [GAP: Business rule not documented in detail]")

                code_lines.extend([
                    "           STOP RUN.",
                    ""
                ])
            else:
                code_lines.extend([
                    f"       {para_name}.",
                    f"      *> [GAP: Logic for {para_name} not documented]",
                    "           EXIT.",
                    ""
                ])

        return "\n".join(code_lines)

    def _is_documented(self, documentation: str, element_name: str) -> bool:
        """Check if element is mentioned in documentation"""
        # Simple keyword search
        return element_name.lower() in documentation.lower()

    def _has_picture_documented(self, documentation: str, field_name: str) -> bool:
        """Check if PICTURE clause is documented for field"""
        # Look for "field_name PIC" or "field_name: type"
        pattern = f"{field_name}.*?(?:PIC|type|format)"
        return bool(re.search(pattern, documentation, re.IGNORECASE))

    def _build_gap_aware_prompt(self, expected_structure: Dict) -> str:
        """Build system prompt for gap-aware generation"""
        return """You are a COBOL code generator with STRICT gap detection.

CRITICAL RULES:
1. Generate ONLY from explicitly documented information
2. Use [GAP: description] for ANY missing information
3. NEVER infer, assume, or hallucinate details
4. Be conservative - when in doubt, mark as gap

Gap markers are REQUIRED for:
- Missing data types (PICTURE clauses)
- Missing business logic (calculations, formulas)
- Missing control flow (loop conditions, branching)
- Missing error handling
- Vague specifications

Example good output:
       01  CUSTOMER-RECORD.
      *> [GAP: Record structure not documented - field layout unknown]

Example bad output:
       01  CUSTOMER-RECORD.
           05  CUST-ID      PIC 9(5).   <-- DO NOT INFER PICTURE CLAUSES!
"""

    def _detect_gaps(self, generated_code: str) -> List[Dict[str, str]]:
        """
        Detect gap markers in generated code.

        Args:
            generated_code: Generated COBOL code

        Returns:
            List of gaps with line number and description
        """
        gaps = []

        lines = generated_code.split('\n')
        for i, line in enumerate(lines, start=1):
            # Look for gap markers: [GAP: ...]
            match = re.search(r'\[GAP:\s*([^\]]+)\]', line)
            if match:
                gaps.append({
                    "line_number": i,
                    "description": match.group(1).strip(),
                    "code_line": line.strip()
                })

        logger.info(f"Detected {len(gaps)} gap markers in generated code")

        return gaps

    def _calculate_gap_percentage(self, generated_code: str,
                                  gaps: List[Dict[str, str]]) -> float:
        """
        Calculate percentage of code that is gaps.

        Args:
            generated_code: Generated code
            gaps: List of detected gaps

        Returns:
            Gap percentage (0.0 to 1.0)
        """
        lines = [line.strip() for line in generated_code.split('\n') if line.strip()]

        # Remove comment-only and empty lines
        code_lines = [
            line for line in lines
            if not line.startswith('*>') and not line.startswith('*')
        ]

        total_loc = len(code_lines)

        if total_loc == 0:
            return 1.0  # 100% gaps if no code

        # Count gap lines
        gap_lines = len(gaps)

        gap_percentage = gap_lines / total_loc if total_loc > 0 else 1.0

        logger.info(f"Gap analysis: {gap_lines}/{total_loc} lines are gaps ({gap_percentage:.1%})")

        return gap_percentage
