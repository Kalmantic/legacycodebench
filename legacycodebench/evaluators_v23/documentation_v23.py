"""
Documentation Evaluator V2.3

Evaluates the quality and completeness of AI-generated documentation.
Weight: 25% of total score

Sub-components:
- Structural Completeness (40%): Required sections present
- Semantic Quality (35%): LLM-as-judge for quality
- Traceability (25%): Line number citations
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple

from .config_v23 import V23_CONFIG


@dataclass
class DocumentationResult:
    """Result of documentation evaluation."""
    overall: float                  # Weighted overall score (0-1)
    structural_score: float         # Structure completeness
    semantic_score: float           # Quality assessment
    traceability_score: float       # Citation accuracy
    missing_sections: List[str]     # Required sections not found
    invalid_citations: List[str]    # Citations that don't exist
    details: Dict = field(default_factory=dict)


class DocumentationEvaluatorV23:
    """
    V2.3 Documentation Evaluator
    
    Evaluates documentation quality with structural analysis and LLM judge.
    """
    
    # Required documentation sections
    REQUIRED_SECTIONS = [
        "overview", "purpose", "summary",                    # General
        "business", "logic", "rules",                        # Business Logic
        "data", "structure", "variable", "field",            # Data
        "flow", "control", "process",                        # Control Flow
    ]
    
    OPTIONAL_SECTIONS = [
        "error", "exception", "handling",                    # Error Handling
        "call", "external", "interface", "dependency",       # External Calls
        "file", "input", "output",                           # I/O
    ]
    
    def __init__(self, llm_client=None):
        self.weights = V23_CONFIG["documentation_weights"]
        self.llm = llm_client
    
    def evaluate(
        self,
        documentation: str,
        source_code: str,
        ground_truth: Dict,
        task_requirements: Dict = None
    ) -> DocumentationResult:
        """
        Evaluate documentation quality.
        
        Args:
            documentation: AI-generated documentation
            source_code: Original COBOL source
            ground_truth: Ground truth data
            task_requirements: Optional task-specific requirements
            
        Returns:
            DocumentationResult with scores and details
        """
        # Evaluate each component
        structural = self._evaluate_structural(documentation, task_requirements)
        semantic = self._evaluate_semantic(documentation, source_code)
        traceability = self._evaluate_traceability(documentation, source_code)
        
        # Calculate weighted score
        overall = (
            self.weights["structural"] * structural["score"] +
            self.weights["semantic"] * semantic["score"] +
            self.weights["traceability"] * traceability["score"]
        )
        
        return DocumentationResult(
            overall=overall,
            structural_score=structural["score"],
            semantic_score=semantic["score"],
            traceability_score=traceability["score"],
            missing_sections=structural["missing"],
            invalid_citations=traceability["invalid"],
            details={
                "structural": structural,
                "semantic": semantic,
                "traceability": traceability
            }
        )
    
    def _evaluate_structural(
        self,
        documentation: str,
        task_requirements: Dict = None
    ) -> Dict:
        """Evaluate structural completeness."""
        doc_lower = documentation.lower()
        
        # Check required sections
        found_required = []
        missing_required = []
        
        for section in self.REQUIRED_SECTIONS:
            if section in doc_lower:
                found_required.append(section)
        
        # Need at least some of each category
        has_overview = any(s in doc_lower for s in ["overview", "purpose", "summary"])
        has_business = any(s in doc_lower for s in ["business", "logic", "rule"])
        has_data = any(s in doc_lower for s in ["data", "structure", "variable", "field"])
        has_flow = any(s in doc_lower for s in ["flow", "control", "process"])
        
        # Calculate section coverage
        categories_present = sum([has_overview, has_business, has_data, has_flow])
        section_score = categories_present / 4
        
        # Check length (reasonable documentation should be substantial)
        word_count = len(documentation.split())
        length_score = min(1.0, word_count / 200)  # At least 200 words expected
        
        # Check for headers/structure
        has_headers = bool(re.search(r'^#+\s+', documentation, re.MULTILINE))
        has_bullets = bool(re.search(r'^[-*]\s+', documentation, re.MULTILINE))
        structure_score = 0.5 + (0.25 if has_headers else 0) + (0.25 if has_bullets else 0)
        
        # Combined score
        score = (section_score * 0.5) + (length_score * 0.3) + (structure_score * 0.2)
        
        missing = []
        if not has_overview:
            missing.append("overview/purpose/summary")
        if not has_business:
            missing.append("business logic/rules")
        if not has_data:
            missing.append("data structures")
        if not has_flow:
            missing.append("control flow")
        
        return {
            "score": score,
            "found": found_required,
            "missing": missing,
            "word_count": word_count,
            "has_headers": has_headers,
            "has_bullets": has_bullets
        }
    
    def _evaluate_semantic(
        self,
        documentation: str,
        source_code: str
    ) -> Dict:
        """
        Evaluate semantic quality of documentation.
        
        Uses rule-based heuristics when LLM not available,
        or LLM-as-judge when available.
        """
        if self.llm:
            return self._evaluate_semantic_llm(documentation, source_code)
        else:
            return self._evaluate_semantic_heuristic(documentation, source_code)
    
    def _evaluate_semantic_heuristic(
        self,
        documentation: str,
        source_code: str
    ) -> Dict:
        """Rule-based semantic quality evaluation."""
        scores = {
            "clarity": 0.0,
            "accuracy": 0.0,
            "completeness": 0.0,
            "organization": 0.0
        }
        
        # Clarity: Sentence structure and readability
        sentences = re.split(r'[.!?]', documentation)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        # Ideal: 10-25 words per sentence
        if 10 <= avg_sentence_length <= 25:
            scores["clarity"] = 1.0
        elif 5 <= avg_sentence_length <= 35:
            scores["clarity"] = 0.7
        else:
            scores["clarity"] = 0.4
        
        # Accuracy: Check if COBOL elements mentioned exist in source
        cobol_refs = re.findall(r'\b([A-Z][A-Z0-9-]+)\b', documentation)
        source_upper = source_code.upper()
        valid_refs = sum(1 for ref in cobol_refs if ref in source_upper)
        scores["accuracy"] = valid_refs / max(len(cobol_refs), 1) if cobol_refs else 0.5
        
        # Completeness: Compare documentation length to source complexity
        source_lines = len(source_code.split('\n'))
        doc_words = len(documentation.split())
        # Rough heuristic: ~1 word of doc per line of code minimum
        expected_words = source_lines * 1
        scores["completeness"] = min(1.0, doc_words / max(expected_words, 50))
        
        # Organization: Check for logical structure
        has_sections = bool(re.search(r'^#+\s+', documentation, re.MULTILINE))
        has_numbered = bool(re.search(r'^\d+\.?\s+', documentation, re.MULTILINE))
        has_bullets = bool(re.search(r'^[-*]\s+', documentation, re.MULTILINE))
        scores["organization"] = 0.4 + (0.2 * sum([has_sections, has_numbered, has_bullets]))
        
        # Average all scores
        avg_score = sum(scores.values()) / len(scores)
        
        return {
            "score": avg_score,
            "sub_scores": scores,
            "method": "heuristic"
        }
    
    def _evaluate_semantic_llm(
        self,
        documentation: str,
        source_code: str
    ) -> Dict:
        """LLM-as-judge semantic quality evaluation."""
        # Truncate for context window
        doc_truncated = documentation[:4000]
        source_truncated = source_code[:4000]
        
        prompt = f"""You are evaluating documentation quality for COBOL code.
Rate the documentation on a scale of 1-5 for each criterion:

1. CLARITY: Is the documentation easy to understand?
2. ACCURACY: Does it correctly describe the code's behavior?
3. COMPLETENESS: Does it cover all important aspects?
4. ORGANIZATION: Is it well-structured?

Source Code:
```cobol
{source_truncated}
```

Documentation:
```
{doc_truncated}
```

Respond with JSON only:
{{"clarity": N, "accuracy": N, "completeness": N, "organization": N}}"""
        
        try:
            response = self.llm.generate(prompt, temperature=0)
            # Parse JSON response
            import json
            scores = json.loads(response)
            
            # Normalize to 0-1
            normalized = {
                k: v / 5.0 for k, v in scores.items()
            }
            
            return {
                "score": sum(normalized.values()) / len(normalized),
                "sub_scores": normalized,
                "method": "llm"
            }
        except Exception as e:
            # Fallback to heuristic
            return self._evaluate_semantic_heuristic(documentation, source_code)
    
    def _evaluate_traceability(
        self,
        documentation: str,
        source_code: str
    ) -> Dict:
        """Evaluate line number citations and references."""
        # Find line references
        line_refs = re.findall(
            r'line[s]?\s*(\d+(?:\s*[-–]\s*\d+)?)', 
            documentation, 
            re.IGNORECASE
        )
        
        # Find paragraph references
        para_refs = re.findall(
            r'paragraph\s+([A-Z0-9-]+)', 
            documentation, 
            re.IGNORECASE
        )
        
        # Validate references
        source_lines = source_code.split('\n')
        max_line = len(source_lines)
        
        valid_line_refs = []
        invalid_line_refs = []
        
        for ref in line_refs:
            # Parse range or single
            if '-' in ref or '–' in ref:
                parts = re.split(r'[-–]', ref)
                try:
                    start, end = int(parts[0].strip()), int(parts[1].strip())
                    if 1 <= start <= max_line and 1 <= end <= max_line:
                        valid_line_refs.append(ref)
                    else:
                        invalid_line_refs.append(ref)
                except ValueError:
                    invalid_line_refs.append(ref)
            else:
                try:
                    line_num = int(ref.strip())
                    if 1 <= line_num <= max_line:
                        valid_line_refs.append(ref)
                    else:
                        invalid_line_refs.append(ref)
                except ValueError:
                    invalid_line_refs.append(ref)
        
        # Validate paragraph references
        valid_para_refs = []
        invalid_para_refs = []
        
        source_upper = source_code.upper()
        for para in para_refs:
            if para.upper() in source_upper:
                valid_para_refs.append(para)
            else:
                invalid_para_refs.append(para)
        
        # Calculate score
        total_refs = len(line_refs) + len(para_refs)
        valid_refs = len(valid_line_refs) + len(valid_para_refs)
        invalid_refs = len(invalid_line_refs) + len(invalid_para_refs)
        
        if total_refs == 0:
            score = 0.3  # Penalty for no citations
        else:
            score = valid_refs / total_refs
            # Additional penalty for invalid refs
            if invalid_refs > 0:
                score -= min(0.3, invalid_refs * 0.1)
        
        return {
            "score": max(0, score),
            "valid_lines": valid_line_refs,
            "valid_paragraphs": valid_para_refs,
            "invalid": invalid_line_refs + invalid_para_refs,
            "total_citations": total_refs
        }
