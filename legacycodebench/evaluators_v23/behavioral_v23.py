"""
Behavioral Evaluator V2.3

Evaluates understanding through code regeneration and execution.
Weight: 35% of total score

Hybrid approach:
- PURE paragraphs: Template-based regeneration + execution
- MIXED paragraphs: Logic extraction + regeneration + BSM for externals
- INFRASTRUCTURE: Preserved (no evaluation)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from .config_v23 import V23_CONFIG
from .paragraph_classifier import (
    ParagraphClassifier, 
    ParagraphType, 
    ClassifiedParagraph
)


@dataclass
class BehavioralResult:
    """Result of behavioral evaluation."""
    overall: float                      # Weighted overall score (0-1)
    pure_score: float                   # Score for PURE paragraphs
    mixed_score: float                  # Score for MIXED paragraphs
    bsm_score: float                    # BSM validation score
    compilation_success: bool           # Did merged code compile?
    execution_success: bool             # Did execution complete?
    cf03_triggered: bool                # Wrong Transformation
    cf05_triggered: bool                # BSM Specification Failure
    paragraphs_evaluated: Dict[str, float]  # Per-paragraph scores
    test_results: Dict = field(default_factory=dict)
    details: Dict = field(default_factory=dict)


class TemplateGenerator:
    """
    Generates template from COBOL source with placeholders for business logic.
    """
    
    def create_template(
        self,
        source_code: str,
        paragraphs_to_replace: List[str]
    ) -> Tuple[str, Dict[str, str]]:
        """
        Create template with placeholders for specified paragraphs.
        
        Args:
            source_code: Original COBOL source
            paragraphs_to_replace: List of paragraph names to replace
            
        Returns:
            (template_code, original_paragraphs_dict)
        """
        template = source_code
        originals = {}
        
        for para_name in paragraphs_to_replace:
            # Find paragraph content
            pattern = rf'(\s*{re.escape(para_name)}\.)'
            match = re.search(pattern, source_code, re.IGNORECASE)
            
            if match:
                # Find paragraph end (next paragraph or division)
                start_pos = match.end()
                end_pattern = rf'\n\s*[A-Z0-9-]+\.\s*\n|\n\s+[A-Z]{{2,}}\s+DIVISION'
                end_match = re.search(end_pattern, source_code[start_pos:])
                
                if end_match:
                    end_pos = start_pos + end_match.start()
                else:
                    end_pos = len(source_code)
                
                # Extract original
                original_content = source_code[match.start():end_pos]
                originals[para_name] = original_content
                
                # Create placeholder
                placeholder = f"""
       {para_name}.
      *> [PLACEHOLDER: {para_name}]
      *> This paragraph will be regenerated from documentation
           CONTINUE.
"""
                template = template.replace(original_content, placeholder)
        
        return template, originals


class ParagraphMerger:
    """
    Merges regenerated paragraphs back into template.
    """
    
    def merge(
        self,
        template: str,
        regenerated: Dict[str, str]
    ) -> str:
        """
        Merge regenerated paragraphs into template.
        
        Args:
            template: Template with placeholders
            regenerated: Dict of paragraph_name -> regenerated_code
            
        Returns:
            Merged source code
        """
        merged = template
        
        for para_name, code in regenerated.items():
            # Find placeholder
            placeholder_pattern = rf'\s*{re.escape(para_name)}\..*?\n\s*CONTINUE\.'
            
            # Format regenerated code
            formatted_code = self._format_cobol(code, para_name)
            
            # Replace placeholder
            merged = re.sub(
                placeholder_pattern,
                formatted_code,
                merged,
                flags=re.IGNORECASE | re.DOTALL
            )
        
        return merged
    
    def _format_cobol(self, code: str, para_name: str) -> str:
        """Format code to COBOL column requirements."""
        lines = code.strip().split('\n')
        formatted_lines = []
        
        # Add paragraph header if not present
        if not lines[0].strip().upper().startswith(para_name.upper()):
            formatted_lines.append(f"       {para_name}.")
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                formatted_lines.append("")
                continue
            
            # Ensure proper column positioning
            stripped = line.lstrip()
            
            # Comments start at column 7
            if stripped.startswith('*'):
                formatted_lines.append(f"      {stripped}")
            else:
                # Code starts at column 12
                formatted_lines.append(f"           {stripped}")
        
        return '\n'.join(formatted_lines)


class BSMValidator:
    """
    Behavioral Specification Matching for external calls.
    Validates that documentation correctly describes external call behavior.
    """
    
    def __init__(self):
        self.patterns = V23_CONFIG["bsm_patterns"]
    
    def validate(
        self,
        documentation: str,
        external_blocks: List[str]
    ) -> Dict:
        """
        Validate external calls against documentation.
        
        Returns:
            Dict with score and details
        """
        if not external_blocks:
            return {"score": 1.0, "patterns_found": 0, "patterns_matched": 0}
        
        total_patterns = 0
        matched_patterns = 0
        pattern_details = []
        
        for block in external_blocks:
            block_upper = block.upper() if isinstance(block, str) else block.content.upper()
            
            for pattern_name, pattern_config in self.patterns.items():
                code_pattern = pattern_config["code_pattern"]
                requirements = pattern_config["requirements"]
                
                match = re.search(code_pattern, block_upper, re.IGNORECASE | re.DOTALL)
                
                if match:
                    total_patterns += 1
                    
                    # Check if requirements are documented
                    doc_lower = documentation.lower()
                    covered = 0
                    
                    for req in requirements:
                        # Check if requirement topic is mentioned
                        req_words = req.replace("_", " ").split()
                        if any(word in doc_lower for word in req_words):
                            covered += 1
                    
                    coverage = covered / len(requirements) if requirements else 1.0
                    
                    if coverage >= 0.7:  # 70% coverage threshold
                        matched_patterns += 1
                    
                    pattern_details.append({
                        "pattern": pattern_name,
                        "coverage": coverage,
                        "matched": coverage >= 0.7
                    })
        
        score = matched_patterns / total_patterns if total_patterns > 0 else 1.0
        
        return {
            "score": score,
            "patterns_found": total_patterns,
            "patterns_matched": matched_patterns,
            "details": pattern_details,
            "cf05_triggered": score < 0.5  # >50% failure
        }


class BehavioralEvaluatorV23:
    """
    V2.3 Behavioral Evaluator
    
    Hybrid evaluation using template-based regeneration for PURE paragraphs
    and BSM validation for external calls in MIXED paragraphs.
    """
    
    def __init__(self, llm_client=None, executor=None):
        self.classifier = ParagraphClassifier()
        self.template_gen = TemplateGenerator()
        self.merger = ParagraphMerger()
        self.bsm = BSMValidator()
        self.llm = llm_client
        self.executor = executor
        self.config = V23_CONFIG
    
    def evaluate(
        self,
        documentation: str,
        source_code: str,
        ground_truth: Dict,
        paragraphs: List[Dict]
    ) -> BehavioralResult:
        """
        Evaluate behavioral fidelity.
        
        Args:
            documentation: AI-generated documentation
            source_code: Original COBOL source
            ground_truth: Ground truth data
            paragraphs: List of parsed paragraphs
            
        Returns:
            BehavioralResult with scores and details
        """
        # Classify paragraphs
        classified = self.classifier.classify_all(paragraphs)
        
        pure_paras = classified["pure"]
        mixed_paras = classified["mixed"]
        
        # Evaluate PURE paragraphs
        pure_score, pure_details = self._evaluate_pure(
            documentation, source_code, pure_paras
        )
        
        # Evaluate MIXED paragraphs
        mixed_score, bsm_score, mixed_details = self._evaluate_mixed(
            documentation, source_code, mixed_paras
        )
        
        # Calculate weighted overall score
        total_testable = len(pure_paras) + len(mixed_paras)
        
        if total_testable == 0:
            overall = 1.0  # No testable paragraphs
        else:
            pure_weight = len(pure_paras) / total_testable
            mixed_weight = len(mixed_paras) / total_testable
            overall = (pure_score * pure_weight) + (mixed_score * mixed_weight)
        
        # Check critical failures
        cf03 = pure_details.get("test_failure_rate", 0) > 0.5 if pure_paras else False
        cf05 = mixed_details.get("cf05_triggered", False)
        
        # Build per-paragraph scores
        para_scores = {}
        for para in pure_paras:
            para_scores[para.name] = pure_details.get(f"score_{para.name}", 0.5)
        for para in mixed_paras:
            para_scores[para.name] = mixed_details.get(f"score_{para.name}", 0.5)
        
        return BehavioralResult(
            overall=overall,
            pure_score=pure_score,
            mixed_score=mixed_score,
            bsm_score=bsm_score,
            compilation_success=pure_details.get("compilation_success", True),
            execution_success=pure_details.get("execution_success", True),
            cf03_triggered=cf03,
            cf05_triggered=cf05,
            paragraphs_evaluated=para_scores,
            test_results=pure_details.get("test_results", {}),
            details={
                "pure": pure_details,
                "mixed": mixed_details,
                "classification": {
                    "pure_count": len(pure_paras),
                    "mixed_count": len(mixed_paras),
                    "infrastructure_count": len(classified["infrastructure"])
                }
            }
        )
    
    def _evaluate_pure(
        self,
        documentation: str,
        source_code: str,
        pure_paras: List[ClassifiedParagraph]
    ) -> Tuple[float, Dict]:
        """Evaluate PURE paragraphs via template-based regeneration."""
        if not pure_paras:
            return 1.0, {"message": "No PURE paragraphs to evaluate"}
        
        details = {
            "compilation_success": True,
            "execution_success": True,
            "test_failure_rate": 0.0
        }
        
        # If no LLM or executor, return mock score
        if not self.llm or not self.executor:
            # Heuristic-based evaluation
            return self._evaluate_pure_heuristic(documentation, pure_paras)
        
        scores = []
        
        for para in pure_paras:
            try:
                # Generate replacement from documentation
                generated = self._generate_paragraph(documentation, para)
                
                # Create template and merge
                template, originals = self.template_gen.create_template(
                    source_code, [para.name]
                )
                merged = self.merger.merge(template, {para.name: generated})
                
                # Compile
                compile_result = self.executor.compile(merged)
                if not compile_result.success:
                    details["compilation_success"] = False
                    scores.append(0.0)
                    details[f"score_{para.name}"] = 0.0
                    continue
                
                # Execute and compare
                score = self._compare_execution(source_code, merged, para)
                scores.append(score)
                details[f"score_{para.name}"] = score
                
            except Exception as e:
                scores.append(0.0)
                details[f"error_{para.name}"] = str(e)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        details["test_failure_rate"] = 1 - avg_score
        
        return avg_score, details
    
    def _evaluate_pure_heuristic(
        self,
        documentation: str,
        pure_paras: List[ClassifiedParagraph]
    ) -> Tuple[float, Dict]:
        """Heuristic evaluation when LLM/executor not available."""
        doc_lower = documentation.lower()
        scores = []
        details = {"method": "heuristic"}
        
        for para in pure_paras:
            # Check if paragraph is mentioned in documentation
            para_mentioned = para.name.lower() in doc_lower
            
            # Check if logic keywords from paragraph are in documentation
            logic_keywords = re.findall(r'\b(COMPUTE|IF|EVALUATE)\b', para.content, re.I)
            keyword_coverage = sum(
                1 for kw in logic_keywords if kw.lower() in doc_lower
            ) / max(len(logic_keywords), 1)
            
            # Score based on coverage
            score = 0.3 if para_mentioned else 0.0
            score += 0.7 * keyword_coverage
            
            scores.append(score)
            details[f"score_{para.name}"] = score
        
        return sum(scores) / len(scores) if scores else 0.5, details
    
    def _evaluate_mixed(
        self,
        documentation: str,
        source_code: str,
        mixed_paras: List[ClassifiedParagraph]
    ) -> Tuple[float, float, Dict]:
        """Evaluate MIXED paragraphs via logic regeneration + BSM."""
        if not mixed_paras:
            return 1.0, 1.0, {"message": "No MIXED paragraphs"}
        
        logic_scores = []
        bsm_scores = []
        details = {}
        
        for para in mixed_paras:
            # Evaluate logic blocks (if present)
            if para.logic_blocks:
                logic_score = self._evaluate_logic_blocks(documentation, para)
                logic_scores.append(logic_score)
            
            # Evaluate external blocks via BSM
            if para.external_blocks:
                external_contents = [b.content for b in para.external_blocks]
                bsm_result = self.bsm.validate(documentation, external_contents)
                bsm_scores.append(bsm_result["score"])
                details[f"bsm_{para.name}"] = bsm_result
        
        avg_logic = sum(logic_scores) / len(logic_scores) if logic_scores else 1.0
        avg_bsm = sum(bsm_scores) / len(bsm_scores) if bsm_scores else 1.0
        
        # Combined MIXED score: 60% logic, 40% BSM
        weights = self.config["behavioral_weights"]
        mixed_score = (
            weights["logic_regeneration"] * avg_logic +
            weights["bsm_validation"] * avg_bsm
        )
        
        details["cf05_triggered"] = avg_bsm < 0.5
        
        return mixed_score, avg_bsm, details
    
    def _evaluate_logic_blocks(
        self,
        documentation: str,
        para: ClassifiedParagraph
    ) -> float:
        """Evaluate logic blocks within a MIXED paragraph."""
        doc_lower = documentation.lower()
        
        # Simple heuristic: check if logic operations are documented
        total_ops = 0
        documented_ops = 0
        
        for block in para.logic_blocks:
            content = block.content if hasattr(block, 'content') else str(block)
            
            # Find operations
            operations = re.findall(
                r'\b(COMPUTE|IF|EVALUATE|ADD|SUBTRACT|MULTIPLY|DIVIDE)\b',
                content,
                re.IGNORECASE
            )
            
            for op in operations:
                total_ops += 1
                if op.lower() in doc_lower:
                    documented_ops += 1
        
        return documented_ops / total_ops if total_ops > 0 else 0.5
    
    def _generate_paragraph(
        self,
        documentation: str,
        para: ClassifiedParagraph
    ) -> str:
        """Generate paragraph code from documentation using LLM."""
        if not self.llm:
            return para.content  # Return original if no LLM
        
        prompt = f"""Based on the following documentation, generate COBOL code for paragraph {para.name}.

Documentation:
{documentation}

Generate ONLY the COBOL code for the paragraph body (not the paragraph name).
Use proper COBOL syntax starting at column 12.
Do not include any explanations, just the code."""
        
        try:
            generated = self.llm.generate(prompt, temperature=0)
            return generated.strip()
        except Exception:
            return para.content  # Fallback to original
    
    def _compare_execution(
        self,
        original_source: str,
        merged_source: str,
        para: ClassifiedParagraph
    ) -> float:
        """Compare execution outputs between original and merged code."""
        if not self.executor:
            return 0.5  # Neutral if no executor
        
        try:
            # Generate test cases
            test_cases = self._generate_test_cases(para)
            
            matches = 0
            for test in test_cases:
                orig_result = self.executor.run(original_source, test)
                merged_result = self.executor.run(merged_source, test)
                
                if self._outputs_match(orig_result, merged_result):
                    matches += 1
            
            return matches / len(test_cases) if test_cases else 0.5
            
        except Exception:
            return 0.0
    
    def _generate_test_cases(self, para: ClassifiedParagraph) -> List[Dict]:
        """Generate test cases for a paragraph."""
        # Simplified: return basic test cases
        # A full implementation would use the test_generator module
        return [
            {"input": "test1"},
            {"input": "test2"},
        ]
    
    def _outputs_match(self, result1: Dict, result2: Dict) -> bool:
        """Compare execution outputs with tolerance."""
        tolerance = self.config["execution"]["numeric_tolerance"]
        
        # Compare return codes
        if result1.get("return_code") != result2.get("return_code"):
            return False
        
        # Compare outputs
        out1 = result1.get("stdout", "")
        out2 = result2.get("stdout", "")
        
        # Exact match
        if out1 == out2:
            return True
        
        # Numeric tolerance check
        try:
            nums1 = [float(x) for x in re.findall(r'-?\d+\.?\d*', out1)]
            nums2 = [float(x) for x in re.findall(r'-?\d+\.?\d*', out2)]
            
            if len(nums1) == len(nums2):
                return all(abs(a - b) <= tolerance for a, b in zip(nums1, nums2))
        except ValueError:
            pass
        
        return False
