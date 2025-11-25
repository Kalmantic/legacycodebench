"""Documentation task evaluator"""

import re
from pathlib import Path
from typing import Dict, List, Optional
import markdown
from bs4 import BeautifulSoup
import logging

from legacycodebench.config import REQUIRED_DOC_SECTIONS, DOCUMENTATION_WEIGHTS, REFERENCES_DIR
from legacycodebench.evaluators.nlp_metrics import NLPMetrics

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available, using fallback similarity methods")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentationEvaluator:
    """Evaluate documentation submissions"""
    
    def __init__(self):
        self.required_sections = REQUIRED_DOC_SECTIONS
        self.weights = DOCUMENTATION_WEIGHTS
        self.nlp_metrics = NLPMetrics()
    
    def evaluate(self, submission_path: Path, task) -> Dict:
        """Evaluate a documentation submission"""
        if not submission_path.exists():
            return {
                "score": 0.0,
                "required_sections_present": 0.0,
                "business_rule_coverage": 0.0,
                "format_quality": 0.0,
                "appropriate_length": 0.0,
                "errors": ["Submission file not found"],
            }
        
        try:
            with open(submission_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {
                "score": 0.0,
                "errors": [f"Failed to read submission: {e}"],
            }
        
        # Check for reference documentation
        reference_path = self._find_reference(task)
        if reference_path and reference_path.exists():
            logger.info(f"Using reference documentation: {reference_path}")
            return self._evaluate_with_reference(submission_path, task, reference_path)
        else:
            logger.info("No reference documentation found, using heuristic evaluation")
            return self._evaluate_without_reference(content, task)
    
    def _evaluate_without_reference(self, content: str, task) -> Dict:
        """Evaluate using heuristics when no reference is available"""
        # Evaluate each metric
        required_sections_score = self._check_required_sections(content)
        business_rule_coverage = self._check_business_rule_coverage(content, task)
        format_quality = self._check_format_quality(content)
        appropriate_length = self._check_appropriate_length(content, task)
        
        # Calculate weighted score
        score = (
            self.weights["required_sections"] * required_sections_score +
            self.weights["business_rule_coverage"] * business_rule_coverage +
            self.weights["format_quality"] * format_quality +
            self.weights["appropriate_length"] * appropriate_length
        )
        
        return {
            "score": round(score, 4),
            "required_sections_present": round(required_sections_score, 4),
            "business_rule_coverage": round(business_rule_coverage, 4),
            "format_quality": round(format_quality, 4),
            "appropriate_length": round(appropriate_length, 4),
            "evaluation_method": "heuristic",
            "details": {
                "sections_found": self._find_sections(content),
                "word_count": len(content.split()),
                "markdown_valid": self._is_valid_markdown(content),
            }
        }
    
    def _evaluate_with_reference(self, submission_path: Path, task, reference_path: Path) -> Dict:
        """Evaluate using reference documentation"""
        try:
            with open(reference_path, 'r', encoding='utf-8') as f:
                reference_content = f.read()
        except Exception as e:
            logger.warning(f"Failed to read reference: {e}, falling back to heuristics")
            with open(submission_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self._evaluate_without_reference(content, task)
        
        with open(submission_path, 'r', encoding='utf-8') as f:
            submission_content = f.read()
        
        # Extract structured information from both documents
        ref_sections = self._extract_sections(reference_content)
        sub_sections = self._extract_sections(submission_content)
        
        # Compare sections
        required_sections_score = self._compare_sections(ref_sections, sub_sections)
        
        # Compare business rules (semantic similarity + ROUGE)
        ref_rules = self._extract_business_rules(reference_content)
        sub_rules = self._extract_business_rules(submission_content)
        business_rule_coverage = self._compare_business_rules(ref_rules, sub_rules)
        
        # Compare edge cases
        ref_edges = self._extract_edge_cases(reference_content)
        sub_edges = self._extract_edge_cases(submission_content)
        edge_coverage = self._compare_edge_cases(ref_edges, sub_edges)
        
        # Calculate NLP metrics (ROUGE, BLEU) for overall comparison
        nlp_metrics = self.nlp_metrics.calculate_all_metrics(
            reference_content,
            submission_content,
            ref_sections=ref_sections,
            sub_sections=sub_sections
        )
        
        # Format quality (still use heuristic)
        format_quality = self._check_format_quality(submission_content)
        
        # Appropriate length (compare to reference)
        appropriate_length = self._compare_length(reference_content, submission_content, task)
        
        # Calculate weighted score (business rule coverage includes edge cases)
        combined_rule_coverage = (business_rule_coverage * 0.7 + edge_coverage * 0.3)
        
        # Incorporate ROUGE and BLEU scores into business rule coverage (weighted combination)
        # ROUGE-L is good for measuring content overlap, BLEU for precision
        rouge_score = nlp_metrics.get("rougeL", 0.0)
        bleu_score = nlp_metrics.get("bleu", 0.0)
        
        # Ensure scores are floats, not dicts or other types
        # ROUGE and BLEU should already be floats from calculate_all_metrics, but validate
        if not isinstance(rouge_score, (int, float)):
            logger.warning(f"ROUGE score is not numeric: {type(rouge_score)}, defaulting to 0.0")
            rouge_score = 0.0
        
        if not isinstance(bleu_score, (int, float)):
            logger.warning(f"BLEU score is not numeric: {type(bleu_score)}, defaulting to 0.0")
            bleu_score = 0.0
        
        # Ensure values are in valid range [0.0, 1.0]
        rouge_score = max(0.0, min(1.0, float(rouge_score)))
        bleu_score = max(0.0, min(1.0, float(bleu_score)))
        
        # Combine semantic similarity with ROUGE and BLEU
        # 50% semantic, 30% ROUGE-L (recall), 20% BLEU (precision)
        enhanced_rule_coverage = (
            combined_rule_coverage * 0.5 + 
            float(rouge_score) * 0.3 + 
            float(bleu_score) * 0.2
        )
        
        score = (
            self.weights["required_sections"] * required_sections_score +
            self.weights["business_rule_coverage"] * enhanced_rule_coverage +
            self.weights["format_quality"] * format_quality +
            self.weights["appropriate_length"] * appropriate_length
        )
        
        return {
            "score": round(score, 4),
            "required_sections_present": round(required_sections_score, 4),
            "business_rule_coverage": round(enhanced_rule_coverage, 4),
            "format_quality": round(format_quality, 4),
            "appropriate_length": round(appropriate_length, 4),
            "evaluation_method": "reference",
            "nlp_metrics": nlp_metrics,  # Include all NLP metrics
            "details": {
                "sections_found": self._find_sections(submission_content),
                "word_count": len(submission_content.split()),
                "markdown_valid": self._is_valid_markdown(submission_content),
                "reference_rules_count": len(ref_rules),
                "submission_rules_count": len(sub_rules),
                "reference_edges_count": len(ref_edges),
                "submission_edges_count": len(sub_edges),
            }
        }
    
    def _check_required_sections(self, content: str) -> float:
        """Check if required sections are present (0.0 to 1.0)"""
        content_lower = content.lower()
        sections_found = 0
        
        # Map section names to keywords
        section_keywords = {
            "business_purpose": ["business purpose", "purpose", "overview", "what this does"],
            "business_rules": ["business rule", "rule", "logic", "condition"],
            "edge_cases": ["edge case", "exception", "error", "boundary"],
            "data_structures": ["data structure", "record", "field", "variable", "data"],
            "algorithm_overview": ["algorithm", "process", "flow", "steps", "procedure"],
        }
        
        for section in self.required_sections:
            keywords = section_keywords.get(section, [section.replace("_", " ")])
            if any(keyword in content_lower for keyword in keywords):
                sections_found += 1
        
        return sections_found / len(self.required_sections)
    
    def _check_business_rule_coverage(self, content: str, task) -> float:
        """Estimate business rule coverage (0.0 to 1.0)"""
        # Look for numbered lists, bullet points, or structured rules
        # This is a heuristic - in real implementation, would compare to reference
        
        # Count rule indicators
        rule_patterns = [
            r'\d+\.\s+',  # Numbered list
            r'[-*]\s+',   # Bullet points
            r'IF\s+',     # COBOL-like conditions
            r'WHEN\s+',   # Conditional logic
            r'RULE\s+\d+', # Explicit rule markers
        ]
        
        rule_count = 0
        for pattern in rule_patterns:
            rule_count += len(re.findall(pattern, content, re.IGNORECASE))
        
        # Normalize: assume good docs have 5-15 rules mentioned
        # This is a rough heuristic
        if rule_count == 0:
            return 0.0
        elif rule_count >= 5:
            return min(1.0, rule_count / 12.0)
        else:
            return rule_count / 5.0
    
    def _check_format_quality(self, content: str) -> float:
        """Check markdown format quality (0.0 to 1.0)"""
        score = 0.0
        
        # Check if valid markdown
        if self._is_valid_markdown(content):
            score += 0.3
        
        # Check for headers
        if re.search(r'^#+\s+', content, re.MULTILINE):
            score += 0.2
        
        # Check for code blocks
        if '```' in content or '`' in content:
            score += 0.2
        
        # Check for lists
        if re.search(r'^[-*]\s+', content, re.MULTILINE) or re.search(r'^\d+\.\s+', content, re.MULTILINE):
            score += 0.2
        
        # Check for paragraphs (not just one block)
        paragraphs = content.split('\n\n')
        if len(paragraphs) >= 3:
            score += 0.1
        
        return min(1.0, score)
    
    def _check_appropriate_length(self, content: str, task) -> float:
        """Check if documentation is appropriate length (0.0 to 1.0)"""
        word_count = len(content.split())
        char_count = len(content)
        
        # Task specifies minimum pages (roughly 500 words per page)
        min_words = task.evaluation_criteria.get("min_length_pages", 3) * 500
        
        if word_count < min_words * 0.5:
            return 0.0
        elif word_count < min_words:
            return word_count / min_words
        elif word_count < min_words * 2:
            return 1.0
        else:
            # Too long is also penalized slightly
            return max(0.7, 1.0 - (word_count - min_words * 2) / min_words)
    
    def _is_valid_markdown(self, content: str) -> bool:
        """Check if content is valid markdown"""
        try:
            html = markdown.markdown(content)
            # Basic check: did it produce HTML?
            return len(html) > 0 and '<' in html
        except:
            return False
    
    def _find_sections(self, content: str) -> List[str]:
        """Find section headers in content"""
        sections = []
        # Look for markdown headers
        for match in re.finditer(r'^#+\s+(.+)$', content, re.MULTILINE):
            sections.append(match.group(1).strip())
        return sections
    
    def _find_reference(self, task) -> Optional[Path]:
        """Find reference documentation for a task"""
        ref_dir = REFERENCES_DIR / "documentation" / task.task_id
        # Look for reference.md (consensus version)
        reference_path = ref_dir / "reference.md"
        if reference_path.exists():
            return reference_path
        # Fallback: look for any .md file in the directory
        if ref_dir.exists():
            md_files = list(ref_dir.glob("*.md"))
            if md_files:
                return md_files[0]
        return None
    
    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extract sections from markdown content"""
        sections = {}
        current_section = None
        current_content = []
        
        lines = content.split('\n')
        for line in lines:
            # Check for markdown header
            header_match = re.match(r'^#+\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                # Start new section
                current_section = header_match.group(1).strip().lower()
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _extract_business_rules(self, content: str) -> List[str]:
        """Extract business rules from documentation"""
        rules = []
        
        # Look for business rules section
        rules_section_match = re.search(
            r'(?:^#+\s+.*business\s+rule.*$)(.*?)(?:^#+\s+|$)',
            content,
            re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        
        if rules_section_match:
            rules_text = rules_section_match.group(1)
            # Extract numbered or bulleted rules
            rule_patterns = [
                r'(?:\d+\.|\*|\-)\s+(.+?)(?=\n(?:\d+\.|\*|\-|\n\n|$))',
                r'^\*\*(.+?)\*\*:?\s*(.+?)(?=\n\*\*|\n\n|$)',
            ]
            
            for pattern in rule_patterns:
                matches = re.findall(pattern, rules_text, re.MULTILINE | re.DOTALL)
                for match in matches:
                    if isinstance(match, tuple):
                        rule_text = ' '.join(match).strip()
                    else:
                        rule_text = match.strip()
                    if rule_text and len(rule_text) > 10:
                        rules.append(rule_text)
        else:
            # Fallback: look for any structured rules
            rule_items = re.findall(r'(?:\d+\.|\*|\-)\s+(.+?)(?=\n(?:\d+\.|\*|\-|\n\n|$))', content, re.MULTILINE)
            rules.extend([r.strip() for r in rule_items if len(r.strip()) > 20])
        
        return rules
    
    def _extract_edge_cases(self, content: str) -> List[str]:
        """Extract edge cases from documentation"""
        edge_cases = []
        
        # Look for edge cases section
        edges_section_match = re.search(
            r'(?:^#+\s+.*edge\s+case.*$)(.*?)(?:^#+\s+|$)',
            content,
            re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        
        if edges_section_match:
            edges_text = edges_section_match.group(1)
            # Extract edge case items
            edge_patterns = [
                r'(?:\d+\.|\*|\-)\s+(.+?)(?=\n(?:\d+\.|\*|\-|\n\n|$))',
                r'^\*\*(.+?)\*\*:?\s*(.+?)(?=\n\*\*|\n\n|$)',
            ]
            
            for pattern in edge_patterns:
                matches = re.findall(pattern, edges_text, re.MULTILINE | re.DOTALL)
                for match in matches:
                    if isinstance(match, tuple):
                        edge_text = ' '.join(match).strip()
                    else:
                        edge_text = match.strip()
                    if edge_text and len(edge_text) > 10:
                        edge_cases.append(edge_text)
        else:
            # Fallback: look for any structured edge cases
            edge_items = re.findall(r'(?:\d+\.|\*|\-)\s+(.+?)(?=\n(?:\d+\.|\*|\-|\n\n|$))', content, re.MULTILINE)
            edge_cases.extend([e.strip() for e in edge_items if len(e.strip()) > 20])
        
        return edge_cases
    
    def _compare_sections(self, ref_sections: Dict[str, str], sub_sections: Dict[str, str]) -> float:
        """Compare section presence between reference and submission"""
        # Map section names to required sections
        section_mapping = {
            "business_purpose": ["business purpose", "purpose", "overview"],
            "business_rules": ["business rule", "rule", "logic"],
            "edge_cases": ["edge case", "exception", "error"],
            "data_structures": ["data structure", "record", "field", "variable"],
            "algorithm_overview": ["algorithm", "process", "flow", "steps"],
        }
        
        found_count = 0
        for required_section in self.required_sections:
            keywords = section_mapping.get(required_section, [required_section.replace("_", " ")])
            
            # Check if reference has this section
            ref_has = any(
                any(kw in section_name.lower() for kw in keywords)
                for section_name in ref_sections.keys()
            )
            
            # Check if submission has this section
            sub_has = any(
                any(kw in section_name.lower() for kw in keywords)
                for section_name in sub_sections.keys()
            )
            
            # Both should have it, or at least submission should match reference
            if ref_has and sub_has:
                found_count += 1
            elif not ref_has:
                # Reference doesn't have it, so it's optional
                found_count += 1
            # If ref has but sub doesn't, don't count
        
        return found_count / len(self.required_sections) if self.required_sections else 0.0
    
    def _compare_business_rules(self, ref_rules: List[str], sub_rules: List[str]) -> float:
        """Compare business rules using semantic similarity"""
        if not ref_rules:
            # No reference rules, fall back to heuristic
            return 1.0 if sub_rules else 0.0
        
        if not sub_rules:
            return 0.0
        
        if HAS_SKLEARN:
            # Use TF-IDF and cosine similarity
            try:
                all_rules = ref_rules + sub_rules
                # TF-IDF requires at least 2 documents
                if len(all_rules) < 2:
                    logger.warning("Insufficient rules for TF-IDF (<2), using fallback")
                    raise ValueError("Insufficient rules for TF-IDF")
                
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                vectors = vectorizer.fit_transform(all_rules)
                
                ref_vectors = vectors[:len(ref_rules)]
                sub_vectors = vectors[len(ref_rules):]
                
                # Calculate similarity for each submission rule to best matching reference rule
                similarities = []
                for sub_vec in sub_vectors:
                    sims = cosine_similarity(sub_vec, ref_vectors)[0]
                    similarities.append(max(sims))
                
                # Average similarity
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
                
                # Coverage: how many reference rules are covered
                coverage = min(1.0, len(sub_rules) / len(ref_rules))
                
                # Combined score: similarity weighted by coverage
                return (avg_similarity * 0.7 + coverage * 0.3)
            except Exception as e:
                logger.warning(f"TF-IDF comparison failed: {e}, using fallback")
        
        # Fallback: simple word overlap
        ref_words = set()
        for rule in ref_rules:
            ref_words.update(rule.lower().split())
        
        sub_words = set()
        for rule in sub_rules:
            sub_words.update(rule.lower().split())
        
        if not ref_words:
            return 1.0 if sub_words else 0.0
        
        overlap = len(ref_words & sub_words) / len(ref_words | sub_words)
        coverage = min(1.0, len(sub_rules) / len(ref_rules))
        
        return (overlap * 0.7 + coverage * 0.3)
    
    def _compare_edge_cases(self, ref_edges: List[str], sub_edges: List[str]) -> float:
        """Compare edge cases using semantic similarity"""
        if not ref_edges:
            return 1.0 if sub_edges else 0.0
        
        if not sub_edges:
            return 0.0
        
        # Similar to business rules comparison
        if HAS_SKLEARN:
            try:
                all_edges = ref_edges + sub_edges
                # TF-IDF requires at least 2 documents
                if len(all_edges) < 2:
                    logger.warning("Insufficient edge cases for TF-IDF (<2), using fallback")
                    raise ValueError("Insufficient edge cases for TF-IDF")
                
                vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
                vectors = vectorizer.fit_transform(all_edges)
                
                ref_vectors = vectors[:len(ref_edges)]
                sub_vectors = vectors[len(ref_edges):]
                
                similarities = []
                for sub_vec in sub_vectors:
                    sims = cosine_similarity(sub_vec, ref_vectors)[0]
                    similarities.append(max(sims))
                
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
                coverage = min(1.0, len(sub_edges) / len(ref_edges))
                
                return (avg_similarity * 0.7 + coverage * 0.3)
            except Exception as e:
                logger.warning(f"TF-IDF comparison failed: {e}, using fallback")
        
        # Fallback: word overlap
        ref_words = set()
        for edge in ref_edges:
            ref_words.update(edge.lower().split())
        
        sub_words = set()
        for edge in sub_edges:
            sub_words.update(edge.lower().split())
        
        if not ref_words:
            return 1.0 if sub_words else 0.0
        
        overlap = len(ref_words & sub_words) / len(ref_words | sub_words)
        coverage = min(1.0, len(sub_edges) / len(ref_edges))
        
        return (overlap * 0.7 + coverage * 0.3)
    
    def _compare_length(self, reference_content: str, submission_content: str, task) -> float:
        """Compare document length to reference"""
        ref_words = len(reference_content.split())
        sub_words = len(submission_content.split())
        
        # Task minimum
        min_words = task.evaluation_criteria.get("min_length_pages", 3) * 500
        
        # Reference is the gold standard, but allow some variance
        if ref_words == 0:
            # Fall back to task minimum
            if sub_words < min_words * 0.5:
                return 0.0
            elif sub_words < min_words:
                return sub_words / min_words
            elif sub_words < min_words * 2:
                return 1.0
            else:
                return max(0.7, 1.0 - (sub_words - min_words * 2) / min_words)
        
        # Compare to reference length (allow 50% to 150% of reference)
        ratio = sub_words / ref_words if ref_words > 0 else 0.0
        
        if ratio < 0.5:
            return ratio / 0.5  # Linear from 0 to 0.5
        elif ratio <= 1.5:
            return 1.0  # Perfect range
        else:
            # Penalize if too long
            return max(0.7, 1.0 - (ratio - 1.5) / 1.0)

