"""
Comprehension Evaluator V2.3

Evaluates whether the AI understands the business purpose of the code.
Weight: 40% of total score

Sub-components:
- Business Rules (40%): Coverage of CRITICAL/IMPORTANT rules
- Data Flow (25%): Understanding of data structures
- Abstraction (20%): WHY vs WHAT explanations (anti-gaming)
- Dependencies (15%): External dependencies documented
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from difflib import SequenceMatcher

from .config_v23 import V23_CONFIG
from .anti_gaming import AbstractionScorer


@dataclass
class BusinessRule:
    """A business rule from ground truth."""
    id: str
    description: str
    priority: str  # CRITICAL, IMPORTANT, TRIVIAL
    source_lines: List[int] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


@dataclass
class ComprehensionResult:
    """Result of comprehension evaluation."""
    overall: float                      # Weighted overall score (0-1)
    business_rules_score: float         # Business rules coverage
    data_flow_score: float              # Data structure understanding
    abstraction_score: float            # WHY vs WHAT
    dependencies_score: float           # External dependencies
    matched_rules: List[str]            # Rule IDs that were matched
    missing_critical_rules: List[str]   # Critical rules not documented
    cf01_triggered: bool                # Missing Core Calculations
    cf02_triggered: bool                # Hallucinated Logic
    cf06_triggered: bool                # Semantic Contradiction
    hallucinations: List[str]           # Detected hallucinations
    contradictions: List[str]           # Detected contradictions
    details: Dict = field(default_factory=dict)


class ComprehensionEvaluatorV23:
    """
    V2.3 Comprehension Evaluator
    
    Evaluates understanding of business logic with anti-gaming measures.
    """
    
    def __init__(self):
        self.weights = V23_CONFIG["comprehension_weights"]
        self.abstraction_scorer = AbstractionScorer()
    
    def evaluate(
        self,
        documentation: str,
        ground_truth: Dict,
        source_code: str
    ) -> ComprehensionResult:
        """
        Evaluate comprehension of source code.
        
        Args:
            documentation: AI-generated documentation
            ground_truth: Ground truth data with business rules, data structures
            source_code: Original COBOL source
            
        Returns:
            ComprehensionResult with scores and details
        """
        # Extract ground truth components
        business_rules = self._parse_business_rules(ground_truth)
        data_structures = ground_truth.get("data_structures", [])
        dependencies = ground_truth.get("dependencies", [])
        error_handlers = ground_truth.get("error_handlers", [])
        
        # Evaluate each component
        br_result = self._evaluate_business_rules(documentation, business_rules)
        df_result = self._evaluate_data_flow(documentation, data_structures)
        abstraction = self.abstraction_scorer.score(documentation)
        dep_result = self._evaluate_dependencies(documentation, dependencies)
        
        # Check for hallucinations and contradictions
        hallucinations = self._detect_hallucinations(documentation, source_code)
        contradictions = self._detect_contradictions(documentation, ground_truth)
        
        # Critical failure detection
        cf01 = br_result["critical_match_rate"] == 0 and br_result["critical_count"] > 0
        cf02 = len(hallucinations) > 0
        cf04 = (len(error_handlers) > 0 and 
                not self._error_handlers_documented(documentation, error_handlers))
        cf06 = len(contradictions) > 0
        
        # Calculate weighted score
        overall = (
            self.weights["business_rules"] * br_result["score"] +
            self.weights["data_flow"] * df_result["score"] +
            self.weights["abstraction"] * abstraction +
            self.weights["dependencies"] * dep_result["score"]
        )
        
        return ComprehensionResult(
            overall=overall,
            business_rules_score=br_result["score"],
            data_flow_score=df_result["score"],
            abstraction_score=abstraction,
            dependencies_score=dep_result["score"],
            matched_rules=br_result["matched"],
            missing_critical_rules=br_result["missing_critical"],
            cf01_triggered=cf01,
            cf02_triggered=cf02,
            cf06_triggered=cf06,
            hallucinations=hallucinations,
            contradictions=contradictions,
            details={
                "business_rules": br_result,
                "data_flow": df_result,
                "dependencies": dep_result,
                "cf04_triggered": cf04
            }
        )
    
    def _parse_business_rules(self, ground_truth: Dict) -> List[BusinessRule]:
        """Parse business rules from ground truth.
        
        Ground truth format:
          business_rules: {
            rules: [
              {id: "BR-001", description: "...", priority: "critical", ...}
            ]
          }
        """
        rules = []
        
        br_data = ground_truth.get("business_rules", {})
        
        # Handle nested structure: business_rules.rules
        if isinstance(br_data, dict):
            rules_list = br_data.get("rules", [])
        elif isinstance(br_data, list):
            rules_list = br_data
        else:
            rules_list = []
        
        for rule_data in rules_list:
            # Handle both dict and string formats
            if isinstance(rule_data, dict):
                # Normalize priority to uppercase
                priority = rule_data.get("priority", "IMPORTANT").upper()
                if priority not in ["CRITICAL", "IMPORTANT", "TRIVIAL"]:
                    priority = "IMPORTANT"
                
                rules.append(BusinessRule(
                    id=rule_data.get("id", ""),
                    description=rule_data.get("description", ""),
                    priority=priority,
                    source_lines=rule_data.get("source_lines", []),
                    keywords=rule_data.get("keywords", [])
                ))
            elif isinstance(rule_data, str):
                # Handle string format (legacy)
                rules.append(BusinessRule(
                    id=f"BR-{len(rules)+1:03d}",
                    description=rule_data,
                    priority="IMPORTANT",
                    source_lines=[],
                    keywords=[]
                ))
        
        return rules
    
    def _evaluate_business_rules(
        self,
        documentation: str,
        rules: List[BusinessRule]
    ) -> Dict:
        """Evaluate coverage of business rules."""
        if not rules:
            return {
                "score": 1.0,
                "matched": [],
                "missing_critical": [],
                "critical_count": 0,
                "critical_match_rate": 1.0
            }
        
        doc_lower = documentation.lower()
        matched = []
        missing_critical = []
        
        critical_rules = [r for r in rules if r.priority == "CRITICAL"]
        important_rules = [r for r in rules if r.priority == "IMPORTANT"]
        
        critical_matched = 0
        important_matched = 0
        
        for rule in rules:
            if self._is_rule_documented(documentation, rule):
                matched.append(rule.id)
                if rule.priority == "CRITICAL":
                    critical_matched += 1
                elif rule.priority == "IMPORTANT":
                    important_matched += 1
            elif rule.priority == "CRITICAL":
                missing_critical.append(rule.id)
        
        # Calculate scores
        critical_rate = (critical_matched / len(critical_rules) 
                        if critical_rules else 1.0)
        important_rate = (important_matched / len(important_rules) 
                         if important_rules else 1.0)
        
        # Weighted: 70% critical, 30% important
        score = (0.70 * critical_rate) + (0.30 * important_rate)
        
        return {
            "score": score,
            "matched": matched,
            "missing_critical": missing_critical,
            "critical_count": len(critical_rules),
            "critical_match_rate": critical_rate,
            "important_match_rate": important_rate
        }
    
    def _is_rule_documented(self, documentation: str, rule: BusinessRule) -> bool:
        """Check if a rule is documented (semantic matching)."""
        doc_lower = documentation.lower()
        
        # Check keyword presence
        keyword_matches = sum(
            1 for kw in rule.keywords 
            if kw.lower() in doc_lower
        )
        keyword_ratio = keyword_matches / len(rule.keywords) if rule.keywords else 0
        
        # Check description similarity
        rule_desc_lower = rule.description.lower()
        
        # Split documentation into sentences
        sentences = re.split(r'[.!?\n]', documentation)
        max_similarity = 0
        
        for sentence in sentences:
            if len(sentence.strip()) > 10:
                similarity = SequenceMatcher(
                    None, 
                    rule_desc_lower, 
                    sentence.lower()
                ).ratio()
                max_similarity = max(max_similarity, similarity)
        
        # Match if keywords present OR high description similarity
        return keyword_ratio >= 0.5 or max_similarity >= 0.6
    
    def _evaluate_data_flow(
        self,
        documentation: str,
        data_structures
    ) -> Dict:
        """Evaluate documentation of data structures."""
        # Handle nested structure: data_structures may be a dict with 'structures' key
        if isinstance(data_structures, dict):
            ds_list = data_structures.get("structures", [])
        elif isinstance(data_structures, list):
            ds_list = data_structures
        else:
            ds_list = []
        
        if not ds_list:
            return {"score": 1.0, "documented": [], "missing": []}
        
        doc_lower = documentation.lower()
        documented = []
        missing = []
        
        for ds in ds_list:
            # Handle both dict and string formats
            if isinstance(ds, dict):
                name = ds.get("name", "")
            elif isinstance(ds, str):
                name = ds
            else:
                continue
            
            # Check if structure name is mentioned
            if name.lower() in doc_lower or name.replace("-", " ").lower() in doc_lower:
                documented.append(name)
            else:
                missing.append(name)
        
        score = len(documented) / len(ds_list) if ds_list else 1.0
        
        return {
            "score": score,
            "documented": documented,
            "missing": missing
        }
    
    def _evaluate_dependencies(
        self,
        documentation: str,
        dependencies
    ) -> Dict:
        """Evaluate documentation of external dependencies."""
        # Handle nested structure: dependencies may be a dict with 'files' key
        if isinstance(dependencies, dict):
            # Try to get files from nested structure
            files_data = dependencies.get("files", {})
            if isinstance(files_data, dict):
                dep_list = files_data.get("files", [])
            else:
                dep_list = files_data if isinstance(files_data, list) else []
            
            # Also check for calls
            calls = dependencies.get("calls", [])
            if calls:
                dep_list = dep_list + calls
        elif isinstance(dependencies, list):
            dep_list = dependencies
        else:
            dep_list = []
        
        if not dep_list:
            return {"score": 1.0, "documented": [], "missing": []}
        
        doc_lower = documentation.lower()
        documented = []
        missing = []
        
        for dep in dep_list:
            # Handle both dict and string formats
            if isinstance(dep, dict):
                name = dep.get("name", "")
                dep_type = dep.get("type", "")
            elif isinstance(dep, str):
                name = dep
                dep_type = ""
            else:
                continue
            
            # Check if dependency is mentioned
            if name.lower() in doc_lower:
                documented.append(name)
            elif dep_type and dep_type.lower() in doc_lower:
                documented.append(name)
            else:
                missing.append(name)
        
        score = len(documented) / len(dep_list) if dep_list else 1.0
        
        return {
            "score": score,
            "documented": documented,
            "missing": missing
        }
    
    def _detect_hallucinations(
        self,
        documentation: str,
        source_code: str
    ) -> List[str]:
        """
        Detect hallucinated logic (logic described but not in source).
        
        This is a simplified implementation. A full implementation would
        use LLM-based detection or more sophisticated pattern matching.
        """
        hallucinations = []
        
        # Look for specific calculations mentioned in docs
        # that don't appear in source
        calc_patterns = [
            (r"calculates?\s+(\w+)\s+as\s+(.+)", "calculation"),
            (r"multiplies?\s+by\s+(\d+(?:\.\d+)?)", "multiplier"),
            (r"(\d+(?:\.\d+)?)\s*%\s+(?:discount|tax|rate)", "percentage")
        ]
        
        source_upper = source_code.upper()
        
        for pattern, desc in calc_patterns:
            matches = re.findall(pattern, documentation, re.IGNORECASE)
            for match in matches:
                # Convert to string if tuple
                match_str = str(match) if isinstance(match, tuple) else match
                # Check if this appears in source
                if not re.search(re.escape(match_str), source_upper, re.IGNORECASE):
                    # Check for numeric values
                    numbers = re.findall(r'\d+(?:\.\d+)?', match_str)
                    for num in numbers:
                        if num not in source_code:
                            hallucinations.append(f"Hallucinated {desc}: {match_str}")
                            break
        
        return hallucinations[:5]  # Limit to first 5
    
    def _detect_contradictions(
        self,
        documentation: str,
        ground_truth: Dict
    ) -> List[str]:
        """Detect semantic contradictions between documentation and source."""
        contradictions = []
        
        operations = ground_truth.get("operations", [])
        doc_lower = documentation.lower()
        
        for op in operations:
            op_type = op.get("type", "")
            variable = op.get("variable", "")
            
            # Check for opposite operations described
            if op_type == "ADD":
                if f"subtract" in doc_lower and variable.lower() in doc_lower:
                    contradictions.append(
                        f"Says subtract but code adds: {variable}"
                    )
            elif op_type == "SUBTRACT":
                if "add" in doc_lower and variable.lower() in doc_lower:
                    contradictions.append(
                        f"Says add but code subtracts: {variable}"
                    )
            elif op_type == "MULTIPLY":
                if "divide" in doc_lower and variable.lower() in doc_lower:
                    contradictions.append(
                        f"Says divide but code multiplies: {variable}"
                    )
            elif op_type == "DIVIDE":
                if "multipl" in doc_lower and variable.lower() in doc_lower:
                    contradictions.append(
                        f"Says multiply but code divides: {variable}"
                    )
        
        return contradictions[:5]  # Limit to first 5
    
    def _error_handlers_documented(
        self,
        documentation: str,
        error_handlers: List[Dict]
    ) -> bool:
        """Check if error handlers are documented."""
        doc_lower = documentation.lower()
        
        error_keywords = ["error", "exception", "invalid", "failure", "abort"]
        
        # Check if any error-related terms appear
        return any(kw in doc_lower for kw in error_keywords)
