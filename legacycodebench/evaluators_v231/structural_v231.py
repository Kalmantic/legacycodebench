"""
Structural Completeness Evaluator V2.3.1 (Track 1)

Weight: 30% of total LCB Score

Evaluates whether the AI documented all important elements:
- Business Rules (40%)
- Data Structures (25%)
- Control Flow (20%)
- External Calls (15%)

Innovation: Uses TF-IDF with deterministic synonym expansion
to handle valid vocabulary variations without LLM calls.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import re
import logging
import math
from collections import Counter

from .config_v231 import V231_CONFIG
from .synonyms import expand_synonyms, normalize_for_matching, extract_cobol_identifiers

logger = logging.getLogger(__name__)


@dataclass
class StructuralResult:
    """Result of structural completeness evaluation."""
    score: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    matched_elements: Dict[str, List[str]] = field(default_factory=dict)
    missing_elements: Dict[str, List[str]] = field(default_factory=dict)
    details: Dict = field(default_factory=dict)


class StructuralEvaluatorV231:
    """
    V2.3.1 Structural Completeness Evaluator
    
    Pipeline:
    1. Expand synonyms in both doc and ground truth
    2. Apply TF-IDF matching
    3. Apply keyword matching as backup
    4. Combine: 60% TF-IDF + 40% keyword
    """
    
    def __init__(self):
        self.weights = V231_CONFIG["structural_weights"]
        self.matching = V231_CONFIG["matching"]
    
    def evaluate(
        self,
        documentation: str,
        ground_truth: Dict
    ) -> StructuralResult:
        """
        Evaluate structural completeness of documentation.
        
        Args:
            documentation: AI-generated documentation
            ground_truth: Ground truth with business_rules, data_structures, etc.
            
        Returns:
            StructuralResult with score and breakdown
        """
        logger.debug("Stage 1: Synonym Expansion [...]")
        # Step 1: Normalize and expand synonyms
        normalized_doc = normalize_for_matching(documentation)
        doc_upper = documentation.upper()
        logger.debug("Stage 1: Synonym Expansion [OK]")
        
        logger.debug("Stage 2: Evaluate Categories [...]")
        # Step 2: Evaluate each category
        br_score, br_matched, br_missing = self._evaluate_business_rules(
            normalized_doc, doc_upper, ground_truth.get("business_rules", {})
        )
        
        ds_score, ds_matched, ds_missing = self._evaluate_data_structures(
            normalized_doc, doc_upper, ground_truth.get("data_structures", {})
        )
        
        cf_score, cf_matched, cf_missing = self._evaluate_control_flow(
            normalized_doc, doc_upper, ground_truth.get("control_flow", {})
        )
        
        ec_score, ec_matched, ec_missing = self._evaluate_external_calls(
            normalized_doc, doc_upper, ground_truth.get("dependencies", {})
        )
        logger.debug("Stage 2: Evaluate Categories [OK]")
        
        logger.debug("Stage 3: Calculate Weighted Score [...]")
        # Step 3: Calculate weighted score
        overall = (
            self.weights["business_rules"] * br_score +
            self.weights["data_structures"] * ds_score +
            self.weights["control_flow"] * cf_score +
            self.weights["external_calls"] * ec_score
        )
        logger.debug("Stage 3: Calculate Weighted Score [OK]")
        
        return StructuralResult(
            score=overall,
            breakdown={
                "business_rules": br_score,
                "data_structures": ds_score,
                "control_flow": cf_score,
                "external_calls": ec_score,
            },
            matched_elements={
                "business_rules": br_matched,
                "data_structures": ds_matched,
                "control_flow": cf_matched,
                "external_calls": ec_matched,
            },
            missing_elements={
                "business_rules": br_missing,
                "data_structures": ds_missing,
                "control_flow": cf_missing,
                "external_calls": ec_missing,
            },
            details={
                "weights": self.weights,
                "method": "tfidf_synonym_expansion",
            }
        )
    
    def _evaluate_business_rules(
        self,
        normalized_doc: str,
        doc_upper: str,
        br_data
    ) -> tuple:
        """Evaluate business rules coverage."""
        # Handle nested format: business_rules.rules
        if isinstance(br_data, dict):
            rules_list = br_data.get("rules", [])
        elif isinstance(br_data, list):
            rules_list = br_data
        else:
            return 1.0, [], []
        
        if not rules_list:
            return 1.0, [], []
        
        matched = []
        missing = []
        total_weight = 0
        weighted_score = 0
        
        for rule in rules_list:
            if isinstance(rule, str):
                # String format: use as description
                rule_id = f"BR-{len(matched)+len(missing)+1:03d}"
                description = rule
                priority = "IMPORTANT"
                keywords = []
            elif isinstance(rule, dict):
                rule_id = rule.get("id", f"BR-{len(matched)+len(missing)+1:03d}")
                description = rule.get("description", "")
                priority = rule.get("priority", "IMPORTANT").upper()
                keywords = rule.get("keywords", [])
            else:
                continue
            
            # Weight by priority
            weight = {"CRITICAL": 3, "IMPORTANT": 2, "TRIVIAL": 1}.get(priority, 2)
            total_weight += weight
            
            # Check coverage using combined matching
            is_matched = self._check_coverage(
                normalized_doc, doc_upper, description, keywords
            )
            
            if is_matched:
                matched.append(rule_id)
                weighted_score += weight
            else:
                missing.append(rule_id)
        
        score = weighted_score / total_weight if total_weight > 0 else 1.0
        return score, matched, missing
    
    def _evaluate_data_structures(
        self,
        normalized_doc: str,
        doc_upper: str,
        ds_data
    ) -> tuple:
        """Evaluate data structures coverage."""
        # Handle nested format
        if isinstance(ds_data, dict):
            structures = ds_data.get("structures", [])
        elif isinstance(ds_data, list):
            structures = ds_data
        else:
            return 1.0, [], []
        
        if not structures:
            return 1.0, [], []
        
        matched = []
        missing = []
        
        for ds in structures:
            if isinstance(ds, str):
                name = ds
            elif isinstance(ds, dict):
                name = ds.get("name", "")
            else:
                continue
            
            if not name:
                continue
            
            # Check if name is mentioned
            if name.upper() in doc_upper or name.replace("-", " ").lower() in normalized_doc:
                matched.append(name)
            else:
                missing.append(name)
        
        score = len(matched) / len(structures) if structures else 1.0
        return score, matched, missing
    
    def _evaluate_control_flow(
        self,
        normalized_doc: str,
        doc_upper: str,
        cf_data
    ) -> tuple:
        """Evaluate control flow coverage."""
        # Handle nested format
        if isinstance(cf_data, dict):
            paragraphs = cf_data.get("paragraphs", [])
        elif isinstance(cf_data, list):
            paragraphs = cf_data
        else:
            return 1.0, [], []
        
        if not paragraphs:
            return 1.0, [], []
        
        matched = []
        missing = []
        
        for para in paragraphs:
            if isinstance(para, str):
                name = para
            elif isinstance(para, dict):
                name = para.get("name", "")
            else:
                continue
            
            if not name:
                continue
            
            # Check if paragraph is mentioned
            if name.upper() in doc_upper or name.replace("-", " ").lower() in normalized_doc:
                matched.append(name)
            else:
                missing.append(name)
        
        score = len(matched) / len(paragraphs) if paragraphs else 1.0
        return score, matched, missing
    
    def _evaluate_external_calls(
        self,
        normalized_doc: str,
        doc_upper: str,
        dep_data
    ) -> tuple:
        """Evaluate external calls coverage."""
        calls = []
        
        # Handle nested format
        if isinstance(dep_data, dict):
            # NEW: Check call_categories.external_dependency first
            call_categories = dep_data.get("call_categories", {})
            if isinstance(call_categories, dict):
                external_deps = call_categories.get("external_dependency", [])
                for call in external_deps:
                    if isinstance(call, dict):
                        callee = call.get("callee", "")
                        if callee:
                            calls.append({"name": callee, "type": "call"})
                
                # Also check middleware calls
                middleware = call_categories.get("middleware", [])
                for call in middleware:
                    if isinstance(call, dict):
                        callee = call.get("callee", "")
                        if callee:
                            calls.append({"name": callee, "type": "middleware"})
            
            # Check files.files for file operations
            files_data = dep_data.get("files", {})
            if isinstance(files_data, dict):
                file_calls = files_data.get("files", [])
                for f in file_calls:
                    if isinstance(f, dict):
                        name = f.get("name", f.get("file", ""))
                        if name:
                            calls.append({"name": name, "type": "file"})
                    elif isinstance(f, str):
                        calls.append({"name": f, "type": "file"})
            
            # Also check for explicit calls
            explicit_calls = dep_data.get("calls", [])
            for c in explicit_calls:
                if isinstance(c, dict):
                    calls.append(c)
                elif isinstance(c, str):
                    calls.append({"name": c, "type": "call"})
        elif isinstance(dep_data, list):
            calls = [{"name": c, "type": "call"} if isinstance(c, str) else c for c in dep_data]
        
        if not calls:
            return 1.0, [], []
        
        matched = []
        missing = []
        
        for call in calls:
            if isinstance(call, str):
                name = call
            elif isinstance(call, dict):
                name = call.get("name", "") or call.get("target", "") or call.get("callee", "")
            else:
                continue
            
            if not name:
                continue
            
            # Check if call target is mentioned
            if name.upper() in doc_upper:
                matched.append(name)
            else:
                missing.append(name)
        
        # Deduplicate
        matched = list(set(matched))
        missing = list(set(missing))
        
        score = len(matched) / (len(matched) + len(missing)) if (matched or missing) else 1.0
        return score, matched, missing
    
    def _check_coverage(
        self,
        normalized_doc: str,
        doc_upper: str,
        description: str,
        keywords: List[str]
    ) -> bool:
        """
        Check if an element is covered using combined matching.

        Strategy:
        1. TF-IDF similarity between description and doc
        2. Keyword presence check (with word boundaries)
        3. Combined: 60% TF-IDF + 40% keyword
        """
        # TF-IDF-like similarity
        desc_normalized = normalize_for_matching(description)
        tfidf_score = self._simple_similarity(desc_normalized, normalized_doc)

        # Keyword presence with WORD BOUNDARIES (prevents TAX matching SYNTAX)
        if keywords:
            found_keywords = sum(
                1 for kw in keywords
                if re.search(rf"\b{re.escape(kw.upper())}\b", doc_upper)
            )
            keyword_score = found_keywords / len(keywords)
        else:
            # If no keywords, extract from description
            desc_words = set(desc_normalized.split())
            # Use word boundaries for description words too
            found_words = sum(
                1 for w in desc_words
                if len(w) > 3 and re.search(rf"\b{re.escape(w)}\b", normalized_doc)
            )
            keyword_score = min(found_words / max(len(desc_words), 1), 1.0)
        
        # Combined score
        combined = (
            self.matching["tfidf_weight"] * tfidf_score +
            self.matching["keyword_weight"] * keyword_score
        )
        
        return combined >= self.matching["similarity_threshold"]
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """
        Simple TF-IDF-like similarity without sklearn dependency.
        
        Uses term frequency with inverse document frequency weighting.
        """
        if not text1 or not text2:
            return 0.0
        
        # Tokenize
        words1 = set(w for w in text1.split() if len(w) > 2)
        words2 = set(w for w in text2.split() if len(w) > 2)
        
        if not words1:
            return 0.0
        
        # Count matches
        matches = words1 & words2
        
        # Jaccard-ish similarity weighted by match count
        if not matches:
            return 0.0
        
        score = len(matches) / len(words1)
        return min(score, 1.0)
