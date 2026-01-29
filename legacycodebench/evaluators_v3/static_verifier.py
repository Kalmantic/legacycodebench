"""
Static Verification for Behavioral Fidelity

Verifies documentation claims WITHOUT execution, using:
1. Variable existence checking against ground truth
2. TF-IDF similarity to ground truth rules
3. BSM (Behavioral Specification Matching) for external calls

KEY DIFFERENCE FROM V2.3.1 HEURISTIC:
- This can actually FAIL bad documentation
- 100% pass rate is NOT possible with proper verification
- Every verification step has clear pass/fail criteria
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from collections import Counter
import math

logger = logging.getLogger(__name__)


# ==================== CONFIGURATION ====================

# TF-IDF similarity thresholds
# 
# THRESHOLD RATIONALE:
# - 0.50 (verified): Conservative threshold. At 0.50+ similarity, the claim
#   and ground truth share significant vocabulary overlap, indicating the
#   documentation describes the same concept as the ground truth.
# - 0.30 (partial): Loose threshold for partial credit. At 0.30-0.49, there's
#   some overlap but not enough confidence for full verification.
# - Below 0.30: Insufficient similarity to claim any match.
#
# TODO: These thresholds should be validated empirically with a labeled
# dataset of known-good and known-bad documentation. Current values are
# based on initial testing with ~20 sample claims.
THRESHOLD_VERIFIED = 0.50
THRESHOLD_PARTIAL = 0.30


@dataclass
class ClaimVerificationResult:
    """Result of verifying a single claim."""
    claim_text: str
    status: str  # "verified", "partial", "failed", "unverified"
    confidence: float  # 0.0 - 1.0
    method: str  # How it was verified (variable_check, tfidf_match, etc.)
    reason: str  # Human-readable explanation


@dataclass
class StaticVerificationResult:
    """Complete result of static verification."""
    claim_score: float  # 0.0 - 1.0
    bsm_score: float  # 0.0 - 1.0
    
    claims_verified: int
    claims_failed: int
    claims_total: int
    claim_details: List[ClaimVerificationResult]
    
    bsm_matched: int
    bsm_total: int
    bsm_details: List[Dict]


class StaticVerifier:
    """
    Verify documentation claims using static analysis.
    
    This is the honest replacement for the old _verify_heuristic() that
    would return verified=ALL, failed=0. This verifier can actually FAIL
    bad documentation.
    
    Usage:
        verifier = StaticVerifier()
        result = verifier.verify(claims, ground_truth, source_code, documentation)
        # result.claim_score might be 0.67 (not automatically 1.0!)
    """
    
    def __init__(self):
        """Initialize the static verifier."""
        self.bsm_validator = BSMValidator()
    
    def verify(
        self,
        claims: List[Any],  # Claim objects from ClaimExtractor
        ground_truth: Dict,
        source_code: str,
        documentation: str,
    ) -> StaticVerificationResult:
        """
        Verify claims and external calls statically.
        
        Args:
            claims: List of Claim objects extracted from documentation
            ground_truth: Static analysis ground truth
            source_code: Original COBOL source
            documentation: AI-generated documentation
            
        Returns:
            StaticVerificationResult with scores and details
        """
        logger.info(f"Static verification: {len(claims)} claims")
        
        # Step 1: Verify claims against ground truth
        claim_score, claim_details = self.verify_claims(
            claims, ground_truth, source_code
        )
        
        # Step 2: BSM validation for external calls
        external_calls = ground_truth.get("external_calls", [])
        bsm_result = self.bsm_validator.validate(documentation, external_calls)
        
        # Count verified/failed
        verified = sum(1 for d in claim_details if d.status == "verified")
        partial = sum(1 for d in claim_details if d.status == "partial")
        failed = sum(1 for d in claim_details if d.status == "failed")
        
        # Partial counts as 0.5 verified
        verified_count = verified + (partial * 0.5)
        
        return StaticVerificationResult(
            claim_score=claim_score,
            bsm_score=bsm_result["score"],
            
            claims_verified=int(verified_count),
            claims_failed=failed,
            claims_total=len(claims),
            claim_details=claim_details,
            
            bsm_matched=bsm_result["matched"],
            bsm_total=bsm_result["total"],
            bsm_details=bsm_result.get("details", []),
        )
    
    def verify_claims(
        self,
        claims: List[Any],
        ground_truth: Dict,
        source_code: str
    ) -> Tuple[float, List[ClaimVerificationResult]]:
        """
        Verify extracted claims against ground truth.
        
        Uses three-tier verification:
        1. Variable existence check
        2. TF-IDF similarity to ground truth rules
        3. (Future) Contradiction detection
        
        Args:
            claims: List of Claim objects
            ground_truth: Static analysis ground truth
            source_code: Original COBOL source
            
        Returns:
            (score, details) tuple
        """
        if not claims:
            logger.info("No claims to verify")
            return 1.0, []
        
        # Extract ground truth components
        gt_variables = self._extract_all_variables(ground_truth)
        
        # Handle nested business_rules structure
        # Ground truth stores rules as: gt['business_rules']['business_rules'] (list)
        br = ground_truth.get("business_rules", {})
        if isinstance(br, dict):
            gt_rules = br.get("business_rules", []) or br.get("rules", [])
        elif isinstance(br, list):
            gt_rules = br
        else:
            gt_rules = []
        
        results = []
        verified = 0
        partial = 0
        failed = 0
        
        for claim in claims:
            result = self._verify_single_claim(
                claim, gt_variables, gt_rules, source_code
            )
            results.append(result)
            
            if result.status == "verified":
                verified += 1
            elif result.status == "partial":
                partial += 1
            elif result.status == "failed":
                failed += 1
        
        # V2.4.2: Hybrid claim scoring
        from ..evaluators_v231.config_v231 import get_hybrid_scoring_config, get_claim_target

        config = get_hybrid_scoring_config()
        partial_weight = config.get("partial_weight", 0.5)
        effective = verified + partial * partial_weight

        if config.get("enabled", True):
            target = get_claim_target()
            score = min(effective / target, 1.0) if target > 0 else 0.0
        else:
            # Legacy formula
            total = len(claims)
            score = effective / total if total > 0 else 0.0

        logger.info(
            f"Claim verification: {verified} verified, {partial} partial, "
            f"{failed} failed, effective={effective:.1f}, score={score:.2f}"
        )
        
        return score, results
    
    def _verify_single_claim(
        self,
        claim: Any,
        gt_variables: set,
        gt_rules: List[Dict],
        source_code: str
    ) -> ClaimVerificationResult:
        """
        Verify a single claim.
        
        Args:
            claim: Claim object with text, output_var, input_vars
            gt_variables: Set of variable names from ground truth
            gt_rules: List of business rules from ground truth
            source_code: Original COBOL source
            
        Returns:
            ClaimVerificationResult
        """
        claim_text = getattr(claim, 'text', str(claim))
        output_var = getattr(claim, 'output_var', None)
        input_vars = getattr(claim, 'input_vars', []) or []
        
        # Tier 1: Variable existence check
        if output_var:
            if output_var.upper() not in gt_variables:
                return ClaimVerificationResult(
                    claim_text=claim_text[:100],
                    status="failed",
                    confidence=0.0,
                    method="variable_check",
                    reason=f"Output variable {output_var} not found in ground truth"
                )
        
        if input_vars:
            missing = [v for v in input_vars if v.upper() not in gt_variables]
            if missing:
                return ClaimVerificationResult(
                    claim_text=claim_text[:100],
                    status="failed",
                    confidence=0.0,
                    method="variable_check",
                    reason=f"Input variables not found: {missing}"
                )
        
        # Tier 2: TF-IDF similarity to ground truth rules
        # Include both descriptions AND keywords for better matching
        if gt_rules:
            gt_texts = []
            for r in gt_rules:
                if isinstance(r, dict):
                    # Add description
                    desc = r.get("description", "")
                    if desc:
                        gt_texts.append(desc)
                    
                    # Also add keywords as a text (important for variable matching!)
                    keywords = r.get("keywords", [])
                    if keywords:
                        gt_texts.append(" ".join(str(k) for k in keywords))
            
            if gt_texts:
                similarity = max(
                    self._tfidf_similarity(claim_text, gt_text) 
                    for gt_text in gt_texts
                )
                
                if similarity >= THRESHOLD_VERIFIED:
                    return ClaimVerificationResult(
                        claim_text=claim_text[:100],
                        status="verified",
                        confidence=similarity,
                        method="tfidf_match",
                        reason=f"Strong match to ground truth rule (sim={similarity:.2f})"
                    )
                elif similarity >= THRESHOLD_PARTIAL:
                    return ClaimVerificationResult(
                        claim_text=claim_text[:100],
                        status="partial",
                        confidence=similarity,
                        method="tfidf_partial",
                        reason=f"Partial match to ground truth rule (sim={similarity:.2f})"
                    )
        
        # Tier 3: Variable existence gives partial credit even without TF-IDF match
        # If we have valid variables but no rule match, that's still worth something
        if output_var or input_vars:
            return ClaimVerificationResult(
                claim_text=claim_text[:100],
                status="partial",
                confidence=0.4,
                method="variable_existence_only",
                reason="Variables exist in ground truth, but no matching rule description"
            )
        
        # No ground truth rules AND no variables - can't verify
        return ClaimVerificationResult(
            claim_text=claim_text[:100],
            status="unverified",
            confidence=0.2,
            method="no_verification",
            reason="No variables or matching rules to verify against"
        )
    
    def _extract_all_variables(self, ground_truth: Dict) -> set:
        """
        Extract all variable names from ground truth.
        
        Args:
            ground_truth: Static analysis ground truth
            
        Returns:
            Set of variable names (uppercase)
        """
        variables = set()
        
        # Handle nested data_structures structure
        # Ground truth stores as: gt['data_structures']['data_structures'] (list)
        ds_section = ground_truth.get("data_structures", {})
        if isinstance(ds_section, dict):
            ds_list = ds_section.get("data_structures", [])
        elif isinstance(ds_section, list):
            ds_list = ds_section
        else:
            ds_list = []
        
        # From data structures
        for ds in ds_list:
            if isinstance(ds, dict):
                name = ds.get("name", "")
                if name:
                    variables.add(name.upper())
                
                # Also add field names
                for field in ds.get("fields", []):
                    if isinstance(field, dict):
                        field_name = field.get("name", "")
                        if field_name:
                            variables.add(field_name.upper())
                    elif isinstance(field, str):
                        variables.add(field.upper())
        
        # Handle nested business_rules structure
        br_section = ground_truth.get("business_rules", {})
        if isinstance(br_section, dict):
            br_list = br_section.get("business_rules", []) or br_section.get("rules", [])
        elif isinstance(br_section, list):
            br_list = br_section
        else:
            br_list = []
        
        # From business rules - extract variables from keywords
        for rule in br_list:
            if isinstance(rule, dict):
                for var in rule.get("variables", []):
                    variables.add(var.upper())
                # Also try keywords which contain variable names
                for kw in rule.get("keywords", []):
                    if isinstance(kw, str) and kw.isupper():
                        variables.add(kw)
        
        return variables
    
    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate TF-IDF cosine similarity between two texts.
        
        Simple implementation that tokenizes and computes TF-IDF vectors.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity (0.0 - 1.0)
        """
        # Tokenize
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Build vocabulary
        vocab = set(tokens1) | set(tokens2)
        
        # Calculate TF (term frequency)
        tf1 = Counter(tokens1)
        tf2 = Counter(tokens2)
        
        # Calculate TF-IDF vectors (simplified - no IDF from corpus)
        # Just use TF * log(2) for common terms
        vec1 = []
        vec2 = []
        
        for term in vocab:
            v1 = tf1.get(term, 0) / len(tokens1) if tokens1 else 0
            v2 = tf2.get(term, 0) / len(tokens2) if tokens2 else 0
            vec1.append(v1)
            vec2.append(v2)
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for TF-IDF.
        
        Converts to lowercase, removes punctuation, splits on whitespace.
        Removes common stopwords.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Split on whitespace
        tokens = text.split()
        
        # Remove stopwords
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
            'if', 'or', 'because', 'until', 'while', 'this', 'that',
            'these', 'those', 'it', 'its', 'they', 'them', 'their',
        }
        
        tokens = [t for t in tokens if t and t not in stopwords and len(t) > 1]
        
        return tokens


class BSMValidator:
    """
    Behavioral Specification Matching for external calls.
    
    Validates that documentation correctly describes:
    - SQL operations
    - CICS commands
    - CALL statements
    - File operations
    """
    
    def __init__(self):
        """Initialize BSM validator with default patterns."""
        self.patterns = {
            "sql": {"doc_keywords": ["database", "sql", "query", "select", "insert", "update", "delete", "table"]},
            "cics": {"doc_keywords": ["cics", "transaction", "screen", "map", "send", "receive"]},
            "call": {"doc_keywords": ["call", "invoke", "subroutine", "subprogram", "module"]},
            "file": {"doc_keywords": ["file", "read", "write", "open", "close", "record"]},
        }
    
    def validate(
        self,
        documentation: str,
        external_calls: List[Dict]
    ) -> Dict:
        """
        Validate external calls against documentation.
        
        Args:
            documentation: AI-generated documentation
            external_calls: List of external calls from ground truth
            
        Returns:
            Dict with score and details
        """
        if not external_calls:
            return {"score": 1.0, "matched": 0, "total": 0, "details": []}
        
        doc_upper = documentation.upper()
        doc_lower = documentation.lower()
        
        matched = 0
        details = []
        
        for call in external_calls:
            if isinstance(call, str):
                target = call
                call_type = "unknown"
            elif isinstance(call, dict):
                target = call.get("target", call.get("name", ""))
                call_type = call.get("type", "unknown")
            else:
                continue
            
            if not target:
                continue
            
            # Check if target is mentioned
            target_found = target.upper() in doc_upper
            
            # Fuzzy matching for file operations
            if not target_found:
                target_found = self._fuzzy_match(target, doc_upper, doc_lower)
            
            # Check for pattern keywords
            pattern_info = self.patterns.get(call_type.lower(), {})
            keywords = pattern_info.get("doc_keywords", [])
            keyword_found = any(kw in doc_lower for kw in keywords) if keywords else True
            
            is_matched = target_found and keyword_found
            
            if is_matched:
                matched += 1
            
            details.append({
                "target": target,
                "type": call_type,
                "matched": is_matched,
                "target_found": target_found,
                "keyword_found": keyword_found,
            })
        
        score = matched / len(external_calls)
        
        return {
            "score": score,
            "matched": matched,
            "total": len(external_calls),
            "details": details,
        }
    
    def _fuzzy_match(self, target: str, doc_upper: str, doc_lower: str) -> bool:
        """
        Fuzzy match for file/program names.
        
        Examples:
            CUSTOMER-FILE -> matches CUSTOMERS, CUSTOMER-DATA
            SCUSTOMP -> matches SCUSTOM, CUSTOMER
        """
        target_upper = target.upper()
        
        # Try base name without common suffixes
        base_patterns = [
            target_upper.replace('-FILE', ''),
            target_upper.replace('-RECORD', ''),
            target_upper.replace('-DATA', ''),
            target_upper[:6] if len(target_upper) > 6 else target_upper,  # First 6 chars
        ]
        
        for pattern in base_patterns:
            if pattern and len(pattern) >= 4 and pattern in doc_upper:
                return True
        
        return False
