"""
Claim Extractor V2.3.1

Extracts behavioral claims from documentation for verification.

Claim Types:
1. Calculation: "X is calculated by Y"
2. Conditional: "When X exceeds Y, Z is set"
3. Assignment: "Result is stored in X"
4. Range: "X must be between Y and Z"
5. Error: "If X fails, Y is triggered"

Primary: Regex extraction (deterministic)
Fallback: LLM extraction (capped at 10 claims, 15% impact)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import re
import logging

from .config_v231 import V231_CONFIG


logger = logging.getLogger(__name__)


class ClaimType(Enum):
    """Types of behavioral claims."""
    CALCULATION = "calculation"
    CONDITIONAL = "conditional"
    ASSIGNMENT = "assignment"
    RANGE = "range"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class Claim:
    """A behavioral claim extracted from documentation."""
    claim_id: str
    claim_type: ClaimType
    text: str
    output_var: Optional[str] = None
    input_vars: List[str] = field(default_factory=list)
    components: Dict = field(default_factory=dict)
    source: str = "regex"  # "regex" or "llm"
    confidence: float = 1.0


class ClaimExtractor:
    """
    Extract behavioral claims from documentation.
    
    Pipeline:
    1. Apply regex patterns for each claim type
    2. If < 3 claims, optionally use LLM fallback
    3. Deduplicate and normalize claims
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.patterns = V231_CONFIG["claim_patterns"]
        self.fallback_config = V231_CONFIG["llm_fallback"]
    
    def extract(self, documentation: str) -> List[Claim]:
        """
        Extract all behavioral claims from documentation.
        
        Args:
            documentation: AI-generated documentation
            
        Returns:
            List of extracted Claims
        """
        logger.debug("Stage 1: Regex Claim Extraction [...]")
        # Step 1: Regex extraction
        claims = self._extract_regex(documentation)
        logger.debug(f"Stage 1: Regex Claim Extraction [OK] ({len(claims)} claims)")
        
        # Step 2: LLM fallback if needed
        if (len(claims) < self.fallback_config["trigger_claim_count"] 
            and self.llm is not None 
            and self.fallback_config["enabled"]):
            logger.debug("Stage 2: LLM Fallback Extraction [...]")
            llm_claims = self._extract_llm(documentation)
            claims.extend(llm_claims)
            logger.debug(f"Stage 2: LLM Fallback Extraction [OK] ({len(llm_claims)} additional)")
        
        # Step 3: Deduplicate
        claims = self._deduplicate(claims)
        
        # Step 4: Assign IDs
        for i, claim in enumerate(claims):
            claim.claim_id = f"CLM-{i+1:03d}"
        
        logger.info(f"Extracted {len(claims)} claims ({sum(1 for c in claims if c.source == 'regex')} regex, {sum(1 for c in claims if c.source == 'llm')} LLM)")
        
        return claims
    
    def _extract_regex(self, documentation: str) -> List[Claim]:
        """
        Extract claims using regex patterns.
        """
        claims = []
        
        for claim_type_str, patterns in self.patterns.items():
            claim_type = ClaimType(claim_type_str)
            
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, documentation, re.IGNORECASE | re.MULTILINE)
                    
                    for match in matches:
                        claim = self._parse_match(claim_type, match, pattern)
                        if claim:
                            claims.append(claim)
                except re.error as e:
                    logger.warning(f"Regex error for pattern {pattern}: {e}")
        
        return claims
    
    def _parse_match(
        self,
        claim_type: ClaimType,
        match: re.Match,
        pattern: str
    ) -> Optional[Claim]:
        """
        Parse a regex match into a Claim.
        """
        groups = match.groups()
        text = match.group(0).strip()
        
        # Limit text length
        if len(text) > 500:
            text = text[:497] + "..."
        
        # Extract variables based on claim type
        output_var = None
        input_vars = []
        components = {}
        
        if claim_type == ClaimType.CALCULATION:
            if groups:
                output_var = groups[0] if groups[0] else None
                # Extract input variables from the formula/description
                input_vars = []
                for g in groups[1:]:
                    if g:
                        if self._is_variable(g):
                            input_vars.append(g)
                        else:
                            # Parse formula text to extract variable names
                            formula_vars = self._extract_variables_from_text(g)
                            input_vars.extend(formula_vars)
                components = {"operation": "calculation", "formula": groups[1] if len(groups) > 1 else None}
        
        elif claim_type == ClaimType.CONDITIONAL:
            if len(groups) >= 2:
                input_vars = [groups[0]] if groups[0] else []
                components = {
                    "condition_var": groups[0] if groups[0] else None,
                    "action": groups[-1] if groups[-1] else None,
                }
        
        elif claim_type == ClaimType.ASSIGNMENT:
            if groups:
                output_var = groups[0] if groups[0] else None
                components = {"operation": "assignment"}
        
        elif claim_type == ClaimType.RANGE:
            if len(groups) >= 3:
                output_var = groups[0] if groups[0] else None
                components = {
                    "min_value": groups[1] if len(groups) > 1 else None,
                    "max_value": groups[2] if len(groups) > 2 else None,
                }
        
        elif claim_type == ClaimType.ERROR:
            if groups:
                components = {
                    "trigger": groups[0] if groups[0] else None,
                    "action": groups[1] if len(groups) > 1 else None,
                }
        
        return Claim(
            claim_id="",  # Will be assigned later
            claim_type=claim_type,
            text=text,
            output_var=output_var,
            input_vars=input_vars,
            components=components,
            source="regex",
            confidence=0.9,
        )
    
    def _is_variable(self, text: str) -> bool:
        """
        Check if text looks like a COBOL variable name.
        """
        if not text:
            return False
        # COBOL variables: letters, digits, hyphens, start with letter
        return bool(re.match(r'^[A-Za-z][-A-Za-z0-9]*$', text.strip()))

    def _extract_variables_from_text(self, text: str) -> List[str]:
        """
        Extract COBOL variable names from formula text.

        Examples:
            "multiplying RATE by AMOUNT" → ["RATE", "AMOUNT"]
            "adding TAX to SUBTOTAL" → ["TAX", "SUBTOTAL"]
            "PRICE * QUANTITY" → ["PRICE", "QUANTITY"]

        Args:
            text: Formula description text

        Returns:
            List of variable names found in text
        """
        if not text:
            return []

        # Find all potential variable names (word characters with hyphens)
        # Match: START-WITH-LETTER, followed by letters/digits/hyphens
        potential_vars = re.findall(r'\b([A-Z][A-Z0-9\-]*)\b', text, re.IGNORECASE)

        # Filter out common English words and keywords
        keywords_to_skip = {
            'by', 'from', 'to', 'the', 'of', 'and', 'or', 'is', 'are', 'was', 'were',
            'in', 'on', 'at', 'for', 'with', 'as', 'if', 'then', 'else', 'when',
            'multiplying', 'adding', 'subtracting', 'dividing', 'calculating',
            'multiply', 'add', 'subtract', 'divide', 'calculate', 'computed',
            'determined', 'equals', 'equal', 'result', 'value', 'percent', 'percentage'
        }

        variables = []
        for var in potential_vars:
            var_clean = var.strip()
            # Skip if it's a keyword or too short
            if var_clean.lower() in keywords_to_skip or len(var_clean) < 2:
                continue
            # Skip if it's all digits
            if var_clean.isdigit():
                continue
            # Accept if it looks like a COBOL variable
            if self._is_variable(var_clean):
                variables.append(var_clean)

        return variables
    
    def _extract_llm(self, documentation: str) -> List[Claim]:
        """
        Extract claims using LLM fallback.
        
        IMPORTANT: This is capped at max_claims and logged for audit.
        """
        if not self.llm:
            return []
        
        prompt = """Extract behavioral claims from this COBOL documentation.
Return ONLY a JSON array with structure:
[
  {
    "type": "calculation|conditional|assignment|range|error",
    "text": "exact text from doc",
    "output": "output variable or null",
    "inputs": ["input variables"]
  }
]

Do NOT infer or generate claims not explicitly stated.
Maximum 10 claims.

Documentation:
""" + documentation[:3000]  # Limit input size
        
        try:
            response = self.llm.generate(prompt, temperature=0)
            claims = self._parse_llm_response(response)
            
            # Cap at max
            max_claims = self.fallback_config["max_claims"]
            if len(claims) > max_claims:
                claims = claims[:max_claims]
                logger.info(f"LLM claims capped at {max_claims}")
            
            # Mark as LLM-extracted
            for claim in claims:
                claim.source = "llm"
                claim.confidence = 0.7  # Lower confidence for LLM
            
            return claims
            
        except Exception as e:
            logger.warning(f"LLM claim extraction failed: {e}")
            return []
    
    def _parse_llm_response(self, response: str) -> List[Claim]:
        """
        Parse LLM JSON response into Claims.
        """
        import json
        
        claims = []
        
        try:
            # Find JSON array in response
            match = re.search(r'\[[\s\S]*\]', response)
            if not match:
                return []
            
            data = json.loads(match.group())
            
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                claim_type_str = item.get("type", "unknown")
                try:
                    claim_type = ClaimType(claim_type_str)
                except ValueError:
                    claim_type = ClaimType.UNKNOWN
                
                claims.append(Claim(
                    claim_id="",
                    claim_type=claim_type,
                    text=item.get("text", "")[:500],
                    output_var=item.get("output"),
                    input_vars=item.get("inputs", []),
                    components={},
                    source="llm",
                    confidence=0.7,
                ))
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
        
        return claims
    
    def _deduplicate(self, claims: List[Claim]) -> List[Claim]:
        """
        Remove duplicate claims based on text similarity.
        """
        seen = set()
        unique = []
        
        for claim in claims:
            # Normalize for comparison
            normalized = re.sub(r'\s+', ' ', claim.text.lower().strip())
            
            if normalized not in seen:
                seen.add(normalized)
                unique.append(claim)
        
        return unique
