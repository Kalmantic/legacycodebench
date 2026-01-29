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
                # Extract variables from condition
                if groups[0] and self._is_variable(groups[0]):
                    input_vars = [groups[0]]
                elif groups[0]:
                    # Try to extract variables from condition text
                    input_vars = self._extract_variables_from_text(groups[0])
                else:
                    input_vars = []
                
                # Also extract variables from the action (last group)
                action = groups[-1] if groups[-1] else None
                if action:
                    action_vars = self._extract_variables_from_text(action)
                    # First action var is likely the output
                    if action_vars:
                        output_var = action_vars[0]
                        input_vars.extend(action_vars[1:])
                
                components = {
                    "condition_var": groups[0] if groups[0] else None,
                    "action": action,
                }
        
        elif claim_type == ClaimType.ASSIGNMENT:
            if groups:
                output_var = groups[0] if groups[0] else None
                # Extract input variables from the value being assigned (group 1)
                if len(groups) > 1 and groups[1]:
                    input_vars = self._extract_variables_from_text(groups[1])
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
                # Extract variables from both trigger and action text
                trigger = groups[0] if groups[0] else None
                action = groups[1] if len(groups) > 1 else None
                
                # Try to find variables in trigger (e.g., "if FILE-STATUS fails")
                if trigger:
                    trigger_vars = self._extract_variables_from_text(trigger)
                    input_vars.extend(trigger_vars)
                
                # Extract variables from action text (e.g., "increments WS-NO-CHKP")
                if action:
                    action_vars = self._extract_variables_from_text(action)
                    # First var could be output, rest are inputs
                    if action_vars:
                        output_var = action_vars[0]
                        input_vars.extend(action_vars[1:])
                
                components = {
                    "trigger": trigger,
                    "action": action,
                }
        
        # Validate extracted variables - reject if they're not valid COBOL names
        if output_var and not self._is_variable(output_var):
            output_var = None  # Clear invalid output variable
        
        # Filter input_vars to only include valid COBOL variables
        input_vars = [v for v in input_vars if self._is_variable(v)]
        
        # V3.2 FALLBACK: If no variables found, scan the entire claim text
        # This catches cases where regex groups don't capture the full variable
        # e.g., "CBL-authorization-panel validates user input" -> captures full hyphenated var
        if not output_var and not input_vars:
            all_vars = self._extract_variables_from_text(text)
            if all_vars:
                # First variable is likely the subject/output
                output_var = all_vars[0]
                # Rest are inputs
                input_vars = all_vars[1:] if len(all_vars) > 1 else []
                logger.debug(f"Fallback extracted vars from full text: {all_vars}")
        
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
    
    # Common English words that should NOT be extracted as COBOL variables
    ENGLISH_STOPWORDS = {
        # Articles, prepositions, conjunctions
        'a', 'an', 'the', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'from', 'with',
        'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'shall', 'can', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
        'where', 'which', 'that', 'this', 'these', 'those', 'it', 'its', 'they',
        
        # Common verbs and words in documentation
        'there', 'here', 'all', 'each', 'every', 'some', 'any', 'no', 'not',
        'more', 'most', 'other', 'such', 'only', 'same', 'so', 'than', 'too',
        'very', 'just', 'also', 'now', 'new', 'first', 'last', 'long', 'little',
        'own', 'old', 'right', 'big', 'high', 'different', 'small', 'large',
        'next', 'early', 'young', 'important', 'few', 'public', 'bad', 'good',
        'set', 'used', 'get', 'put', 'made', 'found', 'given', 'shown',
        'additional', 'displayed', 'initialized', 'incremented', 'decremented',
        'several', 'crucial', 'specific', 'various', 'certain', 'particular',
        'calculated', 'computed', 'determined', 'expired', 'processed', 'handled',
        'based', 'stored', 'saved', 'loaded', 'retrieved', 'checked', 'validated',
        'compared', 'matched', 'equals', 'exceeds', 'contains', 'includes',
        'trigger', 'triggers', 'handles', 'counter', 'handling', 'triggered',
        
        # Programming/documentation terms
        'file', 'files', 'data', 'code', 'program', 'programs', 'function',
        'value', 'values', 'result', 'results', 'output', 'input', 'type',
        'structure', 'structures', 'section', 'sections', 'record', 'records',
        'field', 'fields', 'variable', 'variables', 'operation', 'operations',
        'process', 'processing', 'calculation', 'calculations', 'total', 'count',
        'number', 'name', 'status', 'flag', 'error', 'message', 'information',
        'description', 'example', 'note', 'see', 'using', 'following',
        'authorization', 'authorizations', 'transaction', 'transactions',
        'checkpoint', 'checkpoints', 'determine', 'determines', 'increments',
        
        # COBOL keywords (should not be variable names)
        'perform', 'move', 'add', 'subtract', 'multiply', 'divide', 'compute',
        'display', 'accept', 'read', 'write', 'open', 'close', 'call', 'stop',
        'working', 'storage', 'linkage', 'procedure', 'division',

        # Common verbs used in calculation descriptions
        'multiplying', 'dividing', 'adding', 'subtracting', 'calculating',
        'computing', 'determining', 'storing', 'retrieving', 'processing',

        # File extensions and common fragments
        'cbl', 'cpy', 'txt', 'md', 'json', 'xml', 'sql', 'jcl',

        # V2.4 FIX: Common uppercase words that look like COBOL vars but aren't
        # These appear frequently in documentation and get misidentified
        'final', 'stock', 'terminate', 'details', 'runtime', 'issues',
        'undefined', 'invalid', 'complete', 'success', 'failure', 'active',
        'pending', 'current', 'previous', 'default', 'primary', 'secondary',
        'normal', 'special', 'standard', 'custom', 'system', 'user',
        'main', 'temp', 'test', 'debug', 'trace', 'level', 'mode',
        'start', 'begin', 'ending', 'finish', 'done', 'ready', 'wait',
        'true', 'false', 'null', 'none', 'empty', 'blank', 'zero',
        'positive', 'negative', 'numeric', 'alpha', 'string', 'text',
        'length', 'size', 'width', 'height', 'index', 'offset', 'position',
        'source', 'target', 'destination', 'origin', 'path', 'location',
    }
    
    def _is_variable(self, text: str) -> bool:
        """
        Check if text looks like a COBOL variable name.
        
        COBOL variables typically:
        - Start with a letter
        - Contain letters, digits, and hyphens
        - Often contain HYPHENS (WS-AMOUNT, CUST-ID, P-DEBUG-FLAG)
        - Often have 2-letter prefixes (WS-, WK-, PA-, SW-)
        - Are usually UPPERCASE (or extracted from backticks)
        - Are NOT common English words
        """
        if not text:
            return False
        
        text = text.strip()
        
        # Must match basic COBOL variable pattern
        if not re.match(r'^[A-Za-z][-A-Za-z0-9]*$', text):
            return False
        
        # Reject if it's a common English word (case-insensitive)
        if text.lower() in self.ENGLISH_STOPWORDS:
            return False
        
        # Reject single character
        if len(text) < 2:
            return False
        
        # COBOL variables have STRONG indicators:
        # 1. Contains hyphen (most common: WS-VAR, P-DEBUG-FLAG, CUST-ID)
        # 2. Is ALL CAPS (AMOUNT, CUSTID, WS)
        # 3. Contains digit (CUST1, VAR99)
        # 4. Has common COBOL prefix
        has_hyphen = '-' in text
        is_upper = text.isupper()
        has_digit = any(c.isdigit() for c in text)
        
        # Common COBOL prefixes (2-3 chars followed by hyphen)
        cobol_prefixes = ('WS-', 'WK-', 'SW-', 'PA-', 'PV-', 'LS-', 'LK-', 'FD-', 
                         'SD-', 'WA-', 'WB-', 'WC-', 'WD-', 'WE-', 'WF-', 'W0-', 
                         'W1-', 'W2-', 'W3-', 'WI-', 'WO-', 'WR-', 'WP-', 'CI-',
                         'EI-', 'DF-', 'CP-', 'DI-', 'CB-', 'P-', 'H-', 'I-', 'O-')
        has_cobol_prefix = text.upper().startswith(cobol_prefixes)
        
        # If it has COBOL-like characteristics, accept it
        if has_hyphen or has_cobol_prefix or has_digit:
            return True
        
        # For ALL CAPS words, accept if they're at least 4 chars
        # (to avoid false positives like "SQL", "IMS", etc.)
        if is_upper and len(text) >= 4:
            return True

        # V2.4 FIX: Accept any identifier >= 4 chars that passed stopword check
        # This handles mixed-case variables from markdown (StockValue, QtyInStock)
        # COBOL is case-insensitive, so these are valid when normalized to uppercase
        if len(text) >= 4:
            return True

        # Otherwise, reject - it's probably an English word
        return False

    def _extract_variables_from_text(self, text: str) -> List[str]:
        """
        Extract COBOL variable names from formula text.
        
        V3.2 Enhanced: Better extraction of hyphenated variables like CBL-authorization-panel.

        Examples:
            "multiplying RATE by AMOUNT" → ["RATE", "AMOUNT"]
            "adding TAX to SUBTOTAL" → ["TAX", "SUBTOTAL"]
            "PRICE * QUANTITY" → ["PRICE", "QUANTITY"]
            "CBL-authorization-panel validates" → ["CBL-AUTHORIZATION-PANEL"]

        Args:
            text: Formula description text

        Returns:
            List of variable names found in text
        """
        if not text:
            return []

        variables = []
        
        # Strategy 1: Extract hyphenated identifiers (highest priority - most COBOL-like)
        # Pattern: word-word-word (handles mixed case like CBL-authorization-panel)
        hyphenated_pattern = r'\b([A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+)\b'
        hyphenated_vars = re.findall(hyphenated_pattern, text)
        for var in hyphenated_vars:
            var_upper = var.upper()
            if self._is_variable(var_upper) and var_upper not in variables:
                variables.append(var_upper)
        
        # Strategy 2: Extract ALL_CAPS words (common COBOL style)
        caps_pattern = r'\b([A-Z][A-Z0-9]{2,})\b'  # At least 3 chars total
        caps_vars = re.findall(caps_pattern, text)
        for var in caps_vars:
            if self._is_variable(var) and var not in variables:
                variables.append(var)
        
        # Strategy 3: Extract backtick-quoted identifiers (common in markdown docs)
        # e.g., `WS-AMOUNT` or `CUSTOMER-ID`
        backtick_pattern = r'`([A-Za-z][A-Za-z0-9\-]+)`'
        backtick_vars = re.findall(backtick_pattern, text)
        for var in backtick_vars:
            var_upper = var.upper()
            if self._is_variable(var_upper) and var_upper not in variables:
                variables.append(var_upper)
        
        # Strategy 4: Legacy fallback - any COBOL-like word
        # Only if we haven't found anything yet
        if not variables:
            potential_vars = re.findall(r'\b([A-Z][A-Z0-9\-]*)\b', text, re.IGNORECASE)
            for var in potential_vars:
                var_clean = var.strip().upper()
                if var_clean.isdigit():
                    continue
                if self._is_variable(var_clean) and var_clean not in variables:
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
