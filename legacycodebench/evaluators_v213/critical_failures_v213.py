"""
Critical Failure Detector V2.3.1

Detects conditions that result in task failure (score = 0).

Key Innovation: ZERO TOLERANCE for I/O Hallucinations
- CF-02: ANY I/O variable that doesn't exist -> immediate failure
- CF-02b: ≥3 internal variables -> failure (lenient)

Critical Failures:
- CF-01: 0% CRITICAL business rules documented
- CF-02: ANY I/O variable hallucinated (Zero Tolerance)
- CF-02b: ≥3 internal variables hallucinated
- CF-03: ≥50% claims fail execution
- CF-04: Error handlers exist but undocumented
- CF-05: >50% BSM pattern failures
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import re
import logging

from .config_v213 import V213_CONFIG
from .synonyms import extract_cobol_identifiers, fuzzy_match_identifier


logger = logging.getLogger(__name__)


@dataclass
class CriticalFailure:
    """A critical failure that results in score = 0."""
    cf_id: str
    name: str
    description: str
    severity: str = "critical"


class CriticalFailureDetectorV213:
    """
    V2.3.1 Critical Failure Detector
    
    Key Innovation: Zero Tolerance for I/O Hallucinations
    """
    
    def __init__(self):
        self.cf_config = V213_CONFIG["critical_failures"]
        self.hallucination_config = V213_CONFIG["hallucination"]
    
    def detect_all(
        self,
        documentation: str,
        ground_truth: Dict,
        sc_score: float = 1.0,
        bf_result: Optional[Dict] = None
    ) -> List[CriticalFailure]:
        """
        Detect all critical failures.
        
        Args:
            documentation: AI-generated documentation
            ground_truth: Ground truth data
            sc_score: Structural completeness score (for CF-01)
            bf_result: Behavioral fidelity result (for CF-03, CF-05)
            
        Returns:
            List of CriticalFailures (any = score 0)
        """
        failures = []
        
        logger.debug("CF-01 Check: Missing Core Calculations")
        cf01 = self._detect_cf01(documentation, ground_truth)
        if cf01:
            failures.append(cf01)
            logger.debug(f"CF-01 DETECTED: {cf01.description}")
        else:
            logger.debug("CF-01 Check: PASSED")
        
        logger.debug("CF-02 Check: Hallucinated I/O Variables")
        cf02 = self._detect_cf02_zero_tolerance(documentation, ground_truth)
        if cf02:
            failures.append(cf02)
            logger.debug(f"{cf02.cf_id} DETECTED: {cf02.description}")
        else:
            logger.debug("CF-02 Check: PASSED")
        
        logger.debug("CF-03 Check: Behavioral Contradiction")
        cf03 = self._detect_cf03(bf_result)
        if cf03:
            failures.append(cf03)
            logger.debug(f"CF-03 DETECTED: {cf03.description}")
        else:
            logger.debug("CF-03 Check: PASSED")
        
        logger.debug("CF-04 Check: Missing Error Handlers")
        cf04 = self._detect_cf04(documentation, ground_truth)
        if cf04:
            failures.append(cf04)
            logger.debug(f"CF-04 DETECTED: {cf04.description}")
        else:
            logger.debug("CF-04 Check: PASSED")
        
        logger.debug("CF-05 Check: BSM Pattern Failures")
        cf05 = self._detect_cf05(bf_result)
        if cf05:
            failures.append(cf05)
            logger.debug(f"CF-05 DETECTED: {cf05.description}")
        else:
            logger.debug("CF-05 Check: PASSED")
        
        if failures:
            logger.warning(f"Critical failures detected: {[f.cf_id for f in failures]}")
        
        return failures
    
    def _detect_cf01(
        self,
        documentation: str,
        ground_truth: Dict
    ) -> Optional[CriticalFailure]:
        """
        CF-01: Missing Core Calculations (0% CRITICAL rules documented)
        """
        br_data = ground_truth.get("business_rules", {})
        
        if isinstance(br_data, dict):
            rules = br_data.get("rules", [])
        elif isinstance(br_data, list):
            rules = br_data
        else:
            return None
        
        # Filter to CRITICAL rules
        critical_rules = []
        for rule in rules:
            if isinstance(rule, dict):
                if rule.get("priority", "").upper() == "CRITICAL":
                    critical_rules.append(rule)
        
        if not critical_rules:
            return None  # No critical rules to check
        
        doc_upper = documentation.upper()
        
        # Check if ANY critical rule is documented
        documented = 0
        for rule in critical_rules:
            keywords = rule.get("keywords", [])
            description = rule.get("description", "")
            
            # Check keywords using word boundaries to avoid substring trap
            # e.g., "TAX" should not match "SYNTAX"
            if keywords:
                found = 0
                for kw in keywords:
                    # Use word boundary regex to match whole words only
                    pattern = rf"\b{re.escape(kw.upper())}\b"
                    if re.search(pattern, doc_upper):
                        found += 1
                if found >= min(2, len(keywords)):
                    documented += 1
                    continue
            
            # Check description terms using word boundaries
            if description:
                desc_words = set(description.upper().split())
                found = 0
                for w in desc_words:
                    if len(w) > 3:
                        pattern = rf"\b{re.escape(w)}\b"
                        if re.search(pattern, doc_upper):
                            found += 1
                if found >= 3:
                    documented += 1
        
        if documented == 0:
            return CriticalFailure(
                cf_id="CF-01",
                name="Missing Core Calculations",
                description=f"0/{len(critical_rules)} CRITICAL business rules documented",
            )
        
        return None
    
    def _detect_cf02_zero_tolerance(
        self,
        documentation: str,
        ground_truth: Dict
    ) -> Optional[CriticalFailure]:
        """
        CF-02: Zero Tolerance for I/O Hallucinations
        CF-02b: ≥3 Internal Variable Hallucinations
        
        Innovation: Zero tolerance for I/O, lenient for internals.
        """
        # Build set of valid identifiers from ground truth
        valid_vars = self._extract_valid_identifiers(ground_truth)
        
        if not valid_vars:
            return None  # Can't validate without ground truth
        
        # Extract identifiers mentioned in documentation
        doc_identifiers = extract_cobol_identifiers(documentation)
        
        # Find I/O claims - expanded with robust data movement verbs
        io_patterns = [
            # Read/Input patterns
            r"(\w+(?:-\w+)*)\s+is\s+(?:read|input|loaded|retrieved|fetched)\s+from",
            r"(?:reads?|inputs?|loads?|retrieves?|fetches?|gets?)\s+(\w+(?:-\w+)*)\s+from",
            # Write/Output patterns
            r"(\w+(?:-\w+)*)\s+is\s+(?:written|output|saved|stored|placed)\s+to",
            r"(?:writes?|outputs?|saves?|stores?|places?|puts?)\s+(\w+(?:-\w+)*)\s+to",
            # Return/Output patterns
            r"(?:returns?|outputs?|produces?)\s+(\w+(?:-\w+)*)",
            # Container patterns
            r"(\w+(?:-\w+)*)\s+(?:contains?|holds?|stores?)\s+the\s+(?:input|output|result|data)",
            # Receives/Sends patterns
            r"(?:receives?|sends?)\s+(\w+(?:-\w+)*)",
            r"(\w+(?:-\w+)*)\s+is\s+(?:received|sent)\s+(?:from|to)",
        ]
        
        io_vars = set()
        for pattern in io_patterns:
            matches = re.findall(pattern, documentation, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    for m in match:
                        if m:
                            io_vars.add(m.upper())
                else:
                    io_vars.add(match.upper())
        
        # Check I/O variables with zero tolerance
        # ROBUST APPROACH: Only flag identifiers that look like actual COBOL variable names
        # Real COBOL vars typically: contain hyphens (WS-CUSTOMER-ID), are 5+ chars, all UPPER
        
        # Common COBOL paragraph name patterns - NOT I/O variables
        # These are procedural names, not data hallucinations
        paragraph_patterns = [
            "END-OF-", "START-OF-", "BEGIN-", "FINISH-", 
            "INIT-", "TERM-", "PROCESS-", "HANDLE-",
            "-ROUTINE", "-PROCESSING", "-LOGIC", "-SECTION",
            "-PARAGRAPH", "-PROC", "-EXIT", "-ENTRY"
        ]
        
        hallucinated_io = []
        for var in io_vars:
            # Skip short identifiers (likely generic words)
            if len(var) <= 4:
                continue
            
            # COBOL variables typically contain hyphens (e.g., WS-CUSTOMER-ID)
            # Single words like "FILES", "DATA" are likely documentation prose
            if '-' not in var:
                continue
            
            # Skip common paragraph name patterns (not I/O variables)
            is_paragraph = any(p in var for p in paragraph_patterns)
            if is_paragraph:
                logger.debug(f"Skipping paragraph pattern: {var}")
                continue
            
            # NEW: Check if it's actually a COBOL variable or natural language
            if not self._is_likely_cobol_variable(var, documentation):
                logger.debug(f"Skipping natural language term: {var}")
                continue
            
            # Check if it exists in ground truth
            if not fuzzy_match_identifier(var, valid_vars, 
                                         self.hallucination_config["fuzzy_threshold"]):
                hallucinated_io.append(var)
        
        # Zero tolerance: ANY hallucinated I/O with COBOL-like naming pattern
        if hallucinated_io:
            first_hallucinated = hallucinated_io[0]
            logger.warning(f"CF-02: Hallucinated I/O variable: {first_hallucinated}")
            return CriticalFailure(
                cf_id="CF-02",
                name="Hallucinated I/O Variable",
                description=f"Zero Tolerance: '{first_hallucinated}' does not exist in source",
                severity="critical",
            )
        
        # PRODUCTION DECISION: Do NOT check internal variables (CF-02b removed)
        # Rationale: AI documentation will naturally reference concepts and patterns
        # that aren't literal COBOL variable names. This is abstraction, not hallucination.
        # The Structural Completeness (SC) score will naturally penalize poor documentation
        # without causing 99% of submissions to fail with CF-02b.
        #
        # CF-02 (I/O Zero Tolerance) is kept because I/O hallucinations are truly critical.
        
        return None
    
    def _is_likely_cobol_variable(self, var: str, documentation: str) -> bool:
        """
        Check if a term is likely a real COBOL variable, not natural language.
        
        Real COBOL variables typically:
        1. Start with a prefix (WS-, FD-, SD-, LD-, etc.)
        2. Appear in code-like contexts (backticks, after MOVE/INTO/FROM)
        3. Are consistently capitalized throughout documentation
        4. Don't appear in section headers or prose descriptions
        
        Args:
            var: The candidate variable name (uppercase)
            documentation: The full documentation text
            
        Returns:
            True if likely a COBOL variable, False if likely natural language
        """
        # Check #1: COBOL prefixes strongly indicate real variables
        cobol_prefixes = ['WS-', 'FD-', 'SD-', 'LD-', 'CD-', 'RD-', 'SR-', 
                          'LS-', 'IND-', 'SUB-', 'IX-']
        if any(var.upper().startswith(prefix) for prefix in cobol_prefixes):
            return True
        
        # Check #2: Appears in code blocks (between backticks)
        if f"`{var}`" in documentation or f"`{var.upper()}`" in documentation:
            return True
        
        # Check #3: Common natural language patterns (NOT variables)
        # These patterns indicate the term is used as prose, not a variable reference
        natural_language_patterns = [
            f"{var.lower()} ",  # Lowercase in prose (e.g., "non-zero error")
            f" {var.capitalize()} ",  # Title case in prose
            f"**{var}**",  # Bold markdown headers (e.g., "**Variable-length**")
            f"**{var.capitalize()}**",  # Bold title case
            f"### {var}",  # Markdown section headers
            f"## {var}",
            f"# {var}",
            f"#### {var}",  # All heading levels
        ]
        for pattern in natural_language_patterns:
            if pattern in documentation:
                return False  # It's prose, not a variable
        
        # Check #4: Only uppercase references suggest variable
        # Count uppercase vs lowercase/title case usages
        # Be case-insensitive for the search but track pattern usage
        upper_count = len(re.findall(rf'\b{re.escape(var.upper())}\b', documentation))
        lower_count = len(re.findall(rf'\b{re.escape(var.lower())}\b', documentation, re.IGNORECASE))
        
        # If appears mostly in lowercase/title context, it's prose
        if lower_count > upper_count:
            return False
        
        # Check #5: Appears after COBOL keywords (strong indicator)
        cobol_keywords = ['MOVE', 'TO', 'FROM', 'GIVING', 'INTO', 'BY', 'USING']
        for keyword in cobol_keywords:
            if f"{keyword} {var}" in documentation or f"{keyword} `{var}`" in documentation:
                return True
        
        # Default: If it has hyphens and we haven't ruled it out, be conservative
        # (let it through for further validation against ground truth)
        return True
    
    def _detect_cf03(
        self,
        bf_result: Optional[Dict]
    ) -> Optional[CriticalFailure]:
        """
        CF-03: Behavioral Contradiction (≥50% claims fail execution)
        """
        if not bf_result:
            return None
        
        # Check claim verification rate
        verified = bf_result.get("claims_verified", 0)
        failed = bf_result.get("claims_failed", 0)
        
        total = verified + failed
        if total == 0:
            return None
        
        fail_rate = failed / total
        
        if fail_rate >= self.cf_config["CF03"]["threshold"]:
            return CriticalFailure(
                cf_id="CF-03",
                name="Behavioral Contradiction",
                description=f"{fail_rate:.0%} of claims failed verification",
            )
        
        return None
    
    def _detect_cf04(
        self,
        documentation: str,
        ground_truth: Dict
    ) -> Optional[CriticalFailure]:
        """
        CF-04: Error handlers exist but undocumented
        """
        # Get error handlers from ground truth
        error_handlers = ground_truth.get("error_handlers", [])
        
        if not error_handlers:
            return None  # No handlers to check
        
        doc_lower = documentation.lower()
        
        # Check if ANY error handling is mentioned
        error_keywords = ["error", "exception", "invalid", "failure", "handler", "size error", "on error"]
        
        mentioned = any(kw in doc_lower for kw in error_keywords)
        
        if not mentioned:
            return CriticalFailure(
                cf_id="CF-04",
                name="Missing Error Handlers",
                description=f"{len(error_handlers)} error handlers exist but undocumented",
            )
        
        return None
    
    def _detect_cf05(
        self,
        bf_result: Optional[Dict]
    ) -> Optional[CriticalFailure]:
        """
        CF-05: >70% BSM pattern failures (relaxed from 50%)
        
        Only enforced for programs with 5+ external calls to avoid
        penalizing simple utility programs.
        """
        if not bf_result:
            return None
        
        bsm_matched = bf_result.get("bsm_matched", 0)
        bsm_total = bf_result.get("bsm_total", 0)
        
        # If no external calls, can't fail this check
        if bsm_total == 0:
            return None
        
        # NEW: Only enforce CF-05 if there are 5+ external calls
        # Rationale: Programs with 1-4 calls are likely simple utilities
        # where BSM match rate is less critical than overall documentation quality
        if bsm_total < 5:
            logger.debug(f"Skipping CF-05: Only {bsm_total} external calls (minimum 5 required)")
            return None
        
        bsm_score = bf_result.get("bsm_score", 1.0)
        
        # Threshold is now 0.70, so we trigger when bsm_score < 0.30 (30%)
        if bsm_score < (1 - self.cf_config["CF05"]["threshold"]):
            return CriticalFailure(
                cf_id="CF-05",
                name="BSM Pattern Failures",
                description=f"Only {bsm_matched}/{bsm_total} external calls correctly documented",
            )
        
        return None
    
    def _extract_valid_identifiers(self, ground_truth: Dict) -> Set[str]:
        """
        Extract all valid identifiers from ground truth.
        """
        valid = set()
        
        # From data structures
        ds_data = ground_truth.get("data_structures", {})
        if isinstance(ds_data, dict):
            structures = ds_data.get("structures", [])
        elif isinstance(ds_data, list):
            structures = ds_data
        else:
            structures = []
        
        for ds in structures:
            if isinstance(ds, str):
                valid.add(ds.upper())
            elif isinstance(ds, dict):
                name = ds.get("name", "")
                if name:
                    valid.add(name.upper())
                
                # Add fields
                for field in ds.get("fields", []):
                    if isinstance(field, str):
                        valid.add(field.upper())
                    elif isinstance(field, dict):
                        fname = field.get("name", "")
                        if fname:
                            valid.add(fname.upper())
        
        # From control flow (paragraph names)
        cf_data = ground_truth.get("control_flow", {})
        if isinstance(cf_data, dict):
            paragraphs = cf_data.get("paragraphs", [])
        elif isinstance(cf_data, list):
            paragraphs = cf_data
        else:
            paragraphs = []
        
        for para in paragraphs:
            if isinstance(para, str):
                valid.add(para.upper())
            elif isinstance(para, dict):
                name = para.get("name", "")
                if name:
                    valid.add(name.upper())
        
        return valid
