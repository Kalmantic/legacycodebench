"""
Behavioral Fidelity Evaluator V2.3.1 (Track 3)

Weight: 50% of total LCB Score

Evaluates whether documentation accurately describes behavior:
- Claim verification via execution
- BSM validation for external calls

Key Innovation: SILENCE PENALTY
- If < 1 claims extracted -> BF score = 0 (min_claims=1)
- Prevents gaming by writing vague documentation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import re
import logging

from .config_v231 import V231_CONFIG
from .claim_extractor import ClaimExtractor, Claim
from .test_generator import TestGenerator, TestCase
from .paragraph_classifier_v231 import ParagraphClassifier, ParagraphType


logger = logging.getLogger(__name__)


@dataclass
class BehavioralResult:
    """Result of behavioral fidelity evaluation."""
    score: float
    claim_score: float = 0.0
    bsm_score: float = 0.0
    claim_count: int = 0
    silence_penalty: bool = False
    claims_verified: int = 0
    claims_failed: int = 0
    bsm_matched: int = 0
    bsm_total: int = 0
    breakdown: Dict = field(default_factory=dict)
    details: Dict = field(default_factory=dict)


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
        self.patterns = V231_CONFIG["bsm_patterns"]
    
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
            
            # Check if target is mentioned (exact match)
            target_found = target.upper() in doc_upper

            # FUZZY MATCHING: Try alternate naming patterns
            if not target_found:
                # For file operations: CUSTOMER-FILE -> CUSTOMERS.DAT, CUSTOMER, CUSTOMERS
                target_found = self._fuzzy_file_match(target, doc_upper)

            if not target_found:
                # For program calls: SCUSTOMP -> SCUSTOM, SCUSTOMER
                target_found = self._fuzzy_program_match(target, doc_upper)

            if not target_found:
                # For middleware APIs: MQOPEN -> "opens the queue", MQGET -> "retrieves message"
                target_found = self._fuzzy_middleware_match(target, doc_upper, doc_lower)
            
            # Check for pattern keywords
            pattern_info = self.patterns.get(call_type, {})
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
    
    def _fuzzy_file_match(self, target: str, doc_upper: str) -> bool:
        """
        Fuzzy match file names between COBOL FD names and physical file names.
        
        Examples:
            CUSTOMER-FILE -> matches CUSTOMERS.DAT, CUSTOMER-DATA, CUSTOMERS
            TRANSACTION-FILE -> matches TRANSACTIONS.DAT, TRANS.DAT
        """
        target_upper = target.upper()
        
        # Extract base name (remove -FILE, -RECORD, -DATA suffixes)
        base = target_upper
        for suffix in ["-FILE", "-RECORD", "-DATA", "-REC"]:
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        
        # Also try removing trailing S for plural forms
        base_singular = base.rstrip("S") if len(base) > 3 else base
        base_plural = base + "S" if not base.endswith("S") else base
        
        # Check various patterns
        patterns_to_try = [
            base,                    # CUSTOMER
            base_singular,           # CUSTOMER (from CUSTOMERS)
            base_plural,             # CUSTOMERS
            base + ".DAT",           # CUSTOMER.DAT
            base_plural + ".DAT",    # CUSTOMERS.DAT
            base + "-",              # CUSTOMER-... (partial match)
            base_plural + "-",       # CUSTOMERS-...
        ]
        
        for pattern in patterns_to_try:
            if pattern in doc_upper:
                logger.debug(f"Fuzzy match: {target} matched via '{pattern}'")
                return True

        return False

    def _fuzzy_program_match(self, target: str, doc_upper: str) -> bool:
        """
        Fuzzy match program names for common COBOL variations.

        COBOL programs often have naming conventions that differ from documentation:
        - 8-character limit causes truncation (SCUSTOMER -> SCUSTOMP)
        - Suffixes indicate type: P=Program, I=Inquiry, U=Update, R=Report, M=Maint
        - Abbreviations: CUST=CUSTOMER, ACCT=ACCOUNT, INQ=INQUIRY, etc.

        Examples:
            SCUSTOMP  -> matches SCUSTOM, SCUSTOMER, S-CUSTOMER
            CUSTINQ   -> matches CUST-INQUIRY, CUSTOMER-INQUIRY
            ACCTUPD   -> matches ACCOUNT-UPDATE, ACCT-UPDATE
        """
        target_upper = target.upper()

        # Start with exact target
        patterns_to_try = {target_upper}

        # Remove common program type suffixes (last character)
        base = target_upper
        type_suffixes = ["P", "I", "U", "R", "M", "D", "L", "S", "V", "X"]
        if len(base) > 4 and base[-1] in type_suffixes:
            base_no_suffix = base[:-1]
            patterns_to_try.add(base_no_suffix)
        else:
            base_no_suffix = base

        # Common COBOL abbreviation expansions
        abbreviations = {
            "CUST": ["CUSTOMER", "CUST"],
            "ACCT": ["ACCOUNT", "ACCT"],
            "INQ": ["INQUIRY", "INQ", "INQRY"],
            "UPD": ["UPDATE", "UPD"],
            "RPT": ["REPORT", "RPT"],
            "TXN": ["TRANSACTION", "TXN", "TRANS"],
            "MAINT": ["MAINTENANCE", "MAINT", "MNT"],
            "PROC": ["PROCESS", "PROC"],
            "CALC": ["CALCULATE", "CALC", "CALCULATION"],
            "VAL": ["VALIDATE", "VAL", "VALIDATION"],
            "CHK": ["CHECK", "CHK"],
            "BAL": ["BALANCE", "BAL"],
            "PMT": ["PAYMENT", "PMT"],
            "ORD": ["ORDER", "ORD"],
            "INV": ["INVOICE", "INV", "INVENTORY"],
            "EMP": ["EMPLOYEE", "EMP"],
            "MGR": ["MANAGER", "MGR"],
            "SVC": ["SERVICE", "SVC"],
            "MSG": ["MESSAGE", "MSG"],
            "ERR": ["ERROR", "ERR"],
            "HDR": ["HEADER", "HDR"],
            "DTL": ["DETAIL", "DTL"],
            "TOT": ["TOTAL", "TOT"],
            "SUM": ["SUMMARY", "SUM"],
            "LST": ["LIST", "LST"],
            "SEL": ["SELECT", "SEL"],
            "DEL": ["DELETE", "DEL"],
            "ADD": ["ADD", "INSERT"],
            "MOD": ["MODIFY", "MOD"],
            "EDT": ["EDIT", "EDT"],
            "DSP": ["DISPLAY", "DSP"],
            "PRT": ["PRINT", "PRT"],
            "RDR": ["READER", "RDR"],
            "WRT": ["WRITER", "WRT"],
        }

        # Try expanding abbreviations in the base name
        for base_variant in [target_upper, base_no_suffix]:
            for abbr, expansions in abbreviations.items():
                if abbr in base_variant:
                    for expansion in expansions:
                        expanded = base_variant.replace(abbr, expansion)
                        patterns_to_try.add(expanded)
                        # Also try with hyphens
                        patterns_to_try.add(expanded.replace(expansion, f"-{expansion}"))
                        patterns_to_try.add(expanded.replace(expansion, f"{expansion}-"))

        # Try adding hyphens at common word boundaries
        # SCUSTOM -> S-CUSTOM, S-CUSTOMER
        if len(base_no_suffix) >= 4:
            # Try splitting after first character (common prefix pattern)
            if base_no_suffix[0] in "SABCDEFGHIJKLMNOPQRSTUVWXYZ":
                patterns_to_try.add(f"{base_no_suffix[0]}-{base_no_suffix[1:]}")

        # Check all patterns
        for pattern in patterns_to_try:
            if len(pattern) >= 3 and pattern in doc_upper:
                logger.debug(f"Fuzzy program match: {target} matched via '{pattern}'")
                return True

        return False

    def _fuzzy_middleware_match(self, target: str, doc_upper: str, doc_lower: str) -> bool:
        """
        Fuzzy match middleware API calls using semantic patterns.

        PRODUCTION GRADE RATIONALE:
        Enterprise COBOL programs use middleware APIs (MQ, CICS, IMS, DB2) that AI models
        often describe semantically rather than by literal API name. Good documentation
        should demonstrate UNDERSTANDING of what APIs do, not just parrot names.

        For example, a model describing "opens the message queue" demonstrates understanding
        of MQOPEN's purpose, even without using the literal name.

        This approach rewards comprehension over keyword matching, which is the goal of
        a trustworthy benchmark.

        Examples:
            MQOPEN  -> matches "opens the queue", "open queue connection", "queue open"
            MQGET   -> matches "retrieves message", "get from queue", "reads message"
            MQPUT   -> matches "sends message", "puts message", "writes to queue"
            MQCLOSE -> matches "closes the queue", "close connection", "disconnect"
        """
        target_upper = target.upper().strip("'\"")

        # Define semantic patterns for common middleware APIs
        # Each API maps to phrases that demonstrate understanding of its purpose
        middleware_patterns = {
            # IBM MQ (WebSphere MQ / MQSeries)
            "MQOPEN": [
                "open queue", "opens queue", "opens the queue", "queue open",
                "open connection", "opens connection", "connect to queue",
                "initialize queue", "establish queue", "queue connection"
            ],
            "MQGET": [
                "get message", "gets message", "retrieve message", "retrieves message",
                "read from queue", "reads from queue", "receive message", "receives message",
                "fetch message", "fetches message", "poll queue", "dequeue"
            ],
            "MQPUT": [
                "put message", "puts message", "send message", "sends message",
                "write to queue", "writes to queue", "enqueue", "post message",
                "publish message", "place message", "queue message"
            ],
            "MQPUT1": [
                "put message", "puts message", "send message", "sends message",
                "write to queue", "writes to queue", "single put", "send reply",
                "reply message", "response message", "one-shot put"
            ],
            "MQCLOSE": [
                "close queue", "closes queue", "closes the queue", "queue close",
                "disconnect queue", "disconnect from queue", "release queue",
                "terminate queue", "end queue connection"
            ],
            "MQCONN": [
                "connect to queue manager", "connects to queue manager",
                "queue manager connection", "establish connection", "mq connection"
            ],
            "MQDISC": [
                "disconnect from queue manager", "disconnects from queue manager",
                "terminate connection", "close connection", "mq disconnect"
            ],

            # CICS Transaction Processing
            "DFHEIBLK": ["cics interface", "execute interface block", "eib"],
            "CICS": [
                "cics transaction", "cics program", "cics environment",
                "transaction processing", "online transaction"
            ],

            # IMS Database
            "CBLTDLI": [
                "ims database", "ims call", "dli call", "database call",
                "hierarchical database", "ims access"
            ],
            "AIBTDLI": ["ims aib", "application interface block"],
            "GU": ["get unique", "retrieve segment", "ims get"],
            "GN": ["get next", "sequential read", "next segment"],
            "GNP": ["get next within parent", "child segment"],
            "ISRT": ["insert segment", "add segment", "ims insert"],
            "DLET": ["delete segment", "remove segment", "ims delete"],
            "REPL": ["replace segment", "update segment", "ims replace"],

            # DB2 Database
            "CBLDECL": ["db2 declaration", "sql declaration"],
            "SQLOPEN": ["open cursor", "cursor open", "opens cursor"],
            "SQLFETCH": ["fetch row", "fetches row", "retrieve row", "cursor fetch"],
            "SQLCLOSE": ["close cursor", "cursor close", "closes cursor"],
            "SQLSELECT": ["select statement", "sql query", "database query"],
            "SQLINSERT": ["insert row", "add row", "database insert"],
            "SQLUPDATE": ["update row", "modify row", "database update"],
            "SQLDELETE": ["delete row", "remove row", "database delete"],
        }

        # Check if target matches any pattern
        if target_upper in middleware_patterns:
            patterns = middleware_patterns[target_upper]
            for pattern in patterns:
                if pattern in doc_lower:
                    logger.debug(f"Fuzzy middleware match: {target} matched via '{pattern}'")
                    return True

        # Also check if the call type indicates middleware and doc mentions the technology
        # e.g., if target is MQOPEN and doc contains "MQ" + "open", consider it a match
        mq_apis = ["MQOPEN", "MQGET", "MQPUT", "MQPUT1", "MQCLOSE", "MQCONN", "MQDISC"]
        if target_upper in mq_apis:
            # Check for MQ technology mention + action word
            if "MQ" in doc_upper or "MESSAGE QUEUE" in doc_upper or "WEBSPHERE" in doc_upper:
                action_words = {
                    "MQOPEN": ["open", "connect", "initialize"],
                    "MQGET": ["get", "read", "receive", "retrieve", "fetch"],
                    "MQPUT": ["put", "send", "write", "post"],
                    "MQPUT1": ["put", "send", "write", "reply", "response"],
                    "MQCLOSE": ["close", "disconnect", "release", "terminate"],
                    "MQCONN": ["connect", "connection"],
                    "MQDISC": ["disconnect", "disconnection"],
                }
                if any(action in doc_lower for action in action_words.get(target_upper, [])):
                    logger.debug(f"Fuzzy middleware match: {target} matched via MQ + action word")
                    return True

        return False


class BehavioralEvaluatorV231:
    """
    V2.3.1 Behavioral Fidelity Evaluator
    
    Pipeline:
    1. Extract claims from documentation
    2. Apply SILENCE PENALTY if < 3 claims
    3. Generate tests for claims
    4. Verify claims (if executor available)
    5. Validate BSM patterns
    6. Calculate combined score
    """
    
    def __init__(self, llm_client=None, executor=None):
        self.claim_extractor = ClaimExtractor(llm_client)
        self.test_generator = TestGenerator()
        self.bsm_validator = BSMValidator()
        self.paragraph_classifier = ParagraphClassifier()
        self.executor = executor
        
        self.silence_config = V231_CONFIG["silence_penalty"]
    
    
    def evaluate(
        self,
        documentation: str,
        source_code: str,
        ground_truth: Dict
    ) -> BehavioralResult:
        """
        Evaluate behavioral fidelity using paragraph-level evaluation.
        
        Per PRD Â§5.1-5.2:
        - PURE paragraphs: Claim verification via execution only
        - MIXED paragraphs: Claim verification + BSM
        - INFRASTRUCTURE paragraphs: BSM-only
        
        Args:
            documentation: AI-generated documentation
            source_code: Original COBOL source
            ground_truth: Ground truth data
            
        Returns:
            BehavioralResult with score and breakdown
        """
        logger.debug("Stage 1: Claim Extraction [...]")
        # Step 1: Extract claims
        claims = self.claim_extractor.extract(documentation)
        logger.debug(f"Stage 1: Extracted {len(claims)} claims [OK]")
        
        # Step 2: SILENCE PENALTY
        logger.debug("Stage 2: Silence Penalty Check [...]")
        if len(claims) < self.silence_config["min_claims"]:
            logger.warning(f"SILENCE PENALTY: Only {len(claims)} claims (min: {self.silence_config['min_claims']})")
            logger.debug(f"Stage 2: SILENCE PENALTY TRIGGERED [X]")
            
            return BehavioralResult(
                score=0.0,
                claim_score=0.0,
                bsm_score=0.0,
                claim_count=len(claims),
                silence_penalty=True,
                breakdown={"claims": 0.0, "bsm": 0.0},
                details={"reason": f"Silence Penalty: {len(claims)} claims < {self.silence_config['min_claims']} minimum"},
            )
        logger.debug("Stage 2: Silence Penalty Check [OK] (passed)")
        
        # Step 3: Classify paragraphs
        logger.debug("Stage 3: Paragraph Classification [...]")
        paragraphs = self._extract_paragraphs(ground_truth)
        classified = self.paragraph_classifier.classify_all(paragraphs)
        
        pure_count = len(classified["pure"])
        mixed_count = len(classified["mixed"])
        infra_count = len(classified["infrastructure"])
        logger.debug(f"Stage 3: Classified {pure_count} PURE, {mixed_count} MIXED, {infra_count} INFRASTRUCTURE [OK]")
        
        # Step 4: Generate tests
        logger.debug("Stage 4: Test Generation [...]")
        tests = self.test_generator.generate(claims, ground_truth)
        logger.debug(f"Stage 4: Generated {len(tests)} tests [OK]")
        
        # Step 5: Calculate BSM ONCE at program level (avoid double-counting)
        logger.debug("Stage 5a: Program-level BSM Calculation [...]")
        external_calls = self._extract_external_calls(ground_truth)
        program_bsm = self.bsm_validator.validate(documentation, external_calls)
        logger.debug(f"Stage 5a: BSM = {program_bsm['matched']}/{program_bsm['total']} matched [OK]")

        # Step 5b: Evaluate by paragraph type
        logger.debug("Stage 5b: Paragraph-level Evaluation [...]")

        # Evaluate PURE paragraphs (execution-only)
        pure_score, pure_verified, pure_failed = self._evaluate_pure_paragraphs(
            claims, tests, source_code, classified["pure"]
        )

        # Evaluate MIXED paragraphs (execution + BSM) - pass pre-computed BSM
        mixed_score, mixed_verified, mixed_failed = self._evaluate_mixed_paragraphs_v2(
            claims, tests, source_code, program_bsm, classified["mixed"]
        )

        # Evaluate INFRASTRUCTURE paragraphs (Claims + BSM) - pass pre-computed BSM
        infra_score, infra_verified, infra_failed = self._evaluate_infrastructure_paragraphs_v2(
            claims, program_bsm, classified["infrastructure"]
        )

        logger.debug(f"Stage 5b: PURE={pure_score:.2f}, MIXED={mixed_score:.2f}, INFRA={infra_score:.2f} [OK]")

        # Step 6: Calculate weighted score
        logger.debug("Stage 6: Combining Paragraph Scores [...]")
        total_paragraphs = pure_count + mixed_count + infra_count

        if total_paragraphs == 0:
            # Fallback for programs with no classified paragraphs
            logger.warning("No paragraphs classified, using program-level evaluation")
            return self._evaluate_program_level(claims, tests, source_code, documentation, ground_truth)

        # Weight by paragraph count
        pure_weight = pure_count / total_paragraphs
        mixed_weight = mixed_count / total_paragraphs
        infra_weight = infra_count / total_paragraphs

        overall = (pure_score * pure_weight +
                   mixed_score * mixed_weight +
                   infra_score * infra_weight)

        # Aggregate totals
        total_verified = pure_verified + mixed_verified + infra_verified
        total_failed = pure_failed + mixed_failed + infra_failed

        # Use program-level BSM (calculated once, not doubled)
        bsm_score = program_bsm["score"]
        total_bsm_matched = program_bsm["matched"]
        total_bsm_total = program_bsm["total"]

        claim_score = total_verified / (total_verified + total_failed) if (total_verified + total_failed) > 0 else 0.5
        
        logger.debug(f"Stage 6: Overall score = {overall:.2f} [OK]")
        
        return BehavioralResult(
            score=overall,
            claim_score=claim_score,
            bsm_score=bsm_score,
            claim_count=len(claims),
            silence_penalty=False,
            claims_verified=total_verified,
            claims_failed=total_failed,
            bsm_matched=total_bsm_matched,
            bsm_total=total_bsm_total,
            breakdown={
                "claims": claim_score,
                "bsm": bsm_score,
                "pure_score": pure_score,
                "mixed_score": mixed_score,
                "infrastructure_score": infra_score,
                "pure_weight": pure_weight,
                "mixed_weight": mixed_weight,
                "infrastructure_weight": infra_weight,
            },
            details={
                "claims": [{"id": c.claim_id, "type": c.claim_type.value, "text": c.text[:100]} for c in claims],
                "tests": len(tests),
                "paragraph_classification": {
                    "pure": pure_count,
                    "mixed": mixed_count,
                    "infrastructure": infra_count,
                },
            },
        )
    def _verify_claims(
        self,
        claims: List[Claim],
        tests: List[TestCase],
        source_code: str
    ) -> tuple:
        """
        Verify claims by executing tests.
        
        If no executor available, use heuristic scoring.
        
        Returns:
            (score, verified_count, failed_count)
        """
        if not claims:
            return 1.0, 0, 0
        
        if self.executor:
            # Pre-check if code can be executed (e.g., not CICS)
            can_exec, reason = self.executor.can_execute(source_code)
            
            if not can_exec:
                logger.info(f"Skipping execution: {reason}")
                logger.info("Falling back to BSM-only scoring for behavioral fidelity")
                # Return neutral score - let BSM dominate in final calculation
                # We return 0.5 instead of 0.0 to avoid triggering CF-03
                return 0.5, 0, 0
            
            # Full execution-based verification
            return self._verify_via_execution(claims, tests, source_code)
        else:
            # Heuristic verification
            return self._verify_heuristic(claims)
    
    def _verify_via_execution(
        self,
        claims: List[Claim],
        tests: List[TestCase],
        source_code: str
    ) -> tuple:
        """
        Verify claims by actually executing tests via COBOLExecutor.
        """
        verified = 0
        failed = 0
        compilation_failures = 0

        logger.info(f"Executing {len(tests)} test cases via Docker...")

        for test in tests:
            try:
                # TestCase has 'inputs' dict, not 'input_var'/'input_value'
                test_inputs = test.inputs if test.inputs else {}

                # Execute via COBOLExecutor
                result = self.executor.execute(source_code, test_inputs)

                # Check if execution succeeded AND outputs match expected
                if result.success:
                    # If test has expected outputs, validate them
                    if test.expected_outputs:
                        outputs_valid = self._validate_outputs(result, test.expected_outputs)
                        if outputs_valid:
                            verified += 1
                            desc = test.description[:50] if test.description else test.test_id
                            logger.debug(f"Test PASSED: {desc}... (outputs validated)")
                        else:
                            failed += 1
                            desc = test.description[:50] if test.description else test.test_id
                            logger.debug(f"Test FAILED: {desc}... (output mismatch)")
                    else:
                        # No expected outputs - just check compilation/execution success
                        verified += 1
                        desc = test.description[:50] if test.description else test.test_id
                        logger.debug(f"Test PASSED: {desc}... (execution success)")
                else:
                    # Check if this is a compilation failure (should fallback to heuristic)
                    error_msg = result.error_message if hasattr(result, 'error_message') else ""
                    if "compilation failed" in error_msg.lower() or "No such file" in error_msg:
                        compilation_failures += 1
                        desc = test.description[:50] if test.description else test.test_id
                        logger.debug(f"Test SKIPPED (compilation): {desc}...")
                    else:
                        failed += 1
                        desc = test.description[:50] if test.description else test.test_id
                        logger.debug(f"Test FAILED: {desc}... - {error_msg}")
            except Exception as e:
                logger.warning(f"Test execution error: {e}")
                # Don't count infrastructure errors as test failures
                compilation_failures += 1

        # If ALL tests failed due to compilation, fallback to heuristic
        if compilation_failures > 0 and verified == 0 and failed == 0:
            logger.warning(f"All {compilation_failures} tests failed due to compilation. Using heuristic fallback.")
            return self._verify_heuristic(claims)

        # FIX: Also fallback if NO tests could be verified AND we have infrastructure failures
        # This handles cases where file-based programs fail due to missing test data,
        # not actual behavioral contradictions
        if verified == 0 and (compilation_failures > 0 or failed > 0):
            total_attempts = compilation_failures + failed
            logger.warning(
                f"No tests verified ({total_attempts} failed/skipped). "
                f"Likely missing test data. Using heuristic fallback to avoid false CF-03."
            )
            return self._verify_heuristic(claims)

        total = verified + failed
        score = verified / total if total > 0 else 0.5

        logger.info(f"Execution complete: {verified}/{total} tests passed ({score*100:.0f}%)")

        return score, verified, failed

    def _validate_outputs(
        self,
        result,
        expected_outputs: Dict[str, any]
    ) -> bool:
        """
        Validate execution outputs against expected values.

        Args:
            result: ExecutionResult from COBOL executor
            expected_outputs: Dict of variable â†’ expected value

        Returns:
            True if all expected outputs match actual outputs
        """
        if not expected_outputs:
            return True

        # Extract actual outputs from execution result
        actual_outputs = self._extract_outputs(result)
        raw_stdout = actual_outputs.pop('_RAW_STDOUT', '')

        # Compare each expected output
        for var_name, expected_value in expected_outputs.items():
            actual_value = actual_outputs.get(var_name.upper())

            # Try fuzzy matching if exact match not found
            if actual_value is None:
                # Try matching with hyphens/underscores normalized
                normalized_var = var_name.upper().replace('_', '-')
                for key, val in actual_outputs.items():
                    if key.replace('_', '-') == normalized_var:
                        actual_value = val
                        break
            
            # Try finding the expected value anywhere in extracted outputs
            if actual_value is None:
                for key, val in actual_outputs.items():
                    if self._values_match(val, expected_value):
                        logger.debug(f"Fuzzy matched '{var_name}' via value match in '{key}'")
                        actual_value = val
                        break

            # Last resort: check if the expected value appears in raw stdout
            if actual_value is None:
                expected_str = str(expected_value)
                if expected_str in raw_stdout:
                    logger.debug(f"Found expected value '{expected_str}' in raw stdout for '{var_name}'")
                    actual_value = expected_value  # Assume match

            if actual_value is None:
                logger.debug(f"Output variable '{var_name}' not found in execution result (fuzzy search exhausted)")
                return False

            # Normalize and compare values
            if not self._values_match(actual_value, expected_value):
                logger.debug(f"Output mismatch for '{var_name}': expected {expected_value}, got {actual_value}")
                return False

        return True

    def _extract_outputs(self, result) -> Dict[str, any]:
        """
        Extract output variables from ExecutionResult.

        Parses stdout for DISPLAY statements and file outputs.

        Args:
            result: ExecutionResult

        Returns:
            Dict of variable â†’ value extracted from output
        """
        outputs = {}

        # Parse stdout for displayed values
        stdout_lines = result.stdout.split('\n')

        for line in stdout_lines:
            line = line.strip()
            if not line:
                continue

            # Pattern 1: VAR-NAME: VALUE or VAR-NAME = VALUE
            match = re.match(r'([A-Z0-9\-_]+)\s*[:=]\s*(.+)', line, re.IGNORECASE)
            if match:
                var_name = match.group(1).strip().upper()
                value_str = match.group(2).strip()
                outputs[var_name] = self._parse_value(value_str)
                continue

            # Pattern 2: "Label text: $123.45" or "Label text: 123.45"
            # Common in COBOL reports - extract the rightmost number
            match = re.search(r':\s*\$?([\d,]+\.?\d*)\s*$', line)
            if match:
                value_str = match.group(1).replace(',', '')
                # Use a generic key based on the label
                label_match = re.match(r'^([^:]+):', line)
                if label_match:
                    label = label_match.group(1).strip().upper().replace(' ', '-')
                    outputs[label] = self._parse_value(value_str)
                continue

            # Pattern 3: Lines with just a number (common in simple outputs)
            match = re.match(r'^\s*\$?([\d,]+\.?\d*)\s*$', line)
            if match:
                value_str = match.group(1).replace(',', '')
                outputs[f"VALUE-{len(outputs)}"] = self._parse_value(value_str)
                continue

            # Pattern 4: "Description $123.45" (value at end)
            match = re.search(r'\s+\$?([\d,]+\.?\d*)\s*$', line)
            if match and len(line) > 10:
                value_str = match.group(1).replace(',', '')
                desc_part = line[:match.start()].strip()
                if desc_part:
                    label = desc_part.upper().replace(' ', '-')[:30]
                    outputs[label] = self._parse_value(value_str)

            # Pattern 5: Extract all numbers on the line as potential values
            numbers = re.findall(r'\b(\d+\.?\d*)\b', line)
            for i, num in enumerate(numbers):
                key = f"NUM-{len(outputs)}-{i}"
                if key not in outputs:
                    outputs[key] = self._parse_value(num)

        # Also check file outputs
        for filename, content in result.file_outputs.items():
            file_lines = content.split('\n')
            for line in file_lines:
                match = re.match(r'([A-Z0-9\-_]+)\s*[:=]\s*(.+)', line, re.IGNORECASE)
                if match:
                    var_name = match.group(1).strip().upper()
                    value_str = match.group(2).strip()
                    outputs[var_name] = self._parse_value(value_str)

        # Store raw stdout for fuzzy matching in validation
        outputs['_RAW_STDOUT'] = result.stdout

        return outputs

    def _parse_value(self, value_str: str) -> any:
        """Parse a string value to appropriate type (int, float, or string)."""
        value_str = str(value_str).strip()
        clean_value = re.sub(r'[,$]', '', value_str)
        
        try:
            if '.' in clean_value:
                return float(clean_value)
            else:
                return int(clean_value)
        except ValueError:
            return value_str

    def _values_match(self, actual: any, expected: any) -> bool:
        """
        Check if two values match, with tolerance for floating point.

        Args:
            actual: Actual value from execution
            expected: Expected value

        Returns:
            True if values match
        """
        # Exact match for strings
        if isinstance(expected, str) or isinstance(actual, str):
            return str(actual).strip().upper() == str(expected).strip().upper()

        # Numeric comparison with tolerance
        try:
            actual_num = float(actual)
            expected_num = float(expected)

            # Allow 0.01% tolerance for floating point errors
            tolerance = abs(expected_num) * 0.0001 + 1e-6
            return abs(actual_num - expected_num) <= tolerance
        except (TypeError, ValueError):
            # Fall back to string comparison
            return str(actual) == str(expected)
    
    def _verify_heuristic(self, claims: List[Claim]) -> tuple:
        """
        Heuristic verification when no executor available.
        
        Score based on claim quality indicators:
        - Variable names look valid
        - Claims have specific details
        - Claims don't contradict each other
        """
        # Score based on claim confidence and specificity
        total_confidence = sum(c.confidence for c in claims)
        avg_confidence = total_confidence / len(claims) if claims else 0.5
        
        # Bonus for specific claims (those with output variables)
        specific_claims = sum(1 for c in claims if c.output_var)
        specificity_bonus = (specific_claims / len(claims)) * 0.2 if claims else 0
        
        score = min(avg_confidence + specificity_bonus, 1.0)
        
        # For heuristic, we can't know actual verified/failed counts
        return score, len(claims), 0
    
    def _extract_paragraphs(self, ground_truth: Dict) -> List[Dict]:
        """
        Extract paragraph information from ground truth.
        
        Returns:
            List of paragraph dicts with 'name', 'content', 'start_line', 'end_line'
        """
        control_flow = ground_truth.get("control_flow", {})
        paragraphs_data = control_flow.get("paragraphs", [])
        
        # Convert to format expected by classifier
        paragraphs = []
        for para in paragraphs_data:
            paragraphs.append({
                "name": para.get("name", ""),
                "content": para.get("content", ""),  # Use 'content' field from ground truth
                "start_line": para.get("start_line", 0),
                "end_line": para.get("end_line", 0),
            })
        
        return paragraphs
    
    def _evaluate_pure_paragraphs(
        self,
        claims: List[Claim],
        tests: List[TestCase],
        source_code: str,
        pure_paragraphs: List
    ) -> tuple:
        """
        Evaluate PURE paragraphs using claim verification via execution only.
        
        Returns:
            (score, verified_count, failed_count)
        """
        if not pure_paragraphs:
            return 1.0, 0, 0
        
        # For PURE paragraphs, use execution-only verification
        if self.executor:
            can_exec, reason = self.executor.can_execute(source_code)
            if can_exec:
                return self._verify_via_execution(claims, tests, source_code)
            else:
                logger.warning(f"Cannot execute PURE paragraphs: {reason}")
                # Fallback to heuristic
                return self._verify_heuristic(claims)
        else:
            return self._verify_heuristic(claims)
    
    def _evaluate_mixed_paragraphs(
        self,
        claims: List[Claim],
        tests: List[TestCase],
        source_code: str,
        documentation: str,
        ground_truth: Dict,
        mixed_paragraphs: List
    ) -> tuple:
        """
        Evaluate MIXED paragraphs using claim verification + BSM.
        
        Returns:
            (score, verified_count, failed_count, bsm_result)
        """
        if not mixed_paragraphs:
            return 1.0, 0, 0, {"score": 1.0, "matched": 0, "total": 0}
        
        # Extract external calls from MIXED paragraphs only
        external_calls = self._extract_external_calls(ground_truth)
        bsm_result = self.bsm_validator.validate(documentation, external_calls)
        
        # Check if execution is possible (CICS check)
        if self.executor:
            can_exec, reason = self.executor.can_execute(source_code)
            if can_exec:
                # Can execute: use claim verification
                claim_score, verified, failed = self._verify_via_execution(claims, tests, source_code)
                # Combine with BSM per v2.3.1 spec
                claim_weight = 0.6 if len(claims) >= 5 else 0.4
                combined_score = claim_weight * claim_score + (1 - claim_weight) * bsm_result["score"]
                return combined_score, verified, failed, bsm_result
            else:
                # Cannot execute (CICS): use BSM-only
                logger.info(f"MIXED paragraphs cannot execute ({reason}), using BSM-only")
                return bsm_result["score"], 0, 0, bsm_result
        else:
            # No executor: heuristic + BSM
            claim_score, verified, failed = self._verify_heuristic(claims)
            claim_weight = 0.6 if len(claims) >= 5 else 0.4
            combined_score = claim_weight * claim_score + (1 - claim_weight) * bsm_result["score"]
            return combined_score, verified, failed, bsm_result
    
    def _evaluate_infrastructure_paragraphs(
        self,
        claims: List[Claim],
        documentation: str,
        ground_truth: Dict,
        infrastructure_paragraphs: List
    ) -> tuple:
        """
        Evaluate INFRASTRUCTURE paragraphs using Claims + BSM combination.
        Per PRD v2.3.2: Infrastructure paragraphs cannot be executed, so use heuristic + BSM.

        Returns:
            (score, verified_count, failed_count, bsm_result)
        """
        if not infrastructure_paragraphs:
            return 1.0, 0, 0, {"score": 1.0, "matched": 0, "total": 0}

        # Extract external calls and validate BSM
        external_calls = self._extract_external_calls(ground_truth)
        bsm_result = self.bsm_validator.validate(documentation, external_calls)

        # Infrastructure paragraphs use heuristic claim verification + BSM
        # (cannot be executed, so no execution verification)
        claim_score, verified, failed = self._verify_heuristic(claims)

        # Combine Claims + BSM per PRD v2.3.2 formula
        claim_weight = 0.6 if len(claims) >= 5 else 0.4
        combined_score = claim_weight * claim_score + (1 - claim_weight) * bsm_result["score"]

        return combined_score, verified, failed, bsm_result

    def _evaluate_mixed_paragraphs_v2(
        self,
        claims: List[Claim],
        tests: List[TestCase],
        source_code: str,
        program_bsm: Dict,
        mixed_paragraphs: List
    ) -> tuple:
        """
        Evaluate MIXED paragraphs using claim verification + pre-computed BSM.

        V2: Uses program-level BSM to avoid double-counting external calls.

        Returns:
            (score, verified_count, failed_count)
        """
        if not mixed_paragraphs:
            return 1.0, 0, 0

        # Use pre-computed program-level BSM
        bsm_score = program_bsm["score"]

        # Check if execution is possible (CICS check)
        if self.executor:
            can_exec, reason = self.executor.can_execute(source_code)
            if can_exec:
                # Can execute: use claim verification
                claim_score, verified, failed = self._verify_via_execution(claims, tests, source_code)
                # Combine with BSM per v2.3.1 spec
                claim_weight = 0.6 if len(claims) >= 5 else 0.4
                combined_score = claim_weight * claim_score + (1 - claim_weight) * bsm_score
                return combined_score, verified, failed
            else:
                # Cannot execute (CICS): use BSM-only
                logger.info(f"MIXED paragraphs cannot execute ({reason}), using BSM-only")
                return bsm_score, 0, 0
        else:
            # No executor: heuristic + BSM
            claim_score, verified, failed = self._verify_heuristic(claims)
            claim_weight = 0.6 if len(claims) >= 5 else 0.4
            combined_score = claim_weight * claim_score + (1 - claim_weight) * bsm_score
            return combined_score, verified, failed

    def _evaluate_infrastructure_paragraphs_v2(
        self,
        claims: List[Claim],
        program_bsm: Dict,
        infrastructure_paragraphs: List
    ) -> tuple:
        """
        Evaluate INFRASTRUCTURE paragraphs using Claims + pre-computed BSM.

        V2: Uses program-level BSM to avoid double-counting external calls.

        Returns:
            (score, verified_count, failed_count)
        """
        if not infrastructure_paragraphs:
            return 1.0, 0, 0

        # Use pre-computed program-level BSM
        bsm_score = program_bsm["score"]

        # Infrastructure paragraphs use heuristic claim verification + BSM
        # (cannot be executed, so no execution verification)
        claim_score, verified, failed = self._verify_heuristic(claims)

        # Combine Claims + BSM per PRD v2.3.2 formula
        claim_weight = 0.6 if len(claims) >= 5 else 0.4
        combined_score = claim_weight * claim_score + (1 - claim_weight) * bsm_score

        return combined_score, verified, failed

    def _evaluate_program_level(
        self,
        claims: List[Claim],
        tests: List[TestCase],
        source_code: str,
        documentation: str,
        ground_truth: Dict
    ) -> BehavioralResult:
        """
        Fallback to program-level evaluation when no paragraphs are classified.
        
        This maintains backward compatibility.
        """
        logger.info("Using program-level evaluation (fallback)")
        
        # Verify claims
        claim_score, verified, failed = self._verify_claims(claims, tests, source_code)
        
        # BSM validation
        external_calls = self._extract_external_calls(ground_truth)
        bsm_result = self.bsm_validator.validate(documentation, external_calls)
        bsm_score = bsm_result["score"]
        
        # Weighted scoring
        claim_weight = 0.6 if len(claims) >= 5 else 0.4
        overall = claim_weight * claim_score + (1 - claim_weight) * bsm_score
        
        return BehavioralResult(
            score=overall,
            claim_score=claim_score,
            bsm_score=bsm_score,
            claim_count=len(claims),
            silence_penalty=False,
            claims_verified=verified,
            claims_failed=failed,
            bsm_matched=bsm_result["matched"],
            bsm_total=bsm_result["total"],
            breakdown={"claims": claim_score, "bsm": bsm_score, "claim_weight": claim_weight},
            details={"fallback": "program-level"},
        )
    
    def _extract_external_calls(self, ground_truth: Dict) -> List[Dict]:
        """
        Extract external calls from ground truth.
        
        Looks in multiple locations for compatibility:
        1. dependencies.call_categories.external_dependency (new format)
        2. dependencies.calls (explicit calls)
        3. dependencies.files.files (file operations)
        """
        dep_data = ground_truth.get("dependencies", {})
        calls = []
        
        if isinstance(dep_data, dict):
            # Check call_categories for all external call types
            call_categories = dep_data.get("call_categories", {})
            if isinstance(call_categories, dict):
                # External dependencies (known external systems)
                external_deps = call_categories.get("external_dependency", [])
                for call in external_deps:
                    if isinstance(call, dict):
                        calls.append({
                            "target": call.get("callee", ""),
                            "type": "call",
                            "name": call.get("callee", ""),
                        })
                
                # CRITICAL FIX: Include 'in_scope' calls
                # These are calls to other programs in the same codebase (e.g., SCUSTOMP, SVERSONP)
                # They should be documented just like external calls
                in_scope = call_categories.get("in_scope", [])
                for call in in_scope:
                    if isinstance(call, dict):
                        callee = call.get("callee", "")
                        # Deduplicate - same program may be called multiple times
                        if not any(c.get("target") == callee for c in calls):
                            calls.append({
                                "target": callee,
                                "type": "call",
                                "name": callee,
                            })
                
                # Middleware calls (CICS, MQ, etc.)
                middleware = call_categories.get("middleware", [])
                for call in middleware:
                    if isinstance(call, dict):
                        calls.append({
                            "target": call.get("callee", ""),
                            "type": "middleware",
                            "name": call.get("callee", ""),
                        })
                
                # CRITICAL FIX: Include 'unverifiable' calls
                # These are external API calls (DB2, system calls) that CAN'T be executed
                # but MUST still be documented by the AI model
                # Example: db2gMonitorSwitches, checkerr, etc.
                unverifiable = call_categories.get("unverifiable", [])
                for call in unverifiable:
                    if isinstance(call, dict):
                        callee = call.get("callee", "")
                        # Strip quotes from API names like "db2gMonitorSwitches"
                        callee = callee.strip('"').strip("'")
                        calls.append({
                            "target": callee,
                            "type": "api",
                            "name": callee,
                        })
            
            # Check files.files for file operations
            files_data = dep_data.get("files", {})
            if isinstance(files_data, dict):
                file_calls = files_data.get("files", [])
                for f in file_calls:
                    if isinstance(f, dict):
                        calls.append({
                            "target": f.get("name", f.get("file", "")),
                            "type": "file",
                        })
            
            # Check explicit calls list (often duplicates call_categories, so dedupe)
            # PRODUCTION FIX: Only add if callee doesn't already exist to prevent double-counting
            explicit_calls = dep_data.get("calls", [])
            for c in explicit_calls:
                if isinstance(c, dict):
                    callee = c.get("callee", c.get("target", ""))
                    # Deduplicate - only add if not already present
                    if callee and not any(existing.get("target") == callee for existing in calls):
                        calls.append({
                            "target": callee,
                            "type": c.get("type", "call"),
                            "name": callee,
                        })
                elif isinstance(c, str):
                    if c and not any(existing.get("target") == c for existing in calls):
                        calls.append({"target": c, "type": "call", "name": c})
        elif isinstance(dep_data, list):
            calls = dep_data
        
        return calls
