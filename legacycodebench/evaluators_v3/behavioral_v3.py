"""
Behavioral Fidelity Evaluator V3.0

The main evaluator class that implements compile-first classification
and honest scoring.

Key features:
1. Compile-first: Try to compile, let compiler tell us the truth
2. Two methods: Executed (tests) or Static (claims + BSM)
3. Full provenance: Every result explains HOW it was calculated
4. No silent fallbacks: No heuristic that pretends all claims passed

Design: docs/v2.3.2/BF_V3_DESIGN.md
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from .bf_result import (
    BFResult, 
    VerificationMode,
    create_executed_result,
    create_static_result,
    create_error_result,
)
from .compile_classifier import (
    classify_compile_error,
    CompileClassification,
)
from .copybook_resolver import CopybookResolver, resolve_copybooks
from .static_verifier import StaticVerifier
from .code_verifier import CodeBasedVerifier  # V3.1: Code-based static verification

# Import from existing modules
try:
    from ..evaluators_v231.claim_extractor import ClaimExtractor
    from ..evaluators_v231.test_generator import TestGenerator
    from ..evaluators_v231.behavioral_v231 import BSMValidator as LegacyBSMValidator
except ImportError:
    ClaimExtractor = None
    TestGenerator = None
    LegacyBSMValidator = None

logger = logging.getLogger(__name__)


# ==================== CONFIGURATION ====================

# Score weights for different verification modes
EXECUTED_CLAIM_WEIGHT = 0.60  # 60% tests/claims, 40% BSM for executed
EXECUTED_BSM_WEIGHT = 0.40

STATIC_CLAIM_WEIGHT = 0.50  # 50/50 split for static (both are static analysis)
STATIC_BSM_WEIGHT = 0.50

# Minimum claims threshold (silence penalty)
MIN_CLAIMS = 1


class BehavioralEvaluatorV3:
    """
    Behavioral Fidelity Evaluator with compile-first classification.
    
    This evaluator:
    1. Resolves copybooks and tries to compile
    2. Routes to execution or static based on compile result
    3. Returns detailed BFResult with full provenance
    
    Usage:
        evaluator = BehavioralEvaluatorV3(executor=COBOLExecutor())
        result = evaluator.evaluate(
            documentation="The program calculates...",
            source_code="IDENTIFICATION DIVISION...",
            ground_truth={"business_rules": [...], ...},
            task_metadata={"task_id": "LCB-T1-001", ...}
        )
        print(result.summary())  # BF: 78.0% (executed, 12/15 tests)
    """
    
    def __init__(
        self,
        executor=None,
        dataset_path: Path = None,
        enable_execution: bool = True,
    ):
        """
        Initialize the evaluator.
        
        Args:
            executor: COBOLExecutor instance for running tests (optional)
            dataset_path: Path to dataset for copybook resolution
            enable_execution: Whether to attempt execution (vs always static)
        """
        self.executor = executor
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.enable_execution = enable_execution
        
        # Initialize components
        self.claim_extractor = ClaimExtractor() if ClaimExtractor else None
        self.test_generator = TestGenerator() if TestGenerator else None
        self.static_verifier = StaticVerifier()  # Legacy TF-IDF verifier (kept for BSM)
        self.code_verifier = CodeBasedVerifier()  # V3.1: Code-based verifier
        
        if not self.claim_extractor:
            logger.warning("ClaimExtractor not available - claim extraction disabled")
        if not self.executor:
            logger.info("No executor provided - execution disabled, will use static verification")
        
        logger.info(f"BehavioralEvaluatorV3 initialized (execution={'enabled' if enable_execution and executor else 'disabled'})")
    
    def evaluate(
        self,
        documentation: str,
        source_code: str,
        ground_truth: Dict,
        task_metadata: Dict = None,
    ) -> BFResult:
        """
        Evaluate behavioral fidelity of documentation.
        
        This is the main entry point. It:
        1. Extracts claims from documentation
        2. Tries to compile if execution is enabled
        3. Routes to appropriate verification method
        4. Returns detailed BFResult
        
        Args:
            documentation: AI-generated documentation
            source_code: Original COBOL source code
            ground_truth: Static analysis ground truth
            task_metadata: Optional task metadata (task_id, tier, etc.)
            
        Returns:
            BFResult with score and full provenance
        """
        task_id = task_metadata.get("task_id", "unknown") if task_metadata else "unknown"
        logger.info(f"BF V3.0 evaluation starting for {task_id}")
        
        # Stage 1: Extract claims from documentation
        logger.debug("Stage 1: Extracting claims ⏳")
        claims = self._extract_claims(documentation)
        logger.debug(f"Stage 1: Extracted {len(claims)} claims ✅")
        
        # Check silence penalty
        if len(claims) < MIN_CLAIMS:
            logger.warning(f"Silence penalty: only {len(claims)} claims extracted (min={MIN_CLAIMS})")
            return BFResult(
                score=0.0,
                verification_mode=VerificationMode.ERROR,
                mode_reason="silence_penalty",
                silence_penalty=True,
                claims_total=len(claims),
            )
        
        # Stage 2: Determine execution capability
        logger.debug("Stage 2: Determining verification method ⏳")
        
        if not self.enable_execution or not self.executor:
            # No executor - use static
            logger.info("Execution disabled - using static verification")
            return self._evaluate_static(
                documentation, source_code, ground_truth, claims,
                reason="execution_disabled"
            )
        
        # Stage 3: Try compilation
        logger.debug("Stage 3: Attempting compilation ⏳")
        compile_result = self._try_compile(source_code, task_metadata)
        
        if compile_result.success:
            # Stage 4a: Execute tests
            logger.debug("Stage 4: Executing tests ⏳")
            return self._evaluate_executed(
                documentation, source_code, ground_truth, claims,
                compile_result
            )
        else:
            # Stage 4b: Static verification
            logger.debug("Stage 4: Static verification (compilation failed) ⏳")
            classification = classify_compile_error(compile_result.error)
            return self._evaluate_static(
                documentation, source_code, ground_truth, claims,
                reason=classification.reason,
                compile_error=compile_result.error,
                fixable=classification.fixable,
            )
    
    def _extract_claims(self, documentation: str) -> List[Any]:
        """
        Extract behavioral claims from documentation.
        
        Args:
            documentation: AI-generated documentation
            
        Returns:
            List of Claim objects
        """
        if not self.claim_extractor:
            # Fallback: create mock claims for testing
            logger.warning("Using mock claims (ClaimExtractor not available)")
            return []
        
        try:
            claims = self.claim_extractor.extract(documentation)
            logger.info(f"Extracted {len(claims)} claims from documentation")
            return claims
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            return []
    
    def _try_compile(self, source_code: str, task_metadata: Dict) -> 'CompileResult':
        """
        Try to compile the COBOL source.
        
        Resolves copybooks and attempts compilation via executor.
        
        Args:
            source_code: COBOL source code
            task_metadata: Task metadata for dataset path
            
        Returns:
            CompileResult with success flag and error message
        """
        # Resolve dataset path
        dataset_path = self._get_dataset_path(task_metadata)
        
        # V2.4: Get language from task metadata (copybooks are COBOL-specific)
        task_language = task_metadata.get("language", "COBOL").lower() if task_metadata else "cobol"
        
        # Resolve copybooks ONLY from task's own dataset AND only for COBOL
        copybook_paths = []
        missing_copybooks = []
        if dataset_path and task_language == "cobol":
            try:
                copybook_paths, missing_copybooks = resolve_copybooks(source_code, dataset_path)
                if copybook_paths:
                    logger.info(f"Resolved {len(copybook_paths)} copybooks from {dataset_path}")
                if missing_copybooks:
                    logger.warning(f"Missing copybooks (not in task dataset): {missing_copybooks[:5]}")
            except Exception as e:
                logger.warning(f"Copybook resolution failed: {e}")
        elif task_language != "cobol":
            logger.debug(f"Skipping copybook resolution for {task_language.upper()} (COBOL-specific feature)")
        else:
            logger.warning("No dataset_path - cannot resolve copybooks")
        
        
        # Try compilation
        try:
            # Use executor's compile method if available (must be a real method, not Mock)
            if hasattr(self.executor, 'compile') and callable(getattr(self.executor, 'compile', None)):
                compile_method = getattr(self.executor, 'compile')
                # Check if it's a real method (not a Mock auto-attribute)
                if not str(type(compile_method)).startswith("<class 'unittest.mock"):
                    # V2.4: Only pass copybook_paths for COBOL
                    if task_language == "cobol":
                        result = compile_method(source_code, copybook_paths)
                    else:
                        result = compile_method(source_code, task_metadata.get("task_id", "unknown"))
                    
                    # Handle tuple return (from unibasic_executor)
                    if isinstance(result, tuple):
                        # UniBasic executor returns (success, output, reason)
                        if len(result) == 3:
                            success, output, _ = result
                        else:
                            success, output = result[:2]
                            
                        return CompileResult(
                            success=success,
                            error=output if not success else None,
                            binary=None
                        )

                    return CompileResult(
                        success=result.success,
                        error=getattr(result, 'error_message', None) or getattr(result, 'stderr', ''),
                        binary=getattr(result, 'binary', None),
                    )
            
            # Fall back to execute with minimal inputs to test compilation
            # V2.4: Pass copybook paths ONLY for COBOL (UniBasic doesn't use them)
            if task_language == "cobol":
                result = self.executor.execute(source_code, {}, copybook_paths=copybook_paths)
            else:
                result = self.executor.execute(source_code, task_metadata.get("task_id", "unknown"))

            
            # Extract error message (handle Mock objects properly)
            error_msg = str(getattr(result, 'error_message', '') or '')
            stderr = str(getattr(result, 'stderr', '') or '')
            combined_error = f"{error_msg}\n{stderr}".strip()
            
            # If execution succeeded, compilation succeeded
            if result.success:
                return CompileResult(success=True, error=None, binary=None)
            
            # Execution failed - check if it's a compile error or runtime error
            # Look for compilation-related keywords
            compile_keywords = [
                'error:', 'compilation failed', 'no such file', 
                "unknown statement", 'syntax error', 'undefined',
                'cannot find', 'fatal error'
            ]
            
            is_compile_error = any(
                kw in combined_error.lower() 
                for kw in compile_keywords
            )
            
            if is_compile_error:
                logger.info(f"Compilation failed: {combined_error[:100]}...")
                return CompileResult(success=False, error=combined_error, binary=None)
            
            # Runtime error (execution failed but compilation likely succeeded)
            # This is rare - most failures during compile test are compile errors
            logger.info("Execution failed but appears to be runtime error")
            return CompileResult(success=True, error=None, binary=None)
            
        except Exception as e:
            logger.error(f"Compilation attempt failed: {e}")
            return CompileResult(success=False, error=str(e), binary=None)
    
    def _evaluate_executed(
        self,
        documentation: str,
        source_code: str,
        ground_truth: Dict,
        claims: List[Any],
        compile_result: 'CompileResult',
    ) -> BFResult:
        """
        Evaluate using test execution.
        
        Args:
            documentation: AI-generated documentation
            source_code: COBOL source
            ground_truth: Static analysis ground truth
            claims: Extracted claims
            compile_result: Successful compilation result
            
        Returns:
            BFResult with execution-verified score
        """
        logger.info("Evaluating via test execution")
        
        # Generate tests from claims
        tests = self._generate_tests(claims, ground_truth)
        
        if not tests:
            logger.warning("No tests generated - falling back to static")
            return self._evaluate_static(
                documentation, source_code, ground_truth, claims,
                reason="no_tests_generated",
                compile_error=None,
                fixable=True,
            )
        
        # Execute tests
        passed = 0
        failed = 0
        test_details = []
        
        for test in tests:
            try:
                test_inputs = test.inputs if hasattr(test, 'inputs') else {}
                result = self.executor.execute(source_code, test_inputs)
                
                if result.success:
                    # Validate outputs if expected
                    if hasattr(test, 'expected_outputs') and test.expected_outputs:
                        if self._validate_outputs(result, test.expected_outputs):
                            passed += 1
                            test_details.append({"test": test.test_id, "status": "passed"})
                        else:
                            failed += 1
                            test_details.append({"test": test.test_id, "status": "failed", "reason": "output_mismatch"})
                    else:
                        passed += 1
                        test_details.append({"test": test.test_id, "status": "passed"})
                else:
                    failed += 1
                    test_details.append({
                        "test": test.test_id, 
                        "status": "failed", 
                        "reason": result.error or "execution_failed"
                    })
            except Exception as e:
                logger.warning(f"Test execution error: {e}")
                failed += 1
                test_details.append({"test": getattr(test, 'test_id', 'unknown'), "status": "error", "reason": str(e)})
        
        # BSM validation
        external_calls = ground_truth.get("external_calls", [])
        bsm_result = self.static_verifier.bsm_validator.validate(documentation, external_calls)
        
        # Calculate scores
        test_score = passed / (passed + failed) if (passed + failed) > 0 else 0.0
        bsm_score = bsm_result["score"]
        
        # Weighted score: 60% tests, 40% BSM for executed
        final_score = (EXECUTED_CLAIM_WEIGHT * test_score) + (EXECUTED_BSM_WEIGHT * bsm_score)
        
        logger.info(f"Execution complete: {passed}/{passed+failed} tests, BF={final_score:.2%}")
        
        return create_executed_result(
            score=final_score,
            tests_passed=passed,
            tests_failed=failed,
            claim_score=test_score,
            bsm_score=bsm_score,
            bsm_matched=bsm_result["matched"],
            bsm_total=bsm_result["total"],
            test_details=test_details,
            bsm_details=bsm_result.get("details", []),
        )
    
    def _evaluate_static(
        self,
        documentation: str,
        source_code: str,
        ground_truth: Dict,
        claims: List[Any],
        reason: str,
        compile_error: str = None,
        fixable: bool = None,
    ) -> BFResult:
        """
        Evaluate using static verification.
        
        V3.1 UPDATE: Uses CODE-BASED verification instead of TF-IDF.
        Claims are verified against actual source code structure, not
        ground truth text descriptions.
        
        Args:
            documentation: AI-generated documentation
            source_code: COBOL source
            ground_truth: Static analysis ground truth
            claims: Extracted claims
            reason: Why we're using static (ibm_construct, copybook_missing, etc.)
            compile_error: Compilation error if any
            fixable: Whether the issue might be fixable
            
        Returns:
            BFResult with static-verified score
        """
        logger.info(f"Evaluating via code-based static verification: {reason}")
        
        # V3.1: Primary verification against SOURCE CODE (not TF-IDF)
        code_result = self.code_verifier.verify(claims, source_code, ground_truth)
        
        # V3.0: BSM validation (external calls) - still uses old verifier
        legacy_result = self.static_verifier.verify(
            claims, ground_truth, source_code, documentation
        )
        bsm_score = legacy_result.bsm_score
        bsm_matched = legacy_result.bsm_matched
        bsm_total = legacy_result.bsm_total
        bsm_details = legacy_result.bsm_details
        
        # V3.1 Weights: 60% code verification, 40% BSM
        CODE_WEIGHT = 0.60
        BSM_WEIGHT = 0.40
        
        final_score = (CODE_WEIGHT * code_result.score) + (BSM_WEIGHT * bsm_score)
        
        logger.info(
            f"Code-based verification: {code_result.verified} verified, "
            f"{code_result.partial} partial, {code_result.failed} failed, "
            f"{code_result.unverified} unverified"
        )
        logger.info(
            f"Static verification complete: {code_result.verified}/{code_result.total} claims, "
            f"BF={final_score:.2%}"
        )
        
        return create_static_result(
            score=final_score,
            reason=reason,
            claims_verified=code_result.verified,
            claims_failed=code_result.failed,
            claims_total=code_result.total,
            claim_score=code_result.score,
            bsm_score=bsm_score,
            bsm_matched=bsm_matched,
            bsm_total=bsm_total,
            compile_error=compile_error,
            fixable=fixable,
            claim_details=[
                {
                    "claim": d.claim_text,
                    "status": d.status,
                    "confidence": d.confidence,
                    "method": d.method,
                    "reason": d.reason,
                }
                for d in code_result.details
            ],
            bsm_details=bsm_details,
            # V2.4.2: Hybrid claim scoring
            claims_partial=code_result.partial,
            effective_claims=code_result.effective_claims,
            claim_target=code_result.target_used,
            scoring_mode=code_result.scoring_mode,
        )
    
    def _generate_tests(self, claims: List[Any], ground_truth: Dict) -> List[Any]:
        """
        Generate test cases from claims.
        
        Args:
            claims: Extracted claims
            ground_truth: Static analysis ground truth
            
        Returns:
            List of TestCase objects
        """
        if not self.test_generator:
            logger.warning("TestGenerator not available")
            return []
        
        try:
            tests = self.test_generator.generate(claims, ground_truth)
            logger.info(f"Generated {len(tests)} tests from {len(claims)} claims")
            return tests[:15]  # Cap at 15 tests
        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            return []
    
    def _validate_outputs(self, result: Any, expected_outputs: Dict) -> bool:
        """
        Validate execution outputs against expected values.
        
        Args:
            result: ExecutionResult from COBOL executor
            expected_outputs: Dict of variable → expected value
            
        Returns:
            True if all expected outputs match
        """
        if not expected_outputs:
            return True
        
        stdout = getattr(result, 'stdout', '') or ''
        
        for var_name, expected in expected_outputs.items():
            expected_str = str(expected).upper()
            if expected_str not in stdout.upper():
                return False
        
        return True
    
    def _get_dataset_path(self, task_metadata: Dict) -> Optional[Path]:
        """
        Get dataset path from task metadata.
        
        Args:
            task_metadata: Task metadata
            
        Returns:
            Path to dataset, or None
        """
        if self.dataset_path:
            return self.dataset_path
        
        if not task_metadata:
            return None
        
        # Try to infer from task metadata
        source_dataset = task_metadata.get("source_dataset")
        if source_dataset:
            # Assume datasets are in standard location
            return Path("datasets") / source_dataset
        
        return None


class CompileResult:
    """Result of compilation attempt."""
    
    def __init__(self, success: bool, error: str = None, binary: Any = None):
        self.success = success
        self.error = error
        self.binary = binary
