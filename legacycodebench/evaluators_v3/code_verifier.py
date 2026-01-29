"""
Code-Based Static Verifier for V3.1

Verifies behavioral claims against actual COBOL source code structure,
replacing TF-IDF text matching with direct code verification.

This is the correct approach to static behavioral verification:
- Behavior is defined by code
- Code structure IS behavior specification
- Verifying claims against code = verifying behavioral fidelity

Usage:
    verifier = CodeBasedVerifier()
    result = verifier.verify(claims, source_code, ground_truth)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging

from .code_matchers import (
    ComputationMatcher,
    ConditionalMatcher,
    CallPerformMatcher,
    VariableFinder,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Result Data Classes
# ============================================================================

@dataclass
class ClaimVerificationResult:
    """Result of verifying a single claim against source code."""
    claim_text: str
    status: str  # "verified", "partial", "failed", "unverified"
    confidence: float  # 0.0 - 1.0
    method: str  # How it was verified
    reason: str  # Human-readable explanation


@dataclass
class CodeVerificationResult:
    """Complete result of code-based verification."""
    score: float  # 0.0 - 1.0
    verified: int
    partial: int
    failed: int
    unverified: int
    total: int
    details: List[ClaimVerificationResult] = field(default_factory=list)
    # V2.4.2 Hybrid scoring fields
    effective_claims: float = 0.0  # verified + partial * 0.5
    target_used: int = 6           # Target for full score
    scoring_mode: str = "hybrid"   # "hybrid" or "legacy"


# ============================================================================
# Action Matching Utilities
# ============================================================================

# V3.2 Enhanced Semantic Action Matching
# Maps action types found in code to claim keywords that indicate them
ACTION_KEYWORDS = {
    # Termination actions
    'stop_run': ['stop', 'terminate', 'end', 'abort', 'halt', 'exit', 'fail', 'error'],
    'goback': ['return', 'goback', 'exit', 'end'],
    'abend': ['abend', 'crash', 'abort', 'fail', 'terminate', 'error', 'fatal'],
    
    # Output actions
    'display': ['display', 'print', 'output', 'show', 'log', 'message', 'report',
                'error', 'warning', 'notify', 'alert', 'invalid', 'expired'],
    'write': ['write', 'output', 'save', 'record', 'store', 'log', 'file'],
    
    # Computation actions
    'compute': ['compute', 'calculate', 'increment', 'decrement', 'add', 
                'subtract', 'multiply', 'divide', 'update', 'count', 'total'],
    'set': ['set', 'assign', 'initialize', 'flag', 'mark', 'indicate'],
    
    # Data movement
    'move': ['move', 'assign', 'copy', 'transfer', 'store', 'save', 'set', 'place'],
    
    # Flow control
    'perform': ['perform', 'call', 'execute', 'invoke', 'run', 'process', 'handle'],
    'call': ['call', 'invoke', 'execute', 'run', 'interface'],
    
    # I/O actions  
    'read': ['read', 'input', 'fetch', 'load', 'retrieve', 'get', 'receive'],
}

# Reverse mapping: claim keywords to actions that satisfy them
# Used when we have domain-specific claim terms
CLAIM_KEYWORD_MATCHES = {
    # Validation/error results
    'expired': ['display', 'move', 'set', 'stop_run', 'goback', 'abend'],
    'invalid': ['display', 'move', 'set', 'stop_run', 'goback', 'abend'],
    'valid': ['move', 'set', 'perform', 'compute'],
    'validated': ['move', 'set', 'perform', 'compute', 'display'],
    'error': ['display', 'move', 'set', 'stop_run', 'goback', 'abend'],
    'fail': ['display', 'move', 'set', 'stop_run', 'goback', 'abend'],
    'success': ['move', 'set', 'perform', 'compute'],
    
    # Process results
    'processed': ['move', 'set', 'perform', 'compute', 'write'],
    'calculated': ['compute', 'move', 'set'],
    'determined': ['compute', 'move', 'set', 'perform'],
    'checked': ['perform', 'compute', 'move', 'set'],
    
    # Display-related
    'displayed': ['display', 'write'],
    'shown': ['display', 'write'],
    'printed': ['display', 'write'],
    'logged': ['display', 'write'],
    'reported': ['display', 'write'],
}


def action_matches(claimed_action: str, actual_actions: List[str]) -> bool:
    """
    Check if claimed action matches any actual action found in code.
    
    V3.2 Enhanced: Handles semantic matching like "expired" -> display/move/abend
    
    Args:
        claimed_action: Action described in claim text
        actual_actions: Actions found in code (e.g., ['display', 'perform'])
        
    Returns:
        True if there's a match
    """
    if not claimed_action or not actual_actions:
        return False
    
    claimed_lower = claimed_action.lower()
    actual_set = set(actual_actions)
    
    # Strategy 1: Direct match - action name in claim text
    for action in actual_actions:
        if action.replace('_', ' ') in claimed_lower:
            return True
    
    # Strategy 2: Action keywords match claim text
    for action in actual_actions:
        keywords = ACTION_KEYWORDS.get(action, [action])
        for kw in keywords:
            if kw in claimed_lower:
                return True
    
    # Strategy 3: Claim keywords map to found actions
    for claim_kw, matching_actions in CLAIM_KEYWORD_MATCHES.items():
        if claim_kw in claimed_lower:
            # If any of the matching actions was found in code, it's a match
            if actual_set & set(matching_actions):
                return True
    
    # Strategy 4: Fuzzy - any action satisfies an ambiguous claim
    # If the claim mentions something vague like "handled" or "processed"
    # and we found any non-trivial action, consider it a partial match
    vague_terms = ['handled', 'processed', 'executed', 'performed', 'done', 'completed']
    for term in vague_terms:
        if term in claimed_lower and len(actual_actions) > 0:
            # At least found some actions - weak match
            return True
    
    return False


# ============================================================================
# Code-Based Verifier
# ============================================================================

class CodeBasedVerifier:
    """
    Verify behavioral claims against actual COBOL source code.
    
    This replaces TF-IDF text matching with direct code structure verification.
    
    Verification Methods:
    - CALCULATION: Find computation statement targeting output_var
    - CONDITIONAL: Find IF/EVALUATE checking condition_var
    - ASSIGNMENT: Find MOVE/assignment to output_var
    - ERROR: Find error handling patterns
    
    Example:
        verifier = CodeBasedVerifier()
        result = verifier.verify(claims, source_code, ground_truth)
        print(f"Verified: {result.verified}/{result.total}")
    """
    
    def __init__(self):
        """Initialize with code matchers."""
        self.computation_matcher = ComputationMatcher()
        self.conditional_matcher = ConditionalMatcher()
        self.call_matcher = CallPerformMatcher()
        self.var_finder = VariableFinder()
    
    def verify(
        self,
        claims: List[Any],
        source_code: str,
        ground_truth: Dict = None,
    ) -> CodeVerificationResult:
        """
        Verify all claims against source code.
        
        Args:
            claims: List of Claim objects from ClaimExtractor
            source_code: COBOL source code
            ground_truth: Optional ground truth (for variable existence fallback)
            
        Returns:
            CodeVerificationResult with scores and details
        """
        logger.info(f"Code-based verification: {len(claims)} claims")
        
        if not claims:
            return CodeVerificationResult(
                score=1.0,
                verified=0,
                partial=0,
                failed=0,
                unverified=0,
                total=0,
                details=[],
            )
        
        results = []
        
        for claim in claims:
            result = self._verify_single_claim(claim, source_code, ground_truth)
            results.append(result)
        
        # Count results by status
        verified = sum(1 for r in results if r.status == 'verified')
        partial = sum(1 for r in results if r.status == 'partial')
        failed = sum(1 for r in results if r.status == 'failed')
        unverified = sum(1 for r in results if r.status == 'unverified')

        # V2.4.2: Hybrid claim scoring
        # Import config here to avoid circular imports
        from ..evaluators_v231.config_v231 import get_hybrid_scoring_config, get_claim_target

        config = get_hybrid_scoring_config()
        partial_weight = config.get("partial_weight", 0.5)
        effective = verified * 1.0 + partial * partial_weight

        if config.get("enabled", True):
            # Hybrid formula: score = min(effective / target, 1.0)
            target = get_claim_target()
            score = min(effective / target, 1.0) if target > 0 else 0.0
            scoring_mode = "hybrid"
        else:
            # Legacy formula: score = effective / total
            score = effective / len(claims) if claims else 0.0
            target = len(claims)
            scoring_mode = "legacy"

        logger.info(
            f"Code verification: {verified} verified, {partial} partial, "
            f"{failed} failed, effective={effective:.1f}, target={target}, "
            f"score={score:.2f} ({scoring_mode})"
        )

        return CodeVerificationResult(
            score=score,
            verified=verified,
            partial=partial,
            failed=failed,
            unverified=unverified,
            total=len(claims),
            details=results,
            effective_claims=effective,
            target_used=target,
            scoring_mode=scoring_mode,
        )
    
    def _verify_single_claim(
        self,
        claim: Any,
        source: str,
        ground_truth: Dict = None,
    ) -> ClaimVerificationResult:
        """
        Verify a single claim against source code.
        
        Routes to appropriate verification method based on claim type.
        """
        # Extract claim attributes
        claim_type = getattr(claim, 'claim_type', None)
        claim_text = getattr(claim, 'text', str(claim))[:100]
        output_var = getattr(claim, 'output_var', None)
        input_vars = getattr(claim, 'input_vars', []) or []
        components = getattr(claim, 'components', {}) or {}
        
        # Get claim type value
        type_value = claim_type.value if hasattr(claim_type, 'value') else str(claim_type)
        
        # Route to appropriate verifier
        if type_value == 'calculation':
            return self._verify_calculation(claim_text, output_var, input_vars, source)
        elif type_value == 'conditional':
            return self._verify_conditional(claim_text, output_var, input_vars, components, source)
        elif type_value == 'assignment':
            return self._verify_assignment(claim_text, output_var, input_vars, source)
        elif type_value == 'error':
            return self._verify_error(claim_text, output_var, input_vars, components, source)
        elif type_value == 'range':
            return self._verify_range(claim_text, output_var, input_vars, source)
        else:
            return self._verify_generic(claim_text, output_var, input_vars, source, ground_truth)
    
    # ========================================================================
    # CALCULATION Claims
    # ========================================================================
    
    def _verify_calculation(
        self,
        claim_text: str,
        output_var: Optional[str],
        input_vars: List[str],
        source: str,
    ) -> ClaimVerificationResult:
        """
        Verify calculation claims.
        
        Claim: "X is calculated from Y and Z"
        Verify: Find computation where X = f(Y, Z)
        """
        if not output_var:
            # No output variable - check if any input vars are used in computations
            if input_vars:
                for var in input_vars:
                    if self.var_finder.variable_exists(source, var):
                        return ClaimVerificationResult(
                            claim_text=claim_text,
                            status='partial',
                            confidence=0.4,
                            method='code_variable_exists',
                            reason=f'Variable {var} found in source but no output specified',
                        )
            
            return ClaimVerificationResult(
                claim_text=claim_text,
                status='unverified',
                confidence=0.0,
                method='code_calculation',
                reason='No output variable specified in claim',
            )
        
        # Find computations targeting output_var
        computations = self.computation_matcher.find_computations(source, output_var)
        
        if not computations:
            # Check if variable at least exists
            if self.var_finder.variable_exists(source, output_var):
                return ClaimVerificationResult(
                    claim_text=claim_text,
                    status='partial',
                    confidence=0.3,
                    method='code_variable_exists',
                    reason=f'{output_var} exists but no computation found',
                )
            
            return ClaimVerificationResult(
                claim_text=claim_text,
                status='failed',
                confidence=0.0,
                method='code_calculation',
                reason=f'No computation found for {output_var}',
            )
        
        # Check if input_vars appear in any computation
        for comp in computations:
            operands = self.computation_matcher.extract_operands(comp)
            operands_upper = [o.upper() for o in operands]
            
            if input_vars:
                matched = sum(1 for v in input_vars if v.upper() in operands_upper)
                total = len(input_vars)
                
                if matched == total:
                    return ClaimVerificationResult(
                        claim_text=claim_text,
                        status='verified',
                        confidence=1.0,
                        method='code_calculation_match',
                        reason=f'Found: {comp.statement} at line {comp.line}',
                    )
                elif matched > 0:
                    return ClaimVerificationResult(
                        claim_text=claim_text,
                        status='partial',
                        confidence=0.5 + (0.3 * matched / total),
                        method='code_calculation_partial',
                        reason=f'Found computation but only {matched}/{total} input vars match',
                    )
            else:
                # No input vars specified - partial credit for finding computation
                return ClaimVerificationResult(
                    claim_text=claim_text,
                    status='partial',
                    confidence=0.7,
                    method='code_output_computed',
                    reason=f'Found computation for {output_var}: {comp.statement}',
                )
        
        # Computation found but input vars don't match
        return ClaimVerificationResult(
            claim_text=claim_text,
            status='partial',
            confidence=0.5,
            method='code_computation_different',
            reason=f'Computation found for {output_var} but with different operands',
        )
    
    # ========================================================================
    # CONDITIONAL Claims
    # ========================================================================
    
    def _verify_conditional(
        self,
        claim_text: str,
        output_var: Optional[str],
        input_vars: List[str],
        components: Dict,
        source: str,
    ) -> ClaimVerificationResult:
        """
        Verify conditional claims.
        
        Claim: "When X = Y, action Z happens"
        Verify: Find IF checking X, with Z in THEN branch
        """
        # Get condition variable
        cond_var = input_vars[0] if input_vars else output_var
        
        if not cond_var:
            return ClaimVerificationResult(
                claim_text=claim_text,
                status='unverified',
                confidence=0.0,
                method='code_conditional',
                reason='No condition variable specified in claim',
            )
        
        # Find conditionals checking this variable
        conditionals = self.conditional_matcher.find_conditionals(source, cond_var)
        
        if not conditionals:
            # Check if variable at least exists
            if self.var_finder.variable_exists(source, cond_var):
                return ClaimVerificationResult(
                    claim_text=claim_text,
                    status='partial',
                    confidence=0.3,
                    method='code_variable_exists',
                    reason=f'{cond_var} exists but not in conditional',
                )
            
            return ClaimVerificationResult(
                claim_text=claim_text,
                status='failed',
                confidence=0.0,
                method='code_conditional',
                reason=f'No conditional found checking {cond_var}',
            )
        
        # Check if claimed action is in any THEN branch (with PERFORM following)
        claimed_action = components.get('action', '') or claim_text
        
        for cond in conditionals:
            # V3.2: Use deep action extraction to follow PERFORM chains
            then_actions = self.conditional_matcher.extract_actions_deep(
                cond, source, max_depth=2
            )
            
            if action_matches(claimed_action, then_actions):
                return ClaimVerificationResult(
                    claim_text=claim_text,
                    status='verified',
                    confidence=1.0,
                    method='code_conditional_match',
                    reason=f'Found IF {cond_var} at line {cond.line} with matching action',
                )
        
        # Conditional exists but action differs
        return ClaimVerificationResult(
            claim_text=claim_text,
            status='partial',
            confidence=0.6,
            method='code_conditional_partial',
            reason=f'{cond_var} is checked in conditional at line {conditionals[0].line}',
        )
    
    # ========================================================================
    # ASSIGNMENT Claims
    # ========================================================================
    
    def _verify_assignment(
        self,
        claim_text: str,
        output_var: Optional[str],
        input_vars: List[str],
        source: str,
    ) -> ClaimVerificationResult:
        """
        Verify assignment claims.
        
        Claim: "X is set to Y" or "X receives value from Y"
        Verify: Find MOVE/assignment to X
        """
        if not output_var:
            # Check input vars
            if input_vars:
                for var in input_vars:
                    if self.var_finder.variable_exists(source, var):
                        return ClaimVerificationResult(
                            claim_text=claim_text,
                            status='partial',
                            confidence=0.4,
                            method='code_variable_exists',
                            reason=f'Variable {var} found in source',
                        )
            
            return ClaimVerificationResult(
                claim_text=claim_text,
                status='unverified',
                confidence=0.0,
                method='code_assignment',
                reason='No target variable specified in claim',
            )
        
        # Find assignments/moves to output_var
        computations = self.computation_matcher.find_computations(source, output_var)
        
        # Filter to MOVE/assignment type
        moves = [c for c in computations if c.operation == 'move']
        
        if moves:
            move = moves[0]
            # Check if source var matches input_vars
            operands = self.computation_matcher.extract_operands(move)
            
            if input_vars and any(v.upper() in [o.upper() for o in operands] for v in input_vars):
                return ClaimVerificationResult(
                    claim_text=claim_text,
                    status='verified',
                    confidence=1.0,
                    method='code_assignment_match',
                    reason=f'Found: {move.statement} at line {move.line}',
                )
            
            return ClaimVerificationResult(
                claim_text=claim_text,
                status='partial',
                confidence=0.7,
                method='code_assignment_found',
                reason=f'Found assignment to {output_var} at line {move.line}',
            )
        
        # Any computation counts as partial
        if computations:
            return ClaimVerificationResult(
                claim_text=claim_text,
                status='partial',
                confidence=0.6,
                method='code_assignment_computed',
                reason=f'{output_var} is computed (not simple assignment)',
            )
        
        # Check if variable exists
        if self.var_finder.variable_exists(source, output_var):
            return ClaimVerificationResult(
                claim_text=claim_text,
                status='partial',
                confidence=0.3,
                method='code_variable_exists',
                reason=f'{output_var} exists in source',
            )
        
        return ClaimVerificationResult(
            claim_text=claim_text,
            status='failed',
            confidence=0.0,
            method='code_assignment',
            reason=f'No assignment found for {output_var}',
        )
    
    # ========================================================================
    # ERROR Claims
    # ========================================================================
    
    def _verify_error(
        self,
        claim_text: str,
        output_var: Optional[str],
        input_vars: List[str],
        components: Dict,
        source: str,
    ) -> ClaimVerificationResult:
        """
        Verify error handling claims.
        
        Claim: "If X fails, Y happens"
        Verify: Find error handling pattern
        """
        # Get relevant variable
        error_var = output_var or (input_vars[0] if input_vars else None)
        
        if error_var:
            # Find conditionals involving this variable
            conditionals = self.conditional_matcher.find_conditionals(source, error_var)
            
            if conditionals:
                for cond in conditionals:
                    actions = self.conditional_matcher.extract_actions(cond)
                    
                    # Error handling typically involves abend, display, or stop
                    if any(a in ['abend', 'stop_run', 'goback', 'display'] for a in actions):
                        return ClaimVerificationResult(
                            claim_text=claim_text,
                            status='verified',
                            confidence=1.0,
                            method='code_error_match',
                            reason=f'Found error handling for {error_var} at line {cond.line}',
                        )
                
                return ClaimVerificationResult(
                    claim_text=claim_text,
                    status='partial',
                    confidence=0.6,
                    method='code_error_partial',
                    reason=f'{error_var} is checked but error handling unclear',
                )
        
        # Check for generic error handling patterns
        error_patterns = [
            (r'ON\s+SIZE\s+ERROR', 'size error'),
            (r'INVALID\s+KEY', 'invalid key'),
            (r'AT\s+END', 'at end'),
            (r'FILE\s+STATUS', 'file status'),
            (r'NOT\s+ON\s+EXCEPTION', 'exception'),
            (r'ON\s+EXCEPTION', 'exception'),
        ]
        
        for pattern, desc in error_patterns:
            if re.search(pattern, source, re.IGNORECASE):
                return ClaimVerificationResult(
                    claim_text=claim_text,
                    status='partial',
                    confidence=0.5,
                    method='code_error_generic',
                    reason=f'Error handling pattern ({desc}) found in code',
                )
        
        # Check if any variable from claim exists
        if error_var and self.var_finder.variable_exists(source, error_var):
            return ClaimVerificationResult(
                claim_text=claim_text,
                status='partial',
                confidence=0.3,
                method='code_variable_exists',
                reason=f'{error_var} exists in source',
            )
        
        return ClaimVerificationResult(
            claim_text=claim_text,
            status='unverified',
            confidence=0.2,
            method='code_error',
            reason='Could not verify error handling claim',
        )
    
    # ========================================================================
    # RANGE Claims
    # ========================================================================
    
    def _verify_range(
        self,
        claim_text: str,
        output_var: Optional[str],
        input_vars: List[str],
        source: str,
    ) -> ClaimVerificationResult:
        """
        Verify range claims.
        
        Claim: "X must be between Y and Z"
        Verify: Find range checking for X
        """
        range_var = output_var or (input_vars[0] if input_vars else None)
        
        if not range_var:
            return ClaimVerificationResult(
                claim_text=claim_text,
                status='unverified',
                confidence=0.0,
                method='code_range',
                reason='No range variable specified in claim',
            )
        
        # Check if variable is used in conditionals (range checking)
        conditionals = self.conditional_matcher.find_conditionals(source, range_var)
        
        if conditionals:
            # Multiple conditionals might indicate range checking
            if len(conditionals) >= 2:
                return ClaimVerificationResult(
                    claim_text=claim_text,
                    status='verified',
                    confidence=0.9,
                    method='code_range_match',
                    reason=f'Found {len(conditionals)} conditionals checking {range_var}',
                )
            
            return ClaimVerificationResult(
                claim_text=claim_text,
                status='partial',
                confidence=0.6,
                method='code_range_partial',
                reason=f'{range_var} is checked in conditional',
            )
        
        if self.var_finder.variable_exists(source, range_var):
            return ClaimVerificationResult(
                claim_text=claim_text,
                status='partial',
                confidence=0.3,
                method='code_variable_exists',
                reason=f'{range_var} exists in source',
            )
        
        return ClaimVerificationResult(
            claim_text=claim_text,
            status='failed',
            confidence=0.0,
            method='code_range',
            reason=f'No range checking found for {range_var}',
        )
    
    # ========================================================================
    # GENERIC Claims
    # ========================================================================
    
    def _verify_generic(
        self,
        claim_text: str,
        output_var: Optional[str],
        input_vars: List[str],
        source: str,
        ground_truth: Dict = None,
    ) -> ClaimVerificationResult:
        """
        Verify generic/unknown claim types.
        
        Falls back to checking if mentioned variables exist in source.
        """
        all_vars = []
        if output_var:
            all_vars.append(output_var)
        all_vars.extend(input_vars or [])
        
        if not all_vars:
            return ClaimVerificationResult(
                claim_text=claim_text,
                status='unverified',
                confidence=0.0,
                method='code_generic',
                reason='No variables to verify',
            )
        
        # Check which variables exist
        existing = [v for v in all_vars if self.var_finder.variable_exists(source, v)]
        
        if len(existing) == len(all_vars):
            return ClaimVerificationResult(
                claim_text=claim_text,
                status='partial',
                confidence=0.5,
                method='code_all_vars_exist',
                reason=f'All variables exist: {existing}',
            )
        elif existing:
            return ClaimVerificationResult(
                claim_text=claim_text,
                status='partial',
                confidence=0.3,
                method='code_some_vars_exist',
                reason=f'{len(existing)}/{len(all_vars)} variables found',
            )
        
        return ClaimVerificationResult(
            claim_text=claim_text,
            status='failed',
            confidence=0.0,
            method='code_generic',
            reason='No claimed variables found in source',
        )
