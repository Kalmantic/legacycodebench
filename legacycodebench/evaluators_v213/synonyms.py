"""
COBOL Synonym Expansion for Deterministic Matching

This module provides a frozen synonym dictionary for COBOL terminology
to enable smart matching without the non-determinism of embeddings.

Innovation: Synonyms are expanded BEFORE TF-IDF matching, ensuring
that "calculate" and "compute" are treated as equivalent.
"""

from typing import Dict, List
import re

from .config_v213 import V213_CONFIG


# Frozen synonym dictionary - DO NOT MODIFY after deployment
COBOL_SYNONYMS: Dict[str, List[str]] = V213_CONFIG["cobol_synonyms"]


def expand_synonyms(text: str) -> str:
    """
    Expand text by replacing synonyms with their base form.
    
    This ensures "calculate", "compute", and "determine" all become "compute".
    
    Args:
        text: Input text to expand
        
    Returns:
        Text with all synonyms normalized to base forms
        
    Example:
        >>> expand_synonyms("The premium is calculated by multiplying rate")
        'the premium is computed by multiplying rate'
    """
    result = text.lower()
    
    # Replace each synonym with its base form
    for base, alternatives in COBOL_SYNONYMS.items():
        for alt in alternatives:
            # Use word boundary matching to avoid partial replacements
            pattern = r'\b' + re.escape(alt) + r'\b'
            result = re.sub(pattern, base, result, flags=re.IGNORECASE)
    
    return result


def normalize_for_matching(text: str) -> str:
    """
    Normalize text for consistent matching.
    
    Steps:
    1. Convert to lowercase
    2. Expand synonyms
    3. Normalize whitespace
    4. Remove punctuation except hyphens (for COBOL variable names)
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text ready for matching
    """
    # Step 1: Lowercase
    result = text.lower()
    
    # Step 2: Expand synonyms
    result = expand_synonyms(result)
    
    # Step 3: Normalize whitespace
    result = ' '.join(result.split())
    
    # Step 4: Remove punctuation except hyphens
    result = re.sub(r'[^\w\s-]', ' ', result)
    result = ' '.join(result.split())
    
    return result


def extract_cobol_identifiers(text: str) -> set:
    """
    Extract COBOL-style identifiers from text.
    
    COBOL identifiers typically:
    - Start with a letter
    - Contain letters, digits, and hyphens
    - Are uppercase in code but may be mixed case in docs
    
    Args:
        text: Input text
        
    Returns:
        Set of normalized identifier names
    """
    # Pattern for COBOL identifiers (WS-VARIABLE-NAME style)
    pattern = r'\b([A-Za-z][-A-Za-z0-9]*(?:-[A-Za-z0-9]+)*)\b'
    matches = re.findall(pattern, text)
    
    # Filter out common English words and short matches
    common_words = {
        'the', 'and', 'for', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'shall', 'to', 'of', 'in',
        'on', 'at', 'by', 'with', 'from', 'as', 'if', 'when', 'then', 'else',
        'this', 'that', 'these', 'those', 'it', 'its', 'or', 'not', 'no', 'yes',
        'true', 'false', 'null', 'value', 'data', 'file', 'record', 'field',
    }
    
    identifiers = set()
    for match in matches:
        normalized = match.upper()
        # Keep if it's long enough and not a common word
        if len(normalized) > 2 and normalized.lower() not in common_words:
            identifiers.add(normalized)
    
    return identifiers


def fuzzy_match_identifier(candidate: str, valid_set: set, threshold: float = 0.8) -> bool:
    """
    Check if candidate identifier fuzzy-matches any valid identifier.
    
    Uses simple character-based similarity for determinism.
    
    Args:
        candidate: Identifier to check
        valid_set: Set of valid identifiers
        threshold: Minimum similarity (default 0.8)
        
    Returns:
        True if candidate matches any valid identifier above threshold
    """
    candidate_upper = candidate.upper()
    
    # Exact match
    if candidate_upper in valid_set:
        return True
    
    # Fuzzy match
    for valid in valid_set:
        similarity = _calculate_similarity(candidate_upper, valid)
        if similarity >= threshold:
            return True
    
    return False


def _calculate_similarity(s1: str, s2: str) -> float:
    """
    Calculate character-based similarity between two strings.
    
    Uses Jaccard similarity on character bigrams for determinism.
    """
    if not s1 or not s2:
        return 0.0
    
    if s1 == s2:
        return 1.0
    
    # Generate bigrams
    def bigrams(s):
        return set(s[i:i+2] for i in range(len(s)-1))
    
    b1 = bigrams(s1)
    b2 = bigrams(s2)
    
    if not b1 or not b2:
        return 0.0
    
    intersection = len(b1 & b2)
    union = len(b1 | b2)
    
    return intersection / union if union > 0 else 0.0
