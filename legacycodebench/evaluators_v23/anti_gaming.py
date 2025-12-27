"""
Anti-Gaming Mechanisms for V2.3

Detects and penalizes gaming attempts:
1. Keyword Stuffing - Keywords without proper context
2. Parroting - Copying source code comments
3. Abstraction Scoring - Requires WHY not just WHAT
4. Contamination Checking - Training data leakage
"""

import re
from dataclasses import dataclass
from typing import List, Set, Dict
from difflib import SequenceMatcher

from .config_v23 import V23_CONFIG


@dataclass
class AntiGamingResult:
    """Result of anti-gaming analysis."""
    keyword_stuffing_score: float  # 0 = no stuffing, 1 = heavy stuffing
    parroting_score: float         # 0 = original, 1 = complete copy
    abstraction_score: float       # 0 = just WHAT, 1 = explains WHY
    contamination_detected: bool
    penalties_applied: Dict[str, float]
    details: Dict
    
    @property
    def total_penalty(self) -> float:
        """Total penalty to apply to comprehension score."""
        return sum(self.penalties_applied.values())
    
    @property
    def is_gaming_detected(self) -> bool:
        """Was any form of gaming detected?"""
        config = V23_CONFIG["anti_gaming"]
        return (
            self.keyword_stuffing_score > config["keyword_stuffing_threshold"] or
            self.parroting_score > config["parroting_threshold"] or
            self.abstraction_score < config["abstraction_minimum"] or
            self.contamination_detected
        )


class KeywordStuffingDetector:
    """Detects keywords mentioned without proper context."""
    
    # COBOL keywords that should have context
    COBOL_KEYWORDS = [
        "COMPUTE", "IF", "EVALUATE", "PERFORM", "MOVE", "CALL",
        "EXEC", "READ", "WRITE", "ADD", "SUBTRACT", "MULTIPLY",
        "DIVIDE", "OPEN", "CLOSE", "ACCEPT", "DISPLAY", "STOP",
        "GO", "ALTER", "INSPECT", "STRING", "UNSTRING", "SEARCH"
    ]
    
    def detect(self, documentation: str, source_code: str) -> float:
        """
        Detect keyword stuffing.
        
        Returns score from 0 (no stuffing) to 1 (heavy stuffing).
        """
        doc_lower = documentation.lower()
        
        # Count keyword mentions
        mentions = {}
        for kw in self.COBOL_KEYWORDS:
            count = len(re.findall(rf'\b{kw}\b', documentation, re.IGNORECASE))
            if count > 0:
                mentions[kw] = count
        
        if not mentions:
            return 0.0
        
        # Check if keywords have proper context (in a sentence with >5 words)
        contextual = 0
        total = sum(mentions.values())
        
        # Split into sentences
        sentences = re.split(r'[.!?]', documentation)
        
        for kw in mentions:
            for sentence in sentences:
                if re.search(rf'\b{kw}\b', sentence, re.IGNORECASE):
                    words = sentence.split()
                    if len(words) >= 5:  # Has context
                        contextual += 1
                        break
        
        # Stuffing = keywords without context
        stuffing_ratio = 1 - (contextual / len(mentions)) if mentions else 0
        
        return max(0, min(1, stuffing_ratio))


class ParrotingDetector:
    """Detects when documentation copies source code comments."""
    
    def detect(self, documentation: str, source_code: str) -> float:
        """
        Detect parroting (copying source comments).
        
        Returns score from 0 (original) to 1 (complete copy).
        """
        # Extract comments from COBOL source
        comments = self._extract_comments(source_code)
        
        if not comments:
            return 0.0
        
        # Check how many comments appear in documentation
        doc_lower = documentation.lower()
        copied_count = 0
        
        for comment in comments:
            comment_clean = comment.strip().lower()
            # Only check substantial comments (>10 chars)
            if len(comment_clean) > 10:
                if comment_clean in doc_lower:
                    copied_count += 1
                elif self._is_similar(comment_clean, doc_lower, threshold=0.8):
                    copied_count += 0.5
        
        substantial_comments = [c for c in comments if len(c.strip()) > 10]
        if not substantial_comments:
            return 0.0
        
        return min(1.0, copied_count / len(substantial_comments))
    
    def _extract_comments(self, source: str) -> List[str]:
        """Extract comments from COBOL source."""
        comments = []
        
        # Fixed-format comment lines (column 7 = *)
        comments.extend(re.findall(r'^\s{6}\*(.+)$', source, re.MULTILINE))
        
        # Inline comments (*>)
        comments.extend(re.findall(r'\*>(.+)$', source, re.MULTILINE))
        
        return [c.strip() for c in comments if c.strip()]
    
    def _is_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if text1 appears similar to any part of text2."""
        # Sliding window comparison
        window_size = len(text1)
        text2_words = text2.split()
        
        for i in range(len(text2_words) - window_size // 4):
            window = ' '.join(text2_words[i:i + window_size // 2])
            ratio = SequenceMatcher(None, text1, window).ratio()
            if ratio >= threshold:
                return True
        
        return False


class AbstractionScorer:
    """Scores documentation based on abstraction level (WHY vs WHAT)."""
    
    # Indicators of explaining WHY (high abstraction)
    WHY_INDICATORS = [
        "because", "in order to", "purpose", "ensures", "validates",
        "business requirement", "calculates for", "determines whether",
        "to prevent", "to allow", "enables", "facilitates", "requirement",
        "business rule", "policy", "according to", "must be", "should be",
        "to verify", "to confirm", "to check", "business logic",
        "regulatory", "compliance", "rule states", "when applicable"
    ]
    
    # Indicators of just describing WHAT (low abstraction)
    WHAT_INDICATORS = [
        "performs", "executes", "runs", "does", "calls", "moves",
        "computes", "reads", "writes", "opens", "closes", "adds",
        "subtracts", "multiplies", "divides", "sets", "assigns",
        "increments", "decrements", "initializes", "terminates"
    ]
    
    def score(self, documentation: str) -> float:
        """
        Score abstraction level.
        
        Returns score from 0 (just WHAT) to 1 (explains WHY).
        """
        doc_lower = documentation.lower()
        
        why_count = sum(1 for ind in self.WHY_INDICATORS if ind in doc_lower)
        what_count = sum(1 for ind in self.WHAT_INDICATORS if ind in doc_lower)
        
        total = why_count + what_count
        if total == 0:
            return 0.5  # Neutral if neither found
        
        return why_count / total


class ContaminationChecker:
    """Checks for potential training data contamination."""
    
    # Known canary strings that would indicate contamination
    CANARY_STRINGS = [
        "LCB-CANARY-2025",
        "LEGACYCODEBENCH-TEST-STRING",
        "XYZZY-MAGIC-PLUGH"
    ]
    
    def check(self, documentation: str, task_id: str = None) -> bool:
        """
        Check for contamination indicators.
        
        Returns True if contamination is suspected.
        """
        doc_upper = documentation.upper()
        
        # Check for canary strings
        for canary in self.CANARY_STRINGS:
            if canary in doc_upper:
                return True
        
        # Check for suspiciously specific details
        # (This would be expanded with actual held-out task content)
        
        return False


class AntiGamingAnalyzer:
    """Main anti-gaming analysis coordinator."""
    
    def __init__(self):
        self.keyword_detector = KeywordStuffingDetector()
        self.parroting_detector = ParrotingDetector()
        self.abstraction_scorer = AbstractionScorer()
        self.contamination_checker = ContaminationChecker()
        self.config = V23_CONFIG["anti_gaming"]
    
    def analyze(
        self,
        documentation: str,
        source_code: str,
        task_id: str = None
    ) -> AntiGamingResult:
        """
        Run all anti-gaming checks.
        
        Returns AntiGamingResult with scores and penalties.
        """
        # Run detectors
        keyword_score = self.keyword_detector.detect(documentation, source_code)
        parroting_score = self.parroting_detector.detect(documentation, source_code)
        abstraction_score = self.abstraction_scorer.score(documentation)
        contamination = self.contamination_checker.check(documentation, task_id)
        
        # Calculate penalties
        penalties = {}
        
        if keyword_score > self.config["keyword_stuffing_threshold"]:
            excess = keyword_score - self.config["keyword_stuffing_threshold"]
            penalties["keyword_stuffing"] = min(
                excess * self.config["keyword_stuffing_max_penalty"] / 
                (1 - self.config["keyword_stuffing_threshold"]),
                self.config["keyword_stuffing_max_penalty"]
            )
        
        if parroting_score > self.config["parroting_threshold"]:
            excess = parroting_score - self.config["parroting_threshold"]
            penalties["parroting"] = min(
                excess * self.config["parroting_max_penalty"] /
                (1 - self.config["parroting_threshold"]),
                self.config["parroting_max_penalty"]
            )
        
        if abstraction_score < self.config["abstraction_minimum"]:
            penalties["abstraction"] = self.config["abstraction_penalty"]
        
        return AntiGamingResult(
            keyword_stuffing_score=keyword_score,
            parroting_score=parroting_score,
            abstraction_score=abstraction_score,
            contamination_detected=contamination,
            penalties_applied=penalties,
            details={
                "keyword_threshold": self.config["keyword_stuffing_threshold"],
                "parroting_threshold": self.config["parroting_threshold"],
                "abstraction_minimum": self.config["abstraction_minimum"]
            }
        )
