"""
Documentation Quality Evaluator V2.3.1 (Track 2)

Weight: 20% of total LCB Score

Evaluates documentation quality using algorithmic metrics ONLY:
- Structure (30%): Required sections present
- Traceability (30%): Line citations valid
- Readability (20%): Flesch-Kincaid grade
- Abstraction (20%): WHY vs WHAT ratio

NO LLM CALLS - All metrics are deterministic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re
import logging

from .config_v213 import V213_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class DocumentationResult:
    """Result of documentation quality evaluation."""
    score: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    details: Dict = field(default_factory=dict)


class DocumentationEvaluatorV213:
    """
    V2.3.1 Documentation Quality Evaluator
    
    All metrics are purely algorithmic - no LLM calls.
    """
    
    # Required sections for complete documentation
    REQUIRED_SECTIONS = [
        (r"(?:program\s+)?(?:description|overview|purpose|summary)", "description"),
        (r"(?:input|data\s+)?input", "input"),
        (r"(?:output|data\s+)?output", "output"),
        (r"(?:processing|logic|business\s+rules?)", "processing"),
    ]
    
    # WHY-words indicate explanation (good)
    WHY_WORDS = {
        "because", "since", "therefore", "thus", "hence", "consequently",
        "in order to", "so that", "to ensure", "to prevent", "to allow",
        "enables", "ensures", "prevents", "validates", "purpose", "reason",
    }
    
    # WHAT-words indicate description (neutral)
    WHAT_WORDS = {
        "is", "are", "contains", "has", "includes", "consists", "stores",
        "holds", "represents", "defines", "specifies", "sets", "gets",
    }
    
    def __init__(self):
        self.weights = V213_CONFIG["documentation_weights"]
        self.readability_config = V213_CONFIG["readability"]
    
    def evaluate(
        self,
        documentation: str,
        source_code: str = ""
    ) -> DocumentationResult:
        """
        Evaluate documentation quality.
        
        Args:
            documentation: AI-generated documentation
            source_code: Original COBOL source (for traceability validation)
            
        Returns:
            DocumentationResult with score and breakdown
        """
        logger.debug("Stage 1: Structure Evaluation [...]")
        structure_score, structure_details = self._evaluate_structure(documentation)
        logger.debug("Stage 1: Structure Evaluation [OK]")
        
        logger.debug("Stage 2: Traceability Evaluation [...]")
        trace_score, trace_details = self._evaluate_traceability(documentation, source_code)
        logger.debug("Stage 2: Traceability Evaluation [OK]")
        
        logger.debug("Stage 3: Readability Evaluation [...]")
        read_score, read_details = self._evaluate_readability(documentation)
        logger.debug("Stage 3: Readability Evaluation [OK]")
        
        logger.debug("Stage 4: Abstraction Evaluation [...]")
        abstract_score, abstract_details = self._evaluate_abstraction(documentation)
        logger.debug("Stage 4: Abstraction Evaluation [OK]")
        
        # Calculate weighted score
        overall = (
            self.weights["structure"] * structure_score +
            self.weights["traceability"] * trace_score +
            self.weights["readability"] * read_score +
            self.weights["abstraction"] * abstract_score
        )
        
        return DocumentationResult(
            score=overall,
            breakdown={
                "structure": structure_score,
                "traceability": trace_score,
                "readability": read_score,
                "abstraction": abstract_score,
            },
            details={
                "structure": structure_details,
                "traceability": trace_details,
                "readability": read_details,
                "abstraction": abstract_details,
            }
        )
    
    def _evaluate_structure(self, documentation: str) -> Tuple[float, Dict]:
        """
        Evaluate if required sections are present.
        
        Returns:
            (score, details)
        """
        doc_lower = documentation.lower()
        found_sections = []
        missing_sections = []
        
        for pattern, name in self.REQUIRED_SECTIONS:
            if re.search(pattern, doc_lower, re.IGNORECASE):
                found_sections.append(name)
            else:
                missing_sections.append(name)
        
        score = len(found_sections) / len(self.REQUIRED_SECTIONS)
        
        return score, {
            "found": found_sections,
            "missing": missing_sections,
            "total": len(self.REQUIRED_SECTIONS),
        }
    
    def _evaluate_traceability(
        self,
        documentation: str,
        source_code: str
    ) -> Tuple[float, Dict]:
        """
        Evaluate line citation validity.
        
        Citations are like: "line 42", "lines 100-150", "(L123)", etc.
        
        Returns:
            (score, details)
        """
        # Find all line citations
        citations = re.findall(
            r'(?:line\s*|L)(\d+)(?:\s*-\s*(\d+))?',
            documentation,
            re.IGNORECASE
        )
        
        if not citations:
            # No citations = partial credit (can't validate but not necessarily bad)
            return 0.5, {"citations": 0, "valid": 0, "invalid": 0, "note": "No citations found"}
        
        # Count source lines
        source_lines = len(source_code.split('\n')) if source_code else 0
        
        valid_count = 0
        invalid_count = 0
        
        for match in citations:
            start = int(match[0])
            end = int(match[1]) if match[1] else start
            
            # Check if lines are valid
            if source_lines > 0:
                if 1 <= start <= source_lines and 1 <= end <= source_lines:
                    valid_count += 1
                else:
                    invalid_count += 1
            else:
                # Can't validate without source, assume valid if reasonable
                if start < 10000 and end < 10000:
                    valid_count += 1
        
        total = valid_count + invalid_count
        score = valid_count / total if total > 0 else 0.5
        
        return score, {
            "citations": total,
            "valid": valid_count,
            "invalid": invalid_count,
        }
    
    def _evaluate_readability(self, documentation: str) -> Tuple[float, Dict]:
        """
        Evaluate readability using Flesch-Kincaid grade level.
        
        Target: Grade 8-12 (accessible to non-experts)
        
        Returns:
            (score, details)
        """
        # Clean text
        text = re.sub(r'[^\w\s\.]', ' ', documentation)
        text = ' '.join(text.split())
        
        if not text:
            return 0.5, {"grade": 0, "note": "Empty documentation"}
        
        # Count sentences (end with . ! ?)
        sentences = len(re.findall(r'[.!?]+', text)) or 1
        
        # Count words
        words = text.split()
        word_count = len(words) or 1
        
        # Count syllables (approximate)
        syllable_count = sum(self._count_syllables(w) for w in words)
        
        # Flesch-Kincaid Grade Level formula
        # 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
        avg_words_per_sentence = word_count / sentences
        avg_syllables_per_word = syllable_count / word_count
        
        grade = 0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word - 15.59
        grade = max(0, min(grade, 20))  # Clamp to reasonable range
        
        # Score based on target range
        target_min = self.readability_config["target_grade_min"]
        target_max = self.readability_config["target_grade_max"]
        penalty_per_grade = self.readability_config["penalty_per_grade"]
        
        if target_min <= grade <= target_max:
            score = 1.0
        elif grade < target_min:
            deviation = target_min - grade
            score = max(0, 1.0 - deviation * penalty_per_grade)
        else:
            deviation = grade - target_max
            score = max(0, 1.0 - deviation * penalty_per_grade)
        
        return score, {
            "grade": round(grade, 1),
            "target_range": f"{target_min}-{target_max}",
            "sentences": sentences,
            "words": word_count,
        }
    
    def _count_syllables(self, word: str) -> int:
        """
        Approximate syllable count for a word.
        """
        word = word.lower()
        if len(word) <= 3:
            return 1
        
        # Count vowel groups
        vowels = 'aeiou'
        count = 0
        prev_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        
        # Subtract silent e
        if word.endswith('e'):
            count = max(1, count - 1)
        
        # Special cases
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
        
        return max(1, count)
    
    def _evaluate_abstraction(self, documentation: str) -> Tuple[float, Dict]:
        """
        Evaluate abstraction level: WHY-words vs WHAT-words.
        
        Good documentation explains WHY, not just describes WHAT.
        Target: >= 40% WHY-words ratio
        
        Returns:
            (score, details)
        """
        doc_lower = documentation.lower()
        
        # Count WHY-words
        why_count = sum(
            len(re.findall(r'\b' + re.escape(w) + r'\b', doc_lower))
            for w in self.WHY_WORDS
        )
        
        # Count WHAT-words
        what_count = sum(
            len(re.findall(r'\b' + re.escape(w) + r'\b', doc_lower))
            for w in self.WHAT_WORDS
        )
        
        total = why_count + what_count
        
        if total == 0:
            return 0.5, {"ratio": 0, "why_count": 0, "what_count": 0}
        
        ratio = why_count / total
        
        # Target: >= 40% WHY-words
        target = 0.40
        if ratio >= target:
            score = 1.0
        else:
            score = ratio / target  # Linear scale below target
        
        return score, {
            "ratio": round(ratio, 2),
            "why_count": why_count,
            "what_count": what_count,
            "target": target,
        }
