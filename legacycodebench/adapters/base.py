"""
Language Adapter Base Class for LegacyCodeBench V2.4

Abstract interface that all language-specific adapters must implement.
Enables multi-language support without changing core evaluator logic.

Specification Reference: TDD_V2.4.md Section 3.1
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from pathlib import Path

from legacycodebench.models.enums import Language


class LanguageAdapter(ABC):
    """
    Abstract interface for language-specific processing.
    
    Each supported language (COBOL, UniBasic) implements this interface.
    The adapter provides language-specific:
    - BSM patterns for external call validation
    - Synonyms for TF-IDF expansion
    - Source code parsing
    - Critical failure configuration
    - Executor for compilation/execution
    """

    @property
    @abstractmethod
    def language(self) -> Language:
        """Return the language this adapter handles."""
        pass

    @property
    @abstractmethod
    def file_extensions(self) -> List[str]:
        """
        Valid file extensions for this language.
        
        Returns:
            List of extensions without leading dot, e.g., ["cbl", "cob"]
        """
        pass

    @abstractmethod
    def get_bsm_patterns(self) -> List[Dict]:
        """
        Return BSM patterns for external call validation.
        
        Returns:
            List of pattern dicts with keys: id, regex, category
        """
        pass

    @abstractmethod
    def get_synonyms(self) -> Dict[str, List[str]]:
        """
        Return synonym dictionary for TF-IDF expansion.
        
        Used to expand documentation text before TF-IDF matching.
        Example: {"compute": ["calculate", "determine", "derive"]}
        
        Returns:
            Dict mapping word to list of synonyms
        """
        pass

    @abstractmethod
    def extract_paragraphs(self, source: str) -> List[Dict]:
        """
        Extract paragraph/subroutine names and content from source.
        
        Args:
            source: Source code text
            
        Returns:
            List of dicts with keys: name, type, start_line, end_line, content
        """
        pass

    @abstractmethod
    def extract_data_structures(self, source: str) -> List[Dict]:
        """
        Extract data structure definitions from source.
        
        Args:
            source: Source code text
            
        Returns:
            List of dicts with keys: name, level, line_number, pic_clause, children
        """
        pass

    @abstractmethod
    def detect_blocking_constructs(self, source: str) -> List[str]:
        """
        Detect constructs that prevent execution.
        
        Args:
            source: Source code text
            
        Returns:
            List of blocking construct names (e.g., ["CICS", "DB2"])
        """
        pass

    @abstractmethod
    def get_critical_failure_config(self) -> Dict[str, bool]:
        """
        Return which critical failures apply to this language.
        
        Returns:
            Dict mapping CF-ID to enabled status:
            {
                "CF-01": True,   # Complete Silence
                "CF-02": True,   # Hallucinated Structure
                "CF-03": True,   # Behavioral Contradiction
                "CF-04": True,   # Missing Error Handling
                "CF-05": True,   # External Call Misspecification
            }
        """
        pass

    @abstractmethod
    def get_executor(self) -> Any:
        """
        Return the executor for this language.
        
        Returns:
            Language-specific executor (COBOLExecutor, UniBasicExecutor, StubExecutor)
        """
        pass

    @abstractmethod
    def is_comment(self, line: str) -> bool:
        """
        Check if a line is a comment.
        
        Args:
            line: Single line of source code
            
        Returns:
            True if line is a comment
        """
        pass

    # Optional method with default implementation
    def classify_paragraph(self, paragraph: Dict) -> str:
        """
        Classify paragraph as PURE, MIXED, or INFRASTRUCTURE.
        
        Default implementation returns "PURE".
        Override in language-specific adapters for better classification.
        
        Args:
            paragraph: Paragraph dict from extract_paragraphs
            
        Returns:
            One of: "PURE", "MIXED", "INFRASTRUCTURE"
        """
        return "PURE"

    def get_variable_pattern(self) -> str:
        """
        Return regex pattern for matching variable names.
        
        Used for claim extraction and hallucination detection.
        Default returns COBOL-style pattern.
        """
        return r"[A-Z][A-Z0-9-]*"
