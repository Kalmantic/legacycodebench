"""
Language Adapter Registry for LegacyCodeBench V2.4

Provides adapter registration, lookup, and language detection.

Specification Reference: TDD_V2.4.md Section 3
"""

from typing import Dict, Type
import logging

from legacycodebench.models.enums import Language
from .base import LanguageAdapter
from .cobol_adapter import COBOLAdapter
from .unibasic_adapter import UniBasicAdapter

logger = logging.getLogger(__name__)

# Adapter registry - maps Language enum to adapter class
_ADAPTERS: Dict[Language, Type[LanguageAdapter]] = {
    Language.COBOL: COBOLAdapter,
    Language.UNIBASIC: UniBasicAdapter,
}

# Adapter instance cache (singletons)
_ADAPTER_INSTANCES: Dict[Language, LanguageAdapter] = {}


def get_adapter(language: Language) -> LanguageAdapter:
    """
    Get the adapter for a language.
    
    Adapters are cached as singletons.
    
    Args:
        language: Language enum value
        
    Returns:
        LanguageAdapter instance for the language
        
    Raises:
        ValueError: If no adapter exists for the language
    """
    if language not in _ADAPTERS:
        raise ValueError(f"No adapter registered for language: {language}")
    
    # Return cached instance if available
    if language not in _ADAPTER_INSTANCES:
        adapter_class = _ADAPTERS[language]
        _ADAPTER_INSTANCES[language] = adapter_class()
        logger.debug(f"Created adapter instance for {language.value}")
    
    return _ADAPTER_INSTANCES[language]


def detect_language(task_id: str) -> Language:
    """
    Detect language from task ID prefix.
    
    Decision table:
    - LCB-UB-* → UniBasic
    - LCB-T* → COBOL
    
    Args:
        task_id: Task identifier (e.g., "LCB-T1-001" or "LCB-UB-T1-001")
        
    Returns:
        Language enum value
        
    Raises:
        ValueError: If task_id has unknown prefix
    """
    if task_id.startswith("LCB-UB-"):
        return Language.UNIBASIC
    elif task_id.startswith("LCB-T"):
        return Language.COBOL
    else:
        raise ValueError(f"Unknown task prefix: {task_id}")


def register_adapter(language: Language, adapter_class: Type[LanguageAdapter]) -> None:
    """
    Register a new adapter for a language.
    
    Used for extending with new languages.
    
    Args:
        language: Language enum value
        adapter_class: Adapter class (must extend LanguageAdapter)
    """
    _ADAPTERS[language] = adapter_class
    # Clear any cached instance
    if language in _ADAPTER_INSTANCES:
        del _ADAPTER_INSTANCES[language]
    logger.info(f"Registered adapter {adapter_class.__name__} for {language.value}")


def get_supported_languages() -> list:
    """
    Get list of supported languages.
    
    Returns:
        List of Language enum values with registered adapters
    """
    return list(_ADAPTERS.keys())


def is_language_supported(language: Language) -> bool:
    """
    Check if a language is supported.
    
    Args:
        language: Language enum value
        
    Returns:
        True if an adapter is registered for the language
    """
    return language in _ADAPTERS


# Public exports
__all__ = [
    "LanguageAdapter",
    "COBOLAdapter",
    "UniBasicAdapter",
    "Language",
    "get_adapter",
    "detect_language",
    "register_adapter",
    "get_supported_languages",
    "is_language_supported",
]
