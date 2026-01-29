"""
Stub Executor for LegacyCodeBench V2.4

A placeholder executor for languages that don't have execution support yet.
Always fails compilation, forcing static verification.

Specification Reference: V2.4_IMPLEMENTATION_PLAN.md Phase 3
"""

import logging
from typing import Tuple, List, Any
from legacycodebench.models.enums import Language, CompileFailureReason

logger = logging.getLogger(__name__)


class StubExecutor:
    """
    Stub executor for languages without execution support.
    
    This executor always fails compilation, which causes the
    evaluation pipeline to fall back to static verification.
    
    Used for:
    - UniBasic (until ScarletDME integration is complete)
    - Future languages during initial development
    """

    def __init__(self, language: Language):
        """
        Initialize the stub executor.
        
        Args:
            language: The language this executor "supports"
        """
        self.language = language
        logger.info(f"StubExecutor initialized for {language.value}")

    def compile(
        self, 
        source_code: str, 
        task_id: str
    ) -> Tuple[bool, str, CompileFailureReason]:
        """
        Always fail compilation, forcing static verification.
        
        Args:
            source_code: Source code to "compile"
            task_id: Task identifier
            
        Returns:
            Tuple of (success=False, error_message, failure_reason)
        """
        logger.debug(f"StubExecutor.compile called for {task_id}")
        
        return (
            False,
            f"Execution not supported for {self.language.value}. Using static verification.",
            CompileFailureReason.VENDOR_API,
        )

    def execute(
        self, 
        task_id: str, 
        tests: List[Any]
    ) -> None:
        """
        Execution is not supported.
        
        Args:
            task_id: Task identifier
            tests: Test cases to run
            
        Raises:
            NotImplementedError: Always raised
        """
        raise NotImplementedError(
            f"Execution not supported for {self.language.value}"
        )

    def run_tests(
        self, 
        tests: List[Any]
    ) -> dict:
        """
        Test execution is not supported.
        
        Args:
            tests: Test cases to run
            
        Raises:
            NotImplementedError: Always raised
        """
        raise NotImplementedError(
            f"Test execution not supported for {self.language.value}"
        )

    def cleanup(self) -> None:
        """
        No cleanup needed for stub executor.
        """
        pass
