"""
Custom Exceptions for LegacyCodeBench V2.4

Centralized exception definitions for proper error handling.

Specification Reference: V2.4_IMPLEMENTATION_PLAN.md Phase 9
"""


class LegacyCodeBenchError(Exception):
    """Base exception for all LegacyCodeBench errors."""
    pass


# ============================================================
# Task-related Errors
# ============================================================

class TaskNotFoundError(LegacyCodeBenchError):
    """Raised when a task cannot be found."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        super().__init__(f"Task not found: {task_id}")


class TaskFrozenError(LegacyCodeBenchError):
    """Raised when attempting to modify a frozen task."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        super().__init__(f"Task is frozen and cannot be modified: {task_id}")


class InvalidTaskIdError(LegacyCodeBenchError):
    """Raised when a task ID has invalid format."""
    
    def __init__(self, task_id: str, reason: str = "Invalid format"):
        self.task_id = task_id
        super().__init__(f"Invalid task ID '{task_id}': {reason}")


# ============================================================
# Source/Dataset Errors
# ============================================================

class SourceNotFoundError(LegacyCodeBenchError):
    """Raised when source files cannot be found."""
    
    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Source file(s) not found: {path}")


class DatasetIntegrityError(LegacyCodeBenchError):
    """Raised when dataset integrity check fails."""
    
    def __init__(self, dataset: str, expected_hash: str, actual_hash: str):
        self.dataset = dataset
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        super().__init__(
            f"Dataset integrity check failed for '{dataset}': "
            f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
        )


class CopybookNotFoundError(LegacyCodeBenchError):
    """Raised when a required copybook is not found."""
    
    def __init__(self, copybook_name: str, searched_paths: list = None):
        self.copybook_name = copybook_name
        self.searched_paths = searched_paths or []
        paths_str = ", ".join(str(p) for p in self.searched_paths[:3])
        super().__init__(f"Copybook not found: {copybook_name}. Searched: {paths_str}")


# ============================================================
# Evaluation Errors
# ============================================================

class EvaluationError(LegacyCodeBenchError):
    """Base exception for evaluation-related errors."""
    pass


class GroundTruthError(EvaluationError):
    """Raised when ground truth generation fails."""
    
    def __init__(self, task_id: str, reason: str):
        self.task_id = task_id
        super().__init__(f"Ground truth generation failed for {task_id}: {reason}")


class ClaimExtractionError(EvaluationError):
    """Raised when claim extraction fails."""
    
    def __init__(self, reason: str):
        super().__init__(f"Claim extraction failed: {reason}")


class CompilationError(EvaluationError):
    """Raised when source code compilation fails."""
    
    def __init__(self, language: str, error_message: str):
        self.language = language
        self.error_message = error_message
        super().__init__(f"{language} compilation failed: {error_message}")


class ExecutionError(EvaluationError):
    """Raised when test execution fails."""
    
    def __init__(self, task_id: str, error_message: str):
        self.task_id = task_id
        super().__init__(f"Execution failed for {task_id}: {error_message}")


class ExecutionTimeoutError(ExecutionError):
    """Raised when execution exceeds timeout."""
    
    def __init__(self, task_id: str, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        super().__init__(task_id, f"Timeout after {timeout_seconds}s")


# ============================================================
# Adapter Errors
# ============================================================

class AdapterError(LegacyCodeBenchError):
    """Base exception for adapter-related errors."""
    pass


class UnsupportedLanguageError(AdapterError):
    """Raised when a language is not supported."""
    
    def __init__(self, language: str):
        self.language = language
        super().__init__(f"Unsupported language: {language}")


class LanguageDetectionError(AdapterError):
    """Raised when language cannot be detected from task ID."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        super().__init__(f"Cannot detect language from task ID: {task_id}")


# ============================================================
# Vectorizer/Reproducibility Errors
# ============================================================

class VectorizerError(LegacyCodeBenchError):
    """Raised when vectorizer operations fail."""
    pass


class VectorizerNotFoundError(VectorizerError):
    """Raised when a vectorizer file cannot be found."""
    
    def __init__(self, vectorizer_path: str):
        self.vectorizer_path = vectorizer_path
        super().__init__(f"Vectorizer not found: {vectorizer_path}")


class VectorizerIntegrityError(VectorizerError):
    """Raised when vectorizer checksum verification fails."""
    
    def __init__(self, vectorizer_name: str, expected_hash: str, actual_hash: str):
        self.vectorizer_name = vectorizer_name
        super().__init__(
            f"Vectorizer integrity check failed for '{vectorizer_name}'"
        )


# ============================================================
# Configuration Errors
# ============================================================

class ConfigurationError(LegacyCodeBenchError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, key: str, reason: str = "Missing or invalid"):
        self.key = key
        super().__init__(f"Configuration error for '{key}': {reason}")


# ============================================================
# Critical Failure Markers
# ============================================================

class CriticalFailureError(EvaluationError):
    """
    Raised when a critical failure is detected.
    
    This is NOT an exception that should crash the program,
    but rather a marker for when evaluation should return score=0.
    """
    
    def __init__(self, cf_id: str, name: str, description: str):
        self.cf_id = cf_id
        self.name = name
        self.description = description
        super().__init__(f"{cf_id}: {name} - {description}")
