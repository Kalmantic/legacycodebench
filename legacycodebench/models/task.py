"""
Task Model for LegacyCodeBench V2.4

Specification Reference: TDD_V2.4.md Section 2.2
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict
from .enums import Language


@dataclass
class Task:
    """
    Represents a benchmark task.
    
    Task IDs follow the pattern:
    - COBOL: LCB-T{tier}-{sequence} (e.g., LCB-T1-001)
    - UniBasic: LCB-UB-T{tier}-{sequence} (e.g., LCB-UB-T1-001)
    """
    task_id: str                      # LCB-T1-001 or LCB-UB-T1-001
    tier: str                         # T1, T2, T3, T4
    source_path: Path                 # Path to source file
    ground_truth_path: Path           # Path to ground truth JSON
    
    # Derived from task_id
    language: Language = field(init=False)
    
    # Optional pre-classification (hint only, not authoritative)
    execution_hint: Optional[Dict] = None
    
    # Optional dataset name for copybook resolution
    source_dataset: Optional[str] = None
    
    def __post_init__(self):
        """Derive language from task_id after initialization."""
        self.language = self.detect_language(self.task_id)
        
        # Convert string paths to Path objects if needed
        if isinstance(self.source_path, str):
            self.source_path = Path(self.source_path)
        if isinstance(self.ground_truth_path, str):
            self.ground_truth_path = Path(self.ground_truth_path)
    
    @staticmethod
    def detect_language(task_id: str) -> Language:
        """
        Detect language from task_id prefix.
        
        Decision table:
        - LCB-UB-* → UniBasic
        - LCB-T* → COBOL
        
        Args:
            task_id: Task identifier
            
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
    
    @property
    def sequence(self) -> int:
        """Extract sequence number from task_id."""
        return int(self.task_id.split("-")[-1])
    
    def __repr__(self) -> str:
        return f"Task({self.task_id}, {self.language.value}, tier={self.tier})"
