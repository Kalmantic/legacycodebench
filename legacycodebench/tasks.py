"""Task definitions and management"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

from legacycodebench.config import TASKS_DIR, DATASETS_DIR
from legacycodebench.dataset_loader import DatasetLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# V2.4 Task Protection and Language Detection
# ============================================================

# Frozen task IDs - these cannot be modified after deployment
FROZEN_TASK_IDS = set()  # Populated after initial deployment

def is_frozen(task_id: str) -> bool:
    """
    Check if a task is frozen (cannot be modified).
    
    Frozen tasks are protected after deployment to ensure
    reproducible benchmark results.
    
    Args:
        task_id: Task identifier
        
    Returns:
        True if task is frozen
    """
    return task_id in FROZEN_TASK_IDS


def detect_task_language(task_id: str) -> str:
    """
    Detect language from task ID prefix.
    
    - LCB-UB-* → unibasic
    - LCB-T* → cobol
    
    Args:
        task_id: Task identifier
        
    Returns:
        Language string ("cobol" or "unibasic")
    """
    if task_id.startswith("LCB-UB-"):
        return "unibasic"
    return "cobol"


def filter_tasks_by_language(task_ids: List[str], language: str) -> List[str]:
    """
    Filter task IDs by language.
    
    Args:
        task_ids: List of task IDs
        language: "cobol", "unibasic", or "all"
        
    Returns:
        Filtered list of task IDs
    """
    if language == "all":
        return task_ids
    
    return [tid for tid in task_ids if detect_task_language(tid) == language]


@dataclass
class Task:
    """Represents a single benchmark task"""
    task_id: str
    category: str  # "documentation" or "understanding"
    difficulty: str  # "easy", "medium", "hard"
    language: str  # "COBOL"
    domain: str  # "banking", "finance", etc.
    source_dataset: str  # Which dataset folder (e.g., "aws-carddemo")
    input_files: List[str]  # Relative paths to COBOL files
    task_description: str
    evaluation_criteria: Dict
    reference_solution: Optional[str] = None  # Path to reference solution
    tier: Optional[str] = None  # Complexity tier: T1, T2, T3, T4
    complexity_scoring: Optional[Dict] = None  # Multi-factor scoring breakdown
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Task":
        """Create from dictionary"""
        # V2.4 Compatibility: Map task_category to category if needed
        if "task_category" in data and "category" not in data:
            data["category"] = data.pop("task_category")
        return cls(**data)
    
    def save(self, tasks_dir: Path = TASKS_DIR):
        """Save task to JSON file"""
        tasks_dir.mkdir(parents=True, exist_ok=True)
        task_file = tasks_dir / f"{self.task_id}.json"
        with open(task_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved task {self.task_id} to {task_file}")
    
    @classmethod
    def load(cls, task_id: str, tasks_dir: Path = TASKS_DIR) -> "Task":
        """Load task from JSON file"""
        task_file = tasks_dir / f"{task_id}.json"
        if not task_file.exists():
            raise FileNotFoundError(f"Task {task_id} not found at {task_file}")
        
        with open(task_file, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_input_files_absolute(self, datasets_dir: Path = DATASETS_DIR) -> List[Path]:
        """Get absolute paths to input files using source_dataset"""
        absolute_files = []
        dataset_path = datasets_dir / self.source_dataset
        
        for file_path in self.input_files:
            full_path = dataset_path / file_path
            if full_path.exists():
                absolute_files.append(full_path)
            else:
                logger.warning(f"File not found: {full_path}")
        
        return absolute_files


class TaskManager:
    """Manage benchmark tasks"""
    
    def __init__(self, tasks_dir: Path = TASKS_DIR):
        self.tasks_dir = tasks_dir
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
    
    def create_tasks_from_datasets(self, datasets_dir: Path = DATASETS_DIR,
                                   use_intelligent_selection: bool = True,
                                   language: str = None, start_id: int = 1) -> List[Task]:
        """Create tasks from datasets with intelligent or simple selection

        Args:
            datasets_dir: Path to datasets directory
            use_intelligent_selection: Use intelligent tier-based selection
            language: "cobol", "unibasic", or None (default: cobol)
            start_id: Starting ID for task generation (safe additive mode)

        Returns:
            List of created Task objects
        """
        # V2.4: Default to COBOL for backward compatibility
        if language is None:
            language = "cobol"

        if use_intelligent_selection:
            return self.create_tasks_intelligent(datasets_dir, language=language, start_id=start_id)
        else:
            return self.create_tasks_simple(datasets_dir, language=language)
    
    def create_tasks_intelligent(self, datasets_dir: Path = DATASETS_DIR,
                                  language: str = "cobol", start_id: int = 1) -> List[Task]:
        """Create tasks using intelligent PRD v2.0 aligned selection

        v2.0 Approach:
        - ALL tasks are documentation tasks
        - Tasks are categorized by complexity TIER (T1-T4)
        - No separate "understanding" tasks
        - Understanding is validated through Behavioral Fidelity (execution)

        V2.4: Multi-language support (COBOL + UniBasic)
        """
        from legacycodebench.task_generator import TaskCandidateGenerator
        from legacycodebench.config import TASK_SELECTION_CONFIG

        lang_upper = language.upper()
        logger.info(f"Using v2.0 intelligent task selection for {lang_upper} (documentation only, tier-based)")

        # Initialize generator with language
        generator = TaskCandidateGenerator(datasets_dir)

        # Get configuration
        config = TASK_SELECTION_CONFIG
        min_loc = config["loc_ranges"]["min"]
        max_loc = config["loc_ranges"]["max"]

        # V2.4: Adjust total tasks based on language
        if language == "unibasic":
            total_tasks = 50  # UniBasic: 50 tasks
        else:
            total_tasks = config["task_distribution"]["total_tasks"]  # COBOL: 200 tasks

        # Generate all candidates (documentation only)
        logger.info(f"Analyzing {lang_upper} files (LOC range: {min_loc}-{max_loc})...")
        all_candidates = generator.generate_all_candidates(min_loc, max_loc, language=language)

        if not all_candidates:
            logger.warning(f"No suitable {lang_upper} candidates found, falling back to simple selection")
            return self.create_tasks_simple(datasets_dir, language=language)

        # Select best tasks by tier distribution (v2.0)
        logger.info(f"Selecting {total_tasks} {lang_upper} documentation tasks by tier...")
        selected_candidates = generator.select_best_tasks(all_candidates, total_tasks)

        # Convert candidates to tasks
        tasks = []

        # Group by tier for sequential numbering
        by_tier = {"T1": [], "T2": [], "T3": [], "T4": []}
        for candidate in selected_candidates:
            by_tier[candidate.tier].append(candidate)

        # V2.4: Create tasks with language-aware IDs
        # COBOL: LCB-T1-001, LCB-T2-001, etc.
        # UniBasic: LCB-UB-T1-001, LCB-UB-T2-001, etc.
        id_prefix = "LCB-UB" if language == "unibasic" else "LCB"

        # SAFETY: Check for existing tasks to avoid overwrite
        existing_tasks = self.list_tasks()

        for tier in ["T1", "T2", "T3", "T4"]:
            tier_candidates = by_tier[tier]

            # Determine starting index provided or auto-detect
            current_index = start_id

            for candidate in tier_candidates:
                # Find next safe ID
                while True:
                    task_id = f"{id_prefix}-{tier}-{current_index:03d}"
                    if task_id not in existing_tasks:
                        break
                    current_index += 1

                task = self._candidate_to_task(candidate, task_id, datasets_dir, language=language)
                tasks.append(task)
                loc = candidate.analysis.get('loc', 0) if candidate.analysis else 0
                logger.info(f"  Created {task.task_id}: {task.difficulty} - {task.domain} - {loc} LOC")
                
                current_index += 1  # Increment for next task

        logger.info(f"Created {len(tasks)} {lang_upper} documentation tasks using v2.0 selection")
        logger.info(f"  Tier distribution: T1={len(by_tier['T1'])}, T2={len(by_tier['T2'])}, T3={len(by_tier['T3'])}, T4={len(by_tier['T4'])}")

        return tasks
    
    def _candidate_to_task(self, candidate, task_id: str, datasets_dir: Path,
                           language: str = "cobol") -> Task:
        """Convert TaskCandidate to Task (v2.0: all tasks are documentation)

        V2.4: Added language parameter for multi-language support
        """
        # V2.4: Set display language
        lang_display = "UniBasic" if language == "unibasic" else "COBOL"
        
        # Get relative path from dataset root
        dataset_dir = datasets_dir / candidate.dataset_name
        rel_path = candidate.main_file.relative_to(dataset_dir)
        
        # Collect all input files
        input_files = [str(rel_path)]
        for related_file in candidate.related_files:
            try:
                rel_related = related_file.relative_to(dataset_dir)
                input_files.append(str(rel_related))
            except ValueError:
                # File not in same dataset, skip
                pass
                pass
        
        # v2.0: All tasks are documentation tasks
        # Tier determines complexity and evaluation rigor
        tier = getattr(candidate, 'tier', 'T1')
        loc = candidate.analysis.get('loc', 0) if candidate.analysis else 0
        
        # Create tier-appropriate task description
        if tier == "T1":
            task_description = (
                f"Generate clear documentation for {candidate.main_file.name} "
                f"explaining its business purpose, main data structures, and key business rules."
            )
        elif tier == "T2":
            task_description = (
                f"Generate comprehensive documentation for {candidate.main_file.name} "
                f"including business purpose, data structures, control flow, file operations, and business rules."
            )
        elif tier == "T3":
            task_description = (
                f"Generate detailed technical documentation for {candidate.main_file.name} "
                f"covering business purpose, all data structures, control flow, external dependencies, "
                f"business rules, and error handling."
            )
        else:  # T4
            task_description = (
                f"Generate thorough documentation for the complex program {candidate.main_file.name} "
                f"explaining business purpose, complete data structures, control flow (including GO TO patterns), "
                f"all external interfaces, business rules, error handling, and edge cases."
            )
        
        if len(input_files) > 1:
            task_description += f" Include analysis of {len(input_files)-1} related copybook(s)."
        
        # v2.0 evaluation criteria (aligned with SC + BF + SQ + TR)
        evaluation_criteria = {
            "version": "2.0",
            "tier": tier,
            "required_sections": ["business_purpose", "business_rules", "data_structures"],
            "format": "markdown",
            "evaluation_method": "ground_truth_based",
            "scoring_weights": {
                "structural_completeness": 0.30,
                "behavioral_fidelity": 0.35,
                "semantic_quality": 0.25,
                "traceability": 0.10,
            },
        }
        
        # Add tier-specific requirements
        if tier in ["T2", "T3", "T4"]:
            evaluation_criteria["required_sections"].extend(["control_flow", "file_operations"])
        if tier in ["T3", "T4"]:
            evaluation_criteria["required_sections"].extend(["dependencies", "error_handling"])
        if tier == "T4":
            evaluation_criteria["required_sections"].append("edge_cases")
        
        # Build complexity scoring info
        analysis = candidate.analysis or {}
        complexity_scoring = {
            "method": "multi-factor-v2.1",
            "breakdown": {
                "exec_cics": analysis.get("exec_cics_count", 0),
                "exec_sql": analysis.get("exec_sql_count", 0),
                "goto_count": analysis.get("goto_count", 0),
                "call_count": analysis.get("dependencies", {}).get("total", 0),
                "loc": loc,
                "final_tier": tier,
            }
        }

        return Task(
            task_id=task_id,
            category="documentation",  # v2.0: All tasks are documentation
            difficulty=candidate.difficulty_level,
            language=lang_display,  # V2.4: Use correct language (COBOL or UniBasic)
            domain=candidate.domain,
            source_dataset=candidate.dataset_name,
            input_files=input_files,
            task_description=task_description,
            evaluation_criteria=evaluation_criteria,
            tier=tier,  # v2.1: Complexity tier
            complexity_scoring=complexity_scoring,  # v2.1: Multi-factor scoring breakdown
        )
    
    def create_tasks_simple(self, datasets_dir: Path = DATASETS_DIR,
                            language: str = "cobol") -> List[Task]:
        """Simple sequential task selection (V2.4 format)
        
        V2.4 ID Patterns:
        - COBOL: LCB-T{tier}-{sequence} (e.g., LCB-T1-001)
        - UniBasic: LCB-UB-T{tier}-{sequence} (e.g., LCB-UB-T1-001)
        
        All tasks are documentation tasks (no DOC/UND split).
        """
        lang_upper = language.upper()
        lang_lower = language.lower()
        logger.info(f"Using V2.4 task selection for {lang_upper}")

        loader = DatasetLoader(datasets_dir)
        tasks = []

        # Find all datasets
        datasets = {}
        for dataset_dir in datasets_dir.iterdir():
            if dataset_dir.is_dir():
                # V2.4: Find source files based on language
                if lang_lower == "unibasic":
                    source_files = loader.find_unibasic_files(dataset_dir)
                else:
                    source_files = loader.find_cobol_files(dataset_dir)
                if source_files:
                    datasets[dataset_dir.name] = source_files
        
        # V2.4: Task ID prefix based on language
        if lang_lower == "unibasic":
            id_prefix = "LCB-UB-T1"
            lang_display = "UniBasic"
        else:
            id_prefix = "LCB-T1"
            lang_display = "COBOL"
        
        # Create tasks (V2.4: all documentation, tier-based)
        task_counter = 1
        
        for dataset_id, source_files in list(datasets.items())[:5]:  # Use first 5 datasets
            dataset_path = datasets_dir / dataset_id
            
            for i in range(min(4, len(source_files))):  # 4 per dataset max
                if task_counter > 20:  # Max 20 tasks in simple mode
                    break
                
                file_path = source_files[i]
                rel_path = file_path.relative_to(dataset_path)
                
                # V2.4: Create task with tier-based ID
                task_id = f"{id_prefix}-{task_counter:03d}"
                
                task = Task(
                    task_id=task_id,
                    category="documentation",  # V2.4: All tasks are documentation
                    difficulty="easy" if task_counter <= 5 else "medium",
                    language=lang_display,  # V2.4: Correct language
                    domain="enterprise" if lang_lower == "unibasic" else ("banking" if "bank" in dataset_id.lower() else "finance"),
                    source_dataset=dataset_id,
                    input_files=[str(rel_path)],
                    task_description=f"Generate comprehensive documentation for {rel_path.name} explaining business purpose, business rules, data structures, and key functionality.",
                    evaluation_criteria={
                        "version": "2.4",
                        "required_sections": ["business_purpose", "business_rules", "data_structures"],
                        "format": "markdown",
                    },
                    tier="T1",  # Simple mode defaults to T1
                )
                tasks.append(task)
                task_counter += 1
        
        logger.info(f"Created {len(tasks)} {lang_display} tasks with V2.4 ID format")
        return tasks
    
    def list_tasks(self) -> List[str]:
        """List all task IDs"""
        task_files = list(self.tasks_dir.glob("*.json"))
        return sorted([f.stem for f in task_files])
    
    def get_task(self, task_id: str) -> Task:
        """Get a specific task"""
        return Task.load(task_id, self.tasks_dir)
    
    def save_all(self, tasks: List[Task]):
        """Save all tasks"""
        for task in tasks:
            task.save(self.tasks_dir)


def main():
    """CLI entry point"""
    manager = TaskManager()
    tasks = manager.create_tasks_from_datasets()
    manager.save_all(tasks)
    
    print(f"\n✓ Created {len(tasks)} tasks:")
    for task in tasks:
        print(f"  {task.task_id}: {task.category} - {task.difficulty}")


if __name__ == "__main__":
    main()

