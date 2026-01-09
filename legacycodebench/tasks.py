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
                                   use_intelligent_selection: bool = True) -> List[Task]:
        """Create tasks from datasets with intelligent or simple selection"""
        if use_intelligent_selection:
            return self.create_tasks_intelligent(datasets_dir)
        else:
            return self.create_tasks_simple(datasets_dir)
    
    def create_tasks_intelligent(self, datasets_dir: Path = DATASETS_DIR) -> List[Task]:
        """Create tasks using intelligent PRD v2.0 aligned selection
        
        v2.0 Approach:
        - ALL tasks are documentation tasks
        - Tasks are categorized by complexity TIER (T1-T4)
        - No separate "understanding" tasks
        - Understanding is validated through Behavioral Fidelity (execution)
        """
        from legacycodebench.task_generator import TaskCandidateGenerator
        from legacycodebench.config import TASK_SELECTION_CONFIG
        
        logger.info("Using v2.0 intelligent task selection (documentation only, tier-based)")
        
        # Initialize generator
        generator = TaskCandidateGenerator(datasets_dir)
        
        # Get configuration
        config = TASK_SELECTION_CONFIG
        min_loc = config["loc_ranges"]["min"]
        max_loc = config["loc_ranges"]["max"]
        total_tasks = config["task_distribution"]["total_tasks"]
        
        # Generate all candidates (documentation only)
        logger.info(f"Analyzing COBOL files (LOC range: {min_loc}-{max_loc})...")
        all_candidates = generator.generate_all_candidates(min_loc, max_loc)
        
        if not all_candidates:
            logger.warning("No suitable candidates found, falling back to simple selection")
            return self.create_tasks_simple(datasets_dir)
        
        # Select best tasks by tier distribution (v2.0)
        logger.info(f"Selecting {total_tasks} documentation tasks by tier...")
        selected_candidates = generator.select_best_tasks(all_candidates, total_tasks)
        
        # Convert candidates to tasks
        tasks = []
        
        # Group by tier for sequential numbering
        by_tier = {"T1": [], "T2": [], "T3": [], "T4": []}
        for candidate in selected_candidates:
            by_tier[candidate.tier].append(candidate)
        
        # Create tasks with tier-based IDs: LCB-T1-001, LCB-T2-001, etc.
        for tier in ["T1", "T2", "T3", "T4"]:
            tier_candidates = by_tier[tier]
            for i, candidate in enumerate(tier_candidates, 1):
                task_id = f"LCB-{tier}-{i:03d}"
                task = self._candidate_to_task(candidate, task_id, datasets_dir)
                tasks.append(task)
                logger.info(f"  Created {task.task_id}: {task.difficulty} - {task.domain} - {candidate.analysis['loc']} LOC")
        
        logger.info(f"Created {len(tasks)} documentation tasks using v2.0 selection")
        logger.info(f"  Tier distribution: T1={len(by_tier['T1'])}, T2={len(by_tier['T2'])}, T3={len(by_tier['T3'])}, T4={len(by_tier['T4'])}")
        
        return tasks
    
    def _candidate_to_task(self, candidate, task_id: str, datasets_dir: Path) -> Task:
        """Convert TaskCandidate to Task (v2.0: all tasks are documentation)"""
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
            language="COBOL",
            domain=candidate.domain,
            source_dataset=candidate.dataset_name,
            input_files=input_files,
            task_description=task_description,
            evaluation_criteria=evaluation_criteria,
            tier=tier,  # v2.1: Complexity tier
            complexity_scoring=complexity_scoring,  # v2.1: Multi-factor scoring breakdown
        )
    
    def create_tasks_simple(self, datasets_dir: Path = DATASETS_DIR) -> List[Task]:
        """Simple sequential task selection (legacy method)"""
        logger.info("Using simple task selection (legacy)")
        
        loader = DatasetLoader(datasets_dir)
        tasks = []
        
        # Find all datasets
        datasets = {}
        for dataset_dir in datasets_dir.iterdir():
            if dataset_dir.is_dir():
                cobol_files = loader.find_cobol_files(dataset_dir)
                if cobol_files:
                    datasets[dataset_dir.name] = cobol_files
        
        # Create 10 tasks (5 documentation + 5 understanding)
        task_counter = {"doc": 1, "und": 1}
        
        for dataset_id, cobol_files in list(datasets.items())[:3]:  # Use first 3 datasets
            dataset_path = datasets_dir / dataset_id
            
            # Create documentation tasks (5)
            for i in range(min(2, len(cobol_files))):  # 2 per dataset
                if task_counter["doc"] > 5:
                    break
                
                file_path = cobol_files[i]
                rel_path = file_path.relative_to(dataset_path)
                
                task = Task(
                    task_id=f"LCB-DOC-{task_counter['doc']:03d}",
                    category="documentation",
                    difficulty="medium" if task_counter["doc"] % 2 == 0 else "easy",
                    language="COBOL",
                    domain="banking" if "bank" in dataset_id.lower() else "finance",
                    source_dataset=dataset_id,
                    input_files=[str(rel_path)],
                    task_description=f"Generate comprehensive documentation for {rel_path.name} explaining business purpose, business rules, edge cases, and data structures.",
                    evaluation_criteria={
                        "required_sections": ["business_purpose", "business_rules", "edge_cases", "data_structures"],
                        "min_length_pages": 3,
                        "format": "markdown",
                    }
                )
                tasks.append(task)
                task_counter["doc"] += 1
            
            # Create understanding tasks (5)
            for i in range(min(2, len(cobol_files))):  # 2 per dataset
                if task_counter["und"] > 5:
                    break
                
                file_path = cobol_files[min(i+2, len(cobol_files)-1)]  # Different files
                rel_path = file_path.relative_to(dataset_path)
                
                task = Task(
                    task_id=f"LCB-UND-{task_counter['und']:03d}",
                    category="understanding",
                    difficulty="medium" if task_counter["und"] % 2 == 0 else "easy",
                    language="COBOL",
                    domain="banking" if "bank" in dataset_id.lower() else "finance",
                    source_dataset=dataset_id,
                    input_files=[str(rel_path)],
                    task_description=f"Extract dependency graph, business rules, and data flow from {rel_path.name}.",
                    evaluation_criteria={
                        "output_format": "json",
                        "required_fields": ["dependencies", "business_rules", "data_flow"],
                    }
                )
                tasks.append(task)
                task_counter["und"] += 1
        
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

