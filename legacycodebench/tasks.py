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
    category: str  # "documentation", "understanding", or "conversion"
    difficulty: str  # "easy", "medium", "hard"
    language: str  # "COBOL"
    domain: str  # "banking", "finance", etc.
    input_files: List[str]  # Relative paths to COBOL files
    task_description: str
    evaluation_criteria: Dict
    reference_solution: Optional[str] = None  # Path to reference solution
    target_language: Optional[str] = None  # For conversion tasks: "java", "python", "csharp"
    test_cases: Optional[List[Dict]] = None  # For conversion tasks: input/output test cases
    
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
        """Get absolute paths to input files"""
        absolute_files = []
        for file_path in self.input_files:
            # Try to find the file in any dataset
            for dataset_dir in datasets_dir.iterdir():
                if dataset_dir.is_dir():
                    full_path = dataset_dir / file_path
                    if full_path.exists():
                        absolute_files.append(full_path)
                        break
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
        """Create tasks using intelligent PRD-aligned selection"""
        from legacycodebench.task_generator import TaskCandidateGenerator
        from legacycodebench.config import TASK_SELECTION_CONFIG
        
        logger.info("Using intelligent task selection (PRD-aligned)")
        
        # Initialize generator
        generator = TaskCandidateGenerator(datasets_dir)
        
        # Get configuration
        config = TASK_SELECTION_CONFIG
        min_loc = config["loc_ranges"]["min"]
        max_loc = config["loc_ranges"]["max"]
        num_doc = config["task_distribution"]["documentation_tasks"]
        num_und = config["task_distribution"]["understanding_tasks"]
        num_cnv = config["task_distribution"].get("conversion_tasks", 0)
        
        # Generate all candidates
        logger.info(f"Analyzing COBOL files (LOC range: {min_loc}-{max_loc})...")
        all_candidates = generator.generate_all_candidates(min_loc, max_loc)
        
        if not all_candidates:
            logger.warning("No suitable candidates found, falling back to simple selection")
            return self.create_tasks_simple(datasets_dir)
        
        # Select best tasks
        logger.info(f"Selecting top {num_doc} documentation + {num_und} understanding + {num_cnv} conversion tasks...")
        doc_candidates, und_candidates = generator.select_best_tasks(
            all_candidates, num_doc, num_und
        )

        # For conversion, select from same pool with different criteria
        # Prefer files with clear business logic for conversion
        cnv_candidates = generator.select_conversion_candidates(all_candidates, num_cnv)

        # Convert candidates to tasks
        tasks = []

        # Documentation tasks
        for i, candidate in enumerate(doc_candidates, 1):
            task = self._candidate_to_task(
                candidate, f"LCB-DOC-{i:03d}", datasets_dir
            )
            tasks.append(task)
            logger.info(f"  Created {task.task_id}: {task.difficulty} - {task.domain} - {candidate.analysis['loc']} LOC")

        # Understanding tasks
        for i, candidate in enumerate(und_candidates, 1):
            task = self._candidate_to_task(
                candidate, f"LCB-UND-{i:03d}", datasets_dir
            )
            tasks.append(task)
            logger.info(f"  Created {task.task_id}: {task.difficulty} - {task.domain} - {candidate.analysis['dependencies']['total']} dependencies")

        # Conversion tasks
        target_dist = config["task_distribution"].get("conversion_target_distribution", {"java": 1.0})
        target_counts = {lang: int(num_cnv * pct) for lang, pct in target_dist.items()}

        cnv_idx = 1
        for target_lang, count in target_counts.items():
            for candidate in cnv_candidates[:count]:
                task = self._candidate_to_conversion_task(
                    candidate, f"LCB-CNV-{cnv_idx:03d}", datasets_dir, target_lang
                )
                tasks.append(task)
                logger.info(f"  Created {task.task_id}: {task.difficulty} - COBOL→{target_lang} - {candidate.analysis['loc']} LOC")
                cnv_idx += 1
                cnv_candidates = cnv_candidates[1:]  # Remove used candidate

        logger.info(f"Created {len(tasks)} tasks using intelligent selection")
        return tasks
    
    def _candidate_to_task(self, candidate, task_id: str, datasets_dir: Path) -> Task:
        """Convert TaskCandidate to Task"""
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
        
        # Create task description
        if candidate.category == "documentation":
            task_description = (
                f"Generate comprehensive documentation for {candidate.main_file.name} "
                f"explaining business purpose, business rules, edge cases, and data structures."
            )
            if len(input_files) > 1:
                task_description += f" Analyze the main program and {len(input_files)-1} related copybook(s)."
            
            evaluation_criteria = {
                "required_sections": ["business_purpose", "business_rules", "edge_cases", "data_structures"],
                "min_length_pages": 3 if candidate.difficulty_level == "easy" else 5,
                "format": "markdown",
            }
        else:  # understanding
            task_description = (
                f"Extract dependency graph, business rules, and data flow from {candidate.main_file.name}."
            )
            if len(input_files) > 1:
                task_description += f" Analyze relationships across {len(input_files)} files."
            
            evaluation_criteria = {
                "output_format": "json",
                "required_fields": ["dependencies", "business_rules", "data_flow"],
            }
        
        return Task(
            task_id=task_id,
            category=candidate.category,
            difficulty=candidate.difficulty_level,
            language="COBOL",
            domain=candidate.domain,
            input_files=input_files,
            task_description=task_description,
            evaluation_criteria=evaluation_criteria,
        )

    def _candidate_to_conversion_task(self, candidate, task_id: str, datasets_dir: Path, target_language: str) -> Task:
        """Convert TaskCandidate to Conversion Task"""
        from legacycodebench.config import CONVERSION_TARGETS

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
                pass

        # Get target language config
        target_config = CONVERSION_TARGETS.get(target_language, {})
        target_ext = target_config.get("extension", ".java")

        # Create task description
        task_description = (
            f"Convert {candidate.main_file.name} from COBOL to {target_language.capitalize()}. "
            f"Preserve all business logic, data structures, and program flow. "
            f"The converted code must be functionally equivalent to the original COBOL program."
        )
        if len(input_files) > 1:
            task_description += f" Include conversions for {len(input_files)-1} related copybook(s)."

        # Evaluation criteria for conversion
        evaluation_criteria = {
            "target_language": target_language,
            "output_format": target_ext,
            "requirements": [
                "compilation_success",
                "functional_equivalence",
                "proper_data_types",
                "business_logic_preserved",
            ],
            "code_quality": [
                "no_hardcoded_values",
                "proper_naming_conventions",
                "appropriate_error_handling",
            ],
        }

        return Task(
            task_id=task_id,
            category="conversion",
            difficulty=candidate.difficulty_level,
            language="COBOL",
            domain=candidate.domain,
            input_files=input_files,
            task_description=task_description,
            evaluation_criteria=evaluation_criteria,
            target_language=target_language,
            test_cases=[],  # Will be populated with actual test cases
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

