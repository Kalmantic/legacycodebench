"""Intelligent task candidate generation"""

from pathlib import Path
from typing import Dict, List, Tuple
import logging

from legacycodebench.file_analyzer import COBOLFileAnalyzer
from legacycodebench.difficulty_calibrator import DifficultyCalibrator
from legacycodebench.domain_detector import DomainDetector
from legacycodebench.config import DATASETS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskCandidate:
    """Represents a potential benchmark task"""
    
    def __init__(self, main_file: Path, category: str, dataset_name: str):
        self.main_file = main_file
        self.category = category
        self.dataset_name = dataset_name
        self.related_files = []
        self.analysis = None
        self.difficulty_score = 0.0
        self.difficulty_level = "medium"
        self.domain = "enterprise"
        self.interestingness_score = 0.0
    
    def __repr__(self):
        return f"TaskCandidate({self.main_file.name}, {self.category}, {self.difficulty_level}, score={self.interestingness_score:.1f})"


class TaskCandidateGenerator:
    """Generate intelligent task candidates from COBOL datasets"""
    
    def __init__(self, datasets_dir: Path = DATASETS_DIR):
        self.datasets_dir = datasets_dir
        self.analyzer_cache = {}  # Cache file analyses
        self.calibrator = DifficultyCalibrator()
        self.domain_detector = DomainDetector()
    
    def generate_all_candidates(self, min_loc: int = 300, max_loc: int = 3000) -> List[TaskCandidate]:
        """Generate all possible task candidates from datasets"""
        all_candidates = []
        
        # Process each dataset
        for dataset_dir in self.datasets_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            logger.info(f"Analyzing dataset: {dataset_name}")
            
            # Find all COBOL files
            cobol_files = self._find_cobol_files(dataset_dir)
            logger.info(f"Found {len(cobol_files)} COBOL files in {dataset_name}")
            
            # Analyze each file
            for cobol_file in cobol_files:
                analyzer = self._get_analyzer(cobol_file)
                analysis = analyzer.analyze()
                
                # Filter by LOC
                if not (min_loc <= analysis["loc"] <= max_loc):
                    continue
                
                # Create documentation candidate
                doc_candidate = self._create_documentation_candidate(
                    cobol_file, analysis, dataset_name
                )
                if doc_candidate:
                    all_candidates.append(doc_candidate)
                
                # Create understanding candidate if it has dependencies
                if analysis["dependencies"]["total"] > 0:
                    und_candidate = self._create_understanding_candidate(
                        cobol_file, analysis, dataset_name
                    )
                    if und_candidate:
                        all_candidates.append(und_candidate)
        
        logger.info(f"Generated {len(all_candidates)} total candidates")
        return all_candidates
    
    def select_best_tasks(self, candidates: List[TaskCandidate], 
                         num_doc: int = 8, num_und: int = 7) -> Tuple[List[TaskCandidate], List[TaskCandidate]]:
        """Select best tasks with balanced difficulty and diversity"""
        
        # Separate by category
        doc_candidates = [c for c in candidates if c.category == "documentation"]
        und_candidates = [c for c in candidates if c.category == "understanding"]
        
        # Sort by interestingness score
        doc_candidates.sort(key=lambda c: c.interestingness_score, reverse=True)
        und_candidates.sort(key=lambda c: c.interestingness_score, reverse=True)
        
        # Select with diversity
        selected_doc = self._select_diverse_tasks(doc_candidates, num_doc)
        selected_und = self._select_diverse_tasks(und_candidates, num_und)
        
        logger.info(f"Selected {len(selected_doc)} documentation + {len(selected_und)} understanding tasks")
        
        return selected_doc, selected_und
    
    def _create_documentation_candidate(self, main_file: Path, analysis: Dict, 
                                       dataset_name: str) -> TaskCandidate:
        """Create documentation task candidate"""
        candidate = TaskCandidate(main_file, "documentation", dataset_name)
        candidate.analysis = analysis
        
        # Find related copybooks
        candidate.related_files = self._find_related_copybooks(main_file, analysis)
        
        # Calculate scores
        analyzer = self._get_analyzer(main_file)
        candidate.interestingness_score = analyzer.calculate_interestingness_score()
        candidate.difficulty_score = self.calibrator.calculate_difficulty_score(
            analysis, "documentation"
        )
        candidate.difficulty_level = self.calibrator.assign_difficulty_level(
            candidate.difficulty_score
        )
        candidate.domain = self.domain_detector.detect_domain(analysis, dataset_name)
        
        # Must have some business logic for documentation
        if analysis["business_rules"] < 2:
            return None
        
        return candidate
    
    def _create_understanding_candidate(self, main_file: Path, analysis: Dict,
                                       dataset_name: str) -> TaskCandidate:
        """Create understanding task candidate"""
        candidate = TaskCandidate(main_file, "understanding", dataset_name)
        candidate.analysis = analysis
        
        # Find related files (called programs + copybooks)
        candidate.related_files = self._find_related_files_for_understanding(
            main_file, analysis
        )
        
        # Calculate scores
        analyzer = self._get_analyzer(main_file)
        candidate.interestingness_score = analyzer.calculate_interestingness_score()
        
        # Boost score for understanding if many dependencies
        dep_boost = min(analysis["dependencies"]["total"] / 5, 1.0) * 10
        candidate.interestingness_score += dep_boost
        
        candidate.difficulty_score = self.calibrator.calculate_difficulty_score(
            analysis, "understanding"
        )
        candidate.difficulty_level = self.calibrator.assign_difficulty_level(
            candidate.difficulty_score
        )
        candidate.domain = self.domain_detector.detect_domain(analysis, dataset_name)
        
        return candidate
    
    def _select_diverse_tasks(self, candidates: List[TaskCandidate], num_tasks: int) -> List[TaskCandidate]:
        """Select tasks with diverse difficulty levels and domains"""
        if not candidates:
            return []
        
        selected = []
        used_files = set()
        
        # Target distribution: 33% easy, 47% medium, 20% hard
        target_easy = int(num_tasks * 0.33)
        target_medium = int(num_tasks * 0.47)
        target_hard = num_tasks - target_easy - target_medium
        
        counts = {"easy": 0, "medium": 0, "hard": 0}
        
        # Group by difficulty
        by_difficulty = {"easy": [], "medium": [], "hard": []}
        for candidate in candidates:
            by_difficulty[candidate.difficulty_level].append(candidate)
        
        # Select from each difficulty tier
        for difficulty, target in [("easy", target_easy), ("medium", target_medium), ("hard", target_hard)]:
            available = by_difficulty[difficulty]
            
            for candidate in available:
                if len(selected) >= num_tasks:
                    break
                
                # Avoid duplicate files
                if str(candidate.main_file) in used_files:
                    continue
                
                # Avoid overlapping file sets
                if any(str(f) in used_files for f in candidate.related_files):
                    continue
                
                selected.append(candidate)
                used_files.add(str(candidate.main_file))
                used_files.update(str(f) for f in candidate.related_files)
                
                counts[difficulty] += 1
                
                if counts[difficulty] >= target:
                    break
        
        # Fill remaining slots with best available (if we didn't hit targets)
        if len(selected) < num_tasks:
            remaining = [c for c in candidates 
                        if str(c.main_file) not in used_files]
            
            for candidate in remaining:
                if len(selected) >= num_tasks:
                    break
                
                if any(str(f) in used_files for f in candidate.related_files):
                    continue
                
                selected.append(candidate)
                used_files.add(str(candidate.main_file))
                used_files.update(str(f) for f in candidate.related_files)
        
        return selected
    
    def _find_related_copybooks(self, main_file: Path, analysis: Dict) -> List[Path]:
        """Find copybook files referenced by main file"""
        copybooks = analysis["dependencies"]["copies"]
        dataset_dir = self._get_dataset_dir(main_file)
        
        found_copybooks = []
        for copybook_name in copybooks[:5]:  # Limit to 5
            # Look for copybook file
            copybook_file = self._find_file_in_dataset(
                dataset_dir, copybook_name, [".cpy", ".CPY"]
            )
            if copybook_file:
                found_copybooks.append(copybook_file)
        
        return found_copybooks
    
    def _find_related_files_for_understanding(self, main_file: Path, analysis: Dict) -> List[Path]:
        """Find related files for understanding task (called programs + copybooks)"""
        related = []
        dataset_dir = self._get_dataset_dir(main_file)
        
        # Find called programs
        calls = analysis["dependencies"]["calls"]
        for program_name in calls[:3]:  # Limit to 3 called programs
            program_file = self._find_file_in_dataset(
                dataset_dir, program_name, [".cbl", ".CBL", ".cob", ".COB"]
            )
            if program_file:
                related.append(program_file)
        
        # Find copybooks
        copies = analysis["dependencies"]["copies"]
        for copybook_name in copies[:3]:  # Limit to 3 copybooks
            copybook_file = self._find_file_in_dataset(
                dataset_dir, copybook_name, [".cpy", ".CPY"]
            )
            if copybook_file:
                related.append(copybook_file)
        
        return related[:5]  # Max 5 related files total
    
    def _find_file_in_dataset(self, dataset_dir: Path, filename: str, 
                             extensions: List[str]) -> Path:
        """Find a file by name (without extension) in dataset"""
        for ext in extensions:
            # Try exact match
            for file_path in dataset_dir.rglob(f"{filename}{ext}"):
                return file_path
            
            # Try case-insensitive match
            for file_path in dataset_dir.rglob(f"*{ext}"):
                if file_path.stem.upper() == filename.upper():
                    return file_path
        
        return None
    
    def _get_dataset_dir(self, file_path: Path) -> Path:
        """Get dataset directory for a file"""
        # Assuming structure: datasets/dataset-name/...
        parts = file_path.parts
        datasets_idx = parts.index("datasets")
        return Path(*parts[:datasets_idx + 2])
    
    def _find_cobol_files(self, dataset_dir: Path) -> List[Path]:
        """Find all COBOL files in dataset"""
        cobol_files = []
        for ext in ["*.cbl", "*.cob", "*.COB", "*.CBL"]:
            cobol_files.extend(dataset_dir.rglob(ext))
        return cobol_files
    
    def _get_analyzer(self, file_path: Path) -> COBOLFileAnalyzer:
        """Get cached analyzer or create new one"""
        file_str = str(file_path)
        if file_str not in self.analyzer_cache:
            self.analyzer_cache[file_str] = COBOLFileAnalyzer(file_path)
        return self.analyzer_cache[file_str]

