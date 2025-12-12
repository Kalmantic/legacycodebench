"""Intelligent task candidate generation - v2.0 aligned

v2.0 Approach:
- ALL tasks are DOCUMENTATION tasks (no separate "understanding" tasks)
- Tasks are categorized by COMPLEXITY TIER (T1-T4), not task type
- Understanding is validated through Behavioral Fidelity (execution-based)
- 200 tasks total: T1(80) + T2(70) + T3(40) + T4(10)
"""

from pathlib import Path
from typing import Dict, List, Tuple
import logging

from legacycodebench.file_analyzer import COBOLFileAnalyzer
from legacycodebench.difficulty_calibrator import DifficultyCalibrator
from legacycodebench.domain_detector import DomainDetector
from legacycodebench.config import DATASETS_DIR, TASK_SELECTION_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskCandidate:
    """Represents a potential benchmark task (v2.0: documentation only)"""
    
    def __init__(self, main_file: Path, dataset_name: str, tier: str = "T1"):
        self.main_file = main_file
        self.category = "documentation"  # v2.0: All tasks are documentation
        self.dataset_name = dataset_name
        self.tier = tier  # v2.0: T1, T2, T3, T4
        self.related_files = []
        self.analysis = None
        self.difficulty_score = 0.0
        self.difficulty_level = TASK_SELECTION_CONFIG["task_distribution"]["tier_to_difficulty"].get(tier, "medium")
        self.domain = "enterprise"
        self.interestingness_score = 0.0
    
    def __repr__(self):
        return f"TaskCandidate({self.main_file.name}, {self.tier}, {self.difficulty_level}, score={self.interestingness_score:.1f})"


class TaskCandidateGenerator:
    """Generate intelligent task candidates from COBOL datasets (v2.0 aligned)
    
    v2.0 Approach:
    - ALL tasks are documentation tasks
    - Tasks are categorized by complexity TIER (T1-T4)
    - Selection based on LOC ranges per tier
    - Understanding is validated through execution, not separate tasks
    """
    
    def __init__(self, datasets_dir: Path = DATASETS_DIR):
        self.datasets_dir = datasets_dir
        self.analyzer_cache = {}  # Cache file analyses
        self.calibrator = DifficultyCalibrator()
        self.domain_detector = DomainDetector()
        
        # v2.0 tier configuration
        self.tier_config = TASK_SELECTION_CONFIG["task_distribution"]
        self.tier_targets = self.tier_config["tier_distribution"]
        self.tier_loc_ranges = self.tier_config["tier_loc_ranges"]
    
    def generate_all_candidates(self, min_loc: int = 100, max_loc: int = 5000) -> List[TaskCandidate]:
        """Generate all possible task candidates from datasets (v2.0: documentation only)"""
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
                loc = analysis["loc"]
                
                # Filter by overall LOC range
                if not (min_loc <= loc <= max_loc):
                    continue
                
                # Determine tier based on LOC
                tier = self._determine_tier(loc, analysis)
                
                # Create documentation candidate with tier
                candidate = self._create_candidate(cobol_file, analysis, dataset_name, tier)
                if candidate:
                    all_candidates.append(candidate)
        
        logger.info(f"Generated {len(all_candidates)} total candidates (documentation only)")
        
        # Log tier distribution
        tier_counts = {"T1": 0, "T2": 0, "T3": 0, "T4": 0}
        for c in all_candidates:
            tier_counts[c.tier] += 1
        logger.info(f"Tier distribution: {tier_counts}")
        
        return all_candidates
    
    def _determine_tier(self, loc: int, analysis: Dict) -> str:
        """Determine complexity tier based on LOC and code analysis"""
        # Primary: LOC-based tier assignment (per PRD Section 8)
        if loc <= 500:
            base_tier = "T1"
        elif loc <= 1000:
            base_tier = "T2"
        elif loc <= 2000:
            base_tier = "T3"
        else:
            base_tier = "T4"
        
        # Secondary: Complexity adjustments
        # Upgrade tier if code has complex patterns
        complexity_score = analysis.get("complexity_score", 0)
        has_goto = analysis.get("goto_count", 0) > 5
        has_external_calls = analysis.get("dependencies", {}).get("total", 0) > 3
        
        if base_tier == "T1" and (has_goto or has_external_calls):
            return "T2"
        elif base_tier == "T2" and has_goto and has_external_calls:
            return "T3"
        elif base_tier == "T3" and has_goto and complexity_score > 80:
            return "T4"
        
        return base_tier
    
    def select_best_tasks(self, candidates: List[TaskCandidate], 
                         total_tasks: int = 200) -> List[TaskCandidate]:
        """Select best tasks with balanced tier distribution (v2.0)
        
        Args:
            candidates: All candidate tasks
            total_tasks: Total number of tasks to select (default: 200)
        
        Returns:
            List of selected TaskCandidate objects (documentation only)
        """
        # Calculate targets per tier based on distribution
        tier_targets = {
            "T1": int(total_tasks * 0.40),   # 40% basic
            "T2": int(total_tasks * 0.35),   # 35% moderate
            "T3": int(total_tasks * 0.20),   # 20% complex
            "T4": int(total_tasks * 0.05),   # 5% enterprise
        }
        
        # Group candidates by tier
        by_tier = {"T1": [], "T2": [], "T3": [], "T4": []}
        for candidate in candidates:
            by_tier[candidate.tier].append(candidate)
        
        # Sort each tier by interestingness score
        for tier in by_tier:
            by_tier[tier].sort(key=lambda c: c.interestingness_score, reverse=True)
        
        # Select from each tier
        selected = []
        used_files = set()
        
        for tier in ["T1", "T2", "T3", "T4"]:
            target = tier_targets[tier]
            available = by_tier[tier]
            count = 0
            
            for candidate in available:
                if count >= target:
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
                count += 1
            
            logger.info(f"  {tier}: Selected {count}/{target} tasks (available: {len(available)})")
        
        # Fill remaining slots from any tier if we're short
        total_selected = len(selected)
        if total_selected < total_tasks:
            remaining = [c for c in candidates if str(c.main_file) not in used_files]
            remaining.sort(key=lambda c: c.interestingness_score, reverse=True)
            
            for candidate in remaining:
                if len(selected) >= total_tasks:
                    break
                selected.append(candidate)
                used_files.add(str(candidate.main_file))
        
        logger.info(f"Selected {len(selected)} documentation tasks (target: {total_tasks})")
        
        return selected
    
    def _create_candidate(self, main_file: Path, analysis: Dict, 
                         dataset_name: str, tier: str) -> TaskCandidate:
        """Create documentation task candidate (v2.0: all tasks are documentation)"""
        candidate = TaskCandidate(main_file, dataset_name, tier)
        candidate.analysis = analysis
        
        # Find related copybooks
        candidate.related_files = self._find_related_copybooks(main_file, analysis)
        
        # Calculate scores
        analyzer = self._get_analyzer(main_file)
        candidate.interestingness_score = analyzer.calculate_interestingness_score()
        candidate.difficulty_score = self.calibrator.calculate_difficulty_score(
            analysis, "documentation"
        )
        candidate.domain = self.domain_detector.detect_domain(analysis, dataset_name)
        
        # Must have some business logic for documentation
        if analysis.get("business_rules", 0) < 1:
            return None
        
        return candidate
    
    def _select_diverse_tasks(self, candidates: List[TaskCandidate], num_tasks: int) -> List[TaskCandidate]:
        """Select tasks with diverse difficulty levels and domains (legacy compatibility)"""
        # This method is kept for backward compatibility but delegates to select_best_tasks
        return self.select_best_tasks(candidates, num_tasks)
    
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

