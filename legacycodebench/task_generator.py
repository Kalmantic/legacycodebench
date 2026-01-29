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
import hashlib

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
        # FIXED (Issue 2.4): Added richer task metadata
        self.expected_challenges = []  # List of expected challenges for this task
        self.baseline_score = None  # Expected baseline score (for validation)
        # REPRODUCIBILITY: File hash for deterministic sorting
        self._file_hash = None
    
    @property
    def file_hash(self) -> str:
        """Get SHA256 hash of main file for deterministic sorting"""
        if self._file_hash is None:
            try:
                self._file_hash = hashlib.sha256(self.main_file.read_bytes()).hexdigest()
            except:
                self._file_hash = str(self.main_file)  # Fallback to path
        return self._file_hash
    
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
    
    def generate_all_candidates(self, min_loc: int = 100, max_loc: int = 5000,
                                 language: str = "cobol") -> List[TaskCandidate]:
        """Generate all possible task candidates from datasets (v2.0: documentation only)

        V2.4: Added language parameter for multi-language support
        """
        all_candidates = []
        lang_upper = language.upper()

        # Process each dataset
        for dataset_dir in self.datasets_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name
            logger.info(f"Analyzing dataset: {dataset_name}")

            # V2.4: Find source files based on language
            source_files = self._find_source_files(dataset_dir, language=language)
            logger.info(f"Found {len(source_files)} {lang_upper} files in {dataset_name}")

            # Analyze each file
            for source_file in source_files:
                try:
                    # V2.4: Use language-appropriate analyzer
                    if language == "unibasic":
                        analysis = self._analyze_unibasic_file(source_file)
                    else:
                        analyzer = self._get_analyzer(source_file)
                        analysis = analyzer.analyze()

                    loc = analysis.get("loc", 0)

                    # Filter by overall LOC range
                    if not (min_loc <= loc <= max_loc):
                        continue

                    # Determine tier based on LOC
                    tier = self._determine_tier(loc, analysis)

                    # Create documentation candidate with tier (V2.4: pass language)
                    candidate = self._create_candidate(source_file, analysis, dataset_name, tier, language=language)
                    if candidate:
                        all_candidates.append(candidate)
                except Exception as e:
                    logger.warning(f"Failed to analyze {source_file}: {e}")
                    continue

        logger.info(f"Generated {len(all_candidates)} total {lang_upper} candidates (documentation only)")

        # Log tier distribution
        tier_counts = {"T1": 0, "T2": 0, "T3": 0, "T4": 0}
        for c in all_candidates:
            tier_counts[c.tier] += 1
        logger.info(f"Tier distribution: {tier_counts}")

        return all_candidates

    def _analyze_unibasic_file(self, file_path: Path) -> Dict:
        """Analyze a UniBasic file for task generation (V2.4)

        Simple analysis for UniBasic files since we don't have a full parser yet.
        """
        content = file_path.read_text(encoding='utf-8', errors='replace')
        lines = content.splitlines()

        # Count lines of code (excluding blank lines and comments)
        loc = 0
        comment_lines = 0
        content_upper = content.upper()
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # UniBasic comments start with * or REM
            if stripped.startswith('*') or stripped.upper().startswith('REM '):
                comment_lines += 1
            else:
                loc += 1

        # Count subroutines (GOSUB targets, labels)
        subroutine_count = 0
        for line in lines:
            stripped = line.strip()
            # Labels are typically numeric or alphanumeric at start of line
            if stripped and (stripped[0].isdigit() or ':' in stripped[:20]):
                subroutine_count += 1

        # Complexity Features Detection
        # File I/O
        has_file_io = any(x in content_upper for x in ['OPEN ', 'READ ', 'WRITE ', 'DELETE ', 'MATREAD ', 'MATWRITE '])
        
        # System / C Interop
        has_c_interop = 'CALLC ' in content_upper or 'CALL_PYTHON' in content_upper
        
        # Networking / Advanced
        has_network = 'SOCKET' in content_upper
        
        # Enterprise Logic (Transactions, Phantoms)
        has_enterprise = 'TRANSACTION' in content_upper or 'PHANTOM' in content_upper

        # Detect external calls
        external_calls = []
        for pattern in ['READ ', 'WRITE ', 'MATREAD ', 'MATWRITE ', 'EXECUTE ', 'CALL ']:
            if pattern in content_upper:
                external_calls.append(pattern.strip())

        # Count external calls for dependencies
        call_count = content_upper.count('CALL ')
        execute_count = content_upper.count('EXECUTE ')

        return {
            "loc": loc,
            "file_size": file_path.stat().st_size,
            "paragraphs": subroutine_count,  # Treat labels as "paragraphs"
            "external_calls": external_calls,
            "comment_lines": comment_lines,
            "has_cics": False,  # UniBasic doesn't have CICS
            "has_sql": False,   # Different SQL syntax
            "goto_count": content_upper.count('GO TO') + content_upper.count('GOTO'),
            "exec_cics_count": 0,  # UniBasic doesn't have CICS
            "exec_sql_count": 0,   # Different SQL handling
            "dependencies": {"total": call_count + execute_count},
            # UniBasic Specific Features
            "has_file_io": has_file_io,
            "has_c_interop": has_c_interop,
            "has_network": has_network,
            "has_enterprise": has_enterprise,
        }
    
    def _determine_tier(self, loc: int, analysis: Dict) -> str:
        """
        Determine complexity tier using v2.1 multi-factor approach.

        Aligned with regenerate_tasks_v2.1.py for consistency.

        Base tier from file size:
        - T1: 0-20KB
        - T2: 20-40KB
        - T3: 40-80KB
        - T4: 80KB+

        Complexity boosts (tier jumps):
        - EXEC CICS present: +2 tiers
        - EXEC SQL present: +1 tier
        - GO TO >10: +1 tier, >5: +0.5 tier
        - CALL >5: +0.5 tier
        """
        # Size thresholds (same as regenerate_tasks_v2.1.py)
        SIZE_TIERS = {
            'T1': (0, 20000),
            'T2': (20000, 40000),
            'T3': (40000, 80000),
            'T4': (80000, float('inf'))
        }

        # Get file size from analysis
        file_size = analysis.get("file_size", 0)

        # Base tier from size
        base_tier = 'T1'
        for tier, (min_size, max_size) in SIZE_TIERS.items():
            if min_size <= file_size < max_size:
                base_tier = tier
                break

        # Calculate complexity boosts
        boosts = []

        exec_cics = analysis.get("exec_cics_count", 0)
        exec_sql = analysis.get("exec_sql_count", 0)
        goto_count = analysis.get("goto_count", 0)
        calls = analysis.get("dependencies", {}).get("total", 0)

        if exec_cics > 0:
            boosts.append(('cics', 2))
        if exec_sql > 0:
            boosts.append(('db2', 1))
        
        # UniBasic Specific Boosts
        if analysis.get("has_enterprise") or analysis.get("has_network"):
             boosts.append(('enterprise', 2))
        elif analysis.get("has_c_interop"):
             boosts.append(('interop', 1.5))
        elif analysis.get("has_file_io"):
             boosts.append(('io', 1))

        if goto_count > 10:
            boosts.append(('goto_heavy', 1))
        elif goto_count > 5:
            boosts.append(('goto_moderate', 0.5))
        if calls > 5:
            boosts.append(('external_calls', 0.5))

        # Apply boosts
        total_boost = sum(b[1] for b in boosts)
        tiers = ['T1', 'T2', 'T3', 'T4']
        current_idx = tiers.index(base_tier)
        boosted_idx = min(current_idx + int(total_boost), len(tiers) - 1)

        return tiers[boosted_idx]
    
    def select_best_tasks(self, candidates: List[TaskCandidate], 
                         total_tasks: int = 200) -> List[TaskCandidate]:
        """Select best tasks with balanced tier distribution and DATASET DIVERSITY (v2.4)
        
        Args:
            candidates: All candidate tasks
            total_tasks: Total number of tasks to select (default: 200)
        
        Returns:
            List of selected TaskCandidate objects
        """
        # Calculate targets per tier based on v2.1 distribution
        # Multi-factor scoring naturally produces more T4 tasks (CICS/SQL/GOTO)
        tier_targets = {
            "T1": int(total_tasks * 0.25),   # 25% basic
            "T2": int(total_tasks * 0.205) + 1,  # 20.5% moderate (adjusted for rounding)
            "T3": int(total_tasks * 0.205),  # 20.5% complex
            "T4": int(total_tasks * 0.34),   # 34% enterprise
        }
        
        # Group candidates by tier
        by_tier = {"T1": [], "T2": [], "T3": [], "T4": []}
        for candidate in candidates:
            by_tier[candidate.tier].append(candidate)
        
        # Sort candidates within each tier by score (Descending) then file hash
        for tier in by_tier:
            by_tier[tier].sort(key=lambda c: (-c.interestingness_score, c.file_hash))
        
        # Select from each tier with Round-Robin Diversity
        selected = []
        used_files = set()
        
        for tier in ["T1", "T2", "T3", "T4"]:
            target = tier_targets[tier]
            available = by_tier[tier]
            
            # Group available by dataset
            by_dataset = {}
            for c in available:
                if c.dataset_name not in by_dataset:
                    by_dataset[c.dataset_name] = []
                by_dataset[c.dataset_name].append(c)
                
            datasets_list = sorted(list(by_dataset.keys())) # Verified deterministic order
            
            selected_for_tier = []
            
            # Round Robin selection
            added_any = True
            while len(selected_for_tier) < target and added_any:
                added_any = False
                for dataset in datasets_list:
                    if len(selected_for_tier) >= target:
                        break
                    
                    dataset_candidates = by_dataset[dataset]
                    
                    # Find first valid candidate from this dataset
                    for cand in list(dataset_candidates): # Iterate copy to modify original if needed
                        if str(cand.main_file) in used_files:
                            dataset_candidates.remove(cand)
                            continue
                        
                        # Add it
                        selected_for_tier.append(cand)
                        used_files.add(str(cand.main_file))
                        dataset_candidates.remove(cand) # Remove so we don't pick it again
                        added_any = True
                        break # Move to next dataset
            
            # If we ran out of diverse candidates but still need more (unlikely with deep pools),
            # we just pick best remaining regardless of dataset
            if len(selected_for_tier) < target:
                 remaining = [c for c in available if str(c.main_file) not in used_files]
                 remaining.sort(key=lambda c: -c.interestingness_score)
                 for cand in remaining:
                     if len(selected_for_tier) >= target: break
                     selected_for_tier.append(cand)
                     used_files.add(str(cand.main_file))
            
            logger.info(f"  {tier}: Selected {len(selected_for_tier)}/{target} tasks")
            selected.extend(selected_for_tier)
        
        return selected
    
    def _create_candidate(self, main_file: Path, analysis: Dict, 
                         dataset_name: str, tier: str, language: str = "cobol") -> TaskCandidate:
        """Create documentation task candidate (v2.0: all tasks are documentation)
        
        V2.4: Added language parameter to skip COBOL-specific operations for UniBasic
        """
        candidate = TaskCandidate(main_file, dataset_name, tier)
        candidate.analysis = analysis
        
        # V2.4: COBOL-specific operations (skip for UniBasic)
        if language == "cobol":
            # Find related copybooks
            candidate.related_files = self._find_related_copybooks(main_file, analysis)
            
            # Calculate scores using COBOL analyzer
            analyzer = self._get_analyzer(main_file)
            candidate.interestingness_score = analyzer.calculate_interestingness_score()
            candidate.difficulty_score = self.calibrator.calculate_difficulty_score(
                analysis, "documentation"
            )
        else:
            # UniBasic: No copybooks, simpler scoring
            candidate.related_files = []
            candidate.interestingness_score = 5.0  # Default score
            candidate.difficulty_score = 5.0  # Will be normalized by tier
        
        # Domain detection works for all languages
        candidate.domain = self.domain_detector.detect_domain(analysis, dataset_name)
        
        # V2.4: UniBasic doesn't track business_rules the same way
        # Skip business logic check for non-COBOL
        if language == "cobol":
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
        # FIXED (Issue 2.2): Sort for deterministic file discovery
        for ext in extensions:
            # Try exact match
            for file_path in sorted(dataset_dir.rglob(f"{filename}{ext}"), key=str):
                return file_path

            # Try case-insensitive match
            for file_path in sorted(dataset_dir.rglob(f"*{ext}"), key=str):
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
        # FIXED (Issue 2.2): Sort for deterministic task selection
        cobol_files = []
        for ext in ["*.cbl", "*.cob", "*.COB", "*.CBL"]:
            cobol_files.extend(dataset_dir.rglob(ext))
        return sorted(cobol_files, key=str)
    
    def _find_unibasic_files(self, dataset_dir: Path) -> List[Path]:
        """Find all UniBasic files in dataset (V2.4)

        Handles both:
        1. Standard extensions: .bp, .bas, .b
        2. Pick/MultiValue convention: files inside *.BP or BP directories (no extension)
        """
        unibasic_files = []

        # Standard extensions (e.g., mvbasic, full-stack-with-pick-tutorial)
        for ext in ["*.bp", "*.bas", "*.b", "*.BP", "*.BAS", "*.B"]:
            # Filter out directories - only include actual files
            unibasic_files.extend(f for f in dataset_dir.rglob(ext) if f.is_file())

        # Pick/MultiValue convention: files inside *.BP or BP directories
        # These directories are "files" in Pick parlance containing programs as "items"
        
        # Find all directories that are "BP" (exact) or end in ".BP"
        potential_bp_dirs = []
        for path in dataset_dir.rglob("*"):
            if path.is_dir():
                if path.name.upper() == "BP" or path.name.upper().endswith(".BP"):
                     potential_bp_dirs.append(path)

        for bp_dir in potential_bp_dirs:
            # Each file inside is a UniBasic program (often no extension)
            for prog_file in bp_dir.iterdir():
                if prog_file.is_file():
                    # Check for binary files or hidden files to exclude
                    if prog_file.name.startswith('.'):
                        continue
                    # Skip common non-source files
                    if prog_file.suffix.lower() in ['.xml', '.json', '.md', '.txt', '.py']:
                        continue
                    
                    # Avoid duplicates if already found by extension
                    if prog_file not in unibasic_files:
                        unibasic_files.append(prog_file)

        return sorted(unibasic_files, key=str)
    
    def _find_source_files(self, dataset_dir: Path, language: str = "cobol") -> List[Path]:
        """Find all source files for a given language (V2.4)"""
        if language == "unibasic":
            return self._find_unibasic_files(dataset_dir)
        else:
            return self._find_cobol_files(dataset_dir)
    
    def _get_analyzer(self, file_path: Path) -> COBOLFileAnalyzer:
        """Get cached analyzer or create new one"""
        file_str = str(file_path)
        if file_str not in self.analyzer_cache:
            self.analyzer_cache[file_str] = COBOLFileAnalyzer(file_path)
        return self.analyzer_cache[file_str]

