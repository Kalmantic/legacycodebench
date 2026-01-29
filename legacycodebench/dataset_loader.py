"""Dataset loader for COBOL files from GitHub repositories

LegacyCodeBench v2.0 - Supports tier-based loading for 200-task benchmark
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import logging
from git import Repo
import requests
from zipfile import ZipFile
import io
import json

from legacycodebench.config import (
    DATASETS_DIR, 
    DATASET_SOURCES,
    UNIBASIC_DATASET_SOURCES,  # V2.4: UniBasic datasets
    get_datasets_by_tier,
    get_total_estimated_files,
    TASK_SELECTION_CONFIG,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load COBOL datasets from GitHub repositories
    
    Supports tier-based loading for LegacyCodeBench v2.0:
    - T1: Basic (80 tasks) - Educational, well-documented
    - T2: Moderate (70 tasks) - PERFORM loops, file operations
    - T3: Complex (40 tasks) - External CALLs, business rules
    - T4: Enterprise (10 tasks) - GO TO spaghetti, CICS/DB2
    """
    
    def __init__(self, datasets_dir: Path = DATASETS_DIR):
        self.datasets_dir = datasets_dir
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def load_all_datasets(self, tier: str = None) -> Dict[str, Path]:
        """Load datasets from configured sources
        
        Args:
            tier: Optional tier filter ('T1', 'T2', 'T3', 'T4') or None for all
            
        Returns:
            Dictionary mapping source_id to dataset path
        """
        sources = get_datasets_by_tier(tier) if tier else DATASET_SOURCES
        loaded = {}
        
        logger.info(f"Loading {len(sources)} dataset(s)" + (f" for tier {tier}" if tier else ""))
        
        for source_id, source_info in sources.items():
            try:
                dataset_path = self.load_dataset(source_id, source_info["url"], source_info)
                if dataset_path:
                    loaded[source_id] = {
                        "path": dataset_path,
                        "tier": source_info.get("tier", "T1"),
                        "description": source_info.get("description", ""),
                    }
                    logger.info(f"✓ Loaded {source_id} [{source_info.get('tier', 'T1')}]")
            except Exception as e:
                logger.error(f"✗ Failed to load {source_id}: {e}")
        return loaded
    
    def load_by_tier(self, tier: str) -> Dict[str, Path]:
        """Load all datasets for a specific tier"""
        return self.load_all_datasets(tier=tier)
    
    def load_dataset(self, source_id: str, repo_url: str, source_info: Dict = None) -> Optional[Path]:
        """Load a single dataset from GitHub
        
        Args:
            source_id: Unique identifier for the dataset
            repo_url: GitHub repository URL
            source_info: Optional dictionary with dataset metadata (may contain 'subpath')
        """
        target_dir = self.datasets_dir / source_id
        subpath = source_info.get("subpath") if source_info else None
        
        # Determine the actual path to use (root or subpath)
        actual_path = target_dir / subpath if subpath else target_dir
        
        # Skip if already exists with COBOL or UniBasic files
        if actual_path.exists():
            cobol_files = list(actual_path.rglob("*.cbl")) + list(actual_path.rglob("*.cob"))
            # V2.4: Also check for UniBasic files
            unibasic_files = []
            for ext in ["*.bp", "*.bas", "*.b", "*.BP", "*.BAS", "*.B"]:
                unibasic_files.extend(actual_path.rglob(ext))
            
            # Also check for "BP" directories (Pick convention)
            for path in actual_path.rglob("*"):
                 if path.is_dir() and (path.name == "BP" or path.name.endswith(".BP")):
                     if any(f.is_file() for f in path.iterdir()):
                         unibasic_files.append(path)

            if cobol_files or unibasic_files:
                logger.info(f"Dataset {source_id} already exists ({len(cobol_files) + len(unibasic_files)} files), skipping")
                return actual_path
        
        # FIXED: Clean up corrupted/partial downloads before retrying
        if target_dir.exists():
            # Check if directory exists but has no source files (corrupted/partial download)
            cobol_files = list(target_dir.rglob("*.cbl")) + list(target_dir.rglob("*.cob"))
            
            unibasic_files = []
            for ext in ["*.bp", "*.bas", "*.b", "*.BP", "*.BAS", "*.B"]:
                unibasic_files.extend(target_dir.rglob(ext))
            
            # Also check for "BP" directories
            for path in target_dir.rglob("*"):
                 if path.is_dir() and (path.name == "BP" or path.name.endswith(".BP")):
                     if any(f.is_file() for f in path.iterdir()):
                         unibasic_files.append(path)

            if not cobol_files and not unibasic_files:
                logger.info(f"Removing incomplete dataset {source_id} and retrying...")
                try:
                    shutil.rmtree(target_dir)
                except Exception as e:
                    logger.warning(f"Could not remove {target_dir}: {e}")
            else:
                # If files exist but we are here, it means we passed the first check? 
                # Actually, the first check returns early. So if we are here, something is wrong or I logic is slightly off.
                # The first check verified 'actual_path', this check verifies 'target_dir'.
                # In most cases they are same. If they differ, we might be here.
                # Let's just return if we find files to be safe.
                return target_dir
        
        # Try cloning with git
        try:
            logger.info(f"Cloning {repo_url}...")
            Repo.clone_from(repo_url, target_dir, depth=1)
            logger.info(f"✓ Cloned {source_id}")
            
            # If subpath is specified, return the subpath directory
            if subpath:
                subpath_dir = target_dir / subpath
                if subpath_dir.exists():
                    logger.info(f"Using subpath: {subpath}")
                    return subpath_dir
                else:
                    logger.warning(f"Subpath '{subpath}' not found in {source_id}, using root directory")
            
            return target_dir
        except Exception as e:
            logger.warning(f"Git clone failed: {e}, trying zip download...")
            
            # Fallback: download as zip
            try:
                # Try main branch first, then master
                for branch in ["main", "master"]:
                    zip_url = f"{repo_url}/archive/refs/heads/{branch}.zip"
                    response = requests.get(zip_url, timeout=60)
                    if response.status_code == 200:
                        target_dir.mkdir(parents=True, exist_ok=True)
                        with ZipFile(io.BytesIO(response.content)) as zip_file:
                            # Extract to temp, then move contents
                            zip_file.extractall(self.datasets_dir / "_temp")
                        
                        # Move extracted folder contents to target
                        temp_dirs = list((self.datasets_dir / "_temp").iterdir())
                        if temp_dirs:
                            source_dir = temp_dirs[0]
                            # Move all items from the extracted folder to the target directory
                            for item in source_dir.iterdir():
                                shutil.move(str(item), str(target_dir))
                        shutil.rmtree(self.datasets_dir / "_temp", ignore_errors=True)
                        
                        logger.info(f"✓ Downloaded {source_id} as zip")
                        
                        # If subpath is specified, return the subpath directory
                        if subpath:
                            subpath_dir = target_dir / subpath
                            if subpath_dir.exists():
                                logger.info(f"Using subpath: {subpath}")
                                return subpath_dir
                            else:
                                logger.warning(f"Subpath '{subpath}' not found in {source_id}, using root directory")
                        
                        return target_dir
                
                logger.error(f"No valid branch found for {source_id}")
                return None
            except Exception as e2:
                logger.error(f"Zip download also failed: {e2}")
                return None
    
    def find_cobol_files(self, dataset_path: Path) -> List[Path]:
        """Find all COBOL files in a dataset"""
        cobol_files = []
        for ext in ["*.cbl", "*.cob", "*.CBL", "*.COB"]:
            cobol_files.extend(dataset_path.rglob(ext))
        # Exclude copybooks for main program count
        return sorted([f for f in cobol_files if not f.suffix.lower() == '.cpy'])
    
    def find_copybooks(self, dataset_path: Path) -> List[Path]:
        """Find all COBOL copybooks in a dataset"""
        copybooks = []
        for ext in ["*.cpy", "*.CPY"]:
            copybooks.extend(dataset_path.rglob(ext))
        return sorted(copybooks)
    
    def find_unibasic_files(self, dataset_path: Path) -> List[Path]:
        """Find all UniBasic files in a dataset (V2.4)"""
        unibasic_files = []
        for ext in ["*.bp", "*.bas", "*.b", "*.BP", "*.BAS", "*.B"]:
            unibasic_files.extend(dataset_path.rglob(ext))
        return sorted(unibasic_files, key=str)
    
    def load_unibasic_datasets(self) -> Dict[str, Path]:
        """Load UniBasic datasets from configured sources (V2.4)
        
        Returns:
            Dictionary mapping source_id to dataset path
        """
        loaded = {}
        
        logger.info(f"Loading {len(UNIBASIC_DATASET_SOURCES)} UniBasic dataset(s)")
        
        for source_id, source_info in UNIBASIC_DATASET_SOURCES.items():
            try:
                dataset_path = self.load_dataset(source_id, source_info["url"], source_info)
                if dataset_path:
                    loaded[source_id] = {
                        "path": dataset_path,
                        "tier": source_info.get("tier", "T1"),
                        "language": "unibasic",
                        "description": source_info.get("description", ""),
                    }
                    logger.info(f"✓ Loaded UniBasic: {source_id} [{source_info.get('tier', 'T1')}]")
            except Exception as e:
                logger.error(f"✗ Failed to load UniBasic {source_id}: {e}")
        return loaded
    
    def load_all_datasets_v24(self, language: str = "all") -> Dict[str, Path]:
        """Load datasets for V2.4 with language filter
        
        Args:
            language: "cobol", "unibasic", or "all"
            
        Returns:
            Dictionary mapping source_id to dataset info
        """
        loaded = {}
        
        if language in ["cobol", "all"]:
            cobol_datasets = self.load_all_datasets()
            for k, v in cobol_datasets.items():
                if isinstance(v, dict):
                    v["language"] = "cobol"
                loaded[k] = v
        
        if language in ["unibasic", "all"]:
            unibasic_datasets = self.load_unibasic_datasets()
            loaded.update(unibasic_datasets)
        
        return loaded
    
    def get_dataset_stats(self, dataset_path: Path) -> Dict:
        """Get detailed statistics about a dataset"""
        cobol_files = self.find_cobol_files(dataset_path)
        copybooks = self.find_copybooks(dataset_path)
        
        total_lines = 0
        total_loc = 0  # Lines of code (excluding comments/blanks)
        file_stats = []
        
        for file_path in cobol_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
                    
                    # Count non-comment, non-blank lines
                    loc = sum(1 for line in lines 
                             if line.strip() and 
                             not line.strip().startswith('*') and
                             len(line) > 6 and line[6] != '*')
                    total_loc += loc
                    
                    file_stats.append({
                        "file": str(file_path.name),
                        "lines": len(lines),
                        "loc": loc,
                    })
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
        
        return {
            "total_programs": len(cobol_files),
            "total_copybooks": len(copybooks),
            "total_lines": total_lines,
            "total_loc": total_loc,
            "avg_loc_per_file": total_loc // len(cobol_files) if cobol_files else 0,
            "files": file_stats[:20],  # First 20 for preview
        }
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all loaded datasets, grouped by tier"""
        stats = {"T1": [], "T2": [], "T3": [], "T4": []}
        totals = {"T1": 0, "T2": 0, "T3": 0, "T4": 0}
        
        for source_id, source_info in DATASET_SOURCES.items():
            dataset_path = self.datasets_dir / source_id
            tier = source_info.get("tier", "T1")
            
            if dataset_path.exists():
                ds_stats = self.get_dataset_stats(dataset_path)
                stats[tier].append({
                    "source_id": source_id,
                    "description": source_info.get("description", ""),
                    **ds_stats
                })
                totals[tier] += ds_stats["total_programs"]
        
        return {
            "by_tier": stats,
            "totals": totals,
            "grand_total": sum(totals.values()),
        }
    
    def export_manifest(self, output_path: Path = None) -> Path:
        """Export a manifest of all available COBOL files"""
        output_path = output_path or self.datasets_dir / "manifest.json"
        
        manifest = {
            "version": "2.0",
            "total_target": TASK_SELECTION_CONFIG["task_distribution"]["total_tasks"],
            "tiers": {},
        }
        
        for tier in ["T1", "T2", "T3", "T4"]:
            sources = get_datasets_by_tier(tier)
            tier_files = []
            
            for source_id in sources:
                dataset_path = self.datasets_dir / source_id
                if dataset_path.exists():
                    for cobol_file in self.find_cobol_files(dataset_path):
                        tier_files.append({
                            "source": source_id,
                            "file": str(cobol_file.relative_to(self.datasets_dir)),
                            "name": cobol_file.stem,
                        })
            
            manifest["tiers"][tier] = {
                "target_tasks": TASK_SELECTION_CONFIG["task_distribution"]["tier_distribution"].get(
                    f"{tier}_{'basic' if tier == 'T1' else 'moderate' if tier == 'T2' else 'complex' if tier == 'T3' else 'enterprise'}", 0
                ),
                "available_files": len(tier_files),
                "files": tier_files,
            }
        
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"✓ Exported manifest to {output_path}")
        return output_path


def main():
    """CLI entry point for dataset loading"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LegacyCodeBench v2.0 Dataset Loader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m legacycodebench.dataset_loader                    # Load all datasets
  python -m legacycodebench.dataset_loader --tier T1          # Load only T1 (Basic) datasets
  python -m legacycodebench.dataset_loader --stats            # Show statistics only
  python -m legacycodebench.dataset_loader --manifest         # Export file manifest
        """
    )
    parser.add_argument("--tier", choices=["T1", "T2", "T3", "T4"], 
                        help="Load only datasets for a specific tier")
    parser.add_argument("--stats", action="store_true",
                        help="Show statistics for loaded datasets")
    parser.add_argument("--manifest", action="store_true",
                        help="Export manifest of all COBOL files")
    parser.add_argument("--summary", action="store_true",
                        help="Show configuration summary")
    parser.add_argument("--language", choices=["cobol", "unibasic", "all"], default="cobol",
                        help="Language to load datasets for (default: cobol)")
    
    args = parser.parse_args()
    
    loader = DatasetLoader()
    
    # Show summary if requested
    if args.summary:
        from legacycodebench.config import print_dataset_summary
        print_dataset_summary()
        return
    
    # Show stats if requested
    if args.stats:
        print("\n" + "=" * 60)
        print("LegacyCodeBench v2.0 Dataset Statistics")
        print("=" * 60)
        
        all_stats = loader.get_all_stats()
        
        for tier in ["T1", "T2", "T3", "T4"]:
            tier_name = {"T1": "Basic", "T2": "Moderate", "T3": "Complex", "T4": "Enterprise"}[tier]
            tier_stats = all_stats["by_tier"][tier]
            tier_total = all_stats["totals"][tier]
            
            print(f"\n{tier}: {tier_name} ({tier_total} programs)")
            print("-" * 40)
            
            for ds in tier_stats:
                print(f"  {ds['source_id']}: {ds['total_programs']} programs, {ds['total_loc']} LOC")
        
        print("\n" + "=" * 60)
        print(f"Grand Total: {all_stats['grand_total']} programs")
        print("=" * 60)
        return
    
    # Export manifest if requested
    if args.manifest:
        manifest_path = loader.export_manifest()
        print(f"✓ Manifest exported to {manifest_path}")
        return
    
    # Load datasets
    if args.language == "cobol" and args.tier:
        # Legacy tier filtering only supported for COBOL currently
        loaded = loader.load_all_datasets(tier=args.tier)
    else:
        # Use V2.4 loader
        loaded = loader.load_all_datasets_v24(language=args.language)
    
    print(f"\n" + "=" * 60)
    print(f"Loaded {len(loaded)} dataset(s)")
    print("=" * 60)
    
    tier_counts = {"T1": 0, "T2": 0, "T3": 0, "T4": 0}
    
    # Handle encoding for output
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    for source_id, info in loaded.items():
        if isinstance(info, dict):
            path = info["path"]
            tier = info["tier"]
        else:
            path = info
            tier = DATASET_SOURCES.get(source_id, {}).get("tier", "T1")
        
        if args.language == "unibasic":
            cobol_files = loader.find_unibasic_files(path)
            total_loc = 0 # Simple count for preview
            count = len(cobol_files)
        else:
            stats = loader.get_dataset_stats(path)
            total_loc = stats['total_loc']
            count = stats['total_programs']
            
        tier_counts[tier] += count
        print(f"  [{tier}] {source_id}: {count} programs")
    
    print("\n" + "-" * 40)
    print("Summary by Tier:")
    for tier, count in tier_counts.items():
        if count > 0:
            target = TASK_SELECTION_CONFIG["task_distribution"]["tier_distribution"].get(
                f"{tier}_{'basic' if tier == 'T1' else 'moderate' if tier == 'T2' else 'complex' if tier == 'T3' else 'enterprise'}", 0
            )
            status = "✓" if count >= target else " "
            print(f"  {tier}: {count} programs (target: {target}) {status}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

