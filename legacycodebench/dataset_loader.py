"""Dataset loader for COBOL files from GitHub repositories"""

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

from legacycodebench.config import DATASETS_DIR, DATASET_SOURCES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load COBOL datasets from GitHub repositories"""
    
    def __init__(self, datasets_dir: Path = DATASETS_DIR):
        self.datasets_dir = datasets_dir
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def load_all_datasets(self) -> Dict[str, Path]:
        """Load all datasets from configured sources"""
        loaded = {}
        for source_id, source_info in DATASET_SOURCES.items():
            try:
                dataset_path = self.load_dataset(source_id, source_info["url"])
                if dataset_path:
                    loaded[source_id] = dataset_path
                    logger.info(f"✓ Loaded {source_id} to {dataset_path}")
            except Exception as e:
                logger.error(f"✗ Failed to load {source_id}: {e}")
        return loaded
    
    def load_dataset(self, source_id: str, repo_url: str) -> Optional[Path]:
        """Load a single dataset from GitHub"""
        target_dir = self.datasets_dir / source_id
        
        # Skip if already exists
        if target_dir.exists() and list(target_dir.glob("**/*.cbl")):
            logger.info(f"Dataset {source_id} already exists, skipping")
            return target_dir
        
        # Try cloning with git
        try:
            logger.info(f"Cloning {repo_url}...")
            Repo.clone_from(repo_url, target_dir, depth=1)
            logger.info(f"✓ Cloned {source_id}")
            return target_dir
        except Exception as e:
            logger.warning(f"Git clone failed: {e}, trying zip download...")
            
            # Fallback: download as zip
            try:
                zip_url = repo_url.replace("github.com", "github.com/archive/refs/heads/main.zip")
                if "main" not in zip_url:
                    zip_url = repo_url.replace("github.com", "github.com/archive/refs/heads/master.zip")
                
                response = requests.get(zip_url, timeout=30)
                if response.status_code == 200:
                    with ZipFile(io.BytesIO(response.content)) as zip_file:
                        zip_file.extractall(self.datasets_dir)
                    logger.info(f"✓ Downloaded {source_id} as zip")
                    return target_dir
            except Exception as e2:
                logger.error(f"Zip download also failed: {e2}")
                return None
    
    def find_cobol_files(self, dataset_path: Path, subpath: Optional[str] = None) -> List[Path]:
        """Find all COBOL files in a dataset, optionally within a subpath"""
        # If subpath is specified, search only within that directory
        search_path = dataset_path
        if subpath:
            search_path = dataset_path / subpath
            if not search_path.exists():
                logger.warning(f"Subpath {subpath} does not exist in {dataset_path}")
                # Fall back to searching entire dataset
                search_path = dataset_path

        cobol_files = []
        # Extended list of COBOL file extensions
        extensions = [
            "*.cbl", "*.cob", "*.cpy", "*.cobol",  # lowercase
            "*.CBL", "*.COB", "*.CPY", "*.COBOL",  # uppercase
        ]
        for ext in extensions:
            cobol_files.extend(search_path.rglob(ext))
        return sorted(cobol_files)
    
    def get_subpath_for_dataset(self, source_id: str) -> Optional[str]:
        """Get the subpath configuration for a dataset, if any"""
        if source_id in DATASET_SOURCES:
            return DATASET_SOURCES[source_id].get("subpath")
        return None

    def get_dataset_stats(self, dataset_path: Path, source_id: Optional[str] = None) -> Dict:
        """Get statistics about a dataset"""
        subpath = self.get_subpath_for_dataset(source_id) if source_id else None
        cobol_files = self.find_cobol_files(dataset_path, subpath)
        total_lines = 0
        total_files = len(cobol_files)
        
        for file_path in cobol_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    total_lines += len(f.readlines())
            except:
                pass
        
        return {
            "total_files": total_files,
            "total_lines": total_lines,
            "files": [str(f.relative_to(dataset_path)) for f in cobol_files[:10]],  # First 10
        }


def main():
    """CLI entry point"""
    loader = DatasetLoader()
    loaded = loader.load_all_datasets()

    print(f"\n✓ Loaded {len(loaded)} datasets:")
    for source_id, path in loaded.items():
        stats = loader.get_dataset_stats(path, source_id)
        subpath_info = f" (subpath: {loader.get_subpath_for_dataset(source_id)})" if loader.get_subpath_for_dataset(source_id) else ""
        print(f"  {source_id}: {stats['total_files']} files, {stats['total_lines']} lines{subpath_info}")


if __name__ == "__main__":
    main()

