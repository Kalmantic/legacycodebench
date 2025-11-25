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
    
    def find_cobol_files(self, dataset_path: Path) -> List[Path]:
        """Find all COBOL files in a dataset"""
        cobol_files = []
        for ext in ["*.cbl", "*.cob", "*.cpy", "*.COB", "*.CBL", "*.CPY"]:
            cobol_files.extend(dataset_path.rglob(ext))
        return sorted(cobol_files)
    
    def get_dataset_stats(self, dataset_path: Path) -> Dict:
        """Get statistics about a dataset"""
        cobol_files = self.find_cobol_files(dataset_path)
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
        stats = loader.get_dataset_stats(path)
        print(f"  {source_id}: {stats['total_files']} files, {stats['total_lines']} lines")


if __name__ == "__main__":
    main()

