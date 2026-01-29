"""
Copybook Resolver

Resolves COBOL COPY statements by finding and staging copybook files.
Uses disk caching for performance on large datasets.

Features:
- Recursive copybook resolution (copybooks can COPY other copybooks)
- Cycle detection (prevents infinite loops)
- Disk caching (avoids re-scanning datasets)
- Multiple COPY syntax support (COPY X, COPY 'X', COPY X OF LIB)
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CopybookResolver:
    """
    Resolve COBOL COPY statements to actual copybook files.
    
    Usage:
        resolver = CopybookResolver(dataset_path)
        copybooks, missing = resolver.resolve(source_code)
        # copybooks: List[Path] - paths to copybook files found
        # missing: List[str] - names of copybooks not found
    """
    
    # Cache index for all instances (class-level)
    _global_index: Dict[str, Dict[str, Path]] = {}
    
    def __init__(self, dataset_path: Path, max_depth: int = 5):
        """
        Initialize resolver for a dataset.
        
        Args:
            dataset_path: Root path of the dataset
            max_depth: Maximum recursion depth for nested COPY statements
        """
        self.dataset_path = Path(dataset_path)
        self.max_depth = max_depth
        self._ensure_index()
    
    def _ensure_index(self) -> None:
        """Ensure copybook index is loaded for this dataset."""
        key = str(self.dataset_path)
        if key not in CopybookResolver._global_index:
            CopybookResolver._global_index[key] = self._load_or_build_index()
    
    @property
    def index(self) -> Dict[str, Path]:
        """Get the copybook index for this dataset."""
        return CopybookResolver._global_index[str(self.dataset_path)]
    
    def resolve(self, source_code: str) -> Tuple[List[Path], List[str]]:
        """
        Resolve all COPY statements in source code.
        
        Recursively resolves copybooks (copybooks can COPY other copybooks).
        Detects and handles cycles.
        
        Args:
            source_code: COBOL source code
            
        Returns:
            (resolved, missing) tuple:
            - resolved: List of copybook Paths found
            - missing: List of copybook names not found
        """
        resolved = []
        missing = []
        seen = set()
        
        def _resolve_recursive(code: str, depth: int) -> None:
            if depth > self.max_depth:
                logger.warning(f"Max copybook depth {self.max_depth} reached, stopping recursion")
                return
            
            # Extract COPY statements
            copybook_names = self.extract_copy_statements(code)
            
            for name in copybook_names:
                if name in seen:
                    continue  # Already processed (cycle detection)
                seen.add(name)
                
                # Find copybook file
                copybook_path = self.find_copybook(name)
                
                if copybook_path:
                    resolved.append(copybook_path)
                    
                    # Recursively resolve nested copies
                    try:
                        copybook_content = copybook_path.read_text(errors='ignore')
                        _resolve_recursive(copybook_content, depth + 1)
                    except Exception as e:
                        logger.warning(f"Failed to read copybook {copybook_path}: {e}")
                else:
                    if name not in missing:
                        missing.append(name)
        
        _resolve_recursive(source_code, 0)
        
        logger.info(f"Copybook resolution: {len(resolved)} found, {len(missing)} missing")
        if missing:
            logger.debug(f"Missing copybooks: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        
        return resolved, missing
    
    def extract_copy_statements(self, source_code: str) -> Set[str]:
        """
        Extract copybook names from COPY statements.
        
        Handles COBOL COPY variants:
        - COPY NAME.
        - COPY 'NAME'.
        - COPY NAME REPLACING ...
        - COPY NAME OF LIBRARY.
        
        Args:
            source_code: COBOL source code
            
        Returns:
            Set of copybook names (uppercase, no extension)
        """
        # Remove comments first
        clean_code = self._remove_comments(source_code)
        
        # Extract COPY statements with various patterns
        patterns = [
            r"COPY\s+([A-Z0-9_\-]+)(?:\s|\.)",          # COPY NAME.
            r"COPY\s+['\"]([A-Z0-9_\-]+)['\"]",         # COPY 'NAME'
            r"COPY\s+([A-Z0-9_\-]+)\s+(?:OF|IN)\s+\w+", # COPY NAME OF LIBRARY
            r"COPY\s+([A-Z0-9_\-]+)\s+REPLACING",       # COPY NAME REPLACING
        ]
        
        names = set()
        for pattern in patterns:
            for match in re.finditer(pattern, clean_code, re.IGNORECASE):
                names.add(match.group(1).upper())
        
        return names
    
    def _remove_comments(self, source_code: str) -> str:
        """
        Remove COBOL comments from source code.
        
        COBOL comments:
        - Column 7 = '*' (comment line)
        - Column 7 = '/' (page break, treated as comment)
        - Column 7 = 'D' (debug line, often treated as comment)
        """
        lines = source_code.split('\n')
        clean_lines = []
        
        for line in lines:
            # Skip if line is too short
            if len(line) < 7:
                clean_lines.append(line)
                continue
            
            # Check column 7 (0-indexed = position 6)
            indicator = line[6] if len(line) > 6 else ' '
            if indicator in ('*', '/', 'D', 'd'):
                continue  # Skip comment line
            
            clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    def find_copybook(self, name: str) -> Optional[Path]:
        """
        Find a copybook file by name.
        
        Searches the index for:
        - NAME.cpy
        - NAME.CPY
        - NAME.copy
        - NAME (no extension)
        
        Args:
            name: Copybook name (case-insensitive)
            
        Returns:
            Path to copybook file, or None if not found
        """
        return self.index.get(name.upper())
    
    def _load_or_build_index(self) -> Dict[str, Path]:
        """
        Load copybook index from disk cache, or build and persist it.
        
        Cache file: {dataset_path}/.copybook_index.json
        
        Returns:
            Dict mapping copybook names (uppercase) to file paths
        """
        cache_file = self.dataset_path / ".copybook_index.json"
        
        # Try to load from disk cache
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                # Convert string paths back to Path objects
                index = {k: Path(v) for k, v in cached.items()}
                logger.info(f"Loaded copybook index from cache: {len(index)} entries")
                return index
            except Exception as e:
                logger.warning(f"Failed to load copybook cache: {e}")
        
        # Build index by scanning dataset
        logger.info(f"Building copybook index for {self.dataset_path}...")
        index = self._build_index()
        
        # Persist to disk for future runs
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                # Convert Path objects to strings for JSON
                json.dump({k: str(v) for k, v in index.items()}, f, indent=2)
            logger.info(f"Saved copybook index to cache: {len(index)} entries")
        except Exception as e:
            logger.warning(f"Failed to save copybook cache: {e}")
        
        return index
    
    def _build_index(self) -> Dict[str, Path]:
        """
        Build copybook index by scanning the dataset directory.
        
        Scans for files with extensions: .cpy, .CPY, .copy, .COPY
        Also includes files without extensions in cpy/copy directories.
        
        Returns:
            Dict mapping copybook names (uppercase) to file paths
        """
        index = {}
        
        # Extensions to consider as copybooks
        copybook_extensions = {'.cpy', '.copy', '.CPY', '.COPY'}
        
        # Also look in directories commonly named for copybooks
        copybook_dirs = {'cpy', 'copy', 'copybook', 'copybooks', 'CPY', 'COPY'}
        
        for path in self.dataset_path.rglob("*"):
            if not path.is_file():
                continue
            
            # Check if it's a copybook by extension
            is_copybook = path.suffix.lower() in {'.cpy', '.copy'}
            
            # Or if it's in a copybook directory
            if not is_copybook:
                for parent in path.parents:
                    if parent.name.lower() in {'cpy', 'copy', 'copybook', 'copybooks'}:
                        is_copybook = True
                        break
            
            if is_copybook:
                # Use uppercase stem as key
                key = path.stem.upper()
                if key not in index:  # First found wins
                    index[key] = path
        
        logger.info(f"Built copybook index: {len(index)} entries")
        return index
    
    @classmethod
    def clear_cache(cls, dataset_path: Path = None) -> None:
        """
        Clear the in-memory and disk cache.
        
        Args:
            dataset_path: Clear cache for specific dataset, or all if None
        """
        if dataset_path:
            key = str(dataset_path)
            if key in cls._global_index:
                del cls._global_index[key]
            cache_file = dataset_path / ".copybook_index.json"
            if cache_file.exists():
                cache_file.unlink()
        else:
            cls._global_index.clear()


def resolve_copybooks(
    source_code: str,
    dataset_path: Path,
    max_depth: int = 5
) -> Tuple[List[Path], List[str]]:
    """
    Convenience function to resolve copybooks.
    
    Creates a CopybookResolver instance and resolves copybooks.
    
    Args:
        source_code: COBOL source code
        dataset_path: Root path of the dataset
        max_depth: Maximum recursion depth
        
    Returns:
        (resolved, missing) tuple
    """
    resolver = CopybookResolver(dataset_path, max_depth)
    return resolver.resolve(source_code)
