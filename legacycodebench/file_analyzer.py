"""COBOL file analyzer for intelligent task selection"""

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class COBOLFileAnalyzer:
    """Analyze COBOL files for complexity, dependencies, and characteristics"""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.content = None
        self.analysis = None
        
    def analyze(self) -> Dict:
        """Perform complete analysis of COBOL file"""
        if self.analysis:
            return self.analysis
            
        # Read file
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                self.content = f.read()
        except Exception as e:
            logger.warning(f"Failed to read {self.file_path}: {e}")
            return self._empty_analysis()
        
        # Perform all analyses
        self.analysis = {
            "file_path": str(self.file_path),
            "file_name": self.file_path.name,
            "file_size": self.file_path.stat().st_size,  # v2.1: File size for tier calculation
            "loc": self._count_loc(),
            "complexity": self._analyze_complexity(),
            "dependencies": self._find_dependencies(),
            "business_rules": self._count_business_rules(),
            "file_operations": self._find_file_operations(),
            "has_comments": self._has_documentation_comments(),
            "domain_keywords": self._find_domain_keywords(),
            # v2.1: Multi-factor tier scoring metrics
            "exec_cics_count": self._count_exec_cics(),
            "exec_sql_count": self._count_exec_sql(),
            "goto_count": self._count_goto(),
        }
        
        return self.analysis
    
    def _count_loc(self) -> int:
        """Count lines of code (excluding comments and blank lines)"""
        if not self.content:
            return 0
        
        lines = self.content.split('\n')
        loc = 0
        
        for line in lines:
            stripped = line.strip()
            # Skip blank lines
            if not stripped:
                continue
            # Skip comment lines (COBOL comments start with * or / in column 7)
            if len(stripped) > 0 and stripped[0] in ['*', '/']:
                continue
            # Skip pure whitespace
            if stripped:
                loc += 1
        
        return loc
    
    def _analyze_complexity(self) -> Dict:
        """Analyze code complexity metrics"""
        if not self.content:
            return {"cyclomatic": 0, "nesting_depth": 0, "branches": 0}
        
        # Count decision points (IF, EVALUATE, PERFORM UNTIL)
        if_count = len(re.findall(r'\bIF\b', self.content, re.IGNORECASE))
        evaluate_count = len(re.findall(r'\bEVALUATE\b', self.content, re.IGNORECASE))
        perform_until = len(re.findall(r'\bPERFORM\s+.*\bUNTIL\b', self.content, re.IGNORECASE))
        
        # Cyclomatic complexity approximation
        cyclomatic = 1 + if_count + evaluate_count + perform_until
        
        # Estimate nesting depth (rough heuristic)
        max_nesting = 0
        current_nesting = 0
        for line in self.content.split('\n'):
            if re.search(r'\bIF\b', line, re.IGNORECASE):
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            if re.search(r'\bEND-IF\b', line, re.IGNORECASE):
                current_nesting = max(0, current_nesting - 1)
        
        return {
            "cyclomatic": cyclomatic,
            "nesting_depth": max_nesting,
            "branches": if_count + evaluate_count,
        }
    
    def _find_dependencies(self) -> Dict:
        """Find CALL and COPY dependencies"""
        if not self.content:
            return {"calls": [], "copies": [], "total": 0}
        
        # Find CALL statements
        call_matches = re.findall(r'CALL\s+[\'"]?(\w+)[\'"]?', self.content, re.IGNORECASE)
        # Remove duplicates while preserving order
        calls = list(dict.fromkeys(call_matches))
        
        # Find COPY statements
        copy_matches = re.findall(r'COPY\s+(\w+)', self.content, re.IGNORECASE)
        copies = list(dict.fromkeys(copy_matches))
        
        return {
            "calls": calls,
            "copies": copies,
            "total": len(calls) + len(copies),
        }
    
    def _count_business_rules(self) -> int:
        """Count business rules (IF statements with conditions)"""
        if not self.content:
            return 0
        
        # Find IF statements
        if_matches = re.findall(r'IF\s+(.+?)\s+(THEN|NEXT|CONTINUE|\n)', 
                               self.content, re.IGNORECASE | re.DOTALL)
        
        return len(if_matches)
    
    def _find_file_operations(self) -> Dict:
        """Find file I/O operations"""
        if not self.content:
            return {"operations": [], "files": [], "total": 0}
        
        # Find file operations
        file_ops = re.findall(r'(OPEN|READ|WRITE|CLOSE)\s+(\w+)', 
                             self.content, re.IGNORECASE)
        
        operations = [{"operation": op.upper(), "file": file} 
                     for op, file in file_ops]
        
        # Unique files
        files = list(dict.fromkeys([file for _, file in file_ops]))
        
        return {
            "operations": operations[:20],  # Limit to first 20
            "files": files,
            "total": len(file_ops),
        }
    
    def _has_documentation_comments(self) -> bool:
        """Check if file has documentation comments"""
        if not self.content:
            return False
        
        # Look for documentation comment blocks
        doc_patterns = [
            r'\*\s*(PURPOSE|DESCRIPTION|AUTHOR|DATE)',
            r'\*{5,}',  # Lines of asterisks
            r'/\*.*DESCRIPTION.*\*/',
        ]
        
        for pattern in doc_patterns:
            if re.search(pattern, self.content, re.IGNORECASE):
                return True
        
        return False
    
    def _find_domain_keywords(self) -> Dict[str, int]:
        """Find domain-specific keywords in code"""
        if not self.content:
            return {}
        
        content_lower = self.content.lower()
        
        domain_keywords = {
            "banking": ["account", "balance", "deposit", "withdrawal", "transaction", 
                       "card", "atm", "branch", "customer", "savings", "checking"],
            "finance": ["interest", "loan", "payment", "credit", "debit", "rate",
                       "principal", "amortization", "portfolio", "investment"],
            "insurance": ["policy", "claim", "premium", "coverage", "underwriting",
                         "beneficiary", "insured", "risk", "liability"],
            "retail": ["inventory", "order", "customer", "product", "sale", "purchase",
                      "price", "quantity", "warehouse", "shipment"],
            "hr": ["employee", "payroll", "salary", "department", "hire", "benefit",
                  "time", "attendance", "leave", "pension"],
        }
        
        domain_counts = {}
        for domain, keywords in domain_keywords.items():
            count = sum(1 for keyword in keywords if keyword in content_lower)
            if count > 0:
                domain_counts[domain] = count
        
        return domain_counts
    
    def calculate_interestingness_score(self, weights: Dict = None) -> float:
        """Calculate how 'interesting' this file is for a benchmark task"""
        if not self.analysis:
            self.analyze()
        
        if weights is None:
            weights = {
                "business_logic": 10,
                "dependencies": 5,
                "file_io": 5,
                "in_loc_range": 10,
                "has_comments": -5,
                "complexity": 3,
            }
        
        score = 0.0
        
        # Business logic (rules)
        if self.analysis["business_rules"] > 0:
            score += weights["business_logic"] * min(self.analysis["business_rules"] / 10, 1.0)
        
        # Dependencies
        score += weights["dependencies"] * min(self.analysis["dependencies"]["total"] / 5, 1.0)
        
        # File I/O
        if self.analysis["file_operations"]["total"] > 0:
            score += weights["file_io"]
        
        # LOC in range (500-2000)
        loc = self.analysis["loc"]
        if 500 <= loc <= 2000:
            score += weights["in_loc_range"]
        elif 300 <= loc < 500 or 2000 < loc <= 3000:
            score += weights["in_loc_range"] * 0.5
        
        # Has documentation comments (negative - already documented)
        if self.analysis["has_comments"]:
            score += weights["has_comments"]
        
        # Complexity
        complexity = self.analysis["complexity"]["cyclomatic"]
        score += weights["complexity"] * min(complexity / 20, 1.0)
        
        return score
    
    def _count_exec_cics(self) -> int:
        """Count EXEC CICS statements (v2.1 multi-factor scoring)"""
        if not self.content:
            return 0
        return len(re.findall(r'EXEC\s+CICS', self.content, re.IGNORECASE))

    def _count_exec_sql(self) -> int:
        """Count EXEC SQL statements (v2.1 multi-factor scoring)"""
        if not self.content:
            return 0
        return len(re.findall(r'EXEC\s+SQL', self.content, re.IGNORECASE))

    def _count_goto(self) -> int:
        """Count GO TO statements (v2.1 multi-factor scoring)"""
        if not self.content:
            return 0
        return len(re.findall(r'\bGO\s+TO\b', self.content, re.IGNORECASE))

    def _empty_analysis(self) -> Dict:
        """Return empty analysis for failed reads"""
        return {
            "file_path": str(self.file_path),
            "file_name": self.file_path.name,
            "file_size": 0,
            "loc": 0,
            "complexity": {"cyclomatic": 0, "nesting_depth": 0, "branches": 0},
            "dependencies": {"calls": [], "copies": [], "total": 0},
            "business_rules": 0,
            "file_operations": {"operations": [], "files": [], "total": 0},
            "has_comments": False,
            "domain_keywords": {},
            "exec_cics_count": 0,
            "exec_sql_count": 0,
            "goto_count": 0,
        }

