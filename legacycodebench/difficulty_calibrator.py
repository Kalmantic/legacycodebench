"""Difficulty calibration for benchmark tasks"""

from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DifficultyCalibrator:
    """Calculate and assign difficulty levels to benchmark tasks"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        """Default difficulty thresholds"""
        return {
            "thresholds": {
                "easy": 30,
                "medium": 60,
                "hard": 100,
            },
            "weights": {
                "loc": 0.3,
                "complexity": 0.3,
                "dependencies": 0.2,
                "business_rules": 0.2,
            }
        }
    
    def calculate_difficulty_score(self, file_analysis: Dict, 
                                   task_category: str = "documentation") -> float:
        """Calculate difficulty score (0-100+)"""
        
        score = 0.0
        weights = self.config["weights"]
        
        # LOC contribution (0-30 points)
        loc = file_analysis.get("loc", 0)
        loc_score = self._score_loc(loc)
        score += loc_score * weights["loc"] * 100
        
        # Complexity contribution (0-30 points)
        complexity = file_analysis.get("complexity", {})
        complexity_score = self._score_complexity(complexity)
        score += complexity_score * weights["complexity"] * 100
        
        # Dependencies contribution (0-20 points)
        dependencies = file_analysis.get("dependencies", {})
        dep_score = self._score_dependencies(dependencies)
        score += dep_score * weights["dependencies"] * 100
        
        # Business rules contribution (0-20 points)
        rules = file_analysis.get("business_rules", 0)
        rules_score = self._score_business_rules(rules)
        score += rules_score * weights["business_rules"] * 100
        
        # Adjust for task category
        if task_category == "understanding":
            # Understanding tasks emphasize dependencies more
            dep_bonus = min(dependencies.get("total", 0) / 10, 1.0) * 10
            score += dep_bonus
        elif task_category == "documentation":
            # Documentation tasks emphasize business logic more
            rules_bonus = min(rules / 15, 1.0) * 10
            score += rules_bonus
        
        return round(score, 2)
    
    def assign_difficulty_level(self, difficulty_score: float) -> str:
        """Assign difficulty level based on score"""
        thresholds = self.config["thresholds"]
        
        if difficulty_score < thresholds["easy"]:
            return "easy"
        elif difficulty_score < thresholds["medium"]:
            return "medium"
        else:
            return "hard"
    
    def _score_loc(self, loc: int) -> float:
        """Score based on lines of code (0.0 to 1.0)"""
        if loc < 300:
            return 0.0
        elif loc < 500:
            return 0.2
        elif loc < 800:
            return 0.4
        elif loc < 1200:
            return 0.6
        elif loc < 1800:
            return 0.8
        else:
            return 1.0
    
    def _score_complexity(self, complexity: Dict) -> float:
        """Score based on complexity metrics (0.0 to 1.0)"""
        cyclomatic = complexity.get("cyclomatic", 0)
        nesting = complexity.get("nesting_depth", 0)
        
        # Cyclomatic complexity scoring
        if cyclomatic < 5:
            cyc_score = 0.0
        elif cyclomatic < 10:
            cyc_score = 0.3
        elif cyclomatic < 20:
            cyc_score = 0.6
        elif cyclomatic < 40:
            cyc_score = 0.8
        else:
            cyc_score = 1.0
        
        # Nesting depth scoring
        if nesting < 2:
            nest_score = 0.0
        elif nesting < 4:
            nest_score = 0.5
        else:
            nest_score = 1.0
        
        # Combined (weighted average)
        return 0.7 * cyc_score + 0.3 * nest_score
    
    def _score_dependencies(self, dependencies: Dict) -> float:
        """Score based on dependencies (0.0 to 1.0)"""
        total = dependencies.get("total", 0)
        
        if total == 0:
            return 0.0
        elif total < 3:
            return 0.3
        elif total < 6:
            return 0.6
        elif total < 10:
            return 0.8
        else:
            return 1.0
    
    def _score_business_rules(self, rule_count: int) -> float:
        """Score based on business rules (0.0 to 1.0)"""
        if rule_count < 3:
            return 0.0
        elif rule_count < 8:
            return 0.4
        elif rule_count < 15:
            return 0.7
        else:
            return 1.0
    
    def explain_difficulty(self, file_analysis: Dict, task_category: str = "documentation") -> Dict:
        """Provide detailed difficulty explanation"""
        score = self.calculate_difficulty_score(file_analysis, task_category)
        level = self.assign_difficulty_level(score)
        
        return {
            "level": level,
            "score": score,
            "factors": {
                "loc": file_analysis.get("loc", 0),
                "cyclomatic_complexity": file_analysis.get("complexity", {}).get("cyclomatic", 0),
                "nesting_depth": file_analysis.get("complexity", {}).get("nesting_depth", 0),
                "dependencies": file_analysis.get("dependencies", {}).get("total", 0),
                "business_rules": file_analysis.get("business_rules", 0),
            },
            "explanation": self._generate_explanation(score, level, file_analysis)
        }
    
    def _generate_explanation(self, score: float, level: str, analysis: Dict) -> str:
        """Generate human-readable explanation"""
        loc = analysis.get("loc", 0)
        complexity = analysis.get("complexity", {}).get("cyclomatic", 0)
        deps = analysis.get("dependencies", {}).get("total", 0)
        rules = analysis.get("business_rules", 0)
        
        if level == "easy":
            return f"Simple program ({loc} LOC) with basic logic ({complexity} complexity), few dependencies ({deps}), and straightforward business rules ({rules})."
        elif level == "medium":
            return f"Moderate program ({loc} LOC) with some complexity ({complexity}), several dependencies ({deps}), and multiple business rules ({rules})."
        else:
            return f"Complex program ({loc} LOC) with high complexity ({complexity}), many dependencies ({deps}), and intricate business logic ({rules})."

