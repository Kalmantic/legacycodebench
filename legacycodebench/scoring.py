"""Scoring system for LegacyCodeBench"""

from typing import Dict, List
from pathlib import Path
import json
import logging

from legacycodebench.config import OVERALL_WEIGHTS, PERFORMANCE_TIERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScoringSystem:
    """Calculate overall scores from evaluation results"""
    
    def __init__(self):
        self.weights = OVERALL_WEIGHTS
        self.tiers = PERFORMANCE_TIERS
    
    def calculate_overall_score(self, doc_score: float, und_score: float) -> float:
        """Calculate overall LegacyCodeBench score"""
        return (
            self.weights["documentation"] * doc_score +
            self.weights["understanding"] * und_score
        )
    
    def get_performance_tier(self, score: float) -> str:
        """Get performance tier for a score"""
        if score >= self.tiers["excellent"]:
            return "excellent"
        elif score >= self.tiers["good"]:
            return "good"
        elif score >= self.tiers["fair"]:
            return "fair"
        else:
            return "poor"
    
    def aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results across multiple tasks"""
        doc_scores = []
        und_scores = []
        overall_scores = []
        
        for result in results:
            if result.get("category") == "documentation":
                doc_scores.append(result.get("score", 0.0))
            elif result.get("category") == "understanding":
                und_scores.append(result.get("score", 0.0))
            
            overall_scores.append(result.get("overall_score", 0.0))
        
        doc_avg = sum(doc_scores) / len(doc_scores) if doc_scores else 0.0
        und_avg = sum(und_scores) / len(und_scores) if und_scores else 0.0
        overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        
        return {
            "documentation_avg": round(doc_avg, 4),
            "understanding_avg": round(und_avg, 4),
            "overall_avg": round(overall_avg, 4),
            "documentation_tasks": len(doc_scores),
            "understanding_tasks": len(und_scores),
            "total_tasks": len(results),
        }

