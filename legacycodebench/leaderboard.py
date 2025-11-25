"""Leaderboard generation for LegacyCodeBench"""

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import logging

from legacycodebench.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Leaderboard:
    """Generate and manage leaderboard"""
    
    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, output_path: Path = None) -> Dict:
        """Generate leaderboard from all results"""
        if output_path is None:
            output_path = self.results_dir / "leaderboard.json"
        
        # Load all results
        all_results = []
        for result_file in self.results_dir.glob("*.json"):
            if result_file.name == "leaderboard.json":
                continue
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to load {result_file}: {e}")
        
        # Group by submitter
        submissions = defaultdict(lambda: {
            "results": [],
            "doc_scores": [],
            "und_scores": [],
            "overall_scores": [],
        })
        
        for result in all_results:
            submitter = result.get("submitter", {}).get("name", "Unknown")
            model = result.get("submitter", {}).get("model", "unknown")
            category = result.get("submitter", {}).get("category", "unknown")
            
            key = f"{submitter}::{model}::{category}"
            submissions[key]["results"].append(result)
            
            task_result = result.get("result", {})
            if result.get("task_category") == "documentation":
                doc_score = task_result.get("score", 0.0)
                submissions[key]["doc_scores"].append(doc_score)
            elif result.get("task_category") == "understanding":
                und_score = task_result.get("score", 0.0)
                submissions[key]["und_scores"].append(und_score)
            
            overall = result.get("overall_score", 0.0)
            submissions[key]["overall_scores"].append(overall)
        
        # Calculate averages and create leaderboard entries
        leaderboard = []
        for key, data in submissions.items():
            submitter, model, category = key.split("::")
            
            doc_avg = sum(data["doc_scores"]) / len(data["doc_scores"]) if data["doc_scores"] else 0.0
            und_avg = sum(data["und_scores"]) / len(data["und_scores"]) if data["und_scores"] else 0.0
            overall_avg = sum(data["overall_scores"]) / len(data["overall_scores"]) if data["overall_scores"] else 0.0
            
            leaderboard.append({
                "submitter": submitter,
                "model": model,
                "category": category,
                "overall_score": round(overall_avg, 4),
                "documentation_score": round(doc_avg, 4),
                "understanding_score": round(und_avg, 4),
                "tasks_completed": len(data["results"]),
            })
        
        # Sort by overall score (descending)
        leaderboard.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Add rank
        for i, entry in enumerate(leaderboard, 1):
            entry["rank"] = i
        
        # Create leaderboard structure
        leaderboard_data = {
            "version": "1.0",
            "total_submissions": len(leaderboard),
            "leaderboard": leaderboard,
        }
        
        # Save
        with open(output_path, 'w') as f:
            json.dump(leaderboard_data, f, indent=2)
        
        logger.info(f"Generated leaderboard with {len(leaderboard)} entries")
        return leaderboard_data
    
    def print_leaderboard(self, leaderboard_data: Dict = None):
        """Print formatted leaderboard to console"""
        if leaderboard_data is None:
            leaderboard_data = self.generate()
        
        print("\n" + "=" * 80)
        print("LegacyCodeBench v1.0 Leaderboard")
        print("=" * 80)
        print(f"{'Rank':<6} {'System':<25} {'Category':<12} {'Overall':<10} {'Doc':<10} {'Und':<10} {'Tasks':<8}")
        print("-" * 80)
        
        for entry in leaderboard_data["leaderboard"]:
            rank = entry["rank"]
            system = f"{entry['submitter']} {entry['model']}"[:24]
            category = entry["category"][:11]
            overall = f"{entry['overall_score']*100:.1f}%"
            doc = f"{entry['documentation_score']*100:.1f}%"
            und = f"{entry['understanding_score']*100:.1f}%"
            tasks = entry["tasks_completed"]
            
            print(f"{rank:<6} {system:<25} {category:<12} {overall:<10} {doc:<10} {und:<10} {tasks:<8}")
        
        print("=" * 80 + "\n")

