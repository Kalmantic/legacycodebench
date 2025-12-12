"""
Leaderboard generation for LegacyCodeBench v2.0

LCB Score-Based Evaluation:
- LCB Score: Weighted composite score (0-100)
  - 35% Behavioral Fidelity (BF)
  - 30% Structural Completeness (SC)
  - 25% Semantic Quality (SQ)
  - 10% Traceability (TR)
- Tier breakdown (T1-T4) for detailed analysis
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime
import logging

from legacycodebench.config import (
    RESULTS_DIR,
    EVALUATION_WEIGHTS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Leaderboard:
    """
    Generate and manage leaderboard with LCB Score-based evaluation.

    LCB Score = (0.35 × BF) + (0.30 × SC) + (0.25 × SQ) + (0.10 × TR)
    
    Displayed in web UI:
    - LCB Score (0-100)
    - Component scores: SC, BF, SQ, TR
    - Tier breakdown: T1 (Basic), T4 (Enterprise)
    """

    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, output_path: Path = None) -> Dict:
        """
        Generate leaderboard with LCB Score-based evaluation.

        Scoring: LCB Score = (0.35 × BF) + (0.30 × SC) + (0.25 × SQ) + (0.10 × TR)
        
        This aligns with the web UI which displays:
        - LCB Score (0-100)
        - Component scores: SC, BF, SQ, TR
        - Tier breakdown: T1, T4

        Args:
            output_path: Path to save leaderboard JSON

        Returns:
            dict: Leaderboard data with LCB Score-based metrics
        """
        if output_path is None:
            output_path = self.results_dir / "leaderboard.json"
        
        # Load all results
        all_results = []
        for result_file in self.results_dir.glob("*.json"):
            if result_file.name in ["leaderboard.json", "summary.csv"]:
                continue
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to load {result_file}: {e}")
        
        # Group by model
        submissions = defaultdict(lambda: {
            "results": [],
            "tasks_total": 0,
            # Scores for average calculation
            "lcb_scores": [],
            "sc_scores": [],
            "bf_scores": [],
            "sq_scores": [],
            "tr_scores": [],
            # Critical failures
            "critical_failures_count": 0,
            "critical_failure_types": defaultdict(int),
            # Tier breakdown
            "by_tier": {
                "T1": {"total": 0, "scores": []},
                "T2": {"total": 0, "scores": []},
                "T3": {"total": 0, "scores": []},
                "T4": {"total": 0, "scores": []},
            },
        })
        
        for result in all_results:
            submitter = result.get("submitter", {}).get("name", "Unknown")
            model = result.get("submitter", {}).get("model", "unknown")
            
            # Normalize: Use model as primary identifier
            if not model or model == "unknown":
                model = submitter if submitter != "Unknown" else "unknown"
            
            key = model
            sub = submissions[key]
            sub["results"].append(result)
            sub["tasks_total"] += 1
            
            task_result = result.get("result", {})
            task_id = result.get("task_id", "")
            overall_score = result.get("overall_score", 0.0)
            
            # Extract scores
            sc = task_result.get("structural_completeness", 0.0)
            bf = task_result.get("behavioral_fidelity", 0.0)
            sq = task_result.get("semantic_quality", 0.0)
            tr = task_result.get("traceability", 0.0)
            critical_failures = task_result.get("critical_failures", [])
            
            # Filter out CF-05 from old results if it wouldn't trigger with new logic
            tr_details = task_result.get("details", {}).get("traceability", {})
            total_refs = tr_details.get("total_references", 0)
            
            had_cf05 = any(cf.startswith("CF-05") for cf in critical_failures)
            
            # Remove CF-05 if it's in the list but wouldn't trigger with new logic
            if total_refs == 0:
                critical_failures = [cf for cf in critical_failures if not cf.startswith("CF-05")]
            elif tr >= 0.20:
                critical_failures = [cf for cf in critical_failures if not cf.startswith("CF-05")]
            
            # If we removed CF-05 and score is 0.0, recalculate from components
            if had_cf05 and len(critical_failures) == 0 and overall_score == 0.0:
                overall_score = (
                    EVALUATION_WEIGHTS["structural_completeness"] * sc +
                    EVALUATION_WEIGHTS["behavioral_fidelity"] * bf +
                    EVALUATION_WEIGHTS["semantic_quality"] * sq +
                    EVALUATION_WEIGHTS["traceability"] * tr
                )
            
            # Track scores
            sub["lcb_scores"].append(overall_score)
            sub["sc_scores"].append(sc)
            sub["bf_scores"].append(bf)
            sub["sq_scores"].append(sq)
            sub["tr_scores"].append(tr)
            
            # Track critical failures
            if critical_failures:
                sub["critical_failures_count"] += 1
                for cf in critical_failures:
                    cf_type = cf.split(":")[0] if ":" in cf else cf
                    sub["critical_failure_types"][cf_type] += 1
            
            # Track by tier
            for tier in ["T1", "T2", "T3", "T4"]:
                if f"-{tier}-" in task_id:
                    sub["by_tier"][tier]["total"] += 1
                    sub["by_tier"][tier]["scores"].append(overall_score)
                    break
        
        # Build leaderboard entries
        leaderboard = []
        for model, data in submissions.items():
            # Get most common submitter name for this model
            submitter_names = [r.get("submitter", {}).get("name", "Unknown") 
                              for r in data["results"]]
            from collections import Counter
            submitter = Counter(submitter_names).most_common(1)[0][0] if submitter_names else "Unknown"
            
            total = data["tasks_total"]
            
            # Calculate averages
            def avg(lst): return sum(lst) / len(lst) if lst else 0.0

            # Calculate tier stats (avg_score for each tier)
            tier_stats = {}
            for tier, tier_data in data["by_tier"].items():
                tier_stats[tier] = {
                    "total": tier_data["total"],
                    "avg_score": avg(tier_data["scores"]),
                }

            leaderboard.append({
                "submitter": submitter,
                "model": model,
                "rank": 0,  # Will be set after sorting
                # Primary metric: LCB Score
                "avg_lcb_score": round(avg(data["lcb_scores"]), 4),
                # Component scores (as shown in UI)
                "avg_sc": round(avg(data["sc_scores"]), 4),
                "avg_bf": round(avg(data["bf_scores"]), 4),
                "avg_sq": round(avg(data["sq_scores"]), 4),
                "avg_tr": round(avg(data["tr_scores"]), 4),
                # Task count
                "tasks_total": total,
                # Critical failures
                "critical_failures": data["critical_failures_count"],
                "critical_failure_types": dict(data["critical_failure_types"]),
                # Tier breakdown (as shown in UI: T1 and T4)
                "by_tier": tier_stats,
            })
        
        # Sort by LCB Score (primary metric) - matches web UI display
        leaderboard.sort(key=lambda x: x["avg_lcb_score"], reverse=True)
        
        # Add rank
        for i, entry in enumerate(leaderboard, 1):
            entry["rank"] = i
        
        # Create leaderboard structure
        leaderboard_data = {
            "version": "2.0",
            "generated_at": datetime.now().isoformat(),
            "scoring_formula": "LCB Score = (0.35 × BF) + (0.30 × SC) + (0.25 × SQ) + (0.10 × TR)",
            "scoring_weights": EVALUATION_WEIGHTS,
            "total_models": len(leaderboard),
            "leaderboard": leaderboard,
        }
        
        # Save JSON
        with open(output_path, 'w') as f:
            json.dump(leaderboard_data, f, indent=2)
        
        # Also save CSV
        self._save_csv(leaderboard, output_path.parent / "summary.csv")
        
        logger.info(f"Generated leaderboard with {len(leaderboard)} entries (LCB Score-based)")
        return leaderboard_data
    
    def _save_csv(self, leaderboard: List[Dict], csv_path: Path):
        """Save leaderboard summary as CSV"""
        import csv
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header - matches web UI columns
            writer.writerow([
                "rank", "model",
                "lcb_score", "sc", "bf", "sq", "tr",
                "t1_score", "t4_score",
                "total_tasks", "critical_failures"
            ])
            # Data
            for entry in leaderboard:
                t1_score = entry["by_tier"]["T1"]["avg_score"] if entry["by_tier"]["T1"]["total"] > 0 else 0
                t4_score = entry["by_tier"]["T4"]["avg_score"] if entry["by_tier"]["T4"]["total"] > 0 else 0
                
                writer.writerow([
                    entry["rank"],
                    f"{entry['submitter']}/{entry['model']}",
                    f"{entry['avg_lcb_score']*100:.1f}",
                    f"{entry['avg_sc']*100:.1f}",
                    f"{entry['avg_bf']*100:.1f}",
                    f"{entry['avg_sq']*100:.1f}",
                    f"{entry['avg_tr']*100:.1f}",
                    f"{t1_score*100:.1f}",
                    f"{t4_score*100:.1f}",
                    entry["tasks_total"],
                    entry["critical_failures"],
                ])
        
        logger.info(f"Saved CSV summary to {csv_path}")
    
    def print_leaderboard(self, leaderboard_data: Dict = None):
        """Print formatted leaderboard with LCB Score-based metrics (matches web UI)"""
        if leaderboard_data is None:
            leaderboard_data = self.generate()

        scoring_formula = leaderboard_data.get("scoring_formula", "LCB Score = (0.35 × BF) + (0.30 × SC) + (0.25 × SQ) + (0.10 × TR)")

        # Header
        print("\n" + "=" * 110)
        print("                         LegacyCodeBench v2.0 Leaderboard")
        print("=" * 110)
        print(f"Scoring: {scoring_formula}")
        print("=" * 110)
        print()

        # Main table header - matches web UI columns
        print(f"{'#':<4}{'Model':<25}{'LCB Score':<12}{'SC':<8}{'BF':<8}{'SQ':<8}{'TR':<8}{'T1':<8}{'T4':<8}")
        print("-" * 110)

        # Main table
        for entry in leaderboard_data["leaderboard"]:
            rank = entry["rank"]
            # Display model name
            model_display = entry['model']
            if entry['submitter'] != entry['model'] and entry['submitter'] != "Unknown":
                model_display = f"{entry['submitter']}/{entry['model']}"
            model_display = model_display[:24]

            lcb = f"{entry['avg_lcb_score']*100:.0f}"
            sc = f"{entry['avg_sc']*100:.0f}%"
            bf = f"{entry['avg_bf']*100:.0f}%"
            sq = f"{entry['avg_sq']*100:.0f}%"
            tr = f"{entry['avg_tr']*100:.0f}%"
            
            # Tier scores
            t1 = entry["by_tier"]["T1"]["avg_score"]
            t4 = entry["by_tier"]["T4"]["avg_score"]
            t1_str = f"{t1*100:.0f}" if entry["by_tier"]["T1"]["total"] > 0 else "-"
            t4_str = f"{t4*100:.0f}" if entry["by_tier"]["T4"]["total"] > 0 else "-"

            print(f"{rank:<4}{model_display:<25}{lcb:<12}{sc:<8}{bf:<8}{sq:<8}{tr:<8}{t1_str:<8}{t4_str:<8}")

        print("-" * 110)
        print()
        print("Scoring: LCB Score = (0.35 × BF) + (0.30 × SC) + (0.25 × SQ) + (0.10 × TR)")
        print("SC=Structural Completeness, BF=Behavioral Fidelity, SQ=Semantic Quality, TR=Traceability")
        print("T1=Basic Tier Score, T4=Enterprise Tier Score")
        print()
        print("=" * 110 + "\n")
    
    def print_detailed(self, leaderboard_data: Dict = None):
        """Print detailed leaderboard with component scores and tier breakdown"""
        if leaderboard_data is None:
            leaderboard_data = self.generate()
        
        print("\n" + "=" * 120)
        print("                         LegacyCodeBench v2.0 Detailed Results")
        print("=" * 120)
        
        # Detailed table - all tiers
        print(f"{'#':<4} {'Model':<22} {'LCB':<8} {'SC':<8} {'BF':<8} {'SQ':<8} {'TR':<8} {'T1':<8} {'T2':<8} {'T3':<8} {'T4':<8} {'CF':<4}")
        print("-" * 120)
        
        for entry in leaderboard_data["leaderboard"]:
            rank = entry["rank"]
            model_display = entry['model']
            if entry['submitter'] != entry['model'] and entry['submitter'] != "Unknown":
                model_display = f"{entry['submitter']}/{entry['model']}"
            model_display = model_display[:21]
            
            lcb = f"{entry['avg_lcb_score']*100:.0f}"
            sc = f"{entry['avg_sc']*100:.0f}%"
            bf = f"{entry['avg_bf']*100:.0f}%"
            sq = f"{entry['avg_sq']*100:.0f}%"
            tr = f"{entry['avg_tr']*100:.0f}%"
            cf = entry["critical_failures"]
            
            # All tier scores
            tier_strs = []
            for tier in ["T1", "T2", "T3", "T4"]:
                tier_data = entry["by_tier"][tier]
                if tier_data["total"] > 0:
                    tier_strs.append(f"{tier_data['avg_score']*100:.0f}")
                else:
                    tier_strs.append("-")
            
            print(f"{rank:<4} {model_display:<22} {lcb:<8} {sc:<8} {bf:<8} {sq:<8} {tr:<8} {tier_strs[0]:<8} {tier_strs[1]:<8} {tier_strs[2]:<8} {tier_strs[3]:<8} {cf:<4}")
        
        print("-" * 120)
        print("LCB=LCB Score, SC=Structural, BF=Behavioral, SQ=Semantic, TR=Traceability, CF=Critical Failures")
        print("T1=Basic, T2=Moderate, T3=Complex, T4=Enterprise")
        print("=" * 120 + "\n")
    
    def export_markdown(self, output_path: Path = None, leaderboard_data: Dict = None) -> str:
        """Export leaderboard as Markdown (matches web UI format)"""
        if leaderboard_data is None:
            leaderboard_data = self.generate()
        
        if output_path is None:
            output_path = self.results_dir / "LEADERBOARD.md"
        
        md = []
        md.append("# LegacyCodeBench v2.0 Leaderboard")
        md.append("")
        md.append(f"*Generated: {leaderboard_data.get('generated_at', 'N/A')}*")
        md.append("")
        md.append("## Scoring Formula")
        md.append("")
        md.append("**LCB Score = (0.35 × BF) + (0.30 × SC) + (0.25 × SQ) + (0.10 × TR)**")
        md.append("")
        md.append("- BF = Behavioral Fidelity (35%)")
        md.append("- SC = Structural Completeness (30%)")
        md.append("- SQ = Semantic Quality (25%)")
        md.append("- TR = Traceability (10%)")
        md.append("")
        md.append("## Leaderboard")
        md.append("")
        md.append("| # | Model | LCB Score | SC | BF | SQ | TR | T1 | T4 |")
        md.append("|---|-------|-----------|----|----|----|----|----|----|")
        
        for entry in leaderboard_data["leaderboard"]:
            model = entry['model']
            if entry['submitter'] != entry['model'] and entry['submitter'] != "Unknown":
                model = f"{entry['submitter']}/{entry['model']}"
            
            t1 = entry["by_tier"]["T1"]
            t4 = entry["by_tier"]["T4"]
            t1_str = f"{t1['avg_score']*100:.0f}" if t1["total"] > 0 else "-"
            t4_str = f"{t4['avg_score']*100:.0f}" if t4["total"] > 0 else "-"
            
            md.append(f"| {entry['rank']} | {model} | "
                     f"{entry['avg_lcb_score']*100:.0f} | "
                     f"{entry['avg_sc']*100:.0f}% | "
                     f"{entry['avg_bf']*100:.0f}% | "
                     f"{entry['avg_sq']*100:.0f}% | "
                     f"{entry['avg_tr']*100:.0f}% | "
                     f"{t1_str} | {t4_str} |")
        
        md.append("")
        md.append("## Tier Breakdown")
        md.append("")
        md.append("Tiers represent task complexity:")
        md.append("- **T1 (Basic)**: Simple, linear programs (LOC 300-500)")
        md.append("- **T2 (Moderate)**: PERFORM loops, file operations (LOC 500-1000)")
        md.append("- **T3 (Complex)**: External calls, business rules (LOC 1000-2000)")
        md.append("- **T4 (Enterprise)**: GO TO spaghetti, CICS/DB2 (LOC 2000+)")
        md.append("")
        
        for entry in leaderboard_data["leaderboard"]:
            model = entry['model']
            if entry['submitter'] != entry['model'] and entry['submitter'] != "Unknown":
                model = f"{entry['submitter']}/{entry['model']}"
            md.append(f"### {model}")
            md.append("")
            md.append("| Tier | Difficulty | Avg Score | Tasks |")
            md.append("|------|------------|-----------|-------|")
            
            tier_names = {"T1": "Basic", "T2": "Moderate", "T3": "Complex", "T4": "Enterprise"}
            for tier in ["T1", "T2", "T3", "T4"]:
                tier_data = entry["by_tier"][tier]
                if tier_data["total"] > 0:
                    md.append(f"| {tier} | {tier_names[tier]} | "
                             f"{tier_data['avg_score']*100:.0f} | "
                             f"{tier_data['total']} |")
            md.append("")
        
        content = "\n".join(md)
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Exported Markdown leaderboard to {output_path}")
        return content


# Convenience function
def generate_leaderboard(results_dir: Path = RESULTS_DIR) -> Dict:
    """Generate and print leaderboard"""
    lb = Leaderboard(results_dir)
    data = lb.generate()
    lb.print_leaderboard(data)
    return data
