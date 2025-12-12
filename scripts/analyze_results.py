#!/usr/bin/env python
"""Analyze benchmark results and generate detailed reports"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

def load_results(results_dir: Path) -> List[Dict]:
    """Load all result JSON files"""
    results = []
    for result_file in results_dir.glob("*.json"):
        if result_file.name == "leaderboard.json":
            continue
        with open(result_file, 'r') as f:
            results.append(json.load(f))
    return results

def analyze_by_model(results: List[Dict]) -> Dict:
    """Analyze results grouped by model"""
    by_model = defaultdict(list)

    for result in results:
        model = result["submitter"]["model"]
        by_model[model].append(result)

    analysis = {}
    for model, model_results in by_model.items():
        scores = [r["overall_score"] for r in model_results]
        sc_scores = [r["result"]["structural_completeness"] for r in model_results]
        bf_scores = [r["result"]["behavioral_fidelity"] for r in model_results]
        sq_scores = [r["result"]["semantic_quality"] for r in model_results]
        tr_scores = [r["result"]["traceability"] for r in model_results]

        critical_failures = sum(1 for r in model_results if r["result"].get("has_critical_failures", False))

        analysis[model] = {
            "total_tasks": len(model_results),
            "avg_overall": sum(scores) / len(scores),
            "avg_sc": sum(sc_scores) / len(sc_scores),
            "avg_bf": sum(bf_scores) / len(bf_scores),
            "avg_sq": sum(sq_scores) / len(sq_scores),
            "avg_tr": sum(tr_scores) / len(tr_scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "critical_failures": critical_failures,
            "critical_failure_rate": critical_failures / len(model_results)
        }

    return analysis

def analyze_by_difficulty(results: List[Dict]) -> Dict:
    """Analyze results grouped by task difficulty"""
    # Note: difficulty not in current result structure, would need to join with tasks
    # Placeholder for now
    return {}

def print_analysis(analysis: Dict):
    """Print detailed analysis"""
    print("=" * 80)
    print("DETAILED RESULTS ANALYSIS")
    print("=" * 80)
    print()

    for model, stats in sorted(analysis.items(), key=lambda x: x[1]["avg_overall"], reverse=True):
        print(f"Model: {model}")
        print("-" * 80)
        print(f"  Total Tasks:           {stats['total_tasks']}")
        print(f"  Average Overall Score: {stats['avg_overall']:.2%}")
        print(f"  Score Range:           {stats['min_score']:.2%} - {stats['max_score']:.2%}")
        print()
        print("  Component Scores:")
        print(f"    Structural Completeness: {stats['avg_sc']:.2%}")
        print(f"    Behavioral Fidelity:     {stats['avg_bf']:.2%}")
        print(f"    Semantic Quality:        {stats['avg_sq']:.2%}")
        print(f"    Traceability:            {stats['avg_tr']:.2%}")
        print()
        print(f"  Critical Failures:     {stats['critical_failures']} ({stats['critical_failure_rate']:.1%})")
        print()
        print("=" * 80)
        print()

def analyze_bf_details(results: List[Dict]):
    """Analyze Behavioral Fidelity in detail"""
    print("=" * 80)
    print("BEHAVIORAL FIDELITY ANALYSIS")
    print("=" * 80)
    print()

    bf_results = defaultdict(list)

    for result in results:
        model = result["submitter"]["model"]
        bf_detail = result["result"].get("details", {}).get("behavioral_fidelity", {})

        if not bf_detail.get("placeholder", True):
            bf_results[model].append({
                "score": result["result"]["behavioral_fidelity"],
                "tests_passed": bf_detail.get("tests_passed", 0),
                "tests_failed": bf_detail.get("tests_failed", 0),
                "total_tests": bf_detail.get("total_tests", 0),
                "gap_percentage": bf_detail.get("gap_percentage", 0)
            })

    for model, bf_list in bf_results.items():
        print(f"Model: {model}")
        print("-" * 80)
        print(f"  Tasks with BF evaluation: {len(bf_list)}")

        if bf_list:
            avg_score = sum(b["score"] for b in bf_list) / len(bf_list)
            avg_pass_rate = sum(b["tests_passed"] / b["total_tests"] for b in bf_list if b["total_tests"] > 0) / len(bf_list)
            avg_gaps = sum(b["gap_percentage"] for b in bf_list) / len(bf_list)

            print(f"  Average BF Score:     {avg_score:.2%}")
            print(f"  Average Pass Rate:    {avg_pass_rate:.2%}")
            print(f"  Average Gap %:        {avg_gaps:.2%}")

            # Count perfect scores
            perfect = sum(1 for b in bf_list if b["score"] >= 0.95)
            print(f"  Perfect Scores (>95%): {perfect} ({perfect/len(bf_list):.1%})")

        print()

def main():
    results_dir = Path("results")

    if not results_dir.exists():
        print("Error: results/ directory not found")
        print("Run benchmark first: legacycodebench run-ai --model gpt-4o")
        sys.exit(1)

    print("Loading results...")
    results = load_results(results_dir)

    if not results:
        print("No results found in results/ directory")
        sys.exit(1)

    print(f"Loaded {len(results)} results\n")

    # Overall analysis
    analysis = analyze_by_model(results)
    print_analysis(analysis)

    # BF-specific analysis
    analyze_bf_details(results)

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
