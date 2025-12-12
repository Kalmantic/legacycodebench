"""CLI interface for LegacyCodeBench"""

import click
import json
from pathlib import Path
from typing import Optional, List
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from legacycodebench.config import (
    TASKS_DIR, DATASETS_DIR, SUBMISSIONS_DIR, RESULTS_DIR,
    get_config
)
from legacycodebench.tasks import Task, TaskManager
from legacycodebench.dataset_loader import DatasetLoader

# v2.0 evaluators (default)
from legacycodebench.evaluators_v2.documentation_v2 import DocumentationEvaluatorV2

# v1.0 evaluators (archived, for backward compatibility)
try:
    from legacycodebench.evaluators_v1.documentation import DocumentationEvaluator
    from legacycodebench.evaluators_v1.understanding import UnderstandingEvaluator
    V1_AVAILABLE = True
except ImportError as e:
    V1_AVAILABLE = False
    logger.warning(f"v1.0 evaluators not available: {e}")

from legacycodebench.scoring import ScoringSystem
from legacycodebench.leaderboard import Leaderboard
from legacycodebench.ai_integration import get_ai_model


@click.group()
@click.version_option(version="2.0.0")
def main():
    """LegacyCodeBench - Benchmark for AI systems on legacy code understanding

    v2.0: Ground truth-based evaluation with behavioral fidelity testing
    """
    pass


@main.command()
def load_datasets():
    """Load COBOL datasets from GitHub repositories"""
    click.echo("Loading datasets from GitHub repositories...")
    loader = DatasetLoader()
    loaded = loader.load_all_datasets()
    
    click.echo(f"\n✓ Loaded {len(loaded)} datasets:")
    for source_id, path in loaded.items():
        stats = loader.get_dataset_stats(path)
        click.echo(f"  {source_id}: {stats['total_files']} files, {stats['total_lines']} lines")


@main.command()
def create_tasks():
    """Create tasks from loaded datasets"""
    click.echo("Creating tasks from datasets...")
    manager = TaskManager()
    tasks = manager.create_tasks_from_datasets()
    manager.save_all(tasks)
    
    click.echo(f"\n✓ Created {len(tasks)} tasks:")
    for task in tasks:
        click.echo(f"  {task.task_id}: {task.category} - {task.difficulty}")


@main.command()
@click.option("--task-id", required=True, help="Task ID (e.g., LCB-DOC-001)")
@click.option("--submission", required=True, type=click.Path(exists=True), help="Path to submission file")
@click.option("--output", type=click.Path(), help="Path to output results file")
@click.option("--submitter-name", default="Unknown", help="Submitter name")
@click.option("--submitter-model", default="unknown", help="Model name")
@click.option("--submitter-category", default="verified", help="Category (bash, verified, full, human+ai)")
@click.option("--evaluator", default="v2", type=click.Choice(["v1", "v2"]),
              help="Evaluator version (default: v2 - recommended)")
@click.option("--enable-execution", is_flag=True, default=True,
              help="Enable behavioral fidelity testing (v2 only, requires Docker)")
@click.option("--judge-model", default="gpt-4o", help="LLM judge model for semantic quality (v2 only)")
def evaluate(task_id: str, submission: Path, output: Optional[Path],
             submitter_name: str, submitter_model: str, submitter_category: str,
             evaluator: str, enable_execution: bool, judge_model: str):
    """Evaluate a submission for a task

    v2.0 (default): Ground truth-based evaluation with behavioral fidelity testing
    v1.0 (legacy): Reference-based ROUGE/BLEU evaluation
    """
    click.echo(f"Evaluating {task_id} with {evaluator.upper()} evaluator...")

    # Load task
    try:
        task = Task.load(task_id, TASKS_DIR)
    except FileNotFoundError:
        click.echo(f"✗ Task {task_id} not found. Run 'legacycodebench create-tasks' first.")
        return

    # Load submission
    submission_path = Path(submission)

    # Select evaluator based on version and category
    if evaluator == "v2":
        # v2.0 evaluator (recommended)
        if task.category == "documentation":
            click.echo(f"  Using v2.0 Documentation Evaluator")
            click.echo(f"  Behavioral Fidelity: {'Enabled' if enable_execution else 'Disabled'}")
            click.echo(f"  Judge Model: {judge_model}")

            eval_instance = DocumentationEvaluatorV2(
                enable_execution=enable_execution,
                results_dir=RESULTS_DIR / "escalations"
            )
            # Override judge model if specified
            eval_instance.sq_evaluator.judge_model_name = judge_model

            result = eval_instance.evaluate(submission_path, task)

            # v2.0 uses different score format
            overall_score = result.get("lcb_score", 0) / 100.0  # Convert to 0-1 scale

        elif task.category == "understanding":
            click.echo(f"✗ v2.0 evaluator for understanding tasks not yet implemented")
            click.echo(f"  v2.0 focuses on documentation tasks only")
            click.echo(f"  Use --evaluator v1 for understanding tasks")
            return
        else:
            click.echo(f"✗ Unknown task category: {task.category}")
            return

    elif evaluator == "v1":
        # v1.0 evaluator (legacy)
        if not V1_AVAILABLE:
            click.echo(f"✗ v1.0 evaluators not available (archived)")
            click.echo(f"  Use --evaluator v2 instead")
            return

        click.echo(f"  Using v1.0 Evaluator (LEGACY)")
        click.echo(f"  Note: v1.0 is deprecated. Consider migrating to v2.0")

        if task.category == "documentation":
            eval_instance = DocumentationEvaluator()
            result = eval_instance.evaluate(submission_path, task)
        elif task.category == "understanding":
            eval_instance = UnderstandingEvaluator()
            result = eval_instance.evaluate(submission_path, task)
        else:
            click.echo(f"✗ Unknown task category: {task.category}")
            return

        overall_score = result["score"]

    else:
        click.echo(f"✗ Unknown evaluator version: {evaluator}")
        return

    # Create result entry
    result_entry = {
        "task_id": task_id,
        "task_category": task.category,
        "evaluator_version": evaluator,
        "submitter": {
            "name": submitter_name,
            "model": submitter_model,
            "category": submitter_category,
        },
        "result": result,
        "overall_score": overall_score,
    }

    # Save result
    if output is None:
        output = RESULTS_DIR / f"{task_id}_{submitter_name}_{submitter_model}_{evaluator}.json"

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(result_entry, f, indent=2)

    # Display results
    click.echo(f"\n{'='*60}")
    click.echo(f"EVALUATION RESULTS: {task_id}")
    click.echo(f"{'='*60}")

    if evaluator == "v2":
        # SWE-bench aligned pass/fail display
        passed = result.get('passed', False)
        pass_status = result.get('pass_status', {})
        
        if passed:
            click.echo(f"  STATUS: ✅ PASSED")
        else:
            click.echo(f"  STATUS: ❌ FAILED")
            click.echo(f"  REASON: {pass_status.get('reason', 'Unknown')}")
        
        click.echo(f"")
        click.echo(f"  LCB Score: {result.get('score', 0)*100:.1f}/100")
        click.echo(f"  ├── Structural Completeness (30%): {result.get('structural_completeness', 0)*100:.1f}%")
        click.echo(f"  ├── Behavioral Fidelity (35%):     {result.get('behavioral_fidelity', 0)*100:.1f}%")
        click.echo(f"  ├── Semantic Quality (25%):        {result.get('semantic_quality', 0)*100:.1f}%")
        click.echo(f"  └── Traceability (10%):            {result.get('traceability', 0)*100:.1f}%")

        if result.get('critical_failures'):
            click.echo(f"")
            click.echo(f"  ⚠ CRITICAL FAILURES:")
            for cf in result['critical_failures']:
                click.echo(f"    - {cf}")
    else:
        click.echo(f"  Score: {overall_score*100:.2f}%")

    click.echo(f"")
    click.echo(f"  Results saved to: {output_path}")
    click.echo(f"{'='*60}")


@main.command()
@click.option("--model", required=True, help="AI model ID (e.g., claude-sonnet-4, gpt-4o)")
@click.option("--task-id", help="Specific task ID (optional, runs all if not specified)")
@click.option("--submitter-name", help="Submitter name (defaults to model name)")
def run_ai(model: str, task_id: Optional[str], submitter_name: Optional[str]):
    """Run AI model on task(s) and evaluate"""
    if submitter_name is None:
        submitter_name = model
    
    # Get AI model
    try:
        ai_model = get_ai_model(model)
    except ValueError as e:
        click.echo(f"✗ {e}")
        return
    
    # Get tasks
    manager = TaskManager()
    if task_id:
        tasks = [manager.get_task(task_id)]
    else:
        task_ids = manager.list_tasks()
        tasks = [manager.get_task(tid) for tid in task_ids]
    
    click.echo(f"Running {model} on {len(tasks)} task(s)...")
    
    # Process each task
    for task in tasks:
        click.echo(f"\nProcessing {task.task_id} ({task.category})...")
        
        # Get input files
        input_files = task.get_input_files_absolute()
        if not input_files:
            click.echo(f"  ✗ No input files found for {task.task_id}")
            continue
        
        # Generate output
        if task.category == "documentation":
            output = ai_model.generate_documentation(task, input_files)
            output_file = SUBMISSIONS_DIR / f"{task.task_id}_{model}.md"
        elif task.category == "understanding":
            output = ai_model.generate_understanding(task, input_files)
            output_file = SUBMISSIONS_DIR / f"{task.task_id}_{model}.json"
        else:
            click.echo(f"  ✗ Unknown category: {task.category}")
            continue
        
        # Save submission
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        
        click.echo(f"  ✓ Generated submission: {output_file}")
        
        # Evaluate
        if task.category == "documentation":
            evaluator = DocumentationEvaluator()
        else:
            evaluator = UnderstandingEvaluator()
        
        result = evaluator.evaluate(output_file, task)
        
        # Save result
        result_file = RESULTS_DIR / f"{task.task_id}_{submitter_name}_{model}.json"
        result_entry = {
            "task_id": task.task_id,
            "task_category": task.category,
            "submitter": {
                "name": submitter_name,
                "model": model,
                "category": "verified",
            },
            "result": result,
            "overall_score": result["score"],
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_entry, f, indent=2)
        
        click.echo(f"  ✓ Score: {result['score']*100:.2f}%")


@main.command()
@click.option("--output", type=click.Path(), help="Path to output JSON file")
@click.option("--print", "print_flag", is_flag=True, default=True, help="Print leaderboard to console")
@click.option("--detailed", is_flag=True, help="Show detailed component scores")
@click.option("--export-md", type=click.Path(), help="Export as Markdown file")
@click.option("--export-csv", type=click.Path(), help="Export as CSV file")
def leaderboard(output: Optional[Path], print_flag: bool, detailed: bool, 
                export_md: Optional[Path], export_csv: Optional[Path]):
    """Generate leaderboard from all results (SWE-bench aligned % Passed format)"""
    click.echo("Generating leaderboard (SWE-bench aligned format)...")
    
    lb = Leaderboard()
    leaderboard_data = lb.generate(output)
    
    if print_flag:
        if detailed:
            lb.print_detailed(leaderboard_data)
        else:
            lb.print_leaderboard(leaderboard_data)
    
    if export_md:
        lb.export_markdown(Path(export_md), leaderboard_data)
        click.echo(f"✓ Exported Markdown to: {export_md}")
    
    if export_csv:
        # CSV is auto-generated, just copy if different path
        import shutil
        shutil.copy(RESULTS_DIR / "summary.csv", export_csv)
        click.echo(f"✓ Exported CSV to: {export_csv}")
    
    if output:
        click.echo(f"\n✓ Leaderboard saved to: {output}")


def _run_benchmark(models_to_test: List[str], header_label: str = "LegacyCodeBench Evaluation",
                   evaluator_version: str = "v2", enable_execution: bool = False,
                   judge_model: str = "gpt-4o", task_limit: int = 3,
                   skip_datasets: bool = False, skip_task_creation: bool = False):
    """Shared routine that runs the full benchmark pipeline
    
    Args:
        models_to_test: List of AI model IDs to evaluate
        header_label: Header for console output
        evaluator_version: "v2" (default, ground-truth based) or "v1" (legacy ROUGE/BLEU)
        enable_execution: Enable behavioral fidelity testing (requires Docker)
        judge_model: LLM model for semantic quality evaluation
        task_limit: Maximum number of tasks to run (for quick testing)
        skip_datasets: Skip dataset loading step (use existing datasets)
        skip_task_creation: Skip task creation step (use existing tasks)
    """
    if not models_to_test:
        click.echo("✗ No models selected. Aborting.")
        return
    
    click.echo("=" * 80)
    click.echo(header_label)
    click.echo("=" * 80)
    click.echo(f"Evaluator: {evaluator_version.upper()} | Execution: {'Enabled' if enable_execution else 'Disabled'} | Judge: {judge_model}")
    click.echo("=" * 80)
    
    # Step 1: Load datasets
    if not skip_datasets:
        click.echo("\n[1/7] Loading datasets from GitHub repositories...")
        loader = DatasetLoader()
        loaded = loader.load_all_datasets()
        click.echo(f"✓ Loaded {len(loaded)} datasets")
    else:
        click.echo("\n[1/7] Skipping dataset loading (using existing datasets)...")
        loader = DatasetLoader()
        loaded = loader.load_all_datasets()  # Still need to get loaded dict
        click.echo(f"✓ Using {len(loaded)} existing datasets")
    
    # Step 2: Create tasks (uses v2.0 intelligent selection - documentation only, tier-based)
    if not skip_task_creation:
        click.echo("\n[2/7] Selecting tasks (v2.0 tier-based intelligent selection)...")
        manager = TaskManager()
        tasks = manager.create_tasks_from_datasets(use_intelligent_selection=True)
        manager.save_all(tasks)
        click.echo(f"✓ Created {len(tasks)} documentation tasks")
    else:
        click.echo("\n[2/7] Skipping task creation (using existing tasks)...")
        manager = TaskManager()
        task_ids = manager.list_tasks()
        tasks = [manager.get_task(tid) for tid in task_ids]
        click.echo(f"✓ Using {len(tasks)} existing tasks")
    
    # Step 3: Generate ground truth (pre-generate for all tasks to show progress)
    click.echo("\n[3/7] Generating ground truth (automated static analysis)...")
    from legacycodebench.static_analysis.ground_truth_generator import GroundTruthGenerator
    gt_generator = GroundTruthGenerator()
    gt_cache_dir = Path("cache/ground_truth")
    gt_cache_dir.mkdir(parents=True, exist_ok=True)
    
    gt_generated = 0
    gt_cached = 0
    for task in tasks:
        source_files = task.get_input_files_absolute()
        if source_files:
            main_file = source_files[0]
            # Check if cached
            cached = gt_generator.load_cached_ground_truth(main_file, gt_cache_dir)
            if cached:
                gt_cached += 1
            else:
                # Generate (will be cached)
                gt_generator.generate(source_files, cache_dir=gt_cache_dir)
                gt_generated += 1
    
    click.echo(f"✓ Ground truth: {gt_generated} generated, {gt_cached} cached (95%+ automation)")
    
    # v2.0: Select representative tasks by TIER (not by doc/understanding split)
    def _select_representative_tasks(all_tasks: List[Task], limit: int = 3) -> List[Task]:
        """Select tasks with balanced tier representation (v2.0)"""
        # Group by tier (extracted from task_id: LCB-T1-001 -> T1)
        by_tier = {"T1": [], "T2": [], "T3": [], "T4": []}
        for task in all_tasks:
            if "-T1-" in task.task_id:
                by_tier["T1"].append(task)
            elif "-T2-" in task.task_id:
                by_tier["T2"].append(task)
            elif "-T3-" in task.task_id:
                by_tier["T3"].append(task)
            elif "-T4-" in task.task_id:
                by_tier["T4"].append(task)
            else:
                # Legacy task IDs (LCB-DOC-xxx), treat as T2
                by_tier["T2"].append(task)
        
        selected: List[Task] = []
        
        # Select one from each tier if available, prioritizing T1 and T2
        for tier in ["T1", "T2", "T3", "T4"]:
            if by_tier[tier] and len(selected) < limit:
                selected.append(by_tier[tier].pop(0))
        
        # Fill remaining slots from T1 and T2 (more common)
        remaining_pool = by_tier["T1"] + by_tier["T2"] + by_tier["T3"] + by_tier["T4"]
        for task in remaining_pool:
            if len(selected) >= limit:
                break
            if task not in selected:
                selected.append(task)

        return selected[:limit]

    balanced_tasks = _select_representative_tasks(tasks, limit=task_limit)
    
    # Log tier distribution
    tier_dist = {"T1": 0, "T2": 0, "T3": 0, "T4": 0, "other": 0}
    for task in balanced_tasks:
        for tier in ["T1", "T2", "T3", "T4"]:
            if f"-{tier}-" in task.task_id:
                tier_dist[tier] += 1
                break
        else:
            tier_dist["other"] += 1
    
    click.echo(f"  Selected {len(balanced_tasks)} tasks: T1={tier_dist['T1']}, T2={tier_dist['T2']}, T3={tier_dist['T3']}, T4={tier_dist['T4']}")

    # Step 4: AI generates documentation
    click.echo("\n[4/7] AI models generating documentation...")
    
    # Ensure judge model is different from models being evaluated
    # Per spec: LLM-as-judge must be different to avoid self-evaluation bias
    def _ensure_different_judge(evaluated_models: List[str], default_judge: str) -> str:
        """Ensure judge model is different from any model being evaluated"""
        import os
        
        # If judge is in the list of models being evaluated, pick a different one
        if default_judge in evaluated_models:
            # Try to find a different model
            if any("gpt" in m for m in evaluated_models) and os.getenv("ANTHROPIC_API_KEY"):
                return "claude-sonnet-4"
            elif any("claude" in m for m in evaluated_models) and os.getenv("OPENAI_API_KEY"):
                return "gpt-4o"
            else:
                # Can't find different model, warn but continue
                logger.warning(
                    f"Judge model '{default_judge}' is same as evaluated model. "
                    f"This may cause self-evaluation bias. Consider using --judge-model to specify a different model."
                )
                return default_judge
        return default_judge
    
    actual_judge = _ensure_different_judge(models_to_test, judge_model)
    if actual_judge != judge_model:
        click.echo(f"  ⚠ Judge model changed from '{judge_model}' to '{actual_judge}' "
                   f"(must be different from evaluated models)")
        judge_model = actual_judge
    
    for model_id in models_to_test:
        click.echo(f"\n  Running {model_id}...")
        click.echo(f"    Judge for SQ evaluation: {judge_model} (different from {model_id})")
        try:
            ai_model = get_ai_model(model_id)
            
            for task in balanced_tasks:
                click.echo(f"    Task: {task.task_id} ({task.category})")
                
                input_files = task.get_input_files_absolute()
                if not input_files:
                    click.echo(f"    ⚠ No input files found, skipping")
                    continue
                
                # v2.0: All tasks are documentation tasks
                output = ai_model.generate_documentation(task, input_files)
                output_file = SUBMISSIONS_DIR / f"{task.task_id}_{model_id}.md"
                
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(output)
                click.echo(f"    ✓ Generated submission")
                
                # Evaluate using selected evaluator version
                if evaluator_version == "v2":
                    # v2.0 evaluator (ground-truth based)
                    eval_instance = DocumentationEvaluatorV2(
                        enable_execution=enable_execution,
                        results_dir=RESULTS_DIR / "escalations"
                    )
                    eval_instance.sq_evaluator.judge_model_name = judge_model
                    result = eval_instance.evaluate(output_file, task)
                    overall_score = result.get("score", 0)
                    
                    click.echo(f"    ✓ v2.0 Score: {result.get('score', 0)*100:.1f}% "
                              f"(SC:{result.get('structural_completeness', 0)*100:.0f}% "
                              f"BF:{result.get('behavioral_fidelity', 0)*100:.0f}% "
                              f"SQ:{result.get('semantic_quality', 0)*100:.0f}% "
                              f"TR:{result.get('traceability', 0)*100:.0f}%)")
                    
                    if result.get('critical_failures'):
                        click.echo(f"    ⚠ Critical failures: {', '.join(result['critical_failures'])}")
                else:
                    # v1.0 evaluator (legacy ROUGE/BLEU)
                    if not V1_AVAILABLE:
                        click.echo(f"    ✗ v1.0 evaluator not available")
                        continue
                    
                    eval_instance = DocumentationEvaluator()
                    result = eval_instance.evaluate(output_file, task)
                    overall_score = result.get("score", 0)
                    click.echo(f"    ✓ v1.0 Score: {overall_score*100:.1f}%")
                
                # Save result
                result_file = RESULTS_DIR / f"{task.task_id}_{model_id}_{evaluator_version}.json"
                # Normalize submitter name: use model name as submitter if not specified
                # This prevents duplicates in leaderboard
                submitter_name = model_id.split("-")[0].title() if "-" in model_id else model_id
                
                result_entry = {
                    "task_id": task.task_id,
                    "task_category": task.category,
                    "evaluator_version": evaluator_version,
                    "submitter": {
                        "name": submitter_name,
                        "model": model_id,  # Always use full model ID
                        "category": "verified",
                    },
                    "result": result,
                    "overall_score": overall_score,
                }
                
                with open(result_file, 'w') as f:
                    json.dump(result_entry, f, indent=2)
        
        except Exception as e:
            click.echo(f"  ✗ Error with {model_id}: {e}")
            logger.exception(f"Error running {model_id}")
    
    click.echo("\n✓ Completed AI model runs")
    
    # Step 5: Evaluation (already done during AI run, but summarize here)
    click.echo("\n[5/7] Evaluation complete (SC, BF, SQ, TR calculated)")
    
    # Step 6: Scoring (already done, but summarize)
    click.echo("\n[6/7] Scoring complete (LCB Score calculated with pass/fail status)")
    
    # Step 7: Generate leaderboard
    click.echo("\n[7/7] Generating leaderboard (SWE-bench aligned: % Passed)...")
    lb = Leaderboard()
    leaderboard_data = lb.generate()
    lb.print_leaderboard(leaderboard_data)
    
    # Summary
    click.echo("\n" + "=" * 80)
    click.echo("BENCHMARK COMPLETE")
    click.echo("=" * 80)
    click.echo(f"  ✓ Datasets loaded: {len(loaded)}")
    click.echo(f"  ✓ Tasks selected: {len(balanced_tasks)} (from {len(tasks)} total)")
    click.echo(f"  ✓ Ground truth: {gt_generated + gt_cached} tasks")
    click.echo(f"  ✓ Models evaluated: {', '.join(models_to_test)}")
    click.echo(f"  ✓ Results saved: {len(list(RESULTS_DIR.glob('*.json')))} files")
    click.echo(f"  ✓ Leaderboard: {RESULTS_DIR / 'leaderboard.json'}")
    click.echo("\n" + "=" * 80)


@main.command(name="run-full-benchmark")
@click.option("--evaluator", default="v2", type=click.Choice(["v1", "v2"]),
              help="Evaluator version: v2 (default, ground-truth) or v1 (legacy ROUGE/BLEU)")
@click.option("--enable-execution", is_flag=True, default=False,
              help="Enable behavioral fidelity testing (requires Docker with GnuCOBOL)")
@click.option("--judge-model", default="gpt-4o",
              help="LLM model for semantic quality evaluation (v2 only)")
@click.option("--task-limit", default=3, type=int,
              help="Number of tasks to run (default: 3 for quick testing, use 200 for full benchmark)")
@click.option("--models", default="claude-sonnet-4,gpt-4o,aws-transform",
              help="Comma-separated list of models to test (e.g., claude-sonnet-4,gpt-4o,aws-transform,docmolt-gpt4o)")
@click.option("--skip-datasets", is_flag=True, default=False,
              help="Skip dataset loading (use existing datasets)")
@click.option("--skip-task-creation", is_flag=True, default=False,
              help="Skip task creation (use existing tasks)")
def run_full_benchmark(evaluator: str, enable_execution: bool, judge_model: str,
                       task_limit: int, models: str, skip_datasets: bool, 
                       skip_task_creation: bool):
    """Run complete benchmark pipeline (SWE-bench aligned)
    
    Single command that orchestrates the full pipeline:
    
    [1] LOAD DATASETS ──→ [2] SELECT TASKS ──→ [3] GROUND TRUTH GENERATION
         │                     │                        │
         ▼                     ▼                        ▼
    GitHub Repos        Intelligent             Static Analysis
    (COBOL Files)        Selection               (Automated 95%)
    
    [4] AI GENERATES DOC ──→ [5] EVALUATION ──→ [6] SCORING ──→ [7] LEADERBOARD
    
    Uses v2.0 ground-truth based evaluation with:
    - Structural Completeness (30%): Element coverage vs auto-extracted ground truth
    - Behavioral Fidelity (35%): Execution-based testing (if --enable-execution)
    - Semantic Quality (25%): LLM-as-judge evaluation
    - Traceability (10%): Reference validation
    
    Primary metric: % Passed (SWE-bench aligned, like "% Resolved")
    
    Examples:
        # Quick test (3 tasks, 3 models including AWS Transform)
        legacycodebench run-full-benchmark

        # Full benchmark (200 tasks)
        legacycodebench run-full-benchmark --task-limit 200 --enable-execution

        # Test specific models
        legacycodebench run-full-benchmark --models "gpt-4o,aws-transform"

        # Skip dataset loading if already done
        legacycodebench run-full-benchmark --skip-datasets --skip-task-creation
    """
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    
    _run_benchmark(
        models_to_test=model_list,
        header_label=f"LegacyCodeBench v2.0 Evaluation ({evaluator.upper()})",
        evaluator_version=evaluator,
        enable_execution=enable_execution,
        judge_model=judge_model,
        task_limit=task_limit,
        skip_datasets=skip_datasets,
        skip_task_creation=skip_task_creation
    )


@main.command()
def interactive():
    """Guided run that prompts for API keys and model selection (v2.0)"""
    click.echo("=" * 60)
    click.echo("LegacyCodeBench v2.0 Interactive Runner")
    click.echo("=" * 60)
    click.echo("\nThis will run the v2.0 ground-truth based evaluation.")
    click.echo("Components: SC (30%) + BF (35%) + SQ (25%) + TR (10%)\n")

    openai_key = click.prompt(
        "Enter OpenAI API key (press Enter to skip)",
        hide_input=True,
        default="",
        show_default=False,
    ).strip()
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        click.echo("  ✓ OpenAI API key set for this session")

    anthropic_key = click.prompt(
        "Enter Anthropic API key (press Enter to skip)",
        hide_input=True,
        default="",
        show_default=False,
    ).strip()
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        click.echo("  ✓ Anthropic API key set for this session")

    # Add DocMolt API key prompt
    docmolt_key = click.prompt(
        "Enter DocMolt API key (press Enter to skip)",
        hide_input=True,
        default="",
        show_default=False,
    ).strip()
    if docmolt_key:
        os.environ["DOCMOLT_API_KEY"] = docmolt_key
        click.echo("  ✓ DocMolt API key set for this session")

    # Add AWS credentials prompt
    aws_access_key = click.prompt(
        "Enter AWS Access Key ID (press Enter to skip)",
        hide_input=True,
        default="",
        show_default=False,
    ).strip()
    if aws_access_key:
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
        aws_secret_key = click.prompt(
            "Enter AWS Secret Access Key",
            hide_input=True,
        ).strip()
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key

        aws_region = click.prompt(
            "Enter AWS Region (press Enter for us-east-1)",
            default="us-east-1",
        ).strip()
        os.environ["AWS_REGION"] = aws_region
        click.echo("  ✓ AWS credentials set for this session")

    available_models: List[str] = []
    if os.getenv("OPENAI_API_KEY"):
        available_models.extend(["gpt-4o", "gpt-4"])
    if os.getenv("ANTHROPIC_API_KEY"):
        available_models.append("claude-sonnet-4")
    if os.getenv("DOCMOLT_API_KEY"):
        available_models.extend(["docmolt-gpt4o", "docmolt-gpt4o-mini", "docmolt-claude"])
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        available_models.append("aws-transform")

    # Deduplicate while preserving order
    seen = []
    models_unique = []
    for model in available_models:
        if model not in seen:
            models_unique.append(model)
            seen.append(model)

    if not models_unique:
        click.echo("\n✗ No API keys detected. Please provide an API key to continue.")
        click.echo("  - Set OPENAI_API_KEY for GPT models (also used for LLM-as-judge)")
        click.echo("  - Set ANTHROPIC_API_KEY for Claude models")
        click.echo("  - Set DOCMOLT_API_KEY for DocMolt models")
        click.echo("  - Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY for AWS Transform")
        click.echo("\nAlternatively, run 'legacycodebench run-ai --model <id>' once keys are configured.")
        return

    model_choice = click.prompt(
        "\nChoose a model to evaluate",
        type=click.Choice(models_unique, case_sensitive=False),
        default=models_unique[0],
    )
    
    # Ask about evaluation options
    enable_exec = click.confirm(
        "Enable behavioral fidelity testing? (requires Docker)",
        default=False
    )
    
    task_count = click.prompt(
        "Number of tasks to run",
        type=int,
        default=3
    )
    
    # Determine judge model (prefer different model than being evaluated)
    if model_choice.startswith("gpt") and os.getenv("ANTHROPIC_API_KEY"):
        judge = "claude-sonnet-4"
    elif model_choice.startswith("claude") and os.getenv("OPENAI_API_KEY"):
        judge = "gpt-4o"
    else:
        judge = "gpt-4o"  # Default
    
    click.echo(f"\n  Judge model for semantic quality: {judge}")

    _run_benchmark(
        models_to_test=[model_choice],
        header_label=f"LegacyCodeBench v2.0 Interactive Run ({model_choice})",
        evaluator_version="v2",
        enable_execution=enable_exec,
        judge_model=judge,
        task_limit=task_count
    )


if __name__ == "__main__":
    main()

