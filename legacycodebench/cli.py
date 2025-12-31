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
    GROUND_TRUTH_CACHE_DIR,  # Unified cache location
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
@click.version_option(version="2.3.1")
def main():
    """LegacyCodeBench - Benchmark for AI systems on legacy code understanding

    v2.3.1: Deterministic evaluation with structural, documentation, and behavioral fidelity
    v2.0: Ground truth-based evaluation with behavioral fidelity testing
    """
    pass


@main.command()
def load_datasets():
    """Load COBOL datasets from GitHub repositories"""
    click.echo("Loading datasets from GitHub repositories...")
    loader = DatasetLoader()
    loaded = loader.load_all_datasets()
    
    click.echo(f"\n[OK] Loaded {len(loaded)} datasets:")
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
    
    click.echo(f"\n[OK] Created {len(tasks)} tasks:")
    for task in tasks:
        click.echo(f"  {task.task_id}: {task.category} - {task.difficulty}")


@main.command()
@click.option("--task-id", required=True, help="Task ID (e.g., LCB-DOC-001)")
@click.option("--submission", required=True, type=click.Path(exists=True), help="Path to submission file")
@click.option("--output", type=click.Path(), help="Path to output results file")
@click.option("--submitter-name", default="Unknown", help="Submitter name")
@click.option("--submitter-model", default="unknown", help="Model name")
@click.option("--submitter-category", default="verified", help="Category (bash, verified, full, human+ai)")
@click.option("--evaluator", default="v2.3.1", type=click.Choice(["v1", "v2", "v2.3.1"]),
              help="Evaluator version (default: v2.3.1 - deterministic)")
@click.option("--enable-execution", is_flag=True, default=False,
              help="Enable behavioral fidelity testing (v2 only, requires Docker)")
@click.option("--judge-model", default="gpt-4o", help="LLM judge model for semantic quality (v2 only)")
def evaluate(task_id: str, submission: Path, output: Optional[Path],
             submitter_name: str, submitter_model: str, submitter_category: str,
             evaluator: str, enable_execution: bool, judge_model: str):
    """Evaluate a submission for a task

    v2.3.1 (default): Deterministic evaluation with SC/DQ/BF (no LLM-as-judge)
    v2.0: Ground truth-based evaluation with LLM-as-judge
    v1.0 (legacy): Reference-based ROUGE/BLEU evaluation
    """
    click.echo(f"Evaluating {task_id} with {evaluator.upper()} evaluator...")

    # Load task
    try:
        task = Task.load(task_id, TASKS_DIR)
    except FileNotFoundError:
        click.echo(f"[ERROR] Task {task_id} not found. Run 'legacycodebench create-tasks' first.")
        return

    # Load submission
    submission_path = Path(submission)

    # Select evaluator based on version and category
    if evaluator == "v2.3.1":
        # v2.3.1 evaluator (deterministic, no LLM-as-judge)
        click.echo(f"  Using V2.3.1 Evaluator (Deterministic)")
        
        try:
            from legacycodebench.evaluators_v231 import EvaluatorV231
            from legacycodebench.static_analysis.ground_truth_generator import GroundTruthGenerator
            
            # Load ground truth
            gt_gen = GroundTruthGenerator()
            input_files = task.get_input_files_absolute()
            gt = gt_gen.generate(input_files, cache_dir=GROUND_TRUTH_CACHE_DIR)
            
            # Read documentation
            with open(submission_path, 'r', encoding='utf-8') as f:
                documentation = f.read()
            
            # Read source code
            source_code = ""
            for f in input_files:
                with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
                    source_code += fp.read() + "\n"
            
            # Create executor if execution mode enabled
            executor = None
            if enable_execution:
                try:
                    from legacycodebench.execution.cobol_executor import COBOLExecutor
                    executor = COBOLExecutor()
                    click.echo(f"  Execution Mode: ENABLED (Docker-based)")
                except ImportError:
                    click.echo(f"  [WARNING] COBOLExecutor not available, using heuristic BF")
                except Exception as e:
                    click.echo(f"  [WARNING] Executor init failed: {e}, using heuristic BF")
            else:
                click.echo(f"  Execution Mode: DISABLED (heuristic BF)")
            
            # Evaluate
            evaluator_instance = EvaluatorV231(executor=executor)
            eval_result = evaluator_instance.evaluate(
                task_id=task_id,
                model=submitter_model,
                documentation=documentation,
                source_code=source_code,
                ground_truth=gt
            )
            
            result = eval_result.to_dict()
            overall_score = eval_result.lcb_score / 100  # Convert to 0-1
            
        except ImportError as e:
            click.echo(f"[ERROR] V2.3.1 evaluators not available: {e}")
            return
            
    elif evaluator == "v2":
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

            # v2.0 uses "score" key (already in 0-1 scale)
            overall_score = result.get("score", 0)

        elif task.category == "understanding":
            click.echo(f"[ERROR] v2.0 evaluator for understanding tasks not yet implemented")
            click.echo(f"  v2.0 focuses on documentation tasks only")
            click.echo(f"  Use --evaluator v1 for understanding tasks")
            return
        else:
            click.echo(f"[ERROR] Unknown task category: {task.category}")
            return

    elif evaluator == "v1":
        # v1.0 evaluator (legacy)
        if not V1_AVAILABLE:
            click.echo(f"[ERROR] v1.0 evaluators not available (archived)")
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
            click.echo(f"[ERROR] Unknown task category: {task.category}")
            return

        overall_score = result["score"]

    else:
        click.echo(f"[ERROR] Unknown evaluator version: {evaluator}")
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
            click.echo(f"  STATUS: [PASSED]")
        else:
            click.echo(f"  STATUS: [FAILED]")
            click.echo(f"  REASON: {pass_status.get('reason', 'Unknown')}")

        click.echo(f"")
        click.echo(f"  LCB Score: {result.get('score', 0)*100:.1f}/100")
        click.echo(f"    - Structural Completeness (30%): {result.get('structural_completeness', 0)*100:.1f}%")

        # ADDED (Issue 7.3): Warn if BF is placeholder
        bf_details = result.get('details', {}).get('behavioral_fidelity', {})
        is_placeholder = bf_details.get('placeholder', False)
        bf_score = result.get('behavioral_fidelity', 0)

        if is_placeholder:
            click.echo(f"    - Behavioral Fidelity (35%):     {bf_score*100:.1f}% [PLACEHOLDER - No execution]")
        else:
            click.echo(f"    - Behavioral Fidelity (35%):     {bf_score*100:.1f}%")

        click.echo(f"    - Semantic Quality (25%):        {result.get('semantic_quality', 0)*100:.1f}%")
        click.echo(f"    - Traceability (10%):            {result.get('traceability', 0)*100:.1f}%")

        if result.get('critical_failures'):
            click.echo(f"")
            click.echo(f"  [WARNING] CRITICAL FAILURES:")
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
@click.option("--mock", is_flag=True, default=False, help="Use mock responses for testing (no API keys required)")
def run_ai(model: str, task_id: Optional[str], submitter_name: Optional[str], mock: bool):
    """Run AI model on task(s) and evaluate"""
    if submitter_name is None:
        submitter_name = model

    if mock:
        click.echo("[MOCK MODE] Using fake API responses for testing")

    # Get AI model
    try:
        ai_model = get_ai_model(model, mock_mode=mock)
    except ValueError as e:
        click.echo(f"[ERROR] {e}")
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
            click.echo(f"  [ERROR] No input files found for {task.task_id}")
            continue
        
        # Generate output
        if task.category == "documentation":
            output = ai_model.generate_documentation(task, input_files)
            output_file = SUBMISSIONS_DIR / f"{task.task_id}_{model}.md"
        elif task.category == "understanding":
            output = ai_model.generate_understanding(task, input_files)
            output_file = SUBMISSIONS_DIR / f"{task.task_id}_{model}.json"
        else:
            click.echo(f"  [ERROR] Unknown category: {task.category}")
            continue
        
        # Save submission
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        
        click.echo(f"  [OK] Generated submission: {output_file}")
        
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

        # ADDED (Issue 6.2): Show score breakdown
        click.echo(f"  [OK] Score: {result['score']*100:.2f}%")

        # Show breakdown if v2.0 result
        if result.get('version') == '2.0':
            # ADDED (Issue 7.3): Warn if BF is placeholder
            bf_details = result.get('details', {}).get('behavioral_fidelity', {})
            is_placeholder = bf_details.get('placeholder', False)
            bf_str = f"{result.get('behavioral_fidelity', 0)*100:.1f}%"
            if is_placeholder:
                bf_str += " (placeholder)"

            click.echo(f"       SC: {result.get('structural_completeness', 0)*100:.1f}% | "
                      f"BF: {bf_str} | "
                      f"SQ: {result.get('semantic_quality', 0)*100:.1f}% | "
                      f"TR: {result.get('traceability', 0)*100:.1f}%")


@main.command()
@click.option("--output", type=click.Path(), help="Path to output JSON file")
@click.option("--print", "print_flag", is_flag=True, default=True, help="Print leaderboard to console")
@click.option("--detailed", is_flag=True, help="Show detailed component scores")
@click.option("--export-md", type=click.Path(), help="Export as Markdown file")
@click.option("--export-csv", type=click.Path(), help="Export as CSV file")
def leaderboard(output: Optional[Path], print_flag: bool, detailed: bool, 
                export_md: Optional[Path], export_csv: Optional[Path]):
    """Generate leaderboard from all results"""
    click.echo("Generating leaderboard...")
    
    lb = Leaderboard()
    leaderboard_data = lb.generate(output)
    
    if print_flag:
        if detailed:
            lb.print_detailed(leaderboard_data)
        else:
            lb.print_leaderboard(leaderboard_data)
    
    if export_md:
        lb.export_markdown(Path(export_md), leaderboard_data)
        click.echo(f"[OK] Exported Markdown to: {export_md}")
    
    if export_csv:
        # CSV is auto-generated, just copy if different path
        import shutil
        shutil.copy(RESULTS_DIR / "summary.csv", export_csv)
        click.echo(f"[OK] Exported CSV to: {export_csv}")
    
    if output:
        click.echo(f"\n[OK] Leaderboard saved to: {output}")


def _run_benchmark(models_to_test: List[str], header_label: str = "LegacyCodeBench Evaluation",
                   evaluator_version: str = "v2", enable_execution: bool = False,
                   judge_model: str = "gpt-4o", task_limit: int = 3,
                   skip_datasets: bool = False, skip_task_creation: bool = False,
                   mock_mode: bool = False, clean_results: bool = False):
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
        mock_mode: Use mock responses for testing (no API keys required)
        clean_results: Clear old results before running (prevents stale data)
    """
    if not models_to_test:
        click.echo("[ERROR] No models selected. Aborting.")
        return
    
    # FIXED: Clean old results if requested (prevents stale data in leaderboard)
    if clean_results:
        import shutil
        results_path = RESULTS_DIR
        if results_path.exists():
            old_files = list(results_path.glob("*.json")) + list(results_path.glob("*.csv"))
            # Don't delete leaderboard.json and summary.csv yet - they'll be regenerated
            for f in old_files:
                try:
                    f.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete {f}: {e}")
            click.echo(f"[CLEAN] Removed {len(old_files)} old result files")
    
    click.echo("=" * 80)
    click.echo(header_label)
    click.echo("=" * 80)
    if evaluator_version == "v2.3.1":
        click.echo(f"Evaluator: V2.3.1 (Deterministic) | Execution: {'Enabled' if enable_execution else 'Disabled'}")
    else:
        click.echo(f"Evaluator: {evaluator_version.upper()} | Execution: {'Enabled' if enable_execution else 'Disabled'} | Judge: {judge_model}")
    click.echo("=" * 80)
    
    # Step 1: Load datasets
    if not skip_datasets:
        click.echo("\n[1/7] Loading datasets from GitHub repositories...")
        loader = DatasetLoader()
        loaded = loader.load_all_datasets()
        click.echo(f"[OK] Loaded {len(loaded)} datasets")
    else:
        click.echo("\n[1/7] Skipping dataset loading (using existing datasets)...")
        loader = DatasetLoader()
        loaded = loader.load_all_datasets()  # Still need to get loaded dict
        click.echo(f"[OK] Using {len(loaded)} existing datasets")
    
    # Step 2: Create tasks (uses v2.0 intelligent selection - documentation only, tier-based)
    if not skip_task_creation:
        click.echo("\n[2/7] Selecting tasks (v2.0 tier-based intelligent selection)...")
        manager = TaskManager()
        tasks = manager.create_tasks_from_datasets(use_intelligent_selection=True)
        manager.save_all(tasks)
        click.echo(f"[OK] Created {len(tasks)} documentation tasks")
    else:
        click.echo("\n[2/7] Skipping task creation (using existing tasks)...")
        manager = TaskManager()
        task_ids = manager.list_tasks()
        tasks = [manager.get_task(tid) for tid in task_ids]
        click.echo(f"[OK] Using {len(tasks)} existing tasks")
    
    # Step 3: Generate ground truth (pre-generate for all tasks to show progress)
    click.echo("\n[3/7] Generating ground truth (automated static analysis)...")
    from legacycodebench.static_analysis.ground_truth_generator import GroundTruthGenerator
    gt_generator = GroundTruthGenerator()
    # Use centralized cache directory from config
    GROUND_TRUTH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    gt_generated = 0
    gt_cached = 0
    for task in tasks:
        source_files = task.get_input_files_absolute()
        if source_files:
            main_file = source_files[0]
            # Check if cached
            cached = gt_generator.load_cached_ground_truth(main_file, GROUND_TRUTH_CACHE_DIR)
            if cached:
                gt_cached += 1
            else:
                # Generate (will be cached)
                gt_generator.generate(source_files, cache_dir=GROUND_TRUTH_CACHE_DIR)
                gt_generated += 1
    
    click.echo(f"[OK] Ground truth: {gt_generated} generated, {gt_cached} cached (95%+ automation)")
    
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
    
    # Only process judge model for non-v2.3.1 (v2.3.1 is deterministic, no LLM-as-judge)
    if evaluator_version != "v2.3.1":
        actual_judge = _ensure_different_judge(models_to_test, judge_model)
        if actual_judge != judge_model:
            click.echo(f"  ⚠ Judge model changed from '{judge_model}' to '{actual_judge}' "
                       f"(must be different from evaluated models)")
            judge_model = actual_judge

    for model_id in models_to_test:
        click.echo(f"\n  Running {model_id}...")
        # Only show judge info for V2.1.3 which uses LLM-as-judge
        if evaluator_version != "v2.3.1":
            click.echo(f"    Judge for SQ evaluation: {judge_model} (different from {model_id})")
        try:
            ai_model = get_ai_model(model_id, mock_mode=mock_mode)
            
            for task in balanced_tasks:
                click.echo(f"    Task: {task.task_id}")
                
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
                click.echo(f"    [OK] Generated submission")
                
                # Evaluate using selected evaluator version
                if evaluator_version == "v2.3.1":
                    # V2.3.1 evaluator (deterministic, 4 patches)
                    try:
                        from legacycodebench.evaluators_v231 import EvaluatorV231

                        # Load ground truth
                        source_files = task.get_input_files_absolute()
                        gt = gt_generator.generate(source_files, cache_dir=GROUND_TRUTH_CACHE_DIR) if source_files else {}

                        # Read documentation and source
                        doc_content = output_file.read_text(encoding='utf-8')
                        source_content = source_files[0].read_text(encoding='utf-8') if source_files else ""

                        # Create executor if execution enabled
                        executor = None
                        if enable_execution:
                            try:
                                from legacycodebench.execution.cobol_executor import COBOLExecutor
                                executor = COBOLExecutor()
                            except Exception as e:
                                logger.warning(f"Executor init failed: {e}, using heuristic BF")

                        # Run V2.3.1 evaluation
                        evaluator_instance = EvaluatorV231(executor=executor)
                        eval_result = evaluator_instance.evaluate(
                            task_id=task.task_id,
                            model=model_id,
                            documentation=doc_content,
                            source_code=source_content,
                            ground_truth=gt
                        )
                        
                        result = eval_result.to_dict()
                        overall_score = eval_result.lcb_score / 100  # Convert to 0-1
                        
                        # Display V2.3.1 format
                        status = "[PASSED]" if eval_result.passed else "[FAILED]"
                        click.echo(f"    [OK] V2.3.1 Score: {eval_result.lcb_score:.1f}% {status}")
                        click.echo(f"         (SC:{eval_result.sc_score*100:.0f}% "
                                  f"DQ:{eval_result.dq_score*100:.0f}% "
                                  f"BF:{eval_result.bf_score*100:.0f}%)")
                        
                        if eval_result.critical_failures:
                            cf_ids = [cf.cf_id for cf in eval_result.critical_failures]
                            click.echo(f"    [WARNING] Critical failures: {', '.join(cf_ids)}")
                    
                    except ImportError as e:
                        click.echo(f"    [ERROR] V2.3.1 evaluators not available: {e}")
                        click.echo(f"    Falling back to V2.3...")
                        evaluator_version = "v2.3"  # Fallback
                
                elif evaluator_version == "v2.3":
                    # V2.3 evaluator (hybrid Template + BSM)
                    try:
                        from legacycodebench.evaluators_v23 import EvaluatorV23
                        
                        # Load ground truth
                        source_files = task.get_input_files_absolute()
                        gt = gt_generator.generate(source_files, cache_dir=GROUND_TRUTH_CACHE_DIR) if source_files else {}
                        
                        # Read documentation and source
                        doc_content = output_file.read_text(encoding='utf-8')
                        source_content = source_files[0].read_text(encoding='utf-8') if source_files else ""
                        
                        # Run V2.3 evaluation
                        evaluator = EvaluatorV23()
                        eval_result = evaluator.evaluate(
                            task_id=task.task_id,
                            model=model_id,
                            documentation=doc_content,
                            source_code=source_content,
                            ground_truth=gt
                        )
                        
                        result = eval_result.to_dict()
                        overall_score = eval_result.score.overall / 100  # Convert to 0-1
                        
                        # Display V2.3 format
                        status = "[PASSED]" if eval_result.score.passed else "[FAILED]"
                        click.echo(f"    [OK] V2.3 Score: {eval_result.score.overall:.1f}% {status}")
                        click.echo(f"         (C:{eval_result.score.comprehension*100:.0f}% "
                                  f"D:{eval_result.score.documentation*100:.0f}% "
                                  f"B:{eval_result.score.behavioral*100:.0f}%)")
                        
                        if eval_result.score.critical_failures:
                            click.echo(f"    [WARNING] Critical failures: {', '.join(eval_result.score.critical_failures)}")
                    
                    except ImportError as e:
                        click.echo(f"    [ERROR] V2.3 evaluators not available: {e}")
                        click.echo(f"    Falling back to V2.1.3...")
                        evaluator_version = "v2"  # Fallback
                
                if evaluator_version == "v2":
                    # v2.1.3 evaluator (IUE + BSM - legacy)
                    eval_instance = DocumentationEvaluatorV2(
                        enable_execution=enable_execution,
                        results_dir=RESULTS_DIR / "escalations"
                    )
                    eval_instance.sq_evaluator.judge_model_name = judge_model
                    result = eval_instance.evaluate(output_file, task)
                    overall_score = result.get("score", 0)

                    # ADDED (Issue 7.3): Warn if BF is placeholder
                    bf_details = result.get('details', {}).get('behavioral_fidelity', {})
                    is_placeholder = bf_details.get('placeholder', False)
                    bf_str = f"{result.get('behavioral_fidelity', 0)*100:.0f}%"
                    if is_placeholder:
                        bf_str += "*"  # Asterisk to indicate placeholder

                    click.echo(f"    [OK] v2.1.3 Score: {result.get('score', 0)*100:.1f}% "
                              f"(SC:{result.get('structural_completeness', 0)*100:.0f}% "
                              f"BF:{bf_str} "
                              f"SQ:{result.get('semantic_quality', 0)*100:.0f}% "
                              f"TR:{result.get('traceability', 0)*100:.0f}%)")
                    
                    if result.get('critical_failures'):
                        click.echo(f"    [WARNING] Critical failures: {', '.join(result['critical_failures'])}")
                elif evaluator_version == "v1":
                    # v1.0 evaluator (legacy ROUGE/BLEU)
                    if not V1_AVAILABLE:
                        click.echo(f"    [ERROR] v1.0 evaluator not available")
                        continue
                    
                    eval_instance = DocumentationEvaluator()
                    result = eval_instance.evaluate(output_file, task)
                    overall_score = result.get("score", 0)
                    click.echo(f"    [OK] v1.0 Score: {overall_score*100:.1f}%")
                
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
            click.echo(f"  [ERROR] Error with {model_id}: {e}")
            logger.exception(f"Error running {model_id}")
    
    click.echo("\n[OK] Completed AI model runs")
    
    # Step 5: Evaluation (already done during AI run, but summarize here)
    if evaluator_version == "v2.3.1":
        click.echo("\n[5/7] Evaluation complete (SC/DQ/BF calculated - V2.3.1)")
    elif evaluator_version == "v2.3":
        click.echo("\n[5/7] Evaluation complete (C/D/B calculated - V2.3)")
    else:
        click.echo("\n[5/7] Evaluation complete (SC, BF, SQ, TR calculated)")
    
    # Step 6: Scoring (already done, but summarize)
    if evaluator_version == "v2.3.1":
        click.echo("\n[6/7] Scoring complete (LCB = 0.30xSC + 0.20xDQ + 0.50xBF)")
    else:
        click.echo("\n[6/7] Scoring complete (LCB Score calculated with pass/fail status)")
    
    # Step 7: Generate leaderboard
    click.echo("\n[7/7] Generating leaderboard...")
    lb = Leaderboard()
    leaderboard_data = lb.generate()
    lb.print_leaderboard(leaderboard_data)
    
    # Summary
    click.echo("\n" + "=" * 80)
    click.echo("BENCHMARK COMPLETE")
    click.echo("=" * 80)
    click.echo(f"  [OK] Datasets loaded: {len(loaded)}")
    click.echo(f"  [OK] Tasks selected: {len(balanced_tasks)} (from {len(tasks)} total)")
    click.echo(f"  [OK] Ground truth: {gt_generated + gt_cached} tasks")
    click.echo(f"  [OK] Models evaluated: {', '.join(models_to_test)}")
    click.echo(f"  [OK] Results saved: {len(list(RESULTS_DIR.glob('*.json')))} files")
    click.echo(f"  [OK] Leaderboard: {RESULTS_DIR / 'leaderboard.json'}")
    click.echo("\n" + "=" * 80)


@main.command(name="run-full-benchmark")
@click.option("--evaluation-version", default="v2.3.1", type=click.Choice(["v2.3.1", "v2.3", "v2.1.3"]),
              help="Evaluation version: v2.3.1 (default, deterministic), v2.3 (hybrid template+BSM), v2.1.3 (legacy IUE+BSM)")
@click.option("--enable-execution", is_flag=True, default=False,
              help="Enable behavioral fidelity testing (requires Docker with GnuCOBOL)")
@click.option("--judge-model", default="gpt-4o",
              help="LLM model for semantic quality evaluation")
@click.option("--task-limit", default=3, type=int,
              help="Number of tasks to run (default: 3 for quick testing, use 200 for full benchmark)")
@click.option("--models", default="claude-sonnet-4,gpt-4o,gemini-2.0-flash",
              help="Comma-separated list of models to test")
@click.option("--skip-datasets", is_flag=True, default=False,
              help="Skip dataset loading (use existing datasets)")
@click.option("--skip-task-creation", is_flag=True, default=False,
              help="Skip task creation (use existing tasks)")
@click.option("--mock", is_flag=True, default=False,
              help="Use mock responses for testing (no API keys required)")
@click.option("--clean", is_flag=True, default=False,
              help="Clear old results before running (prevents stale data in leaderboard)")
def run_full_benchmark(evaluation_version: str, enable_execution: bool, judge_model: str,
                       task_limit: int, models: str, skip_datasets: bool,
                       skip_task_creation: bool, mock: bool, clean: bool):
    """Run complete benchmark pipeline (SWE-bench aligned)
    
    Single command that orchestrates the full pipeline:
    
    [1] LOAD DATASETS -> [2] SELECT TASKS -> [3] GROUND TRUTH GENERATION
          |                    |                       |
          V                    V                       V
    GitHub Repos        Intelligent             Static Analysis
    (COBOL Files)        Selection               (Automated 95%)
    
    [4] AI GENERATES DOC -> [5] EVALUATION -> [6] SCORING -> [7] LEADERBOARD
    
    V2.3 (default) - Hybrid Template + BSM evaluation:
    - Comprehension (40%): Business rules, data flow, abstraction
    - Documentation (25%): Structure, semantic quality, traceability
    - Behavioral (35%): Template-based (PURE) + BSM (MIXED paragraphs)
    - Anti-Gaming: Keyword stuffing, parroting, abstraction scoring
    - 6 Critical Failures: CF-01 through CF-06
    
    V2.1.3 (legacy) - IUE + BSM evaluation:
    - Structural Completeness (30%): Element coverage
    - Behavioral Fidelity (35%): IUE + BSM
    - Semantic Quality (25%): LLM-as-judge
    - Traceability (10%): Reference validation
    
    V2.3.1 (default): LCB = 0.30xSC + 0.20xDQ + 0.50xBF (deterministic, no LLM-as-judge)
    
    Examples:
        # Quick test (1 task, V2.3.1 default)
        legacycodebench run-full-benchmark --task-limit 1

        # Full benchmark with V2.3.1
        legacycodebench run-full-benchmark --task-limit 200 --enable-execution

        # Use legacy V2.1.3 evaluation
        legacycodebench run-full-benchmark --evaluation-version v2.1.3

        # Test specific models
        legacycodebench run-full-benchmark --models "gpt-4o,claude-sonnet-via-bedrock"
    """
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    
    # Map evaluation version to internal evaluator
    if evaluation_version == "v2.3.1":
        evaluator = "v2.3.1"
        header_label = f"LegacyCodeBench V2.3.1 Evaluation (Deterministic, 4 Patches)"
    elif evaluation_version == "v2.3":
        evaluator = "v2.3"
        header_label = f"LegacyCodeBench V2.3 Evaluation (Hybrid Template+BSM)"
    else:
        evaluator = "v2"
        header_label = f"LegacyCodeBench V2.1.3 Evaluation (IUE+BSM)"
    
    _run_benchmark(
        models_to_test=model_list,
        header_label=header_label,
        evaluator_version=evaluator,
        enable_execution=enable_execution,
        judge_model=judge_model,
        task_limit=task_limit,
        skip_datasets=skip_datasets,
        skip_task_creation=skip_task_creation,
        mock_mode=mock,
        clean_results=clean
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
        click.echo("  [OK] OpenAI API key set for this session")

    anthropic_key = click.prompt(
        "Enter Anthropic API key (press Enter to skip)",
        hide_input=True,
        default="",
        show_default=False,
    ).strip()
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        click.echo("  [OK] Anthropic API key set for this session")

    # Add DocMolt API key prompt
    docmolt_key = click.prompt(
        "Enter DocMolt API key (press Enter to skip)",
        hide_input=True,
        default="",
        show_default=False,
    ).strip()
    if docmolt_key:
        os.environ["DOCMOLT_API_KEY"] = docmolt_key
        click.echo("  [OK] DocMolt API key set for this session")

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
        click.echo("  [OK] AWS credentials set for this session")

    available_models: List[str] = []
    if os.getenv("OPENAI_API_KEY"):
        available_models.extend(["gpt-4o", "gpt-4"])
    if os.getenv("ANTHROPIC_API_KEY"):
        available_models.append("claude-sonnet-4")
    if os.getenv("DOCMOLT_API_KEY"):
        available_models.extend(["docmolt-gpt4o", "docmolt-gpt4o-mini", "docmolt-claude"])
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        available_models.append("claude-sonnet-via-bedrock")

    # Deduplicate while preserving order
    seen = []
    models_unique = []
    for model in available_models:
        if model not in seen:
            models_unique.append(model)
            seen.append(model)

    if not models_unique:
        click.echo("\n[ERROR] No API keys detected. Please provide an API key to continue.")
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


@main.command(name="validate-setup")
def validate_setup():
    """Verify all dependencies and configurations are correct"""
    click.echo("=" * 80)
    click.echo("LegacyCodeBench Setup Validation")
    click.echo("=" * 80)

    all_ok = True

    # Check 1: Python version
    import sys
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    click.echo(f"\n[1/6] Python version: {py_version}")
    if sys.version_info.major >= 3 and sys.version_info.minor >= 8:
        click.echo("  [OK] Python 3.8+ detected")
    else:
        click.echo("  [ERROR] Python 3.8+ required")
        all_ok = False

    # Check 2: Required directories
    click.echo("\n[2/6] Directory structure:")
    dirs_to_check = [TASKS_DIR, DATASETS_DIR, SUBMISSIONS_DIR, RESULTS_DIR, GROUND_TRUTH_CACHE_DIR]
    for d in dirs_to_check:
        if d.exists():
            click.echo(f"  [OK] {d.name}/")
        else:
            click.echo(f"  [WARN] {d.name}/ not found (will be created on first use)")

    # Check 3: Docker (for execution mode)
    click.echo("\n[3/6] Docker (required for --enable-execution):")
    try:
        import subprocess
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            click.echo(f"  [OK] {version}")

            # Check for GnuCOBOL image
            result = subprocess.run(["docker", "images", "legacycodebench/gnucobol"],
                                  capture_output=True, text=True, timeout=5)
            if "legacycodebench/gnucobol" in result.stdout:
                click.echo("  [OK] GnuCOBOL Docker image found")
            else:
                click.echo("  [WARN] GnuCOBOL Docker image not found (execution mode unavailable)")
        else:
            click.echo("  [WARN] Docker not found (execution mode unavailable)")
    except Exception as e:
        click.echo(f"  [WARN] Docker check failed: {e}")

    # Check 4: API keys
    click.echo("\n[4/6] API keys:")
    api_keys = {
        "OPENAI_API_KEY": "OpenAI (for gpt-4o, gpt-4, etc.)",
        "ANTHROPIC_API_KEY": "Anthropic (for claude models)",
        "GOOGLE_API_KEY": "Google (for gemini models)"
    }
    found_keys = 0
    for key, desc in api_keys.items():
        if os.getenv(key):
            click.echo(f"  [OK] {desc}")
            found_keys += 1
        else:
            click.echo(f"  [WARN] {key} not set - {desc} unavailable")

    if found_keys == 0:
        click.echo("  [ERROR] No API keys found. Set at least one to run evaluations.")
        all_ok = False

    # Check 5: Ground truth cache
    click.echo("\n[5/6] Ground truth cache:")
    if GROUND_TRUTH_CACHE_DIR.exists():
        gt_files = list(GROUND_TRUTH_CACHE_DIR.glob("*.json"))
        click.echo(f"  [OK] {len(gt_files)} cached ground truth files")
    else:
        click.echo("  [INFO] No cache yet (will be created on first run)")

    # Check 6: Evaluators
    click.echo("\n[6/6] Evaluators:")
    try:
        from legacycodebench.evaluators_v231 import EvaluatorV231
        click.echo("  [OK] V2.3.1 evaluators available")
    except ImportError as e:
        click.echo(f"  [ERROR] V2.3.1 evaluators not available: {e}")
        all_ok = False

    # Summary
    click.echo("\n" + "=" * 80)
    if all_ok:
        click.echo("[OK] Setup validation PASSED - ready to run benchmarks")
    else:
        click.echo("[ERROR] Setup validation FAILED - fix errors above")
    click.echo("=" * 80)


@main.command(name="verify-reproducibility")
@click.option("--model", required=True, help="Model to test (e.g., gpt-4o)")
@click.option("--task-id", default="LCB-T1-003", help="Task ID to test")
@click.option("--runs", default=3, type=int, help="Number of runs (default: 3)")
def verify_reproducibility(model: str, task_id: str, runs: int):
    """Verify that evaluations are 100% reproducible

    Runs the same task N times and verifies scores are identical.
    This validates the deterministic nature of v2.3.1 evaluation.
    """
    click.echo("=" * 80)
    click.echo(f"LegacyCodeBench Reproducibility Test")
    click.echo("=" * 80)
    click.echo(f"Model: {model}")
    click.echo(f"Task: {task_id}")
    click.echo(f"Runs: {runs}")
    click.echo("=" * 80)

    # Load task
    try:
        task = Task.load(task_id, TASKS_DIR)
    except FileNotFoundError:
        click.echo(f"[ERROR] Task {task_id} not found")
        return

    # Generate documentation once
    click.echo("\n[1/3] Generating documentation...")
    try:
        ai_model = get_ai_model(model)
        input_files = task.get_input_files_absolute()
        documentation = ai_model.generate_documentation(task, input_files)
        click.echo(f"  [OK] Documentation generated ({len(documentation)} chars)")
    except Exception as e:
        click.echo(f"  [ERROR] Failed to generate documentation: {e}")
        return

    # Load ground truth
    click.echo("\n[2/3] Loading ground truth...")
    try:
        from legacycodebench.static_analysis.ground_truth_generator import GroundTruthGenerator
        gt_gen = GroundTruthGenerator()
        gt = gt_gen.generate(input_files, cache_dir=GROUND_TRUTH_CACHE_DIR)
        source_code = input_files[0].read_text(encoding='utf-8') if input_files else ""
        click.echo(f"  [OK] Ground truth loaded")
    except Exception as e:
        click.echo(f"  [ERROR] Failed to load ground truth: {e}")
        return

    # Run evaluation N times
    click.echo(f"\n[3/3] Running evaluation {runs} times...")
    from legacycodebench.evaluators_v231 import EvaluatorV231

    results = []
    for i in range(runs):
        evaluator = EvaluatorV231()
        eval_result = evaluator.evaluate(
            task_id=task_id,
            model=model,
            documentation=documentation,
            source_code=source_code,
            ground_truth=gt
        )
        results.append({
            "run": i + 1,
            "lcb_score": eval_result.lcb_score,
            "sc_score": eval_result.sc_score,
            "dq_score": eval_result.dq_score,
            "bf_score": eval_result.bf_score,
            "passed": eval_result.passed,
        })
        click.echo(f"  Run {i+1}: LCB={eval_result.lcb_score:.1f}% "
                  f"(SC={eval_result.sc_score*100:.0f}% "
                  f"DQ={eval_result.dq_score*100:.0f}% "
                  f"BF={eval_result.bf_score*100:.0f}%)")

    # Check reproducibility
    click.echo("\n" + "=" * 80)
    first = results[0]
    all_identical = all(
        r["lcb_score"] == first["lcb_score"] and
        r["sc_score"] == first["sc_score"] and
        r["dq_score"] == first["dq_score"] and
        r["bf_score"] == first["bf_score"] and
        r["passed"] == first["passed"]
        for r in results
    )

    if all_identical:
        click.echo("[OK] REPRODUCIBILITY VERIFIED - All runs produced identical scores")
        click.echo("     This confirms v2.3.1 deterministic evaluation is working correctly.")
    else:
        click.echo("[ERROR] REPRODUCIBILITY FAILED - Scores differ across runs")
        click.echo("        This indicates non-deterministic behavior (bug in evaluator)")
    click.echo("=" * 80)


@main.command(name="compare")
@click.argument("models", nargs=2)
@click.option("--output", type=click.Path(), help="Save comparison to file")
def compare(models, output):
    """Compare results between two models

    Examples:
        legacycodebench compare gpt-4o claude-sonnet-4
        legacycodebench compare gpt-4o claude-sonnet-4 --output comparison.json
    """
    model1, model2 = models

    click.echo("=" * 80)
    click.echo(f"LegacyCodeBench Model Comparison")
    click.echo("=" * 80)
    click.echo(f"Model 1: {model1}")
    click.echo(f"Model 2: {model2}")
    click.echo("=" * 80)

    # Load leaderboard data
    lb = Leaderboard()
    leaderboard_data = lb.generate()

    # Find models in leaderboard
    model1_data = next((m for m in leaderboard_data["models"] if m["model"] == model1), None)
    model2_data = next((m for m in leaderboard_data["models"] if m["model"] == model2), None)

    if not model1_data:
        click.echo(f"\n[ERROR] {model1} not found in results. Run evaluation first.")
        return
    if not model2_data:
        click.echo(f"\n[ERROR] {model2} not found in results. Run evaluation first.")
        return

    # Compare scores
    click.echo("\n--- Overall Scores ---")
    click.echo(f"{model1:30s}: {model1_data['lcb_score']*100:6.1f}%")
    click.echo(f"{model2:30s}: {model2_data['lcb_score']*100:6.1f}%")
    diff = (model1_data['lcb_score'] - model2_data['lcb_score']) * 100
    click.echo(f"{'Difference':30s}: {diff:+6.1f}%")

    # Compare track breakdown
    click.echo("\n--- Track Breakdown ---")
    tracks = [
        ("Structural Completeness", "sc_score"),
        ("Documentation Quality", "dq_score"),
        ("Behavioral Fidelity", "bf_score"),
    ]

    for track_name, track_key in tracks:
        score1 = model1_data.get(track_key, 0) * 100
        score2 = model2_data.get(track_key, 0) * 100
        diff = score1 - score2
        click.echo(f"{track_name:30s}: {score1:6.1f}% vs {score2:6.1f}% ({diff:+6.1f}%)")

    # Compare pass rates
    click.echo("\n--- Pass/Fail Status ---")
    pass1 = model1_data.get("pass_count", 0)
    total1 = model1_data.get("total_tasks", 0)
    pass2 = model2_data.get("pass_count", 0)
    total2 = model2_data.get("total_tasks", 0)

    rate1 = (pass1 / total1 * 100) if total1 > 0 else 0
    rate2 = (pass2 / total2 * 100) if total2 > 0 else 0

    click.echo(f"{model1:30s}: {pass1}/{total1} ({rate1:.1f}%)")
    click.echo(f"{model2:30s}: {pass2}/{total2} ({rate2:.1f}%)")

    # Save comparison if requested
    if output:
        comparison = {
            "model1": model1,
            "model2": model2,
            "model1_data": model1_data,
            "model2_data": model2_data,
            "differences": {
                "lcb_score": diff,
                "sc_score": (model1_data.get("sc_score", 0) - model2_data.get("sc_score", 0)) * 100,
                "dq_score": (model1_data.get("dq_score", 0) - model2_data.get("dq_score", 0)) * 100,
                "bf_score": (model1_data.get("bf_score", 0) - model2_data.get("bf_score", 0)) * 100,
            }
        }
        with open(output, 'w') as f:
            json.dump(comparison, f, indent=2)
        click.echo(f"\n[OK] Comparison saved to: {output}")

    click.echo("\n" + "=" * 80)


if __name__ == "__main__":
    main()

