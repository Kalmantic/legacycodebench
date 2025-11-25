"""CLI interface for LegacyCodeBench"""

import click
import json
from pathlib import Path
from typing import Optional, List
import logging
import os

from legacycodebench.config import (
    TASKS_DIR, DATASETS_DIR, SUBMISSIONS_DIR, RESULTS_DIR,
    get_config
)
from legacycodebench.tasks import Task, TaskManager
from legacycodebench.dataset_loader import DatasetLoader
from legacycodebench.evaluators import DocumentationEvaluator, UnderstandingEvaluator
from legacycodebench.scoring import ScoringSystem
from legacycodebench.leaderboard import Leaderboard
from legacycodebench.ai_integration import get_ai_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0")
def main():
    """LegacyCodeBench - Benchmark for AI systems on legacy code understanding"""
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
def evaluate(task_id: str, submission: Path, output: Optional[Path], 
             submitter_name: str, submitter_model: str, submitter_category: str):
    """Evaluate a submission for a task"""
    click.echo(f"Evaluating {task_id}...")
    
    # Load task
    try:
        task = Task.load(task_id, TASKS_DIR)
    except FileNotFoundError:
        click.echo(f"✗ Task {task_id} not found. Run 'legacycodebench create-tasks' first.")
        return
    
    # Load submission
    submission_path = Path(submission)
    
    # Evaluate based on category
    if task.category == "documentation":
        evaluator = DocumentationEvaluator()
        result = evaluator.evaluate(submission_path, task)
    elif task.category == "understanding":
        evaluator = UnderstandingEvaluator()
        result = evaluator.evaluate(submission_path, task)
    else:
        click.echo(f"✗ Unknown task category: {task.category}")
        return
    
    # Calculate overall score (for this task, it's just the task score)
    scoring = ScoringSystem()
    overall_score = result["score"]
    
    # Create result entry
    result_entry = {
        "task_id": task_id,
        "task_category": task.category,
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
        output = RESULTS_DIR / f"{task_id}_{submitter_name}_{submitter_model}.json"
    
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result_entry, f, indent=2)
    
    click.echo(f"\n✓ Evaluation complete:")
    click.echo(f"  Score: {result['score']*100:.2f}%")
    click.echo(f"  Results saved to: {output_path}")


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
@click.option("--print", "print_flag", is_flag=True, help="Print leaderboard to console")
def leaderboard(output: Optional[Path], print_flag: bool):
    """Generate leaderboard from all results"""
    click.echo("Generating leaderboard...")
    
    lb = Leaderboard()
    leaderboard_data = lb.generate(output)
    
    if print_flag or output is None:
        lb.print_leaderboard(leaderboard_data)
    
    if output:
        click.echo(f"\n✓ Leaderboard saved to: {output}")


def _run_benchmark(models_to_test: List[str], header_label: str = "LegacyCodeBench Evaluation"):
    """Shared routine that runs the full benchmark pipeline"""
    if not models_to_test:
        click.echo("✗ No models selected. Aborting.")
        return
    
    click.echo("=" * 80)
    click.echo(header_label)
    click.echo("=" * 80)
    
    # Step 1: Load datasets
    click.echo("\n[1/5] Loading datasets...")
    loader = DatasetLoader()
    loaded = loader.load_all_datasets()
    click.echo(f"✓ Loaded {len(loaded)} datasets")
    
    # Step 2: Create tasks
    click.echo("\n[2/5] Creating tasks...")
    manager = TaskManager()
    tasks = manager.create_tasks_from_datasets()
    manager.save_all(tasks)
    click.echo(f"✓ Created {len(tasks)} tasks")
    
    # Step 3: Run AI models
    click.echo("\n[3/5] Running AI models...")
    
    for model_id in models_to_test:
        click.echo(f"  Running {model_id}...")
        try:
            ai_model = get_ai_model(model_id)
            
            # Run on first 3 tasks
            for task in tasks[:3]:
                input_files = task.get_input_files_absolute()
                if not input_files:
                    continue
                
                if task.category == "documentation":
                    output = ai_model.generate_documentation(task, input_files)
                    output_file = SUBMISSIONS_DIR / f"{task.task_id}_{model_id}.md"
                else:
                    output = ai_model.generate_understanding(task, input_files)
                    output_file = SUBMISSIONS_DIR / f"{task.task_id}_{model_id}.json"
                
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(output)
                
                # Evaluate
                if task.category == "documentation":
                    evaluator = DocumentationEvaluator()
                else:
                    evaluator = UnderstandingEvaluator()
                
                result = evaluator.evaluate(output_file, task)
                
                # Save result
                result_file = RESULTS_DIR / f"{task.task_id}_{model_id}_{model_id}.json"
                result_entry = {
                    "task_id": task.task_id,
                    "task_category": task.category,
                    "submitter": {
                        "name": model_id.split("-")[0].title(),
                        "model": model_id,
                        "category": "verified",
                    },
                    "result": result,
                    "overall_score": result["score"],
                }
                
                with open(result_file, 'w') as f:
                    json.dump(result_entry, f, indent=2)
        
        except Exception as e:
            click.echo(f"  ✗ Error with {model_id}: {e}")
    
    click.echo("✓ Completed AI model runs")
    
    # Step 4: Generate leaderboard
    click.echo("\n[4/5] Generating leaderboard...")
    lb = Leaderboard()
    leaderboard_data = lb.generate()
    lb.print_leaderboard(leaderboard_data)
    
    # Step 5: Summary
    click.echo("\n[5/5] Evaluation complete!")
    click.echo(f"  - Datasets: {len(loaded)}")
    click.echo(f"  - Tasks: {len(tasks)}")
    click.echo(f"  - Models tested: {', '.join(models_to_test)}")
    click.echo(f"  - Results: {len(list(RESULTS_DIR.glob('*.json')))}")
    click.echo("\n" + "=" * 80)


@main.command()
def evaluate():
    """Run full evaluation with default model suite"""
    default_models = ["claude-sonnet-4", "gpt-4o", "aws-transform"]
    _run_benchmark(default_models)


@main.command()
def interactive():
    """Guided run that prompts for API keys and model selection"""
    click.echo("LegacyCodeBench Interactive Runner")
    click.echo("---------------------------------")
    
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
    
    available_models: List[str] = []
    if os.getenv("OPENAI_API_KEY"):
        available_models.extend(["gpt-4o", "gpt-4"])
    if os.getenv("ANTHROPIC_API_KEY"):
        available_models.append("claude-sonnet-4")
    # Always allow mock AWS path as fallback
    available_models.append("aws-transform")
    # Deduplicate while preserving order
    seen = []
    models_unique = []
    for model in available_models:
        if model not in seen:
            models_unique.append(model)
            seen.append(model)
    
    model_choice = click.prompt(
        "Choose a model to run",
        type=click.Choice(models_unique, case_sensitive=False),
        default=models_unique[0],
    )
    
    _run_benchmark([model_choice], header_label=f"LegacyCodeBench Interactive Run ({model_choice})")


if __name__ == "__main__":
    main()

