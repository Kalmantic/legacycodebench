"""Test infrastructure paragraph evaluation - compare multiple models"""
from pathlib import Path
from legacycodebench.tasks import TaskManager
from legacycodebench.evaluators_v231.evaluator_v231 import EvaluatorV231
from legacycodebench.config import SUBMISSIONS_DIR, GROUND_TRUTH_CACHE_DIR
import json

# Test with LCB-T1-001 and different models
task_id = 'LCB-T1-001'
models = [
    'docmolt-enterprise',
    'gpt-4o',
    'claude-sonnet-4',
    'gemini-2.5-flash'
]

manager = TaskManager()
evaluator = EvaluatorV231(llm_client=None, executor=None)

print(f'Testing {task_id} with fixed infrastructure evaluation...\n')
print('='*80)

try:
    task = manager.get_task(task_id)

    # Get source code from task input files
    input_files = task.get_input_files_absolute()
    if not input_files:
        raise FileNotFoundError(f"No input files found for task {task_id}")

    source_code = ""
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source_code += f.read() + "\n"

    # Load ground truth
    gt_file = GROUND_TRUTH_CACHE_DIR / f"{input_files[0].stem}_ground_truth.json"
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_file}")

    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)

    results = []
    for model in models:
        submission_file = f'{task_id}_{model}.md'
        sub_path = SUBMISSIONS_DIR / submission_file

        if not sub_path.exists():
            print(f'[SKIP] {model}: submission not found')
            continue

        # Load documentation
        with open(sub_path, 'r', encoding='utf-8') as f:
            documentation = f.read()

        result = evaluator.evaluate(
            task_id=task_id,
            model=model,
            documentation=documentation,
            source_code=source_code,
            ground_truth=ground_truth
        )

        results.append({
            'model': model,
            'lcb': result.lcb_score,
            'bf': result.bf_score,
            'sc': result.sc_score,
            'dq': result.dq_score,
            'breakdown': result.bf_breakdown
        })

        print(f'{model:20s} | LCB: {result.lcb_score:5.1f}% | BF: {result.bf_score*100:5.1f}%', end='')
        if result.bf_breakdown:
            claims = result.bf_breakdown.get('claims', 0)
            bsm = result.bf_breakdown.get('bsm', 0)
            infra_score = result.bf_breakdown.get('infrastructure_score', 0)
            claims_verified = result.bf_breakdown.get('claims_verified', 0)
            claims_failed = result.bf_breakdown.get('claims_failed', 0)
            print(f' | Claims: {claims:.2f} | BSM: {bsm:.2f} | Infra: {infra_score:.2f} | V:{claims_verified} F:{claims_failed}')
        else:
            print()

    print('='*80)
    print('\nSummary:')
    print(f'  Models tested: {len(results)}')

    if len(results) > 1:
        bf_scores = [r['bf'] for r in results]
        print(f'  BF Score Range: {min(bf_scores)*100:.1f}% - {max(bf_scores)*100:.1f}%')
        print(f'  BF Score Std Dev: {(sum((s - sum(bf_scores)/len(bf_scores))**2 for s in bf_scores) / len(bf_scores))**0.5 * 100:.2f}%')

        if all(r['bf'] == results[0]['bf'] for r in results):
            print('  [WARNING] All models have IDENTICAL BF scores!')
        else:
            print('  [OK] BF scores DIFFER across models (as expected)')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
