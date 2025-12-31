"""Test v2.0 behavioral fidelity evaluator for comparison"""
from pathlib import Path
from legacycodebench.tasks import TaskManager
from legacycodebench.evaluators_v2.behavioral_fidelity_v2 import BehavioralFidelityEvaluatorV2
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
evaluator_v2 = BehavioralFidelityEvaluatorV2(enable_execution=False)

print(f'Testing {task_id} with v2.0 Behavioral Fidelity Evaluator...\n')
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

        # Use v2 evaluator
        result = evaluator_v2.evaluate(
            source_code=source_code,
            documentation=documentation,
            ground_truth=ground_truth,
            task_id=task_id
        )

        bf_score = result.get('score', 0)
        iue_score = result.get('iue', {}).get('score')
        bsm_score = result.get('bsm', {}).get('score')

        results.append({
            'model': model,
            'bf': bf_score,
            'iue': iue_score,
            'bsm': bsm_score
        })

        print(f'{model:20s} | BF: {bf_score*100:5.1f}% | IUE: {iue_score*100 if iue_score else "N/A":>6s} | BSM: {bsm_score*100 if bsm_score else "N/A":>6s}')

    print('='*80)
    print('\nSummary:')
    print(f'  Models tested: {len(results)}')

    if len(results) > 1:
        bf_scores = [r['bf'] for r in results]
        print(f'  BF Score Range: {min(bf_scores)*100:.1f}% - {max(bf_scores)*100:.1f}%')

        if all(abs(r['bf'] - results[0]['bf']) < 0.001 for r in results):
            print('  [WARNING] All models have IDENTICAL BF scores!')
        else:
            print('  [OK] v2.0 evaluator DIFFERENTIATES BF scores!')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
