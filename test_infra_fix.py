"""Test infrastructure paragraph evaluation fix"""
from pathlib import Path
from legacycodebench.tasks import TaskManager
from legacycodebench.evaluators_v231.evaluator_v231 import EvaluatorV231
from legacycodebench.config import SUBMISSIONS_DIR
import json

# Test with LCB-T1-001
task_id = 'LCB-T1-001'
submission_file = 'LCB-T1-001_docmolt-enterprise.md'

manager = TaskManager()
evaluator = EvaluatorV231(llm_client=None, executor=None)

sub_path = SUBMISSIONS_DIR / submission_file

try:
    task = manager.get_task(task_id)
    print(f'Testing {task_id} with fixed infrastructure evaluation...\n')

    # Load documentation
    with open(sub_path, 'r', encoding='utf-8') as f:
        documentation = f.read()

    # Get source code from task input files
    input_files = task.get_input_files_absolute()
    if not input_files:
        raise FileNotFoundError(f"No input files found for task {task_id}")

    source_code = ""
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source_code += f.read() + "\n"

    # Load ground truth
    from legacycodebench.config import GROUND_TRUTH_CACHE_DIR
    gt_file = GROUND_TRUTH_CACHE_DIR / f"{input_files[0].stem}_ground_truth.json"
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_file}")

    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)

    result = evaluator.evaluate(
        task_id=task_id,
        model='hexaview_insights',
        documentation=documentation,
        source_code=source_code,
        ground_truth=ground_truth
    )

    # Print results
    sc = result.sc_score * 100
    dq = result.dq_score * 100
    bf = result.bf_score * 100
    lcb = result.lcb_score

    print(f'LCB Score: {lcb:.1f}%')
    print(f'  SC: {sc:.1f}%')
    print(f'  DQ: {dq:.1f}%')
    print(f'  BF: {bf:.1f}%')
    print(f'  Passed: {result.passed}')

    # Show BF breakdown
    if result.bf_breakdown:
        bf_details = result.bf_breakdown
        print(f'\nBehavioral Fidelity Breakdown:')
        print(f'  Claim Score: {bf_details.get("claims", 0):.2f}')
        print(f'  BSM Score: {bf_details.get("bsm", 0):.2f}')
        print(f'  Pure Score: {bf_details.get("pure_score", 0):.2f} (weight: {bf_details.get("pure_weight", 0):.2f})')
        print(f'  Mixed Score: {bf_details.get("mixed_score", 0):.2f} (weight: {bf_details.get("mixed_weight", 0):.2f})')
        print(f'  Infrastructure Score: {bf_details.get("infrastructure_score", 0):.2f} (weight: {bf_details.get("infrastructure_weight", 0):.2f})')
        print(f'  Claims Verified: {bf_details.get("claims_verified", 0)}')
        print(f'  Claims Failed: {bf_details.get("claims_failed", 0)}')
        print(f'  BSM Matched: {bf_details.get("bsm_matched", 0)}/{bf_details.get("bsm_total", 0)}')

    print(f'\n[OK] Infrastructure paragraphs now use Claims + BSM combination')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
