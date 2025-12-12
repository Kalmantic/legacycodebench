"""Re-evaluate DocMolt submissions with v2.0 evaluator"""
from pathlib import Path
from legacycodebench.tasks import Task, TaskManager
from legacycodebench.evaluators_v2.documentation_v2 import DocumentationEvaluatorV2
from legacycodebench.config import RESULTS_DIR, SUBMISSIONS_DIR, TASKS_DIR
import json

# Re-evaluate DocMolt submissions with v2.0 evaluator
submissions = [
    'LCB-DOC-001_docmolt-gpt4o.md',
    'LCB-DOC-002_docmolt-gpt4o.md'
]

manager = TaskManager()
evaluator = DocumentationEvaluatorV2(enable_execution=False)

for sub_file in submissions:
    sub_path = SUBMISSIONS_DIR / sub_file
    if not sub_path.exists():
        print(f'Submission not found: {sub_path}')
        continue
    
    # Extract task_id from filename
    task_id = sub_file.split('_')[0]  # LCB-DOC-001
    
    try:
        task = manager.get_task(task_id)
        print(f'Evaluating {task_id} with v2.0 evaluator...')
        
        result = evaluator.evaluate(sub_path, task)
        
        # Create result entry
        result_entry = {
            'task_id': task_id,
            'task_category': task.category,
            'evaluator_version': 'v2',
            'submitter': {
                'name': 'Docmolt',
                'model': 'docmolt-gpt4o',
                'category': 'verified',
            },
            'result': result,
            'overall_score': result.get('score', 0),
        }
        
        # Save result
        result_file = RESULTS_DIR / f'{task_id}_docmolt-gpt4o_v2.json'
        with open(result_file, 'w') as f:
            json.dump(result_entry, f, indent=2)
        
        sc = result.get('structural_completeness', 0) * 100
        bf = result.get('behavioral_fidelity', 0) * 100
        sq = result.get('semantic_quality', 0) * 100
        tr = result.get('traceability', 0) * 100
        score = result.get('score', 0) * 100
        
        print(f'  Score: {score:.1f}% (SC:{sc:.0f}% BF:{bf:.0f}% SQ:{sq:.0f}% TR:{tr:.0f}%)')
        print(f'  Saved to: {result_file}')
        
    except FileNotFoundError as e:
        print(f'Task not found: {task_id}')
    except Exception as e:
        print(f'Error evaluating {task_id}: {e}')
        import traceback
        traceback.print_exc()

print('Done!')
