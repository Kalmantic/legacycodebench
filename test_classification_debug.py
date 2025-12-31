"""Debug paragraph classification"""
from pathlib import Path
from legacycodebench.tasks import TaskManager
from legacycodebench.evaluators_v231.paragraph_classifier_v231 import ParagraphClassifier
from legacycodebench.config import GROUND_TRUTH_CACHE_DIR
import json

task_id = 'LCB-T1-001'

manager = TaskManager()
task = manager.get_task(task_id)

# Get ground truth
input_files = task.get_input_files_absolute()
gt_file = GROUND_TRUTH_CACHE_DIR / f"{input_files[0].stem}_ground_truth.json"

with open(gt_file, 'r') as f:
    ground_truth = json.load(f)

# Extract paragraphs
control_flow = ground_truth.get("control_flow", {})
paragraphs_data = control_flow.get("paragraphs", [])

print(f'Task: {task_id}')
print(f'Total paragraphs in ground truth: {len(paragraphs_data)}\n')

# Classify
classifier = ParagraphClassifier()
classified = classifier.classify_all(paragraphs_data)

print(f'Classification Results:')
print(f'  PURE: {len(classified["pure"])}')
print(f'  MIXED: {len(classified["mixed"])}')
print(f'  INFRASTRUCTURE: {len(classified["infrastructure"])}\n')

print('='*80)
print('Paragraph Details:\n')

for para in paragraphs_data:  # Show all paragraphs
    name = para.get('name', 'UNKNOWN')
    content = para.get('content', '')
    start_line = para.get('start_line', 0)
    end_line = para.get('end_line', 0)

    # Find which category it's in
    para_type = 'UNKNOWN'
    for cat_name, cat_paras in classified.items():
        if any(p.name == name for p in cat_paras):
            para_type = cat_name.upper()
            break

    print(f'{name:30s} | {para_type:15s} | Lines {start_line}-{end_line}')
    if content:
        # Show full content (truncated at 500 chars)
        print(f'  Content:')
        lines = content.split('\n')[:15]  # First 15 lines
        for line in lines:
            print(f'    {line}')
        if len(content.split('\n')) > 15:
            print(f'    ... ({len(content.split(chr(10))) - 15} more lines)')
    print()
