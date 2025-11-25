#!/usr/bin/env python3
"""Test script for intelligent task selection system"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from legacycodebench.tasks import TaskManager
from legacycodebench.config import DATASETS_DIR

def main():
    """Test intelligent task selection"""
    print("=" * 80)
    print("Testing Intelligent Task Selection System")
    print("=" * 80)
    
    manager = TaskManager()
    
    # Test intelligent selection
    print("\n[1/2] Running intelligent task selection...")
    try:
        tasks = manager.create_tasks_intelligent()
        
        print(f"\n✓ Created {len(tasks)} tasks\n")
        
        # Analyze results
        doc_tasks = [t for t in tasks if t.category == "documentation"]
        und_tasks = [t for t in tasks if t.category == "understanding"]
        
        print(f"Documentation tasks: {len(doc_tasks)}")
        print(f"Understanding tasks: {len(und_tasks)}")
        
        # Difficulty distribution
        difficulties = {}
        for task in tasks:
            difficulties[task.difficulty] = difficulties.get(task.difficulty, 0) + 1
        
        print(f"\nDifficulty distribution:")
        for diff, count in sorted(difficulties.items()):
            print(f"  {diff}: {count} tasks")
        
        # Domain distribution
        domains = {}
        for task in tasks:
            domains[task.domain] = domains.get(task.domain, 0) + 1
        
        print(f"\nDomain distribution:")
        for domain, count in sorted(domains.items()):
            print(f"  {domain}: {count} tasks")
        
        # Multi-file tasks
        multi_file = [t for t in tasks if len(t.input_files) > 1]
        print(f"\nMulti-file tasks: {len(multi_file)}/{len(tasks)}")
        
        # Show sample tasks
        print(f"\nSample tasks:")
        for task in tasks[:3]:
            print(f"\n  {task.task_id} ({task.category} - {task.difficulty} - {task.domain})")
            print(f"    Files: {len(task.input_files)}")
            for file in task.input_files[:2]:
                print(f"      - {file}")
        
        print("\n✓ Intelligent selection test passed!")
        
    except Exception as e:
        print(f"\n✗ Intelligent selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test simple selection
    print("\n[2/2] Running simple task selection (for comparison)...")
    try:
        simple_tasks = manager.create_tasks_simple()
        print(f"✓ Simple selection created {len(simple_tasks)} tasks")
        
        # Quick comparison
        simple_diff = {}
        for task in simple_tasks:
            simple_diff[task.difficulty] = simple_diff.get(task.difficulty, 0) + 1
        
        print(f"\nSimple difficulty distribution:")
        for diff, count in sorted(simple_diff.items()):
            print(f"  {diff}: {count} tasks")
        
    except Exception as e:
        print(f"\n✗ Simple selection test failed: {e}")
        return 1
    
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(main())

