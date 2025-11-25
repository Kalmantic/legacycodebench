#!/usr/bin/env python3
"""Create reference documentation template for experts"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from legacycodebench.tasks import Task
from legacycodebench.config import TASKS_DIR, REFERENCES_DIR


def create_documentation_template(task: Task, expert_id: str) -> str:
    """Create template for documentation reference"""
    
    template = f"""# Reference Documentation: {task.task_id}

**Expert ID:** {expert_id}
**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Time Started:** [Record when you start]
**Time Completed:** [Record when you finish]

## Instructions

1. Read the COBOL source code carefully
2. Fill in each section below based on your analysis
3. Be comprehensive - document everything you find
4. If unsure, note it in "Additional Notes"
5. Do NOT look at other experts' work until after submission

---

## Business Purpose

[What does this program do? Why does it exist? What business problem does it solve?]

Example:
"This program processes daily credit card transactions, applying interest calculations
and generating monthly billing statements for cardholders."

---

## Business Rules

[List ALL business rules, conditions, and logic. Use this format for each:]

### Rule 1: [Rule Name]
- **Description:** [What is this rule?]
- **Condition:** [When does this apply?]
- **Action:** [What happens?]
- **Implementation:** [Which COBOL paragraph/section?]
- **Edge cases:** [What unusual situations are handled?]

### Rule 2: [Rule Name]
...

[Continue for all rules found in the code]

---

## Edge Cases

[Document all exception conditions, error handling, boundary cases:]

### Edge Case 1: [Case Name]
- **Trigger:** [What causes this situation?]
- **Detection:** [How is it detected in code?]
- **Handling:** [What action is taken?]
- **Result:** [What is the outcome?]
- **COBOL Location:** [Paragraph/line number]

### Edge Case 2: [Case Name]
...

---

## Data Structures

[Document all data structures, records, and fields:]

### [Structure Name 1]

**Purpose:** [What is this structure used for?]

**Fields:**
- **FIELD-NAME-1** (PIC X(10)): [Description, constraints, valid values]
- **FIELD-NAME-2** (PIC 9(5)V99): [Description, constraints, valid values]
- **FIELD-NAME-3** (PIC X): [Description, constraints, valid values]

**Relationships:** [How does this relate to other structures?]

### [Structure Name 2]
...

---

## Algorithm Overview

[Describe the main processing flow step-by-step:]

### Main Process Flow

1. **Initialization**
   - [What happens first?]
   - COBOL: [Paragraph name]

2. **Input Processing**
   - [How is input read/validated?]
   - COBOL: [Paragraph name]

3. **Business Logic**
   - [What calculations/decisions are made?]
   - COBOL: [Paragraph name]

4. **Output Generation**
   - [What output is produced?]
   - COBOL: [Paragraph name]

5. **Cleanup/Termination**
   - [How does it end?]
   - COBOL: [Paragraph name]

### Sub-Processes

[If there are significant sub-processes, document them here]

---

## File Operations

[Document all file I/O:]

### Input Files
- **[FILE-NAME-1]**: [Purpose, format, when read]
- **[FILE-NAME-2]**: [Purpose, format, when read]

### Output Files
- **[FILE-NAME-1]**: [Purpose, format, when written]
- **[FILE-NAME-2]**: [Purpose, format, when written]

---

## Dependencies

[Document external dependencies:]

### Called Programs
- **[PROGRAM-1]**: [Purpose, when called, parameters]
- **[PROGRAM-2]**: [Purpose, when called, parameters]

### Copybooks Used
- **[COPYBOOK-1]**: [What it contains, why needed]
- **[COPYBOOK-2]**: [What it contains, why needed]

---

## Error Handling

[Document all error conditions and how they're handled:]

### Error Condition 1: [Error Name]
- **Trigger:** [What causes this error?]
- **Detection:** [How is it detected?]
- **Handling:** [What action is taken?]
- **Error Code:** [If applicable]

### Error Condition 2: [Error Name]
...

---

## Additional Notes

[Any observations, assumptions, ambiguities, or uncertainties:]

- [Note 1]
- [Note 2]
- [Areas where code is unclear]
- [Assumptions made]
- [Questions for other experts]

---

## Self-Assessment

**Confidence Level:** [High/Medium/Low]

**Areas of Uncertainty:**
- [List any parts you're not sure about]

**Estimated Code Coverage:** [What % of the code did you understand and document?]

**Would you use this documentation?** [Yes/No and why]

---

## Source Code Reference

**Task ID:** {task.task_id}
**Input Files:** {', '.join(task.input_files)}
**COBOL File Location:** [Record where you found the source]

---

## Completion Checklist

Before submitting, verify:

- [ ] All business rules documented
- [ ] All edge cases identified
- [ ] All data structures explained
- [ ] Algorithm flow is complete
- [ ] File I/O operations listed
- [ ] Dependencies documented
- [ ] Error handling covered
- [ ] Spent at least 2 hours on analysis
- [ ] Confident in accuracy
- [ ] Ready for independent review

---

**Submission Date:** [Fill in when complete]
**Total Time Spent:** [Fill in hours]
"""
    
    return template


def create_understanding_template(task: Task, expert_id: str) -> str:
    """Create template for understanding reference"""
    
    template = f"""# Reference Analysis: {task.task_id}

**Expert ID:** {expert_id}
**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Time Started:** [Record when you start]
**Time Completed:** [Record when you finish]

## Instructions

1. Read all COBOL source files carefully
2. Extract complete dependency graph, business rules, and data flow
3. Output in JSON format (see below)
4. Be exhaustive - capture everything
5. Do NOT look at other experts' work until after submission

---

## Extracted Structure

```json
{{
  "dependencies": [
    {{
      "type": "CALL",
      "source": "[calling-program.cbl]",
      "target": "[called-program]",
      "location": "[line number or paragraph]",
      "parameters": "[if known]",
      "purpose": "[what does this call do?]"
    }},
    {{
      "type": "COPY",
      "source": "[program.cbl]",
      "target": "[copybook]",
      "location": "[line number]",
      "purpose": "[what does this copybook provide?]"
    }}
  ],
  
  "business_rules": [
    {{
      "id": "RULE-001",
      "condition": "[Full IF condition from COBOL]",
      "action": "[What happens when true]",
      "else_action": "[What happens when false]",
      "location": "[paragraph name or line number]",
      "explanation": "[Plain English explanation]"
    }}
  ],
  
  "data_flow": [
    {{
      "operation": "OPEN",
      "file": "[FILE-NAME]",
      "mode": "[INPUT/OUTPUT/I-O]",
      "location": "[paragraph name]",
      "purpose": "[why is this file opened?]"
    }},
    {{
      "operation": "READ",
      "file": "[FILE-NAME]",
      "into": "[RECORD-NAME]",
      "location": "[paragraph name]",
      "purpose": "[what data is being read?]"
    }},
    {{
      "operation": "WRITE",
      "file": "[FILE-NAME]",
      "from": "[RECORD-NAME]",
      "location": "[paragraph name]",
      "purpose": "[what data is being written?]"
    }},
    {{
      "operation": "CLOSE",
      "file": "[FILE-NAME]",
      "location": "[paragraph name]"
    }}
  ],
  
  "program_flow": [
    {{
      "sequence": 1,
      "paragraph": "[PARAGRAPH-NAME]",
      "description": "[What this paragraph does]",
      "calls_to": ["[other paragraphs called from here]"],
      "critical": true
    }}
  ]
}}
```

---

## Analysis Notes

### Complexity Assessment

**Overall Complexity:** [Simple/Moderate/Complex]

**Factors:**
- Number of CALL statements: [count]
- Number of COPY statements: [count]
- Number of IF statements: [count]
- Number of file operations: [count]
- Maximum nesting depth: [depth]
- Cyclomatic complexity: [estimated value]

### Dependency Graph Description

[Describe the overall structure:]

Example:
"The main program MAINPROG calls 3 subprograms: VALIDATE, PROCESS, and OUTPUT.
It uses 2 copybooks: ACCOUNT-REC and ERROR-CODES. The dependency chain is
linear with no circular dependencies."

### Data Flow Description

[Describe how data moves through the system:]

Example:
"The program reads from INPUT-FILE, processes each record through validation
and calculation routines, and writes results to OUTPUT-FILE. Errors are
logged to ERROR-FILE."

### Business Logic Summary

[High-level summary of business rules:]

Example:
"The program implements 8 business rules related to account validation,
transaction limits, and balance calculations. Most rules are simple conditionals,
with 2 complex multi-condition rules in the interest calculation section."

---

## Ambiguities and Assumptions

[Document anything unclear:]

- [Ambiguity 1: Description and how you resolved it]
- [Assumption 1: What you assumed and why]
- [Questions for other experts]

---

## Validation Checklist

Before submitting:

- [ ] All CALL statements found and documented
- [ ] All COPY statements found and documented
- [ ] All IF statements found and documented
- [ ] All file operations found and documented
- [ ] Dependencies form complete graph
- [ ] Business rules have clear explanations
- [ ] Data flow is traceable end-to-end
- [ ] Spent at least 2 hours on analysis
- [ ] Confident in completeness
- [ ] JSON is valid

---

## Metadata

**Source Files Analyzed:** {', '.join(task.input_files)}
**Lines of Code:** [Approximate total]
**Analysis Approach:** [How did you approach this?]
**Tools Used:** [Any tools used for analysis?]

**Submission Date:** [Fill in when complete]
**Total Time Spent:** [Fill in hours]
"""
    
    return template


def main():
    """Generate reference template"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create reference documentation template')
    parser.add_argument('--task-id', required=True, help='Task ID (e.g., LCB-DOC-001)')
    parser.add_argument('--expert-id', required=True, help='Expert identifier')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    # Load task
    try:
        task = Task.load(args.task_id, TASKS_DIR)
    except FileNotFoundError:
        print(f"✗ Task {args.task_id} not found")
        return 1
    
    # Create appropriate template
    if task.category == "documentation":
        template = create_documentation_template(task, args.expert_id)
        ext = ".md"
    else:  # understanding
        template = create_understanding_template(task, args.expert_id)
        ext = ".md"
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        ref_dir = REFERENCES_DIR / task.category / args.task_id
        ref_dir.mkdir(parents=True, exist_ok=True)
        output_path = ref_dir / f"{args.expert_id}{ext}"
    
    # Write template
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"✓ Created reference template:")
    print(f"  Task: {args.task_id} ({task.category})")
    print(f"  Expert: {args.expert_id}")
    print(f"  File: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Send {output_path} to expert")
    print(f"  2. Expert fills in all sections")
    print(f"  3. Expert returns completed file")
    print(f"  4. Run agreement calculation when all experts done")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

