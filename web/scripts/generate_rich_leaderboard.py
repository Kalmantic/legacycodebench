import json
import os
import glob
import re
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(r"C:\legacyCodeBenchmark\results")
OUTPUT_FILE = Path(r"C:\legacyCodeBenchmark\web\results\leaderboard.json")

def detect_language(task_id):
    if task_id.startswith("LCB-UB"):
        return "UniBasic"
    return "COBOL"

def get_tier(task_id):
    # LCB-T1-001 -> T1
    match = re.search(r"LCB-(UB-)?(T\d)", task_id)
    if match:
        return match.group(2)
    return "Unknown"

def calculate_averages(tasks):
    if not tasks:
        return {
            "avg_lcb_score": 0, "avg_sc": 0, "avg_dq": 0, "avg_bf": 0,
            "executed": 0, "static": 0, "verified_executed_pct": 0
        }
    
    total = len(tasks)
    lcb_sum = sum(t['lcb_score'] for t in tasks)
    sc_sum = sum(t['scores']['sc'] for t in tasks)
    dq_sum = sum(t['scores']['dq'] for t in tasks)
    bf_sum = sum(t['scores']['bf'] for t in tasks)
    
    executed = sum(1 for t in tasks if t['verification_mode'] == 'executed')
    static = sum(1 for t in tasks if t['verification_mode'] == 'static')
    
    return {
        "avg_lcb_score": lcb_sum / total / 100, # Convert 0-100 to 0-1
        "avg_sc": sc_sum / total / 100,
        "avg_dq": dq_sum / total / 100,
        "avg_bf": bf_sum / total / 100,
        "executed": executed,
        "static": static,
        "verified_executed_pct": (executed / total * 100) if total > 0 else 0
    }

def main():
    print(f"Scanning {RESULTS_DIR}...")
    
    # helper to find model name from filename
    # LCB-T1-001_IBM_ibm-granite-13b_v3.json
    # LCB-UB-T1-001_gpt-4o_v3.json
    
    data_by_model = defaultdict(list)
    
    files = list(RESULTS_DIR.glob("*_v3.json"))
    print(f"Found {len(files)} result files.")
    
    for f in files:
        try:
            with open(f, 'r') as fd:
                data = json.load(fd)
                
            task_id = data.get('task_id')
            submitter = data.get('submitter', {})
            model = submitter.get('model', 'unknown')
            
            # Extract scores
            result = data.get('result', {})
            scores = result.get('scores', {})
            breakdown = result.get('breakdown', {})
            bf_data = breakdown.get('bf', {})
            
            # Helper to get safe score
            def get_score(key):
                val = scores.get(key, 0)
                return float(val) if val is not None else 0.0

            # Improved Reason Extraction
            reason = bf_data.get('mode_reason', 'Unknown')
            details = f"Score: {get_score('bf')}%"
            
            if reason == "compile_error":
                raw_error = bf_data.get('compile_error') or bf_data.get('v3_provenance', {}).get('compile_error')
                if raw_error:
                    # Try to extract meaningful error
                    if "copybook" in raw_error.lower() or "not found" in raw_error.lower():
                        reason = "Missing Copybook/File"
                    elif "syntax error" in raw_error.lower():
                        reason = "Syntax Error"
                    elif "using clause" in raw_error.lower():
                        reason = "Procedure Using Clause"
                    else:
                        # Take first line of error, truncated
                        first_line = raw_error.split('\n')[0]
                        reason = (first_line[:40] + '..') if len(first_line) > 40 else first_line

            # Construct task entry
            task_entry = {
                "task_id": task_id,
                "language": detect_language(task_id),
                "tier": get_tier(task_id),
                "lcb_score": float(result.get('lcb_score', 0)),
                "bf_score": get_score('bf'), # Added to fix NaN issue
                "scores": {
                    "sc": get_score('sc'),
                    "dq": get_score('dq'),
                    "bf": get_score('bf')
                },
                "verification_mode": bf_data.get('verification_mode', 'error'),
                "mode_reason": reason,
                "details": details 
            }
            
            # improve details string
            if task_entry['verification_mode'] == 'executed':
                prov = bf_data.get('v3_provenance', {})
                if prov.get('execution_succeeded'):
                     task_entry['details'] = "Passed execution"
            elif task_entry['verification_mode'] == 'static':
                 task_entry['details'] = "Static analysis (Fallback)"
            
            data_by_model[model].append(task_entry)
            
        except Exception as e:
            print(f"Error processing {f.name}: {e}")

    # Build Leaderboard
    leaderboard = []
    
    for model, tasks in data_by_model.items():
        stats = calculate_averages(tasks)
        
        # Determine languages supported
        langs = list(set(t['language'] for t in tasks))
        
        entry = {
            "model": model,
            "avg_lcb_score": stats['avg_lcb_score'],
            "avg_sc": stats['avg_sc'],
            "avg_dq": stats['avg_dq'],
            "avg_bf": stats['avg_bf'],
            "executed": stats['executed'],
            "static": stats['static'],
            "languages": langs,
            "tasks": tasks # rich data for frontend
        }
        leaderboard.append(entry)

    # Sort by LCB Score
    leaderboard.sort(key=lambda x: x['avg_lcb_score'], reverse=True)
    
    # Assign ranks
    for i, entry in enumerate(leaderboard):
        entry['rank'] = i + 1

    output = {
        "generated_at": "2026-01-23T12:00:00Z", # You might want dynamic time
        "total_models": len(leaderboard),
        "leaderboard": leaderboard
    }
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
        
    print(f"Successfully generated {OUTPUT_FILE} with {len(leaderboard)} models.")

if __name__ == "__main__":
    main()
