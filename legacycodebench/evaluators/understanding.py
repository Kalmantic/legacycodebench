"""Understanding task evaluator"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import re
import logging

from legacycodebench.config import REFERENCES_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnderstandingEvaluator:
    """Evaluate understanding task submissions (dependency graphs, business rules, data flow)"""
    
    def evaluate(self, submission_path: Path, task) -> Dict:
        """Evaluate an understanding submission"""
        if not submission_path.exists():
            return {
                "score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "errors": ["Submission file not found"],
            }
        
        try:
            with open(submission_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse as JSON
            try:
                submission_data = json.loads(content)
            except json.JSONDecodeError:
                # If not JSON, try to extract structure from text
                submission_data = self._parse_text_submission(content)
        except Exception as e:
            return {
                "score": 0.0,
                "errors": [f"Failed to read submission: {e}"],
            }
        
        # Check for reference documentation
        reference_path = self._find_reference(task)
        if reference_path and reference_path.exists():
            logger.info(f"Using reference documentation: {reference_path}")
            return self._evaluate_with_reference(submission_data, reference_path, task)
        else:
            logger.info("No reference documentation found, using COBOL extraction")
            return self._evaluate_without_reference(submission_data, task)
    
    def _evaluate_without_reference(self, submission_data: Dict, task) -> Dict:
        """Evaluate using COBOL extraction when no reference is available"""
        # Extract ground truth from COBOL files
        ground_truth = self._extract_ground_truth(task)
        
        # Evaluate different aspects
        dependency_score = self._evaluate_dependencies(submission_data, ground_truth)
        business_rules_score = self._evaluate_business_rules(submission_data, ground_truth)
        data_flow_score = self._evaluate_data_flow(submission_data, ground_truth)
        
        # Combined F1 score
        f1_score = (dependency_score["f1"] + business_rules_score["f1"] + data_flow_score["f1"]) / 3.0
        
        return {
            "score": round(f1_score, 4),
            "f1": round(f1_score, 4),
            "precision": round((dependency_score["precision"] + business_rules_score["precision"] + data_flow_score["precision"]) / 3.0, 4),
            "recall": round((dependency_score["recall"] + business_rules_score["recall"] + data_flow_score["recall"]) / 3.0, 4),
            "dependency": dependency_score,
            "business_rules": business_rules_score,
            "data_flow": data_flow_score,
            "evaluation_method": "cobol_extraction",
        }
    
    def _evaluate_with_reference(self, submission_data: Dict, reference_path: Path, task) -> Dict:
        """Evaluate using reference documentation"""
        try:
            with open(reference_path, 'r', encoding='utf-8') as f:
                reference_content = f.read()
            
            # Try to parse as JSON
            try:
                reference_data = json.loads(reference_content)
            except json.JSONDecodeError:
                # If not JSON, try to extract from markdown
                reference_data = self._parse_reference_markdown(reference_content)
        except Exception as e:
            logger.warning(f"Failed to read reference: {e}, falling back to COBOL extraction")
            return self._evaluate_without_reference(submission_data, task)
        
        # Evaluate against reference
        dependency_score = self._evaluate_dependencies(submission_data, reference_data)
        business_rules_score = self._evaluate_business_rules(submission_data, reference_data)
        data_flow_score = self._evaluate_data_flow(submission_data, reference_data)
        
        # Combined F1 score
        f1_score = (dependency_score["f1"] + business_rules_score["f1"] + data_flow_score["f1"]) / 3.0
        
        return {
            "score": round(f1_score, 4),
            "f1": round(f1_score, 4),
            "precision": round((dependency_score["precision"] + business_rules_score["precision"] + data_flow_score["precision"]) / 3.0, 4),
            "recall": round((dependency_score["recall"] + business_rules_score["recall"] + data_flow_score["recall"]) / 3.0, 4),
            "dependency": dependency_score,
            "business_rules": business_rules_score,
            "data_flow": data_flow_score,
            "evaluation_method": "reference",
        }
    
    def _parse_text_submission(self, content: str) -> Dict:
        """Parse a text submission into structured format"""
        data = {
            "dependencies": [],
            "business_rules": [],
            "data_flow": [],
        }
        
        # Try to extract dependencies
        if "CALL" in content.upper() or "dependency" in content.lower():
            # Look for program names
            call_matches = re.findall(r'CALL\s+[\'"]?(\w+)[\'"]?', content, re.IGNORECASE)
            data["dependencies"] = [{"type": "CALL", "target": name} for name in call_matches]
        
        # Try to extract business rules
        if "rule" in content.lower() or "IF" in content:
            # Look for conditional statements
            if_matches = re.findall(r'IF\s+(.+?)\s+THEN', content, re.IGNORECASE)
            data["business_rules"] = [{"condition": match.strip()} for match in if_matches]
        
        return data
    
    def _extract_ground_truth(self, task) -> Dict:
        """Extract ground truth from COBOL source files"""
        ground_truth = {
            "dependencies": [],
            "business_rules": [],
            "data_flow": [],
        }
        
        # Read COBOL files and extract structure
        for file_path_str in task.input_files:
            # Find the actual file
            from legacycodebench.config import DATASETS_DIR
            file_path = None
            for dataset_dir in DATASETS_DIR.iterdir():
                if dataset_dir.is_dir():
                    potential_path = dataset_dir / file_path_str
                    if potential_path.exists():
                        file_path = potential_path
                        break
            
            if file_path and file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        cobol_content = f.read()
                    
                    # Extract CALL statements
                    call_matches = re.findall(r'CALL\s+[\'"]?(\w+)[\'"]?', cobol_content, re.IGNORECASE)
                    for match in call_matches:
                        ground_truth["dependencies"].append({
                            "type": "CALL",
                            "source": file_path.name,
                            "target": match,
                        })
                    
                    # Extract COPY statements
                    copy_matches = re.findall(r'COPY\s+(\w+)', cobol_content, re.IGNORECASE)
                    for match in copy_matches:
                        ground_truth["dependencies"].append({
                            "type": "COPY",
                            "source": file_path.name,
                            "target": match,
                        })
                    
                    # Extract business rules (IF statements)
                    if_matches = re.findall(r'IF\s+(.+?)\s+(THEN|NEXT|CONTINUE)', cobol_content, re.IGNORECASE | re.DOTALL)
                    for match in if_matches:
                        ground_truth["business_rules"].append({
                            "condition": match[0].strip()[:100],  # First 100 chars
                        })
                    
                    # Extract file I/O (data flow)
                    file_matches = re.findall(r'(OPEN|READ|WRITE|CLOSE)\s+(\w+)', cobol_content, re.IGNORECASE)
                    for match in file_matches:
                        ground_truth["data_flow"].append({
                            "operation": match[0],
                            "file": match[1],
                        })
                except Exception as e:
                    logger.warning(f"Failed to parse {file_path}: {e}")
        
        return ground_truth
    
    def _evaluate_dependencies(self, submission: Dict, ground_truth: Dict) -> Dict:
        """Evaluate dependency extraction (F1 score)"""
        pred_deps = set()
        true_deps = set()
        
        # Extract predicted dependencies
        for dep in submission.get("dependencies", []):
            if isinstance(dep, dict):
                target = dep.get("target", "")
                if target:
                    pred_deps.add(target.upper())
            elif isinstance(dep, str):
                pred_deps.add(dep.upper())
        
        # Extract true dependencies
        for dep in ground_truth.get("dependencies", []):
            target = dep.get("target", "")
            if target:
                true_deps.add(target.upper())
        
        # Calculate precision, recall, F1
        if len(pred_deps) == 0 and len(true_deps) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        if len(pred_deps) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        true_positives = len(pred_deps & true_deps)
        precision = true_positives / len(pred_deps) if len(pred_deps) > 0 else 0.0
        recall = true_positives / len(true_deps) if len(true_deps) > 0 else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "predicted": len(pred_deps),
            "actual": len(true_deps),
            "correct": true_positives,
        }
    
    def _evaluate_business_rules(self, submission: Dict, ground_truth: Dict) -> Dict:
        """Evaluate business rule extraction (F1 score)"""
        # Simplified: count rules found
        pred_rules = len(submission.get("business_rules", []))
        true_rules = len(ground_truth.get("business_rules", []))
        
        if pred_rules == 0 and true_rules == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        if pred_rules == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Heuristic: assume some overlap
        # In real implementation, would do semantic matching
        overlap = min(pred_rules, true_rules) * 0.7  # Assume 70% overlap
        
        precision = overlap / pred_rules if pred_rules > 0 else 0.0
        recall = overlap / true_rules if true_rules > 0 else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "predicted": pred_rules,
            "actual": true_rules,
        }
    
    def _evaluate_data_flow(self, submission: Dict, ground_truth: Dict) -> Dict:
        """Evaluate data flow extraction (F1 score)"""
        pred_flows = set()
        true_flows = set()
        
        # Extract predicted data flows
        for flow in submission.get("data_flow", []):
            if isinstance(flow, dict):
                file = flow.get("file", "")
                if file:
                    pred_flows.add(file.upper())
            elif isinstance(flow, str):
                pred_flows.add(flow.upper())
        
        # Extract true data flows
        for flow in ground_truth.get("data_flow", []):
            file = flow.get("file", "")
            if file:
                true_flows.add(file.upper())
        
        # Calculate precision, recall, F1
        if len(pred_flows) == 0 and len(true_flows) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        if len(pred_flows) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        true_positives = len(pred_flows & true_flows)
        precision = true_positives / len(pred_flows) if len(pred_flows) > 0 else 0.0
        recall = true_positives / len(true_flows) if len(true_flows) > 0 else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "predicted": len(pred_flows),
            "actual": len(true_flows),
            "correct": true_positives,
        }
    
    def _find_reference(self, task) -> Optional[Path]:
        """Find reference documentation for a task"""
        ref_dir = REFERENCES_DIR / "understanding" / task.task_id
        # Look for reference.json (consensus version)
        reference_path = ref_dir / "reference.json"
        if reference_path.exists():
            return reference_path
        # Fallback: look for any .json file in the directory
        if ref_dir.exists():
            json_files = list(ref_dir.glob("*.json"))
            if json_files:
                return json_files[0]
        return None
    
    def _parse_reference_markdown(self, content: str) -> Dict:
        """Parse reference documentation from markdown format"""
        data = {
            "dependencies": [],
            "business_rules": [],
            "data_flow": [],
        }
        
        # Extract dependencies section
        deps_match = re.search(
            r'(?:^#+\s+.*dependenc.*$)(.*?)(?:^#+\s+|$)',
            content,
            re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        if deps_match:
            deps_text = deps_match.group(1)
            # Look for CALL, COPY statements
            call_matches = re.findall(r'CALL\s+[\'"]?(\w+)[\'"]?', deps_text, re.IGNORECASE)
            copy_matches = re.findall(r'COPY\s+(\w+)', deps_text, re.IGNORECASE)
            for match in call_matches:
                data["dependencies"].append({"type": "CALL", "target": match})
            for match in copy_matches:
                data["dependencies"].append({"type": "COPY", "target": match})
        
        # Extract business rules section
        rules_match = re.search(
            r'(?:^#+\s+.*business\s+rule.*$)(.*?)(?:^#+\s+|$)',
            content,
            re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        if rules_match:
            rules_text = rules_match.group(1)
            # Extract rule items
            rule_items = re.findall(r'(?:\d+\.|\*|\-)\s+(.+?)(?=\n(?:\d+\.|\*|\-|\n\n|$))', rules_text, re.MULTILINE)
            for item in rule_items:
                if len(item.strip()) > 10:
                    data["business_rules"].append({"condition": item.strip()[:200]})
        
        # Extract data flow section
        flow_match = re.search(
            r'(?:^#+\s+.*data\s+flow.*$)(.*?)(?:^#+\s+|$)',
            content,
            re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        if flow_match:
            flow_text = flow_match.group(1)
            # Look for file operations
            file_matches = re.findall(r'(OPEN|READ|WRITE|CLOSE)\s+(\w+)', flow_text, re.IGNORECASE)
            for match in file_matches:
                data["data_flow"].append({"operation": match[0], "file": match[1]})
        
        return data

