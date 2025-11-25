"""AI model integration for LegacyCodeBench"""

import os
from pathlib import Path
from typing import Dict, Optional, List
import json
import logging

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from legacycodebench.config import AI_MODELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIModelInterface:
    """Interface for AI models"""
    
    def __init__(self, model_id: str):
        if model_id not in AI_MODELS:
            raise ValueError(f"Unknown model: {model_id}")
        
        self.model_id = model_id
        self.config = AI_MODELS[model_id]
        self.provider = self.config["provider"]
    
    def generate_documentation(self, task, input_files: List[Path]) -> str:
        """Generate documentation for a task"""
        # Read input files
        code_content = ""
        for file_path in input_files:
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code_content += f"\n\n=== {file_path.name} ===\n\n"
                        code_content += f.read()
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
        
        # Create prompt
        prompt = self._create_documentation_prompt(task, code_content)
        
        # Call appropriate API (check for API keys first)
        if self.provider == "openai" and OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            return self._call_openai(prompt)
        elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            return self._call_anthropic(prompt)
        elif self.provider == "aws":
            return self._call_aws_transform(prompt)
        else:
            # Fallback: return mock response
            return self._generate_mock_documentation(task)
    
    def generate_understanding(self, task, input_files: List[Path]) -> str:
        """Generate understanding output (JSON) for a task"""
        # Read input files
        code_content = ""
        for file_path in input_files:
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code_content += f"\n\n=== {file_path.name} ===\n\n"
                        code_content += f.read()
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
        
        # Create prompt
        prompt = self._create_understanding_prompt(task, code_content)
        
        # Call appropriate API (check for API keys first)
        if self.provider == "openai" and OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            response = self._call_openai(prompt)
            # Try to extract JSON from response
            return self._extract_json(response)
        elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            response = self._call_anthropic(prompt)
            return self._extract_json(response)
        elif self.provider == "aws":
            response = self._call_aws_transform(prompt)
            return self._extract_json(response)
        else:
            # Fallback: return mock response
            return self._generate_mock_understanding(task, input_files)
    
    def _create_documentation_prompt(self, task, code_content: str) -> str:
        """Create prompt for documentation task"""
        return f"""You are a technical documentation expert. Analyze the following COBOL code and generate comprehensive documentation.

Task: {task.task_description}

COBOL Code:
{code_content[:5000]}  # Limit to avoid token limits

Requirements:
- Explain business purpose
- Document all business rules
- Identify edge cases
- Describe data structures
- Use Markdown format
- Minimum 3-5 pages of documentation

Generate the documentation now:"""
    
    def _create_understanding_prompt(self, task, code_content: str) -> str:
        """Create prompt for understanding task"""
        return f"""You are a code analysis expert. Analyze the following COBOL code and extract its structure.

Task: {task.task_description}

COBOL Code:
{code_content[:5000]}  # Limit to avoid token limits

Requirements:
- Extract dependency graph (CALL and COPY relationships)
- Identify business rules (IF conditions, logic)
- Map data flow (file I/O operations)
- Output as JSON following this schema:
{{
  "dependencies": [{{"type": "CALL|COPY", "source": "...", "target": "..."}}],
  "business_rules": [{{"condition": "..."}}],
  "data_flow": [{{"operation": "OPEN|READ|WRITE|CLOSE", "file": "..."}}]
}}

Generate the JSON output now:"""
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, using mock response")
            return self._generate_mock_response()
        
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}, using mock response")
            return self._generate_mock_response()
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set, using mock response")
            return self._generate_mock_response()
        
        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=self.config["model"],
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}, using mock response")
            return self._generate_mock_response()
    
    def _call_aws_transform(self, prompt: str) -> str:
        """Call AWS Transform (placeholder - would need actual API)"""
        logger.warning("AWS Transform API not implemented, using mock response")
        return self._generate_mock_response()
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text response"""
        # Try to find JSON block
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # If no JSON found, return as-is (evaluator will handle)
        return text
    
    def _generate_mock_documentation(self, task) -> str:
        """Generate mock documentation"""
        return f"""# Documentation for {task.task_id}

## Business Purpose

This COBOL program handles financial transactions in a legacy banking system. It processes account operations including deposits, withdrawals, and interest calculations.

## Business Rules

1. **Interest Calculation Rule**: Interest is calculated monthly at a rate of 2.5% for savings accounts.
2. **Minimum Balance Rule**: Accounts must maintain a minimum balance of $100.
3. **Transaction Limit Rule**: Maximum single transaction amount is $10,000.
4. **Overdraft Rule**: Overdrafts are allowed up to $500 with a fee of $25.

## Edge Cases

- **Zero Balance**: Accounts with zero balance are flagged for review.
- **Negative Balance**: Accounts exceeding overdraft limit are frozen.
- **Large Transactions**: Transactions over $5,000 require additional verification.

## Data Structures

### Account Record
- ACCOUNT-NUMBER: 10-digit account identifier
- ACCOUNT-TYPE: Type of account (CHECKING, SAVINGS)
- BALANCE: Current account balance
- INTEREST-RATE: Annual interest rate percentage

### Transaction Record
- TRANSACTION-ID: Unique transaction identifier
- TRANSACTION-TYPE: Type (DEPOSIT, WITHDRAWAL, TRANSFER)
- AMOUNT: Transaction amount
- TIMESTAMP: Date and time of transaction

## Algorithm Overview

1. Read account information
2. Validate transaction request
3. Check business rules (balance, limits)
4. Process transaction
5. Update account balance
6. Calculate interest if applicable
7. Write updated records
8. Generate confirmation
"""
    
    def _generate_mock_understanding(self, task, input_files: List[Path]) -> str:
        """Generate mock understanding output"""
        # Extract some actual dependencies from files
        dependencies = []
        business_rules = []
        data_flow = []
        
        for file_path in input_files:
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Extract CALL statements
                    import re
                    calls = re.findall(r'CALL\s+[\'"]?(\w+)[\'"]?', content, re.IGNORECASE)
                    for call in calls:
                        dependencies.append({
                            "type": "CALL",
                            "source": file_path.name,
                            "target": call
                        })
                    
                    # Extract COPY statements
                    copies = re.findall(r'COPY\s+(\w+)', content, re.IGNORECASE)
                    for copy in copies:
                        dependencies.append({
                            "type": "COPY",
                            "source": file_path.name,
                            "target": copy
                        })
                    
                    # Extract file operations
                    files = re.findall(r'(OPEN|READ|WRITE|CLOSE)\s+(\w+)', content, re.IGNORECASE)
                    for op, file in files:
                        data_flow.append({
                            "operation": op.upper(),
                            "file": file.upper()
                        })
                except:
                    pass
        
        return json.dumps({
            "dependencies": dependencies[:10],  # Limit output
            "business_rules": [
                {"condition": "IF BALANCE < MINIMUM-BALANCE THEN"},
                {"condition": "IF TRANSACTION-AMOUNT > MAX-TRANSACTION THEN"},
            ],
            "data_flow": data_flow[:10],
        }, indent=2)
    
    def _generate_mock_response(self) -> str:
        """Generate generic mock response"""
        return "Mock AI response (API key not configured)"


def get_ai_model(model_id: str) -> AIModelInterface:
    """Get AI model interface"""
    return AIModelInterface(model_id)

