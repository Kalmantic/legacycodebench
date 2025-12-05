"""AI model integration for LegacyCodeBench"""

import os
import re
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

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

from legacycodebench.config import AI_MODELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Token limits for different models (input context windows)
MODEL_CONTEXT_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "o1": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-opus-4-20250514": 200000,
    "gemini-1.5-pro": 1000000,
    "gemini-2.0-flash-exp": 1000000,
    "transform": 8000,
    "codemolt": 4000,
}

# Approximate chars per token (conservative estimate for COBOL)
CHARS_PER_TOKEN = 3.5

# Maximum input tokens to use (leave room for output)
MAX_INPUT_RATIO = 0.6  # Use 60% of context for input, leave 40% for output


class AIModelInterface:
    """Interface for AI models"""
    
    def __init__(self, model_id: str):
        if model_id not in AI_MODELS:
            raise ValueError(f"Unknown model: {model_id}")
        
        self.model_id = model_id
        self.config = AI_MODELS[model_id]
        self.provider = self.config["provider"]
        
        # Calculate max input chars based on model context window
        model_name = self.config["model"]
        context_limit = MODEL_CONTEXT_LIMITS.get(model_name, 8000)
        max_input_tokens = int(context_limit * MAX_INPUT_RATIO)
        self.max_input_chars = int(max_input_tokens * CHARS_PER_TOKEN)
        logger.info(f"Model {model_id}: context={context_limit}, max_input_chars={self.max_input_chars}")
    
    def generate_documentation(self, task, input_files: List[Path]) -> str:
        """Generate documentation for a task"""
        # Read input files
        code_content = self._read_input_files(input_files)
        
        # Create prompt with dynamic sizing
        prompt = self._create_documentation_prompt(task, code_content)
        
        # Call appropriate API (check for API keys first)
        if self.provider == "openai" and OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            return self._call_openai_with_continuation(prompt, is_documentation=True)
        elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            return self._call_anthropic_with_continuation(prompt, is_documentation=True)
        elif self.provider == "google" and GOOGLE_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
            return self._call_google(prompt)
        elif self.provider == "aws":
            return self._call_aws_transform(prompt)
        elif self.provider == "hexaview":
            logger.warning("CodeMolt (Hexaview) requires manual web UI - no API available")
            return self._generate_mock_documentation(task)
        else:
            # Fallback: return mock response
            return self._generate_mock_documentation(task)
    
    def generate_understanding(self, task, input_files: List[Path]) -> str:
        """Generate understanding output (JSON) for a task"""
        # Read input files
        code_content = self._read_input_files(input_files)

        # Create prompt with dynamic sizing
        prompt = self._create_understanding_prompt(task, code_content)

        # Call appropriate API (check for API keys first)
        if self.provider == "openai" and OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            response = self._call_openai_with_continuation(prompt, is_documentation=False)
            return self._extract_json(response)
        elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            response = self._call_anthropic_with_continuation(prompt, is_documentation=False)
            return self._extract_json(response)
        elif self.provider == "google" and GOOGLE_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
            response = self._call_google(prompt)
            return self._extract_json(response)
        elif self.provider == "aws":
            response = self._call_aws_transform(prompt)
            return self._extract_json(response)
        elif self.provider == "hexaview":
            logger.warning("CodeMolt (Hexaview) requires manual web UI - no API available")
            return self._generate_mock_understanding(task, input_files)
        else:
            # Fallback: return mock response
            return self._generate_mock_understanding(task, input_files)

    def generate_conversion(self, task, input_files: List[Path]) -> str:
        """Generate converted code for a conversion task"""
        # Read input files
        code_content = self._read_input_files(input_files)

        # Create prompt with dynamic sizing
        target_language = task.target_language or "java"
        prompt = self._create_conversion_prompt(task, code_content, target_language)

        # Call appropriate API (check for API keys first)
        if self.provider == "openai" and OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            response = self._call_openai_with_continuation(prompt, is_documentation=False)
            return self._extract_code(response, target_language)
        elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            response = self._call_anthropic_with_continuation(prompt, is_documentation=False)
            return self._extract_code(response, target_language)
        elif self.provider == "google" and GOOGLE_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
            response = self._call_google(prompt)
            return self._extract_code(response, target_language)
        elif self.provider == "aws":
            response = self._call_aws_transform(prompt)
            return self._extract_code(response, target_language)
        elif self.provider == "hexaview":
            logger.warning("CodeMolt (Hexaview) requires manual web UI - no API available")
            return self._generate_mock_conversion(task, target_language)
        else:
            # Fallback: return mock response
            return self._generate_mock_conversion(task, target_language)
    
    def _read_input_files(self, input_files: List[Path]) -> str:
        """Read input files with size limit based on model context"""
        code_content = ""
        total_chars = 0
        
        for file_path in input_files:
            if file_path.exists() and total_chars < self.max_input_chars:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_content = f.read()
                    
                    # Check if we can fit this file
                    header = f"\n\n=== {file_path.name} ===\n\n"
                    remaining_chars = self.max_input_chars - total_chars - len(header)
                    
                    if remaining_chars > 0:
                        code_content += header
                        if len(file_content) <= remaining_chars:
                            code_content += file_content
                            total_chars += len(header) + len(file_content)
                        else:
                            # Truncate this file but add note
                            code_content += file_content[:remaining_chars]
                            code_content += f"\n\n... [File truncated, {len(file_content) - remaining_chars} chars omitted] ..."
                            total_chars = self.max_input_chars
                            logger.info(f"Truncated {file_path.name} to fit context window")
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
        
        logger.info(f"Total input size: {total_chars} chars ({total_chars / CHARS_PER_TOKEN:.0f} est. tokens)")
        return code_content
    
    def _create_documentation_prompt(self, task, code_content: str) -> str:
        """Create prompt for documentation task"""
        return f"""You are a technical documentation expert specializing in legacy COBOL systems. Analyze the following COBOL code and generate comprehensive documentation.

Task: {task.task_description}

COBOL Code:
{code_content}

Requirements:
1. **Business Purpose**: Explain what the program does and its role in the system
2. **Business Rules**: Document ALL business rules, conditions, and logic (be thorough)
3. **Edge Cases**: Identify error handling, boundary conditions, and special cases
4. **Data Structures**: Describe all records, fields, copybooks, and data layouts
5. **Algorithm Overview**: Explain the program flow step by step

Output Format:
- Use Markdown format with clear headers (##)
- Include code snippets where helpful
- Minimum 5 pages of detailed documentation
- Be specific - cite actual variable names and conditions from the code

IMPORTANT: Generate the COMPLETE documentation. Do not stop early or summarize. Include every business rule you find in the code.

Generate the documentation now:"""
    
    def _create_understanding_prompt(self, task, code_content: str) -> str:
        """Create prompt for understanding task"""
        return f"""You are a code analysis expert specializing in legacy COBOL systems. Analyze the following COBOL code and extract its complete structure.

Task: {task.task_description}

COBOL Code:
{code_content}

Requirements:
1. Extract ALL dependency relationships:
   - CALL statements (program calls)
   - COPY statements (copybook includes)
2. Extract ALL business rules:
   - IF/EVALUATE conditions
   - PERFORM logic
   - Calculation rules
3. Extract ALL data flow:
   - File OPEN/READ/WRITE/CLOSE operations
   - Data transformations

Output ONLY valid JSON following this exact schema:
{{
  "dependencies": [
    {{"type": "CALL", "source": "program.cbl", "target": "subprogram-name"}},
    {{"type": "COPY", "source": "program.cbl", "target": "copybook-name"}}
  ],
  "business_rules": [
    {{"condition": "IF ACCOUNT-BALANCE < MINIMUM-BALANCE", "description": "Check minimum balance requirement"}},
    {{"condition": "EVALUATE TRANSACTION-TYPE", "description": "Route based on transaction type"}}
  ],
  "data_flow": [
    {{"operation": "OPEN", "file": "CUSTOMER-FILE", "mode": "INPUT"}},
    {{"operation": "READ", "file": "CUSTOMER-FILE"}},
    {{"operation": "WRITE", "file": "REPORT-FILE"}}
  ]
}}

IMPORTANT: Include ALL dependencies, rules, and data flows you find. Be comprehensive.

Generate the JSON output now:"""

    def _create_conversion_prompt(self, task, code_content: str, target_language: str) -> str:
        """Create prompt for conversion task"""
        lang_specifics = {
            "java": {
                "naming": "Use camelCase for methods/variables, PascalCase for classes",
                "types": "Use appropriate Java types (int, long, double, String, BigDecimal for currency)",
                "structure": "Create a proper class structure with main method",
                "io": "Use BufferedReader/BufferedWriter for file I/O",
            },
            "python": {
                "naming": "Use snake_case for functions/variables, PascalCase for classes",
                "types": "Use type hints where appropriate",
                "structure": "Create a proper module with if __name__ == '__main__' guard",
                "io": "Use context managers (with open(...)) for file I/O",
            },
            "csharp": {
                "naming": "Use PascalCase for methods/classes, camelCase for local variables",
                "types": "Use appropriate C# types (int, long, decimal for currency, string)",
                "structure": "Create a proper class within a namespace",
                "io": "Use StreamReader/StreamWriter for file I/O",
            },
        }

        specifics = lang_specifics.get(target_language, lang_specifics["java"])

        return f"""You are an expert legacy code modernization engineer. Convert the following COBOL program to {target_language.upper()}.

Task: {task.task_description}

COBOL Code to Convert:
{code_content}

Conversion Requirements:
1. **Functional Equivalence**: The converted code MUST produce identical output for identical input
2. **Data Structures**: Convert COBOL records and fields to appropriate {target_language} data structures
3. **Business Logic**: Preserve ALL business rules, calculations, and conditions exactly
4. **File I/O**: Convert file operations to {target_language} equivalents
5. **Error Handling**: Implement proper exception handling for all error conditions

{target_language.capitalize()}-Specific Guidelines:
- {specifics['naming']}
- {specifics['types']}
- {specifics['structure']}
- {specifics['io']}

Output Requirements:
- Generate ONLY the converted {target_language} code
- Include necessary imports/using statements
- Add comments explaining complex business logic
- The code should be complete and compilable/runnable
- Do NOT include explanations outside of code comments

IMPORTANT: Generate the COMPLETE converted code. Do not use placeholders like "// TODO" or "pass". Implement all functionality.

Generate the {target_language} code now:"""

    def _extract_code(self, response: str, target_language: str) -> str:
        """Extract code from response, removing markdown code blocks if present"""
        # Try to extract from markdown code block
        lang_markers = {
            "java": ["java", "Java"],
            "python": ["python", "Python", "py"],
            "csharp": ["csharp", "c#", "cs", "C#"],
        }

        markers = lang_markers.get(target_language, [target_language])

        # Try each marker
        for marker in markers:
            pattern = rf'```{marker}\s*\n(.*?)```'
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Try generic code block
        pattern = r'```\s*\n(.*?)```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # No code block found, return as-is (might already be clean code)
        return response.strip()

    def _generate_mock_conversion(self, task, target_language: str) -> str:
        """Generate mock conversion output for testing"""
        if target_language == "java":
            return '''import java.io.*;
import java.util.*;

public class ConvertedProgram {
    // Mock converted COBOL program
    private static final String INPUT_FILE = "input.dat";
    private static final String OUTPUT_FILE = "output.dat";

    public static void main(String[] args) {
        ConvertedProgram program = new ConvertedProgram();
        program.run();
    }

    public void run() {
        System.out.println("Mock conversion - replace with actual converted code");
        // TODO: Implement actual conversion
    }
}
'''
        elif target_language == "python":
            return '''#!/usr/bin/env python3
"""Mock converted COBOL program"""

INPUT_FILE = "input.dat"
OUTPUT_FILE = "output.dat"

def main():
    """Main entry point"""
    print("Mock conversion - replace with actual converted code")
    # TODO: Implement actual conversion

if __name__ == "__main__":
    main()
'''
        else:  # csharp
            return '''using System;
using System.IO;

namespace ConvertedProgram
{
    class Program
    {
        // Mock converted COBOL program
        private const string InputFile = "input.dat";
        private const string OutputFile = "output.dat";

        static void Main(string[] args)
        {
            Console.WriteLine("Mock conversion - replace with actual converted code");
            // TODO: Implement actual conversion
        }
    }
}
'''
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API (single request)"""
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
    
    def _call_openai_with_continuation(self, prompt: str, is_documentation: bool = True, max_continuations: int = 3) -> str:
        """Call OpenAI API with automatic continuation for long outputs"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, using mock response")
            return self._generate_mock_response()
        
        try:
            client = openai.OpenAI(api_key=api_key)
            messages = [{"role": "user", "content": prompt}]
            full_response = ""
            
            for i in range(max_continuations + 1):
                response = client.chat.completions.create(
                    model=self.config["model"],
                    messages=messages,
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"],
                )
                
                content = response.choices[0].message.content
                full_response += content
                finish_reason = response.choices[0].finish_reason
                
                logger.info(f"OpenAI response {i+1}: {len(content)} chars, finish_reason={finish_reason}")
                
                # Check if output is complete
                if finish_reason == "stop":
                    break
                
                # Check if we should continue (length limit hit)
                if finish_reason == "length" and i < max_continuations:
                    logger.info(f"Output truncated, requesting continuation {i+1}/{max_continuations}")
                    
                    # Add assistant's partial response and continuation request
                    messages.append({"role": "assistant", "content": content})
                    
                    if is_documentation:
                        messages.append({"role": "user", "content": "Continue the documentation from where you left off. Do not repeat previous content."})
                    else:
                        messages.append({"role": "user", "content": "Continue the JSON from where you left off. Ensure valid JSON structure."})
                else:
                    break
            
            logger.info(f"Total OpenAI response: {len(full_response)} chars")
            return full_response
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}, using mock response")
            return self._generate_mock_response()
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API (single request)"""
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
    
    def _call_anthropic_with_continuation(self, prompt: str, is_documentation: bool = True, max_continuations: int = 3) -> str:
        """Call Anthropic API with automatic continuation for long outputs"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set, using mock response")
            return self._generate_mock_response()
        
        try:
            client = anthropic.Anthropic(api_key=api_key)
            messages = [{"role": "user", "content": prompt}]
            full_response = ""
            
            for i in range(max_continuations + 1):
                response = client.messages.create(
                    model=self.config["model"],
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    messages=messages,
                )
                
                content = response.content[0].text
                full_response += content
                stop_reason = response.stop_reason
                
                logger.info(f"Anthropic response {i+1}: {len(content)} chars, stop_reason={stop_reason}")
                
                # Check if output is complete
                if stop_reason == "end_turn":
                    break
                
                # Check if we should continue (max_tokens limit hit)
                if stop_reason == "max_tokens" and i < max_continuations:
                    logger.info(f"Output truncated, requesting continuation {i+1}/{max_continuations}")
                    
                    # Add assistant's partial response and continuation request
                    messages.append({"role": "assistant", "content": content})
                    
                    if is_documentation:
                        messages.append({"role": "user", "content": "Continue the documentation from where you left off. Do not repeat previous content."})
                    else:
                        messages.append({"role": "user", "content": "Continue the JSON from where you left off. Ensure valid JSON structure."})
                else:
                    break
            
            logger.info(f"Total Anthropic response: {len(full_response)} chars")
            return full_response
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}, using mock response")
            return self._generate_mock_response()
    
    def _call_aws_transform(self, prompt: str) -> str:
        """Call AWS Transform (placeholder - would need actual API)"""
        logger.warning("AWS Transform API not implemented, using mock response")
        return self._generate_mock_response()

    def _call_google(self, prompt: str) -> str:
        """Call Google Gemini API"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not set, using mock response")
            return self._generate_mock_response()

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.config["model"])

            generation_config = genai.types.GenerationConfig(
                temperature=self.config["temperature"],
                max_output_tokens=self.config["max_tokens"],
            )

            response = model.generate_content(
                prompt,
                generation_config=generation_config,
            )

            if response.text:
                logger.info(f"Google response: {len(response.text)} chars")
                return response.text
            else:
                logger.warning("Empty response from Google API")
                return self._generate_mock_response()

        except Exception as e:
            logger.error(f"Google API error: {e}, using mock response")
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

