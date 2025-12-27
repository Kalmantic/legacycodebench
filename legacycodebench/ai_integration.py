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
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    ClientError = Exception  # Fallback for type hints
    NoCredentialsError = Exception

try:
    # Suppress deprecation warning for google.generativeai
    # TODO: Migrate to google-genai package when stable
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

from legacycodebench.config import AI_MODELS, DOCMOLT_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom Exceptions (Issue 4.3)
class CredentialsError(Exception):
    """Raised when API credentials are missing or invalid"""
    pass


# Token limits for different models (input context windows)
MODEL_CONTEXT_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4": 8192,
    "claude-sonnet-4-20250514": 200000,
    "anthropic.claude-3-sonnet-20240229-v1:0": 200000,  # AWS Bedrock Claude 3 Sonnet
    "transform": 8000,  # Deprecated, kept for backward compatibility
}

# FIXED (Issue 4.4): Normalize context to smallest for fair comparison
# All models will use the same effective context to ensure fair evaluation
NORMALIZED_CONTEXT_LIMIT = 8000  # 8k tokens - smallest common denominator

# Approximate chars per token (conservative estimate for COBOL)
CHARS_PER_TOKEN = 3.5

# Maximum input tokens to use (leave room for output)
MAX_INPUT_RATIO = 0.6  # Use 60% of context for input, leave 40% for output


class AIModelInterface:
    """Interface for AI models"""

    def __init__(self, model_id: str, mock_mode: bool = False):
        if model_id not in AI_MODELS:
            raise ValueError(f"Unknown model: {model_id}")

        self.model_id = model_id
        self.config = AI_MODELS[model_id]
        self.provider = self.config["provider"]
        self.mock_mode = mock_mode  # ADDED (Issue 4.3): Explicit mock mode flag

        if mock_mode:
            logger.warning(f"[MOCK MODE] ENABLED for {model_id} - Using fake responses for testing")

        # ADDED (Issue 4.2): Enforce temperature = 0 for deterministic outputs
        temperature = self.config.get("temperature", 0)
        if temperature != 0:
            logger.error(f"[CRITICAL] Model {model_id} has temperature={temperature}. "
                        f"LegacyCodeBench requires temperature=0 for deterministic, reproducible results.")
            raise ValueError(f"Invalid temperature {temperature} for {model_id}. Must be 0 for benchmark reproducibility.")

        # FIXED (Issue 4.4): Calculate max input chars using NORMALIZED context
        # This ensures fair comparison across models with different context windows
        model_name = self.config["model"]
        model_context_limit = MODEL_CONTEXT_LIMITS.get(model_name, 8000)
        # Use smaller of model limit or normalized limit for fairness
        effective_context = min(model_context_limit, NORMALIZED_CONTEXT_LIMIT)
        max_input_tokens = int(effective_context * MAX_INPUT_RATIO)
        self.max_input_chars = int(max_input_tokens * CHARS_PER_TOKEN)
        logger.info(f"Model {model_id}: raw_context={model_context_limit}, normalized={effective_context}, max_input_chars={self.max_input_chars}")
    
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
            return self._call_google_gemini(prompt)
        elif self.provider == "aws":
            return self._call_aws_transform(prompt)
        else:
            # FIXED (Issue 4.3): Check mock mode before failing
            if self.mock_mode:
                logger.info(f"Mock mode: returning fake {self.provider} documentation")
                return self._generate_mock_documentation(task)
            raise CredentialsError(
                f"No valid API credentials found for provider '{self.provider}'. "
                f"Please set the appropriate API key environment variable or use --mock flag for testing."
            )
    
    def generate_understanding(self, task, input_files: List[Path]) -> str:
        """Generate understanding output (JSON) for a task"""
        # Read input files
        code_content = self._read_input_files(input_files)
        
        # Create prompt with dynamic sizing
        prompt = self._create_understanding_prompt(task, code_content)
        
        # Call appropriate API (check for API keys first)
        if self.provider == "openai" and OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            response = self._call_openai_with_continuation(prompt, is_documentation=False)
            # Try to extract JSON from response
            return self._extract_json(response)
        elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            response = self._call_anthropic_with_continuation(prompt, is_documentation=False)
            return self._extract_json(response)
        elif self.provider == "google" and GOOGLE_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
            response = self._call_google_gemini(prompt)
            return self._extract_json(response)
        elif self.provider == "aws":
            response = self._call_aws_transform(prompt)
            return self._extract_json(response)
        else:
            # FIXED (Issue 4.3): Check mock mode before failing
            if self.mock_mode:
                logger.info(f"Mock mode: returning fake {self.provider} understanding")
                return self._generate_mock_understanding(task, input_files)
            raise CredentialsError(
                f"No valid API credentials found for provider '{self.provider}'. "
                f"Please set the appropriate API key environment variable or use --mock flag for testing."
            )
    
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
- Minimum 2500 words (~10000 characters) of detailed documentation  # FIXED (Issue 4.1): Replaced vague "5 pages"
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
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API (single request)"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # FIXED (Issue 4.3): Check mock mode before failing
            if self.mock_mode:
                logger.info("Mock mode: returning fake OpenAI response")
                return self._generate_mock_response()
            raise CredentialsError(
                "OPENAI_API_KEY not set. Set the environment variable, use --mock flag for testing, or use a different model. "
                "Example: export OPENAI_API_KEY=sk-..."
            )
        
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
            # FIXED (Issue 4.3): Check mock mode before failing
            if self.mock_mode:
                logger.info("Mock mode: returning fake OpenAI response")
                return self._generate_mock_response()
            raise CredentialsError(
                "OPENAI_API_KEY not set. Set the environment variable, use --mock flag for testing, or use a different model. "
                "Example: export OPENAI_API_KEY=sk-..."
            )
        
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
            # FIXED (Issue 4.3): Check mock mode before failing
            if self.mock_mode:
                logger.info("Mock mode: returning fake Anthropic response")
                return self._generate_mock_response()
            raise CredentialsError(
                "ANTHROPIC_API_KEY not set. Set the environment variable, use --mock flag for testing, or use a different model. "
                "Example: export ANTHROPIC_API_KEY=sk-ant-..."
            )
        
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
            # FIXED (Issue 4.3): Check mock mode before failing
            if self.mock_mode:
                logger.info("Mock mode: returning fake Anthropic response")
                return self._generate_mock_response()
            raise CredentialsError(
                "ANTHROPIC_API_KEY not set. Set the environment variable, use --mock flag for testing, or use a different model. "
                "Example: export ANTHROPIC_API_KEY=sk-ant-..."
            )
        
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
    
    def _call_google_gemini(self, prompt: str) -> str:
        """Call Google Gemini API"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # Check mock mode before failing
            if self.mock_mode:
                logger.info("Mock mode: returning fake Google Gemini response")
                return self._generate_mock_response()
            raise CredentialsError(
                "GOOGLE_API_KEY not set. Set the environment variable, use --mock flag for testing, or use a different model. "
                "Example: export GOOGLE_API_KEY=AIza..."
            )
        
        try:
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Create model with generation config
            generation_config = {
                "temperature": self.config["temperature"],
                "max_output_tokens": self.config["max_tokens"],
            }
            
            model = genai.GenerativeModel(
                model_name=self.config["model"],
                generation_config=generation_config
            )
            
            logger.info(f"Calling Google Gemini: model={self.config['model']}")
            
            # Generate content
            response = model.generate_content(prompt)
            
            # Extract text from response
            result_text = response.text
            logger.info(f"Gemini response: {len(result_text)} chars")
            
            return result_text
            
        except Exception as e:
            logger.error(f"Google Gemini API error: {e}, using mock response")
            return self._generate_mock_response()
    
    def _call_aws_transform(self, prompt: str) -> str:
        """Call AWS Bedrock Converse API for code analysis"""
        if not AWS_AVAILABLE:
            # FIXED (Issue 4.3): Check mock mode before failing
            if self.mock_mode:
                logger.info("Mock mode: returning fake AWS response")
                return self._generate_mock_response()
            raise CredentialsError(
                "boto3 not available. Install AWS SDK: pip install boto3 botocore"
            )

        # Check for AWS credentials
        region = os.getenv("AWS_REGION", self.config.get("region", "us-east-1"))
        model_id = os.getenv("AWS_BEDROCK_MODEL_ID", self.config["model"])

        try:
            # Create Bedrock Runtime client
            bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region
            )

            logger.info(f"Calling AWS Bedrock: model={model_id}, region={region}")

            # Call Converse API
            response = bedrock_client.converse(
                modelId=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                inferenceConfig={
                    "maxTokens": self.config["max_tokens"],
                    "temperature": self.config["temperature"],
                    "topP": 0.9
                }
            )

            # Extract response text
            output_text = response['output']['message']['content'][0]['text']

            # Log metrics
            usage = response.get('usage', {})
            metrics = response.get('metrics', {})
            logger.info(f"AWS Bedrock response: {len(output_text)} chars, "
                       f"tokens={usage.get('totalTokens', 'N/A')}, "
                       f"latency={metrics.get('latencyMs', 'N/A')}ms")

            return output_text

        except NoCredentialsError:
            # FIXED (Issue 4.3): Check mock mode before failing
            if self.mock_mode:
                logger.info("Mock mode: returning fake AWS response (no credentials)")
                return self._generate_mock_response()
            raise CredentialsError(
                "AWS credentials not found. Please configure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY. "
                "Example: export AWS_ACCESS_KEY_ID=AKIA... or use --mock flag for testing"
            )
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_msg = e.response.get('Error', {}).get('Message', str(e))
            logger.error(f"AWS Bedrock API error [{error_code}]: {error_msg}")
            # Re-raise API errors (not credentials issue)
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling AWS Bedrock: {e}")
            # Re-raise unexpected errors
            raise
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text response"""
        # Try to find JSON block
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            # FIXED (Issue 4.5): Validate JSON schema
            if self._validate_understanding_schema(json_str):
                return json_str
            else:
                logger.warning("[OUTPUT VALIDATION] JSON structure incomplete, returning as-is")
        
        # If no JSON found, return as-is (evaluator will handle)
        return text

    def _validate_understanding_schema(self, json_str: str) -> bool:
        """
        Validate understanding output schema (Issue 4.5).
        
        Returns True if JSON has required structure.
        """
        try:
            import json
            data = json.loads(json_str)
            
            # Required top-level fields for understanding output
            required_fields = ["dependencies", "business_rules", "data_flow"]
            has_required = all(field in data for field in required_fields)
            
            if not has_required:
                missing = [f for f in required_fields if f not in data]
                logger.warning(f"[SCHEMA] Understanding output missing fields: {missing}")
                return False
            
            return True
        except json.JSONDecodeError:
            logger.warning("[SCHEMA] Failed to parse JSON from AI output")
            return False
    
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


class DocMoltInterface(AIModelInterface):
    """
    Interface for DocMolt API - Specialized documentation generation service

    DocMolt is a specialized documentation generation service that supports
    multiple backend models (GPT-4o, GPT-4o-mini, Claude, etc.) and various
    documentation formats (technical specs, API docs, TDD, etc.).

    For fair comparison in LegacyCodeBench, we use the same evaluation pipeline
    (DocumentationEvaluatorV2) as other models.
    """

    def __init__(self, model_id: str, mock_mode: bool = False):
        super().__init__(model_id, mock_mode)
        self.api_endpoint = self.config["api_endpoint"]
        self.docmolt_model = self.config["model"]
        self.artefact = self.config.get("artefact", DOCMOLT_CONFIG["default_artefact"])
        self.language = self.config.get("language", DOCMOLT_CONFIG["language"])
        self.timeout = DOCMOLT_CONFIG["timeout_seconds"]
        self.max_retries = DOCMOLT_CONFIG["max_retries"]
        self.retry_delay = DOCMOLT_CONFIG["retry_delay_seconds"]

        logger.info(f"DocMolt initialized: model={self.docmolt_model}, artefact={self.artefact}")

    def generate_documentation(self, task, input_files: List[Path]) -> str:
        """Generate documentation using DocMolt API"""
        # Read input files
        code_content = self._read_input_files(input_files)

        # Determine filename
        filename = input_files[0].name if input_files else "program.cbl"

        # Call DocMolt API with retry logic
        return self._call_docmolt_with_retry(code_content, filename, task)

    def generate_understanding(self, task, input_files: List[Path]) -> str:
        """
        DocMolt is designed for documentation, not understanding tasks.

        NOTE: v2.0 doesn't have understanding tasks anyway (documentation-only).
        This method is here for v1.0 compatibility but should not be used with DocMolt.
        """
        logger.warning("DocMolt is not designed for understanding tasks. Falling back to mock response.")
        return self._generate_mock_understanding(task, input_files)

    def _call_docmolt_with_retry(self, code: str, filename: str, task) -> str:
        """Call DocMolt API with retry logic"""
        import time

        for attempt in range(self.max_retries + 1):
            try:
                result = self._call_docmolt(code, filename)
                return result
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"DocMolt API call failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"DocMolt API failed after {self.max_retries + 1} attempts: {e}")
                    logger.info("Falling back to mock documentation")
                    return self._generate_mock_documentation(task)

    def _call_docmolt(self, code: str, filename: str) -> str:
        """Call DocMolt API with proper error handling"""
        import requests

        # Prepare payload
        payload = {
            "code": code,
            "filename": filename,
            "language": self.language,
            "artefact": self.artefact,
            "model": self.docmolt_model,
        }

        # Prepare headers
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(DOCMOLT_CONFIG["api_key_env_var"])
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            logger.info("Using DocMolt API key from environment")
        else:
            logger.warning(f"No {DOCMOLT_CONFIG['api_key_env_var']} found - attempting unauthenticated request")

        logger.info(f"Calling DocMolt API: model={self.docmolt_model}, artefact={self.artefact}, code_size={len(code)} chars")

        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )

            # Log response details
            logger.info(f"DocMolt response: status={response.status_code}, content_length={len(response.content)}")

            if response.status_code == 200:
                result = response.json()
                documentation = self._extract_documentation_from_response(result)
                logger.info(f"DocMolt documentation extracted: {len(documentation)} chars")
                return documentation
            elif response.status_code == 401:
                raise Exception("Authentication failed - check DOCMOLT_API_KEY")
            elif response.status_code == 429:
                raise Exception("Rate limit exceeded - too many requests")
            elif response.status_code >= 500:
                raise Exception(f"Server error: {response.status_code}")
            else:
                raise Exception(f"Unexpected status code: {response.status_code}, body: {response.text[:200]}")

        except requests.exceptions.Timeout:
            raise Exception(f"Request timed out after {self.timeout} seconds")
        except requests.exceptions.ConnectionError as e:
            raise Exception(f"Connection error: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")

    def _extract_documentation_from_response(self, result: dict) -> str:
        """
        Extract documentation from DocMolt API response

        Note: The actual response structure needs to be determined from API testing.
        This method tries common key patterns.
        """
        # Try common response keys
        for key in ['documentation', 'output', 'content', 'result', 'data', 'text', 'markdown']:
            if key in result:
                content = result[key]
                if isinstance(content, str):
                    return content
                elif isinstance(content, dict):
                    # If nested, try to extract text
                    for subkey in ['text', 'content', 'value']:
                        if subkey in content:
                            return str(content[subkey])

        # If no known key found, try to stringify the whole response
        logger.warning(f"Unknown DocMolt response structure. Keys: {list(result.keys())}")

        # Try to find any string value that looks like documentation
        for key, value in result.items():
            if isinstance(value, str) and len(value) > 100:  # Likely documentation
                logger.info(f"Using response key '{key}' as documentation")
                return value

        # Last resort: return JSON representation
        import json
        logger.warning("Could not extract documentation string, returning JSON representation")
        return json.dumps(result, indent=2)


def get_ai_model(model_id: str, mock_mode: bool = False) -> AIModelInterface:
    """
    Get AI model interface - supports OpenAI, Anthropic, DocMolt, AWS

    Routes to appropriate interface based on provider:
    - openai → AIModelInterface (OpenAI GPT models)
    - anthropic → AIModelInterface (Anthropic Claude models)
    - docmolt → DocMoltInterface (DocMolt documentation service)
    - aws → AIModelInterface (AWS Transform)

    Args:
        model_id: Model identifier from AI_MODELS config
        mock_mode: If True, uses mock responses instead of API calls (Issue 4.3)
    """
    if model_id not in AI_MODELS:
        raise ValueError(f"Unknown model: {model_id}")

    config = AI_MODELS[model_id]
    provider = config["provider"]

    # Route to appropriate interface
    if provider == "docmolt":
        return DocMoltInterface(model_id, mock_mode=mock_mode)
    else:
        return AIModelInterface(model_id, mock_mode=mock_mode)

