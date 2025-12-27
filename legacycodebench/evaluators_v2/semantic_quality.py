"""Semantic Quality Evaluator using LLM-as-Judge (25% weight)

Implements Section 5 of spec: Semantic Quality (SQ)
Measures: Clarity, completeness of explanation, appropriate abstraction

Uses a SEPARATE LLM model as judge (not the same model being evaluated)
"""

import json
import logging
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# Evaluation rubric per Section 4.2 of spec
DOCUMENTATION_RUBRIC = {
    "semantic_accuracy": {
        "question": "Does the explanation match actual code behavior?",
        "scale": "0-100",
        "weight": 0.30,
        "description": "Compare documentation claims to ground truth and source code"
    },
    "completeness": {
        "question": "Are all relevant aspects covered appropriately?",
        "scale": "0-100",
        "weight": 0.25,
        "description": "Check if documentation addresses all key elements from ground truth"
    },
    "clarity": {
        "question": "Would a developer understand without seeing code?",
        "scale": "0-100",
        "weight": 0.25,
        "description": "Assess readability, organization, and understandability"
    },
    "abstraction_level": {
        "question": "Appropriate balance of detail vs. overview?",
        "scale": "0-100",
        "weight": 0.20,
        "description": "Not too detailed (line-by-line) nor too abstract (vague)"
    }
}


class SemanticQualityEvaluator:
    """
    Evaluate semantic quality using LLM-as-judge.

    Per Section 4 of spec:
    - Uses different model than the one being evaluated
    - Temperature=0 for deterministic evaluation
    - Structured JSON output with scores + justifications
    - Escalates to human review if confidence < 70%
    """

    def __init__(self, judge_model_name: str = "gpt-4o",
                results_dir: Optional[Path] = None):
        """
        Initialize semantic quality evaluator.

        Args:
            judge_model_name: Model to use as judge (must be different from evaluated model)
            results_dir: Directory to store escalations for human review
        """
        self.judge_model_name = judge_model_name
        self.results_dir = results_dir or Path("results/escalations")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.rubric = DOCUMENTATION_RUBRIC

    def evaluate(self, submission_content: str, ground_truth: Dict,
                source_code: str, task_id: str = "unknown", 
                evaluated_model: Optional[str] = None) -> Dict:
        """
        Evaluate semantic quality using LLM-as-judge.

        Args:
            submission_content: AI-generated documentation
            ground_truth: Automated ground truth from static analysis
            source_code: Original COBOL source code
            task_id: Task identifier for tracking
            evaluated_model: Model that generated the documentation (for validation)

        Returns:
            Dictionary with:
            - score: Overall SQ score (0.0 to 1.0)
            - dimensions: Scores for each rubric dimension
            - confidence: Judge's confidence in evaluation
            - escalated: Whether escalated for human review
        """
        # FIXED (Issue 5.5): Block same-model judging with error
        if evaluated_model and evaluated_model == self.judge_model_name:
            raise ValueError(
                f"Judge model '{self.judge_model_name}' cannot be same as evaluated model '{evaluated_model}'. "
                f"Self-evaluation bias is not allowed. Use a different judge model."
            )
        
        logger.info(f"Evaluating semantic quality using LLM-as-judge ({self.judge_model_name})")

        # Build judge prompt
        judge_prompt = self._build_judge_prompt(
            submission_content,
            ground_truth,
            source_code
        )

        # Call judge model
        try:
            judge_response = self._call_judge_model(judge_prompt)
        except Exception as e:
            logger.error(f"LLM-as-judge call failed: {e}")
            # Fallback to conservative score
            return {
                "score": 0.5,
                "dimensions": {},
                "confidence": 0.0,
                "escalated": True,
                "error": str(e)
            }

        # Parse response
        scores = judge_response.get("scores", {})
        confidence = judge_response.get("confidence", 100)

        # Check if escalation needed (confidence < 70%)
        escalated = confidence < 70
        if escalated:
            self._escalate_for_human_review(
                task_id,
                submission_content,
                judge_response
            )

        # Calculate weighted score
        total_score = self._calculate_weighted_score(scores)

        logger.info(f"Semantic Quality Score: {total_score:.2%} (confidence: {confidence}%)")

        return {
            "score": total_score,
            "dimensions": scores,
            "confidence": confidence,
            "escalated": escalated,
            "judge_model": self.judge_model_name
        }

    def _build_judge_prompt(self, documentation: str, ground_truth: Dict,
                          source_code: str) -> str:
        """Build prompt for LLM-as-judge"""

        # Truncate source code if too long (keep first 500 lines)
        source_lines = source_code.split('\n')
        if len(source_lines) > 500:
            source_code_snippet = '\n'.join(source_lines[:500]) + "\n\n... (truncated)"
        else:
            source_code_snippet = source_code

        # Extract key ground truth elements for context
        gt_summary = {
            "data_structures": ground_truth.get("element_count", {}).get("data_structures", 0),
            "business_rules": ground_truth.get("element_count", {}).get("business_rules", 0),
            "paragraphs": ground_truth.get("element_count", {}).get("paragraphs", 0),
            "error_handlers": ground_truth.get("element_count", {}).get("error_handlers", 0),
            "total_lines": ground_truth.get("metadata", {}).get("total_lines", 0)
        }

        # Sample business rules for reference
        sample_rules = ground_truth.get("business_rules", {}).get("business_rules", [])[:5]

        prompt = f"""You are evaluating AI-generated documentation for COBOL legacy code.

GROUND TRUTH (from automated static analysis):
{json.dumps(gt_summary, indent=2)}

Sample Business Rules Detected:
{json.dumps(sample_rules, indent=2)}

SOURCE CODE (first 500 lines):
```cobol
{source_code_snippet}
```

AI-GENERATED DOCUMENTATION:
{documentation}

EVALUATION TASK:
Evaluate the documentation using the rubric below.
Compare the documentation against the GROUND TRUTH and SOURCE CODE.

CRITICAL RULES:
1. Base your evaluation ONLY on the ground truth and source code provided
2. Do NOT use your general knowledge of COBOL or business practices
3. Score 0-100 for each dimension with detailed justification
4. Provide an overall confidence score (0-100) for your evaluation
5. Output valid JSON only (no markdown, no explanations outside JSON)

RUBRIC:
{json.dumps(self.rubric, indent=2)}

OUTPUT FORMAT (JSON only):
{{
    "scores": {{
        "semantic_accuracy": {{
            "score": 85,
            "justification": "Detailed explanation comparing docs to ground truth..."
        }},
        "completeness": {{
            "score": 90,
            "justification": "Explanation of what is covered and what is missing..."
        }},
        "clarity": {{
            "score": 75,
            "justification": "Assessment of readability and organization..."
        }},
        "abstraction_level": {{
            "score": 80,
            "justification": "Assessment of detail balance..."
        }}
    }},
    "confidence": 95,
    "overall_assessment": "Brief summary of strengths and weaknesses"
}}

Respond with JSON only:"""

        return prompt

    def _call_judge_model(self, prompt: str) -> Dict:
        """
        Call LLM-as-judge model.

        Uses OpenAI, Anthropic, or Google Gemini API for evaluation.
        Temperature=0 for deterministic evaluation.
        """
        import os

        # Determine which provider to use based on judge model name
        is_gemini = "gemini" in self.judge_model_name.lower()
        is_claude = "claude" in self.judge_model_name.lower()
        
        # Try Google Gemini if judge model is gemini
        if is_gemini and os.getenv("GOOGLE_API_KEY"):
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    import google.generativeai as genai
                
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                
                logger.info(f"Calling Google {self.judge_model_name} as judge...")
                
                generation_config = {
                    "temperature": 0,
                    "max_output_tokens": 2000,
                }
                
                model = genai.GenerativeModel(
                    model_name=self.judge_model_name,
                    generation_config=generation_config
                )
                
                # Add JSON instruction to prompt
                json_prompt = prompt + "\n\nIMPORTANT: Respond with valid JSON only, no markdown."
                response = model.generate_content(json_prompt)
                response_text = response.text
                
                logger.info(f"Judge response received: {len(response_text)} chars")
                
                # Parse JSON response
                cleaned_response = self._clean_json_response(response_text)
                return json.loads(cleaned_response)
                
            except Exception as e:
                logger.error(f"Google Gemini judge call failed: {e}")

        # Try OpenAI if judge model is gpt-* or openai key available
        if not is_gemini and not is_claude and os.getenv("OPENAI_API_KEY"):
            try:
                import openai
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                logger.info(f"Calling OpenAI {self.judge_model_name} as judge...")

                response = client.chat.completions.create(
                    model=self.judge_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,  # Deterministic
                    max_tokens=2000,
                    response_format={"type": "json_object"}  # Request JSON response
                )

                response_text = response.choices[0].message.content
                logger.info(f"Judge response received: {len(response_text)} chars")

                # Parse JSON response
                cleaned_response = self._clean_json_response(response_text)
                return json.loads(cleaned_response)

            except Exception as e:
                logger.error(f"OpenAI judge call failed: {e}")

        # Try Anthropic if judge model is claude or anthropic key available
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

                logger.info(f"Calling Anthropic Claude as judge...")

                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}]
                )

                response_text = response.content[0].text
                logger.info(f"Judge response received: {len(response_text)} chars")

                # Parse JSON response
                cleaned_response = self._clean_json_response(response_text)
                return json.loads(cleaned_response)

            except Exception as e:
                logger.error(f"Anthropic judge call failed: {e}")

        # Fallback to mock if no API available
        logger.warning("No LLM API available for judge, using mock response")
        return self._mock_judge_response()

    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response from LLM (remove markdown, extract JSON)"""
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith('```'):
            # Extract JSON from code block
            lines = response.split('\n')
            json_lines = []
            in_code_block = False
            for line in lines:
                if line.startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or not line.startswith('```'):
                    json_lines.append(line)
            response = '\n'.join(json_lines).strip()

        return response

    def _mock_judge_response(self) -> Dict:
        """Mock response for testing (when AI not available)"""
        return {
            "scores": {
                "semantic_accuracy": {"score": 75, "justification": "Mock evaluation"},
                "completeness": {"score": 80, "justification": "Mock evaluation"},
                "clarity": {"score": 70, "justification": "Mock evaluation"},
                "abstraction_level": {"score": 75, "justification": "Mock evaluation"}
            },
            "confidence": 60,  # Low confidence to trigger escalation in testing
            "overall_assessment": "Mock evaluation - replace with actual LLM judge"
        }

    def _calculate_weighted_score(self, scores: Dict) -> float:
        """Calculate weighted average of dimension scores"""
        total = 0.0

        for dimension, weight in [(k, v["weight"]) for k, v in self.rubric.items()]:
            if dimension in scores:
                dimension_score = scores[dimension].get("score", 0)
                total += (dimension_score / 100) * weight

        return total

    def _escalate_for_human_review(self, task_id: str, documentation: str,
                                  judge_response: Dict):
        """
        Escalate low-confidence evaluations for human review.

        Per Section 4.4 of spec: Human-in-the-Loop Escalation
        """
        import datetime

        escalation_file = self.results_dir / f"escalation_{task_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        escalation_data = {
            "task_id": task_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "reason": f"Low confidence ({judge_response.get('confidence', 0)}%)",
            "judge_response": judge_response,
            "documentation_preview": documentation[:500] + "..." if len(documentation) > 500 else documentation,
            "status": "pending_human_review"
        }

        with open(escalation_file, 'w', encoding='utf-8') as f:
            json.dump(escalation_data, f, indent=2)

        logger.warning(f"Escalated for human review: {escalation_file}")
