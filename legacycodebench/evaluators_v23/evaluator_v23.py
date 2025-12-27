"""
Main Evaluator Orchestrator V2.3

Coordinates all V2.3 evaluation components to produce a complete evaluation result.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging

from .config_v23 import V23_CONFIG
from .paragraph_classifier import ParagraphClassifier
from .anti_gaming import AntiGamingAnalyzer
from .comprehension_v23 import ComprehensionEvaluatorV23
from .documentation_v23 import DocumentationEvaluatorV23
from .behavioral_v23 import BehavioralEvaluatorV23
from .critical_failures_v23 import CriticalFailureDetectorV23
from .scoring_v23 import ScoringEngineV23, LCBScoreV23


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResultV23:
    """Complete evaluation result for V2.3."""
    task_id: str
    model: str
    score: LCBScoreV23
    comprehension_details: Dict = field(default_factory=dict)
    documentation_details: Dict = field(default_factory=dict)
    behavioral_details: Dict = field(default_factory=dict)
    anti_gaming_details: Dict = field(default_factory=dict)
    classification_stats: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "model": self.model,
            "version": "2.3.0",
            "score": self.score.to_dict(),
            "details": {
                "comprehension": self.comprehension_details,
                "documentation": self.documentation_details,
                "behavioral": self.behavioral_details,
                "anti_gaming": self.anti_gaming_details,
                "classification": self.classification_stats
            }
        }


class EvaluatorV23:
    """
    V2.3 Evaluation Orchestrator
    
    Coordinates:
    1. Paragraph classification (PURE/MIXED/INFRASTRUCTURE)
    2. Comprehension evaluation (40%)
    3. Documentation evaluation (25%)
    4. Behavioral evaluation (35%)
    5. Anti-gaming analysis
    6. Critical failure detection
    7. Final scoring
    """
    
    def __init__(self, llm_client=None, executor=None):
        """
        Initialize V2.3 evaluator.
        
        Args:
            llm_client: Optional LLM client for generation/judgment
            executor: Optional COBOL executor for behavioral testing
        """
        self.classifier = ParagraphClassifier()
        self.anti_gaming = AntiGamingAnalyzer()
        self.comprehension = ComprehensionEvaluatorV23()
        self.documentation = DocumentationEvaluatorV23(llm_client)
        self.behavioral = BehavioralEvaluatorV23(llm_client, executor)
        self.cf_detector = CriticalFailureDetectorV23()
        self.scoring = ScoringEngineV23()
        self.llm = llm_client
        self.executor = executor
    
    def evaluate(
        self,
        task_id: str,
        model: str,
        documentation: str,
        source_code: str,
        ground_truth: Dict,
        paragraphs: List[Dict] = None,
        task_requirements: Dict = None
    ) -> EvaluationResultV23:
        """
        Perform complete V2.3 evaluation.
        
        Args:
            task_id: Task identifier
            model: Model name being evaluated
            documentation: AI-generated documentation
            source_code: Original COBOL source code
            ground_truth: Ground truth data
            paragraphs: Parsed paragraphs (optional, will extract if not provided)
            task_requirements: Task-specific requirements (optional)
            
        Returns:
            Complete EvaluationResultV23
        """
        logger.info(f"Starting V2.3 evaluation for task {task_id}")
        
        # Step 1: Parse paragraphs if not provided
        if paragraphs is None:
            paragraphs = self._extract_paragraphs(source_code)
        
        # Step 2: Classify paragraphs
        classification = self.classifier.classify_all(paragraphs)
        stats = self.classifier.get_statistics(paragraphs)
        logger.debug(f"Classification: {stats}")
        
        # Step 3: Run anti-gaming analysis
        anti_gaming_result = self.anti_gaming.analyze(
            documentation, source_code, task_id
        )
        logger.debug(f"Anti-gaming: stuffing={anti_gaming_result.keyword_stuffing_score:.2f}, "
                    f"parroting={anti_gaming_result.parroting_score:.2f}, "
                    f"abstraction={anti_gaming_result.abstraction_score:.2f}")
        
        # Step 4: Evaluate comprehension (40%)
        comp_result = self.comprehension.evaluate(
            documentation, ground_truth, source_code
        )
        logger.debug(f"Comprehension: {comp_result.overall:.2f}")
        
        # Step 5: Evaluate documentation (25%)
        doc_result = self.documentation.evaluate(
            documentation, source_code, ground_truth, task_requirements
        )
        logger.debug(f"Documentation: {doc_result.overall:.2f}")
        
        # Step 6: Evaluate behavioral (35%)
        beh_result = self.behavioral.evaluate(
            documentation, source_code, ground_truth, paragraphs
        )
        logger.debug(f"Behavioral: {beh_result.overall:.2f}")
        
        # Step 7: Detect critical failures
        cf_result = self.cf_detector.detect(
            comprehension_result=comp_result,
            documentation_result=doc_result,
            behavioral_result=beh_result,
            ground_truth=ground_truth,
            documentation=documentation
        )
        logger.debug(f"Critical failures: {cf_result.triggered_list}")
        
        # Step 8: Calculate final score
        score = self.scoring.calculate(
            comprehension_result=comp_result,
            documentation_result=doc_result,
            behavioral_result=beh_result,
            cf_result=cf_result,
            anti_gaming_result=anti_gaming_result
        )
        
        logger.info(f"V2.3 evaluation complete: {score.overall:.1f} "
                   f"({'PASSED' if score.passed else 'FAILED'})")
        
        return EvaluationResultV23(
            task_id=task_id,
            model=model,
            score=score,
            comprehension_details=self._to_dict(comp_result),
            documentation_details=self._to_dict(doc_result),
            behavioral_details=self._to_dict(beh_result),
            anti_gaming_details={
                "keyword_stuffing": anti_gaming_result.keyword_stuffing_score,
                "parroting": anti_gaming_result.parroting_score,
                "abstraction": anti_gaming_result.abstraction_score,
                "penalties": anti_gaming_result.penalties_applied
            },
            classification_stats=stats
        )
    
    def _extract_paragraphs(self, source_code: str) -> List[Dict]:
        """Extract paragraphs from COBOL source code."""
        paragraphs = []
        
        # Find PROCEDURE DIVISION
        proc_match = re.search(
            r'PROCEDURE\s+DIVISION.*?\.',
            source_code,
            re.IGNORECASE | re.DOTALL
        )
        
        if not proc_match:
            return paragraphs
        
        proc_start = proc_match.end()
        proc_content = source_code[proc_start:]
        
        # Find paragraph names (start at column 8, end with period)
        para_pattern = re.compile(
            r'^       ([A-Z0-9][A-Z0-9-]*)\.\s*$',
            re.MULTILINE
        )
        
        matches = list(para_pattern.finditer(proc_content))
        lines = proc_content.split('\n')
        
        for i, match in enumerate(matches):
            para_name = match.group(1)
            
            # Find content until next paragraph
            start_line = proc_content[:match.end()].count('\n')
            
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
                end_line = proc_content[:end_pos].count('\n')
            else:
                end_line = len(lines)
            
            content = '\n'.join(lines[start_line:end_line])
            
            paragraphs.append({
                "name": para_name,
                "content": content,
                "start_line": proc_start // 80 + start_line,  # Approximate
                "end_line": proc_start // 80 + end_line
            })
        
        return paragraphs
    
    def _to_dict(self, result: Any) -> Dict:
        """Convert result object to dictionary."""
        if hasattr(result, 'to_dict'):
            return result.to_dict()
        elif hasattr(result, '__dict__'):
            return {
                k: v for k, v in result.__dict__.items()
                if not k.startswith('_')
            }
        else:
            return {}


# Import re for paragraph extraction
import re


def evaluate_v23(
    task_id: str,
    model: str,
    documentation: str,
    source_code: str,
    ground_truth: Dict,
    llm_client=None,
    executor=None,
    **kwargs
) -> EvaluationResultV23:
    """
    Convenience function for V2.3 evaluation.
    
    Args:
        task_id: Task identifier
        model: Model name
        documentation: AI-generated documentation
        source_code: COBOL source code
        ground_truth: Ground truth data
        llm_client: Optional LLM client
        executor: Optional COBOL executor
        
    Returns:
        Complete evaluation result
    """
    evaluator = EvaluatorV23(llm_client, executor)
    return evaluator.evaluate(
        task_id=task_id,
        model=model,
        documentation=documentation,
        source_code=source_code,
        ground_truth=ground_truth,
        **kwargs
    )
