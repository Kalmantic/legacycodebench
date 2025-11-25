"""NLP metrics for evaluation: BLEU, ROUGE, etc."""

import re
from typing import List, Dict, Optional
import logging

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    logging.warning("rouge-score not available")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    HAS_BLEU = True
except ImportError:
    HAS_BLEU = False
    logging.warning("nltk not available for BLEU")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLPMetrics:
    """Calculate BLEU, ROUGE, and other NLP metrics for text comparison"""
    
    def __init__(self):
        self.rouge_scorer = None
        if HAS_ROUGE:
            try:
                self.rouge_scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
                    use_stemmer=True
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ROUGE scorer: {e}")
                self.rouge_scorer = None
        
        # Download NLTK data if needed
        if HAS_BLEU:
            try:
                import nltk
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
            except Exception as e:
                logger.warning(f"NLTK setup issue: {e}")
    
    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum)"""
        if not self.rouge_scorer:
            return {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "rougeLsum": 0.0,
                "rouge_avg": 0.0,
            }
        
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            
            rouge1 = scores['rouge1'].fmeasure
            rouge2 = scores['rouge2'].fmeasure
            rougeL = scores['rougeL'].fmeasure
            rougeLsum = scores['rougeLsum'].fmeasure
            
            # Average of all ROUGE scores
            rouge_avg = (rouge1 + rouge2 + rougeL + rougeLsum) / 4.0
            
            return {
                "rouge1": round(rouge1, 4),
                "rouge2": round(rouge2, 4),
                "rougeL": round(rougeL, 4),
                "rougeLsum": round(rougeLsum, 4),
                "rouge_avg": round(rouge_avg, 4),
                "rouge1_precision": round(scores['rouge1'].precision, 4),
                "rouge1_recall": round(scores['rouge1'].recall, 4),
                "rouge2_precision": round(scores['rouge2'].precision, 4),
                "rouge2_recall": round(scores['rouge2'].recall, 4),
                "rougeL_precision": round(scores['rougeL'].precision, 4),
                "rougeL_recall": round(scores['rougeL'].recall, 4),
            }
        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {e}")
            return {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "rougeLsum": 0.0,
                "rouge_avg": 0.0,
            }
    
    def calculate_bleu(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate BLEU score"""
        if not HAS_BLEU:
            return {
                "bleu": 0.0,
                "bleu_1": 0.0,
                "bleu_2": 0.0,
                "bleu_3": 0.0,
                "bleu_4": 0.0,
            }
        
        try:
            # Tokenize
            ref_tokens = word_tokenize(reference.lower())
            cand_tokens = word_tokenize(candidate.lower())
            
            if not ref_tokens or not cand_tokens:
                return {
                    "bleu": 0.0,
                    "bleu_1": 0.0,
                    "bleu_2": 0.0,
                    "bleu_3": 0.0,
                    "bleu_4": 0.0,
                }
            
            # Use smoothing function to handle zero n-grams
            smoothing = SmoothingFunction().method1
            
            # Calculate BLEU-1 through BLEU-4
            bleu_1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu_2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu_3 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            bleu_4 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            
            # Standard BLEU-4 is the main score
            bleu = bleu_4
            
            return {
                "bleu": round(bleu, 4),
                "bleu_1": round(bleu_1, 4),
                "bleu_2": round(bleu_2, 4),
                "bleu_3": round(bleu_3, 4),
                "bleu_4": round(bleu_4, 4),
            }
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
            return {
                "bleu": 0.0,
                "bleu_1": 0.0,
                "bleu_2": 0.0,
                "bleu_3": 0.0,
                "bleu_4": 0.0,
            }
    
    def calculate_section_rouge(self, ref_sections: Dict[str, str], 
                               sub_sections: Dict[str, str]) -> Dict[str, float]:
        """Calculate ROUGE scores for each section, then average"""
        if not ref_sections or not sub_sections:
            return {
                "section_rouge_avg": 0.0,
                "section_rouge1": 0.0,
                "section_rouge2": 0.0,
                "section_rougeL": 0.0,
            }
        
        section_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        # Match sections by name similarity
        for ref_section_name, ref_content in ref_sections.items():
            # Find best matching submission section
            best_match = None
            best_score = 0.0
            
            for sub_section_name, sub_content in sub_sections.items():
                # Simple name similarity
                ref_words = set(ref_section_name.lower().split())
                sub_words = set(sub_section_name.lower().split())
                if ref_words and sub_words:
                    similarity = len(ref_words & sub_words) / len(ref_words | sub_words)
                    if similarity > best_score:
                        best_score = similarity
                        best_match = sub_content
            
            if best_match and ref_content:
                # Calculate ROUGE for this section
                rouge_scores = self.calculate_rouge(ref_content, best_match)
                section_scores.append(rouge_scores["rouge_avg"])
                rouge1_scores.append(rouge_scores["rouge1"])
                rouge2_scores.append(rouge_scores["rouge2"])
                rougeL_scores.append(rouge_scores["rougeL"])
        
        if not section_scores:
            return {
                "section_rouge_avg": 0.0,
                "section_rouge1": 0.0,
                "section_rouge2": 0.0,
                "section_rougeL": 0.0,
            }
        
        return {
            "section_rouge_avg": round(sum(section_scores) / len(section_scores), 4),
            "section_rouge1": round(sum(rouge1_scores) / len(rouge1_scores), 4) if rouge1_scores else 0.0,
            "section_rouge2": round(sum(rouge2_scores) / len(rouge2_scores), 4) if rouge2_scores else 0.0,
            "section_rougeL": round(sum(rougeL_scores) / len(rougeL_scores), 4) if rougeL_scores else 0.0,
        }
    
    def calculate_all_metrics(self, reference: str, candidate: str,
                              ref_sections: Optional[Dict[str, str]] = None,
                              sub_sections: Optional[Dict[str, str]] = None) -> Dict:
        """Calculate all available NLP metrics"""
        metrics = {}
        
        # Overall ROUGE
        rouge_scores = self.calculate_rouge(reference, candidate)
        metrics.update(rouge_scores)
        
        # Overall BLEU
        bleu_scores = self.calculate_bleu(reference, candidate)
        metrics.update(bleu_scores)
        
        # Section-level ROUGE if sections provided
        if ref_sections and sub_sections:
            section_rouge = self.calculate_section_rouge(ref_sections, sub_sections)
            metrics.update(section_rouge)
        
        return metrics

