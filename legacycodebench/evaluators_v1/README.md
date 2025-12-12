# LegacyCodeBench v1.0 Evaluators (ARCHIVED)

This directory contains the original v1.0 evaluation modules, archived for reference and backward compatibility.

## Files

- **documentation.py** - v1.0 Documentation Evaluator (reference-based ROUGE/BLEU)
- **understanding.py** - v1.0 Understanding Evaluator (precision/recall)
- **nlp_metrics.py** - NLP metrics utilities (ROUGE, BLEU, etc.)

## Status

**⚠️ DEPRECATED** - These evaluators are no longer actively maintained.

For new implementations, use **`evaluators_v2/`** which implements the v2.0 framework:
- Ground truth-based evaluation (not reference-based)
- Behavioral fidelity testing
- LLM-as-judge semantic quality
- Critical failure detection

## Migration

**Old (v1.0):**
```python
from legacycodebench.evaluators.documentation import DocumentationEvaluator
evaluator = DocumentationEvaluator()
```

**New (v2.0):**
```python
from legacycodebench.evaluators_v2.documentation_v2 import DocumentationEvaluatorV2
evaluator = DocumentationEvaluatorV2(enable_execution=True)
```

## Why v1.0 Was Replaced

v1.0 had fundamental limitations:
1. **Reference dependency** - Required human-written reference documentation
2. **Surface-level matching** - ROUGE/BLEU don't measure semantic correctness
3. **No execution testing** - Couldn't verify behavioral fidelity
4. **Scalability issues** - Required manual reference creation for each task

v2.0 addresses these with:
- Automated ground truth from source code
- Execution-based behavioral testing
- LLM-as-judge for semantic quality
- Fully automated pipeline (no human references needed)

## Keeping for Reference

These files are kept for:
- Backward compatibility with existing results
- Reference for migration
- Historical comparison
- Academic transparency

**Do not use for new evaluations.**

Use **`evaluators_v2/`** instead.
