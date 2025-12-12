"""Evaluation modules for LegacyCodeBench

v1.0 evaluators have been archived to evaluators_v1/
For new implementations, use evaluators_v2/

To use v2.0 evaluators:
    from legacycodebench.evaluators_v2.documentation_v2 import DocumentationEvaluatorV2

To use archived v1.0 evaluators:
    from legacycodebench.evaluators_v1.documentation import DocumentationEvaluator
"""

# For backward compatibility, import v1.0 from archive
try:
    from legacycodebench.evaluators_v1.documentation import DocumentationEvaluator
    from legacycodebench.evaluators_v1.understanding import UnderstandingEvaluator
    __all__ = ["DocumentationEvaluator", "UnderstandingEvaluator"]
except ImportError:
    # If v1 not available, use v2 as fallback
    from legacycodebench.evaluators_v2.documentation_v2 import DocumentationEvaluatorV2 as DocumentationEvaluator
    __all__ = ["DocumentationEvaluator"]

