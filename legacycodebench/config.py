"""Configuration for LegacyCodeBench"""

import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent
TASKS_DIR = BASE_DIR / "tasks"
DATASETS_DIR = BASE_DIR / "datasets"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
REFERENCES_DIR = BASE_DIR / "references"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
for dir_path in [TASKS_DIR, DATASETS_DIR, SUBMISSIONS_DIR, REFERENCES_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset sources (GitHub repos)
DATASET_SOURCES = {
    # Original datasets
    "aws-carddemo": {
        "url": "https://github.com/aws-samples/aws-mainframe-modernization-carddemo",
        "description": "AWS Mainframe Modernization Card Demo",
    },
    "az-legacy": {
        "url": "https://github.com/bhbandam/AZ-Legacy-Engineering",
        "description": "Azure Legacy Engineering",
    },
    "rocket-bank": {
        "url": "https://github.com/RocketSoftwareCOBOLandMainframe/BankDemo",
        "description": "Rocket Software Bank Demo",
    },
    # Tier 1: High Priority - IBM and Enterprise
    "ibm-cobol-fun": {
        "url": "https://github.com/IBM/cobol-is-fun",
        "description": "IBM Developer COBOL tutorials and applications",
    },
    "cobol-banking": {
        "url": "https://github.com/ak55m/cobol-banking-system",
        "description": "GnuCOBOL banking system with transactions",
    },
    "microfocus-bankdemo": {
        "url": "https://github.com/MicroFocus/BankDemo",
        "description": "Micro Focus enterprise banking demo",
    },
    "gnucobol-tests": {
        "url": "https://github.com/OCamlPro/gnucobol",
        "description": "GnuCOBOL compiler with NIST test suite",
        "subpath": "tests/cobol85",  # COBOL test files are in this subdirectory
    },
    "ibm-db2-samples": {
        "url": "https://github.com/IBM/db2-samples",
        "description": "IBM DB2 COBOL integration samples",
        "subpath": "cobol_mf",  # COBOL files are in this subdirectory
    },
    # Tier 2: Secondary - Training and Examples
    "cobol-course": {
        "url": "https://github.com/openmainframeproject/cobol-programming-course",
        "description": "Open Mainframe Project COBOL training",
    },
    "dscobol-projects": {
        "url": "https://github.com/dscobol/Cobol-Projects",
        "description": "IBM Enterprise COBOL course projects",
    },
    "gnucobol-examples": {
        "url": "https://github.com/OlegKunitsyn/gnucobol-examples",
        "description": "Modern COBOL microservice examples",
    },
}

# Evaluation weights
DOCUMENTATION_WEIGHTS = {
    "required_sections": 0.40,
    "business_rule_coverage": 0.30,
    "format_quality": 0.20,
    "appropriate_length": 0.10,
}

UNDERSTANDING_WEIGHTS = {
    "precision": 0.50,
    "recall": 0.50,
}

# Conversion target languages
CONVERSION_TARGETS = {
    "java": {
        "version": "17",
        "extension": ".java",
        "test_framework": "junit",
        "compile_cmd": "javac",
        "run_cmd": "java",
    },
    "python": {
        "version": "3.11",
        "extension": ".py",
        "test_framework": "pytest",
        "compile_cmd": None,  # Interpreted
        "run_cmd": "python3",
    },
    "csharp": {
        "version": "net8.0",
        "extension": ".cs",
        "test_framework": "xunit",
        "compile_cmd": "dotnet build",
        "run_cmd": "dotnet run",
    },
}

# Conversion evaluation weights
CONVERSION_WEIGHTS = {
    "compilation": 0.20,       # Code compiles/runs without errors
    "functional": 0.40,        # Produces correct output for test cases
    "code_quality": 0.20,      # Proper types, structure, naming
    "completeness": 0.20,      # All business logic converted
}

OVERALL_WEIGHTS = {
    "documentation": 0.33,
    "understanding": 0.33,
    "conversion": 0.34,
}

# Required sections for documentation tasks
REQUIRED_DOC_SECTIONS = [
    "business_purpose",
    "business_rules",
    "edge_cases",
    "data_structures",
    "algorithm_overview",
]

# Performance tiers
PERFORMANCE_TIERS = {
    "excellent": 0.60,  # >60%: Practically useful with oversight
    "good": 0.40,       # 40-60%: Research baseline
    "fair": 0.20,       # 20-40%: Expected AI performance
    "poor": 0.00,       # <20%: Not functional
}

# Task selection configuration
TASK_SELECTION_CONFIG = {
    "loc_ranges": {
        "min": 300,
        "max": 3000,
        "preferred_min": 500,
        "preferred_max": 2000,
    },
    "complexity_thresholds": {
        "easy": 30,
        "medium": 60,
        "hard": 100,
    },
    "scoring_weights": {
        "business_logic": 10,
        "dependencies": 5,
        "file_io": 5,
        "in_loc_range": 10,
        "has_comments": -5,
        "complexity": 3,
    },
    "task_distribution": {
        "total_tasks": 300,
        "documentation_tasks": 100,
        "understanding_tasks": 100,
        "conversion_tasks": 100,
        "difficulty_distribution": {
            "easy": 0.33,
            "medium": 0.47,
            "hard": 0.20,
        },
        "conversion_target_distribution": {
            "java": 0.50,      # 50 tasks
            "python": 0.35,   # 35 tasks
            "csharp": 0.15,   # 15 tasks
        }
    }
}

# AI model configurations
# Note: max_tokens is for OUTPUT only. Input uses model's context window.
AI_MODELS = {
    # Anthropic models
    "claude-sonnet-4": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.2,
        "max_tokens": 16000,
    },
    "claude-opus-4": {
        "provider": "anthropic",
        "model": "claude-opus-4-20250514",
        "temperature": 0.2,
        "max_tokens": 16000,
    },
    # OpenAI models
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.2,
        "max_tokens": 16384,
    },
    "gpt-4-turbo": {
        "provider": "openai",
        "model": "gpt-4-turbo",
        "temperature": 0.2,
        "max_tokens": 4096,
    },
    "o1": {
        "provider": "openai",
        "model": "o1",
        "temperature": 1.0,  # o1 requires temperature=1
        "max_tokens": 32768,
    },
    # Google Gemini models
    "gemini-pro": {
        "provider": "google",
        "model": "gemini-2.5-pro",
        "temperature": 0.2,
        "max_tokens": 8192,
    },
    "gemini-flash": {
        "provider": "google",
        "model": "gemini-2.5-flash",
        "temperature": 0.2,
        "max_tokens": 8192,
    },
    # AWS Transform (placeholder - requires S3 workflow)
    "aws-transform": {
        "provider": "aws",
        "model": "transform",
        "temperature": 0.2,
        "max_tokens": 8000,
    },
    # Hexaview CodeMolt (placeholder - web UI only, no API)
    "codemolt": {
        "provider": "hexaview",
        "model": "codemolt",
        "temperature": 0.2,
        "max_tokens": 4000,
    },
}

def get_config() -> Dict[str, Any]:
    """Get full configuration dictionary"""
    return {
        "base_dir": str(BASE_DIR),
        "tasks_dir": str(TASKS_DIR),
        "datasets_dir": str(DATASETS_DIR),
        "submissions_dir": str(SUBMISSIONS_DIR),
        "references_dir": str(REFERENCES_DIR),
        "results_dir": str(RESULTS_DIR),
        "dataset_sources": DATASET_SOURCES,
        "documentation_weights": DOCUMENTATION_WEIGHTS,
        "understanding_weights": UNDERSTANDING_WEIGHTS,
        "conversion_weights": CONVERSION_WEIGHTS,
        "conversion_targets": CONVERSION_TARGETS,
        "overall_weights": OVERALL_WEIGHTS,
        "required_doc_sections": REQUIRED_DOC_SECTIONS,
        "performance_tiers": PERFORMANCE_TIERS,
        "task_selection_config": TASK_SELECTION_CONFIG,
        "ai_models": AI_MODELS,
    }

