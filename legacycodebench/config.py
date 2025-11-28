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

OVERALL_WEIGHTS = {
    "documentation": 0.50,
    "understanding": 0.50,
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
        "total_tasks": 15,
        "documentation_tasks": 8,
        "understanding_tasks": 7,
        "difficulty_distribution": {
            "easy": 0.33,
            "medium": 0.47,
            "hard": 0.20,
        }
    }
}

# AI model configurations
# Note: max_tokens is for OUTPUT only. Input uses model's context window.
AI_MODELS = {
    "claude-sonnet-4": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.2,
        "max_tokens": 16000,  # Claude supports up to 8192 per response, but we use continuation
    },
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.2,
        "max_tokens": 16384,  # GPT-4o max output
    },
    "gpt-4": {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.2,
        "max_tokens": 8192,  # GPT-4 max output
    },
    "aws-transform": {
        "provider": "aws",
        "model": "transform",
        "temperature": 0.2,
        "max_tokens": 8000,
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
        "overall_weights": OVERALL_WEIGHTS,
        "required_doc_sections": REQUIRED_DOC_SECTIONS,
        "performance_tiers": PERFORMANCE_TIERS,
        "task_selection_config": TASK_SELECTION_CONFIG,
        "ai_models": AI_MODELS,
    }

