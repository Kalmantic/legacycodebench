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
GROUND_TRUTH_CACHE_DIR = BASE_DIR / "cache" / "ground_truth"  # Unified cache location

# Create directories if they don't exist
for dir_path in [TASKS_DIR, DATASETS_DIR, SUBMISSIONS_DIR, REFERENCES_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset sources (GitHub repos)
# Curated list of COBOL repositories for 200-task benchmark
# FIXED (Issue 1.1): Added commit_sha for pinning at release time
DATASET_SOURCES = {
    # Original datasets
    "aws-carddemo": {
        "url": "https://github.com/aws-samples/aws-mainframe-modernization-carddemo",
        "description": "AWS Mainframe Modernization Card Demo",
        "tier": "T2",
        "estimated_files": 25,
        "commit_sha": None,  # TODO: Pin before release
    },
    "rocket-bank": {
        "url": "https://github.com/RocketSoftwareCOBOLandMainframe/BankDemo",
        "description": "Rocket Software Bank Demo",
        "tier": "T3",
        "estimated_files": 15,
        "commit_sha": None,  # TODO: Pin before release
    },
    
    # Tier 1: High Priority - IBM and Enterprise
    "ibm-cobol-fun": {
        "url": "https://github.com/IBM/cobol-is-fun",
        "description": "IBM Developer COBOL tutorials and applications",
        "tier": "T1",
        "estimated_files": 50,
        "commit_sha": None,  # TODO: Pin before release
    },
    "cobol-banking": {
        "url": "https://github.com/ak55m/cobol-banking-system",
        "description": "GnuCOBOL banking system with transactions",
        "tier": "T2",
        "estimated_files": 30,
        "commit_sha": None,  # TODO: Pin before release
    },
    "microfocus-bankdemo": {
        "url": "https://github.com/MicroFocus/BankDemo",
        "description": "Micro Focus enterprise banking demo",
        "tier": "T4",
        "estimated_files": 10,
    },
    "gnucobol-tests": {
        "url": "https://github.com/OCamlPro/gnucobol",
        "description": "GnuCOBOL compiler with NIST test suite",
        "subpath": "tests/cobol85",  # COBOL test files are in this subdirectory
        "tier": "T1",
        "estimated_files": 100,
    },
    "ibm-db2-samples": {
        "url": "https://github.com/IBM/db2-samples",
        "description": "IBM DB2 COBOL integration samples",
        "subpath": "cobol_mf",  # COBOL files are in this subdirectory
        "tier": "T3",
        "estimated_files": 20,
    },
    
    # Tier 2: Secondary - Training and Examples
    "cobol-course": {
        "url": "https://github.com/openmainframeproject/cobol-programming-course",
        "description": "Open Mainframe Project COBOL training",
        "tier": "T1",
        "estimated_files": 50,
    },
    "dscobol-projects": {
        "url": "https://github.com/dscobol/Cobol-Projects",
        "description": "IBM Enterprise COBOL course projects",
        "tier": "T2",
        "estimated_files": 40,
    },
    "gnucobol-examples": {
        "url": "https://github.com/OlegKunitsyn/gnucobol-examples",
        "description": "Modern COBOL microservice examples",
        "tier": "T2",
        "estimated_files": 25,
    },
}

# ===========================================
# LegacyCodeBench v2.1 Scoring Weights
# ===========================================
# LCB_Score = (0.25 x SC) + (0.35 x BF) + (0.25 x SQ) + (0.15 x TR) − Critical_Penalty
# BF = IUE (20%) + BSM (15%) for 100% program coverage

EVALUATION_WEIGHTS = {
    "behavioral_fidelity": 0.35,      # BF: IUE (paragraph execution) + BSM (external call validation)
    "structural_completeness": 0.25,   # SC: Element coverage vs ground truth
    "semantic_quality": 0.25,          # SQ: LLM-as-judge evaluation
    "traceability": 0.15,              # TR: Reference validation
}

# ===========================================
# Evaluation Modes
# ===========================================
# Dual-metric evaluation: Execution (does code work?) + Quality (are docs good?)

EVALUATION_MODES = {
    "execution": {
        "name": "Execution-Based",
        "description": "Pass if generated code executes correctly",
        "requires_execution": True,
        "pass_criteria": {
            "bf_min": 0.70  # BF ≥ 70%
        }
    },
    "balanced": {
        "name": "Balanced",
        "description": "Both execution and documentation quality must meet standards",
        "requires_execution": False,  # Optional but recommended
        "pass_criteria": {
            "bf_min": 0.60,              # BF ≥ 60%
            "quality_score_min": 0.55     # Quality ≥ 55%
        }
    },
    "quality": {
        "name": "Quality-Focused",
        "description": "Pass if documentation meets quality standards",
        "requires_execution": False,
        "pass_criteria": {
            "quality_score_min": 0.60  # Quality ≥ 60%
        }
    }
}

DEFAULT_EVALUATION_MODE = "balanced"

# Legacy thresholds (kept for backward compatibility)
PASS_THRESHOLDS = {
    "behavioral_fidelity_min": 0.70,
    "lcb_score_min": 0.60,
    "critical_failures_max": 0,
    "structural_completeness_min": 0.60,
    "semantic_quality_min": 0.50,
}


# ===========================================
# Dual-Metric Functions
# ===========================================

def calculate_quality_score(result: dict) -> float:
    """
    Calculate documentation quality score (separate from execution).

    Quality considers:
    - Structural Completeness (SC): Are all elements documented?
    - Semantic Quality (SQ): Is documentation accurate and clear?
    - Traceability (TR): Are references valid?
    - Critical Failures: Any severe issues?

    Returns:
        float: Quality score 0.0-1.0, or 0.0 if minimum thresholds not met
    """
    sc = result.get("structural_completeness", 0)
    sq = result.get("semantic_quality", 0)
    tr = result.get("traceability", 0)
    cf = result.get("critical_failures", [])

    # Must meet minimum thresholds for each component
    # v2.1.3: Lowered SC threshold from 60% to 50% to align with other components
    meets_sc = sc >= 0.50
    meets_sq = sq >= 0.50
    meets_tr = tr >= 0.50
    no_critical = len(cf) == 0

    # All components must meet minimum threshold
    if not (meets_sc and meets_sq and meets_tr and no_critical):
        return 0.0

    # Calculate weighted quality score
    # Emphasize structural completeness and semantic quality
    quality = (0.40 * sc) + (0.40 * sq) + (0.20 * tr)
    return quality


def is_task_resolved(result: dict, bf_threshold: float = 0.70) -> bool:
    """
    Check if task is resolved (execution-based).

    A task is resolved if code generated from the documentation
    executes correctly and produces expected outputs.

    Args:
        result: Task evaluation result
        bf_threshold: Minimum behavioral fidelity score (default 0.70)

    Returns:
        bool: True if task is resolved
    """
    bf_details = result.get("details", {}).get("behavioral_fidelity", {})
    is_placeholder = bf_details.get("placeholder", False)

    # Cannot be resolved without actual execution
    if is_placeholder:
        return False

    bf = result.get("behavioral_fidelity", 0)
    return bf >= bf_threshold


def is_task_quality(result: dict, quality_threshold: float = 0.55) -> bool:
    """
    Check if task meets quality standards.

    A task has quality if the documentation is structurally complete,
    semantically accurate, and properly traceable.

    Args:
        result: Task evaluation result
        quality_threshold: Minimum quality score (default 0.55)

    Returns:
        bool: True if task meets quality standards
    """
    quality_score = calculate_quality_score(result)
    return quality_score >= quality_threshold


def is_task_passed(result: dict, mode: str = "balanced") -> bool:
    """
    Determine if a task passes evaluation.

    Evaluation modes:
    - execution: Pass if BF >= 70% (code works)
    - balanced: Pass if BF >= 60% AND Quality >= 55% (both matter)
    - quality: Pass if Quality >= 60% (documentation quality)

    Args:
        result: Task evaluation result
        mode: Evaluation mode (execution, balanced, quality)

    Returns:
        bool: True if task passes under the specified mode
    """
    if mode not in EVALUATION_MODES:
        # Fallback to balanced for unknown modes
        mode = "balanced"

    # FIXED (Issue 6.5): Auto-detect mode based on BF availability
    bf_details = result.get("details", {}).get("behavioral_fidelity", {})
    is_placeholder = bf_details.get("placeholder", False)
    
    if is_placeholder and mode in ["execution", "balanced"]:
        # BF is placeholder, switch to quality mode
        mode = "quality"

    criteria = EVALUATION_MODES[mode]["pass_criteria"]

    if mode == "execution":
        # Execution-based: code must work
        bf_min = criteria.get("bf_min", 0.70)
        return is_task_resolved(result, bf_threshold=bf_min)

    elif mode == "quality":
        # Quality-based: documentation must be good
        quality_min = criteria.get("quality_score_min", 0.60)
        return is_task_quality(result, quality_threshold=quality_min)

    elif mode == "balanced":
        # Both execution and quality matter
        bf_min = criteria.get("bf_min", 0.60)
        quality_min = criteria.get("quality_score_min", 0.55)

        resolved = is_task_resolved(result, bf_threshold=bf_min)
        quality = is_task_quality(result, quality_threshold=quality_min)

        # Both criteria must be met
        return resolved and quality

    return False


def get_pass_status(result: dict, mode: str = "balanced") -> dict:
    """
    Get detailed pass/fail status with reason.

    Args:
        result: Task evaluation result
        mode: Evaluation mode (execution, balanced, quality)

    Returns:
        dict: {
            "passed": bool,
            "resolved": bool,
            "quality": bool,
            "reason": str,
            "criteria": dict
        }
    """
    if mode not in EVALUATION_MODES:
        mode = "balanced"

    criteria = EVALUATION_MODES[mode]["pass_criteria"]

    # Calculate both metrics
    bf_threshold = criteria.get("bf_min", 0.70)
    quality_threshold = criteria.get("quality_score_min", 0.55)

    resolved = is_task_resolved(result, bf_threshold)
    quality = is_task_quality(result, quality_threshold)
    passed = is_task_passed(result, mode)

    # Determine failure reason
    if passed:
        reason = "All criteria met"
    elif mode == "execution":
        bf = result.get("behavioral_fidelity", 0)
        if not resolved:
            reason = f"Execution: BF {bf:.1%} < {bf_threshold:.0%}"
        else:
            reason = "Execution criteria not met"
    elif mode == "quality":
        quality_score = calculate_quality_score(result)
        if not quality:
            reason = f"Quality: {quality_score:.1%} < {quality_threshold:.0%}"
        else:
            reason = "Quality criteria not met"
    elif mode == "balanced":
        if not resolved and not quality:
            bf = result.get("behavioral_fidelity", 0)
            quality_score = calculate_quality_score(result)
            reason = f"Both below threshold: BF {bf:.1%}, Quality {quality_score:.1%}"
        elif not resolved:
            bf = result.get("behavioral_fidelity", 0)
            reason = f"Execution: BF {bf:.1%} < {bf_threshold:.0%}"
        elif not quality:
            quality_score = calculate_quality_score(result)
            reason = f"Quality: {quality_score:.1%} < {quality_threshold:.0%}"
        else:
            reason = "Criteria not met"
    else:
        reason = "Unknown mode"

    # Check for critical failures (override reason if present)
    cf = result.get("critical_failures", [])
    if cf:
        reason = f"Critical failure: {cf[0]}"

    return {
        "passed": passed,
        "resolved": resolved,
        "quality": quality,
        "reason": reason,
        "criteria": {
            "execution_met": resolved,
            "quality_met": quality
        }
    }

# Legacy weights (kept for backward compatibility)
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

# Task selection configuration - v2.0 aligned
# All tasks are DOCUMENTATION tasks, differentiated by COMPLEXITY TIER
# Understanding is validated through Behavioral Fidelity (execution-based testing)
TASK_SELECTION_CONFIG = {
    "loc_ranges": {
        "min": 100,
        "max": 5000,
        "preferred_min": 300,
        "preferred_max": 2000,
    },
    "complexity_thresholds": {
        "T1_basic": 20,        # Simple linear flow
        "T2_moderate": 50,     # PERFORM loops, REDEFINES
        "T3_complex": 80,      # External CALLs, nested structures
        "T4_enterprise": 100,  # GO TO spaghetti, CICS/DB2
    },
    "scoring_weights": {
        "business_logic": 10,
        "dependencies": 5,
        "file_io": 5,
        "in_loc_range": 10,
        "has_comments": -5,
        "complexity": 3,
        "goto_density": 8,      # Higher = more complex
        "dead_code_pct": 5,     # Unreachable code
        "ambiguous_names": 3,   # Single-letter variables
    },
    # v2.0: 200 documentation tasks across 4 complexity tiers
    # NO separate "understanding" tasks - understanding is validated via BF score
    "task_distribution": {
        "total_tasks": 200,
        "tier_distribution": {
            "T1_basic": 80,       # 40% - Straightforward programs (LOC: 300-500)
            "T2_moderate": 70,    # 35% - PERFORM loops, file ops (LOC: 500-1000)
            "T3_complex": 40,     # 20% - External calls, business rules (LOC: 1000-2000)
            "T4_enterprise": 10,  # 5%  - GO TO spaghetti, CICS/DB2 (LOC: 2000+)
        },
        # v2.0: Map tiers to difficulty levels for task IDs
        "tier_to_difficulty": {
            "T1": "easy",
            "T2": "medium", 
            "T3": "hard",
            "T4": "expert",
        },
        # v2.0: LOC ranges per tier (aligned with PRD Section 8)
        "tier_loc_ranges": {
            "T1": (300, 500),
            "T2": (500, 1000),
            "T3": (1000, 2000),
            "T4": (2000, 5000),
        },
    }
}

# Anti-pattern injection for synthetic programs (from v2.0 spec)
ANTI_PATTERN_CONFIG = {
    "T1_basic": {
        "goto_density": 0.00,
        "dead_code_pct": 0.00,
        "ambiguous_names_pct": 0.10,
        "filler_bytes": 100,
    },
    "T2_moderate": {
        "goto_density": 0.05,
        "dead_code_pct": 0.05,
        "ambiguous_names_pct": 0.20,
        "filler_bytes": 500,
    },
    "T3_complex": {
        "goto_density": 0.15,
        "dead_code_pct": 0.10,
        "ambiguous_names_pct": 0.35,
        "filler_bytes": 2000,
    },
    "T4_enterprise": {
        "goto_density": 0.25,
        "dead_code_pct": 0.20,
        "ambiguous_names_pct": 0.50,
        "filler_bytes": 5000,
    },
}

# Critical failures (automatic disqualification)
# FIXED (Issue 6.3): Added CF-07 and CF-08
CRITICAL_FAILURES = {
    "CF-01": "Missing primary calculation",
    "CF-02": "Hallucinated module (references non-existent program/paragraph)",
    "CF-03": "Wrong data transformation (≥10% output mismatch)",
    "CF-04": "Missing error handler (FILE STATUS, ON SIZE ERROR)",
    "CF-05": "Broken traceability (≥20% invalid references)",
    "CF-06": "False positive (passes with MISSING/AMBIGUOUS markers)",
    "CF-07": "Unspecified file extension (references file without proper SELECT/ASSIGN)",
    "CF-08": "Wrong specification (documented behavior contradicts actual code)",
}

# AI model configurations
# Note: max_tokens is for OUTPUT only. Input uses model's context window.
AI_MODELS = {
    "claude-sonnet-4": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "temperature": 0,  # FIXED (Issue 4.2): Enforce deterministic outputs
        "max_tokens": 16000,  # Claude supports up to 8192 per response, but we use continuation
    },
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0,  # FIXED (Issue 4.2): Enforce deterministic outputs
        "max_tokens": 16384,  # GPT-4o max output
    },
    "gpt-4": {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0,  # FIXED (Issue 4.2): Enforce deterministic outputs
        "max_tokens": 8192,  # GPT-4 max output
    },
    # FIXED (Issue 7.6): Renamed from 'aws-transform' to clarify this is Claude via Bedrock
    "claude-sonnet-via-bedrock": {
        "provider": "aws",
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",  # AWS Bedrock model ID
        "temperature": 0,  # FIXED (Issue 4.2): Enforce deterministic outputs
        "max_tokens": 8000,
        "region": "us-east-1",  # Can be overridden by AWS_REGION env var
    },
    # DocMolt - Specialized documentation generation service
    "docmolt-gpt4o": {
        "provider": "docmolt",
        "model": "gpt-4o",
        "artefact": "documentation",  # Options: documentation, technical-spec, tdd, api-docs
        "language": "cobol",
        "api_endpoint": "https://docmolt.hexaview.ai/api/docstream",
        "temperature": 0,  # FIXED (Issue 4.2): Enforce deterministic outputs
        "max_tokens": 16384,
    },
    "docmolt-gpt4o-mini": {
        "provider": "docmolt",
        "model": "gpt-4o-mini",
        "artefact": "documentation",
        "language": "cobol",
        "api_endpoint": "https://docmolt.hexaview.ai/api/docstream",
        "temperature": 0,  # FIXED (Issue 4.2): Enforce deterministic outputs
        "max_tokens": 16384,
    },
    "docmolt-claude": {
        "provider": "docmolt",
        "model": "claude-sonnet-4",  # If supported by DocMolt
        "artefact": "documentation",
        "language": "cobol",
        "api_endpoint": "https://docmolt.hexaview.ai/api/docstream",
        "temperature": 0,  # FIXED (Issue 4.2): Enforce deterministic outputs
        "max_tokens": 16000,
    },
    # Google Gemini models
    "gemini-2.5-flash": {
        "provider": "google",
        "model": "models/gemini-2.5-flash",  # Best available flash model
        "temperature": 0,  # Enforce deterministic outputs
        "max_tokens": 8192,
    },
    "gemini-1.5-pro": {
        "provider": "google",
        "model": "models/gemini-2.5-flash",  # FIXED: Use available model (1.5 deprecated)
        "temperature": 0,  # Enforce deterministic outputs
        "max_tokens": 8192,
    },
    "gemini-1.5-flash": {
        "provider": "google",
        "model": "gemini-1.5-flash",
        "temperature": 0,  # Enforce deterministic outputs
        "max_tokens": 8192,
    },
}

# DocMolt-specific configuration
DOCMOLT_CONFIG = {
    "api_endpoint": "https://docmolt.hexaview.ai/api/docstream",
    "timeout_seconds": 300,  # 5 minutes for large COBOL files
    "max_retries": 3,
    "retry_delay_seconds": 2,
    "language": "cobol",
    "default_artefact": "documentation",  # Best for LegacyCodeBench
    "api_key_env_var": "DOCMOLT_API_KEY",
    "supported_artefacts": [
        "documentation",     # Comprehensive technical documentation
        "technical-spec",    # Technical specification document
        "tdd",              # Test-driven development documentation
        "api-docs",         # API documentation
        "readme",           # README file
        "user-guide",       # User guide documentation
    ],
}

def get_datasets_by_tier(tier: str = None) -> Dict[str, Dict]:
    """Get dataset sources filtered by tier
    
    Args:
        tier: One of 'T1', 'T2', 'T3', 'T4' or None for all
        
    Returns:
        Dictionary of dataset sources matching the tier
    """
    if tier is None:
        return DATASET_SOURCES
    
    return {
        source_id: info 
        for source_id, info in DATASET_SOURCES.items() 
        if info.get("tier") == tier
    }


def get_tier_task_count(tier: str) -> int:
    """Get the target number of tasks for a tier"""
    return TASK_SELECTION_CONFIG["task_distribution"]["tier_distribution"].get(tier, 0)


def get_total_estimated_files() -> Dict[str, int]:
    """Get estimated file counts by tier"""
    tier_counts = {"T1": 0, "T2": 0, "T3": 0, "T4": 0}
    for source_id, info in DATASET_SOURCES.items():
        tier = info.get("tier", "T1")
        tier_counts[tier] += info.get("estimated_files", 0)
    return tier_counts


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
        "evaluation_weights": EVALUATION_WEIGHTS,
        "documentation_weights": DOCUMENTATION_WEIGHTS,
        "understanding_weights": UNDERSTANDING_WEIGHTS,
        "overall_weights": OVERALL_WEIGHTS,
        "required_doc_sections": REQUIRED_DOC_SECTIONS,
        "performance_tiers": PERFORMANCE_TIERS,
        "task_selection_config": TASK_SELECTION_CONFIG,
        "anti_pattern_config": ANTI_PATTERN_CONFIG,
        "critical_failures": CRITICAL_FAILURES,
        "ai_models": AI_MODELS,
    }


def print_dataset_summary():
    """Print a summary of configured datasets"""
    print("\n" + "=" * 60)
    print("LegacyCodeBench v2.0 Dataset Configuration")
    print("=" * 60)
    
    tier_counts = get_total_estimated_files()
    tier_targets = TASK_SELECTION_CONFIG["task_distribution"]["tier_distribution"]
    
    for tier in ["T1", "T2", "T3", "T4"]:
        tier_name = {
            "T1": "Basic",
            "T2": "Moderate", 
            "T3": "Complex",
            "T4": "Enterprise"
        }[tier]
        
        sources = get_datasets_by_tier(tier)
        target = tier_targets.get(f"{tier}_{'basic' if tier == 'T1' else 'moderate' if tier == 'T2' else 'complex' if tier == 'T3' else 'enterprise'}", 0)
        
        print(f"\n{tier}: {tier_name} (Target: {target} tasks)")
        print("-" * 40)
        for source_id, info in sources.items():
            print(f"  • {source_id}: ~{info.get('estimated_files', '?')} files")
        print(f"  Total estimated: {tier_counts[tier]} files")
    
    print("\n" + "=" * 60)
    total = sum(tier_targets.values())
    print(f"Total Target Tasks: {total}")
    print("=" * 60)

