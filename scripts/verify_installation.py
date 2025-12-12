#!/usr/bin/env python3
"""
LegacyCodeBench v2.0 Installation Verification Script

This script verifies that all components are properly installed and configured.
"""

import sys
import os
from pathlib import Path
import subprocess
import importlib

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    """Print a section header"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

def print_success(text):
    """Print success message"""
    print(f"{GREEN}✓{RESET} {text}")

def print_error(text):
    """Print error message"""
    print(f"{RED}✗{RESET} {text}")

def print_warning(text):
    """Print warning message"""
    print(f"{YELLOW}⚠{RESET} {text}")

def print_info(text):
    """Print info message"""
    print(f"  {text}")

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 11:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} (3.11+ required)")
        return False

def check_python_packages():
    """Check required Python packages"""
    print_header("Checking Python Packages")

    required_packages = [
        ('openai', 'OpenAI API client (optional but recommended)'),
        ('anthropic', 'Anthropic API client (optional but recommended)'),
        ('requests', 'HTTP requests library'),
    ]

    all_installed = True
    for package, description in required_packages:
        try:
            importlib.import_module(package)
            print_success(f"{package}: {description}")
        except ImportError:
            if package in ['openai', 'anthropic']:
                print_warning(f"{package}: {description} - NOT INSTALLED (optional)")
            else:
                print_error(f"{package}: {description} - NOT INSTALLED")
                all_installed = False

    return all_installed

def check_docker():
    """Check Docker installation"""
    print_header("Checking Docker")

    try:
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            version = result.stdout.strip()
            print_success(f"Docker installed: {version}")

            # Check if Docker is running
            try:
                result = subprocess.run(
                    ['docker', 'ps'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode == 0:
                    print_success("Docker daemon is running")
                    return True
                else:
                    print_error("Docker daemon is not running")
                    print_info("Start Docker Desktop or run: sudo systemctl start docker")
                    return False

            except Exception as e:
                print_error(f"Docker daemon check failed: {e}")
                return False

        else:
            print_error("Docker command failed")
            return False

    except FileNotFoundError:
        print_error("Docker is not installed")
        print_info("Install from: https://www.docker.com/products/docker-desktop")
        return False
    except Exception as e:
        print_error(f"Docker check failed: {e}")
        return False

def check_docker_image():
    """Check if COBOL Docker image exists"""
    print_header("Checking COBOL Docker Image")

    try:
        result = subprocess.run(
            ['docker', 'images', '-q', 'legacycodebench-cobol:latest'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.stdout.strip():
            print_success("COBOL Docker image found: legacycodebench-cobol:latest")

            # Test the image
            try:
                result = subprocess.run(
                    ['docker', 'run', '--rm', 'legacycodebench-cobol:latest', 'cobc', '--version'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0:
                    version = result.stdout.split('\n')[0]
                    print_success(f"GnuCOBOL version: {version}")
                    return True
                else:
                    print_error("Failed to run GnuCOBOL in Docker")
                    return False

            except Exception as e:
                print_error(f"Docker image test failed: {e}")
                return False
        else:
            print_error("COBOL Docker image not found")
            print_info("Build with: docker build -t legacycodebench-cobol:latest docker/cobol-sandbox/")
            return False

    except Exception as e:
        print_error(f"Docker image check failed: {e}")
        return False

def check_api_keys():
    """Check API key configuration"""
    print_header("Checking API Keys")

    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')

    has_keys = False

    if openai_key:
        print_success(f"OPENAI_API_KEY is set (length: {len(openai_key)})")
        has_keys = True
    else:
        print_warning("OPENAI_API_KEY is not set (optional but recommended)")
        print_info("Set with: export OPENAI_API_KEY='sk-...'")

    if anthropic_key:
        print_success(f"ANTHROPIC_API_KEY is set (length: {len(anthropic_key)})")
        has_keys = True
    else:
        print_warning("ANTHROPIC_API_KEY is not set (optional but recommended)")
        print_info("Set with: export ANTHROPIC_API_KEY='sk-ant-...'")

    if not has_keys:
        print_warning("No LLM API keys configured - LLM features will use fallbacks")
        return False

    return True

def check_legacycodebench_modules():
    """Check LegacyCodeBench modules"""
    print_header("Checking LegacyCodeBench Modules")

    modules = [
        ('legacycodebench.execution.code_generator', 'Code Generator'),
        ('legacycodebench.execution.test_generator', 'Test Generator'),
        ('legacycodebench.execution.cobol_executor', 'COBOL Executor'),
        ('legacycodebench.execution.behavior_comparator', 'Behavior Comparator'),
        ('legacycodebench.evaluators_v2.semantic_quality', 'Semantic Quality Evaluator'),
        ('legacycodebench.evaluators_v2.structural_completeness', 'Structural Completeness Evaluator'),
        ('legacycodebench.evaluators_v2.traceability', 'Traceability Evaluator'),
        ('legacycodebench.evaluators_v2.documentation_v2', 'Documentation Evaluator v2.0'),
        ('legacycodebench.scoring', 'Scoring System'),
        ('legacycodebench.config', 'Configuration'),
    ]

    all_imported = True
    for module_name, description in modules:
        try:
            importlib.import_module(module_name)
            print_success(f"{description}: {module_name}")
        except ImportError as e:
            print_error(f"{description}: {module_name} - IMPORT FAILED")
            print_info(f"  Error: {e}")
            all_imported = False

    return all_imported

def test_code_generator():
    """Test code generator initialization"""
    print_header("Testing Code Generator")

    try:
        from legacycodebench.execution.code_generator import ConstrainedCodeGenerator

        generator = ConstrainedCodeGenerator()
        print_success("Code Generator initialized successfully")
        print_info(f"  LLM model: {generator.llm_model}")
        return True

    except Exception as e:
        print_error(f"Code Generator test failed: {e}")
        return False

def test_cobol_executor():
    """Test COBOL executor initialization"""
    print_header("Testing COBOL Executor")

    try:
        from legacycodebench.execution.cobol_executor import COBOLExecutor

        executor = COBOLExecutor()
        print_success("COBOL Executor initialized successfully")
        print_info(f"  Docker image: {executor.docker_image}")
        print_info(f"  Timeout: {executor.timeout_seconds}s")
        print_info(f"  Memory limit: {executor.memory_limit}")
        return True

    except Exception as e:
        print_error(f"COBOL Executor test failed: {e}")
        print_info(f"  Make sure Docker is running and image is built")
        return False

def test_scoring_system():
    """Test scoring system"""
    print_header("Testing Scoring System v2.0")

    try:
        from legacycodebench.scoring import ScoringSystem

        scorer = ScoringSystem()

        # Test v2.0 score calculation
        test_score = scorer.calculate_lcb_v2_score(
            sc=80.0,  # Structural Completeness
            bf=75.0,  # Behavioral Fidelity
            sq=85.0,  # Semantic Quality
            tr=90.0   # Traceability
        )

        expected_score = (0.30 * 80) + (0.35 * 75) + (0.25 * 85) + (0.10 * 90)

        if abs(test_score - expected_score) < 0.01:
            print_success(f"v2.0 scoring formula verified: {test_score:.2f}")
            print_info(f"  Formula: (0.30×SC) + (0.35×BF) + (0.25×SQ) + (0.10×TR)")
            return True
        else:
            print_error(f"v2.0 scoring formula mismatch: {test_score} vs {expected_score}")
            return False

    except Exception as e:
        print_error(f"Scoring system test failed: {e}")
        return False

def print_summary(results):
    """Print summary of checks"""
    print_header("Summary")

    total = len(results)
    passed = sum(1 for r in results.values() if r)
    failed = total - passed

    print(f"\nTotal Checks: {total}")
    print(f"{GREEN}Passed: {passed}{RESET}")
    print(f"{RED}Failed: {failed}{RESET}")

    if failed == 0:
        print(f"\n{GREEN}{'='*60}{RESET}")
        print(f"{GREEN}✓ All checks passed! LegacyCodeBench v2.0 is ready to use.{RESET}")
        print(f"{GREEN}{'='*60}{RESET}")
        print(f"\nNext steps:")
        print(f"  1. Run a sample evaluation: python -m legacycodebench.cli evaluate --help")
        print(f"  2. Read the implementation guide: V2_IMPLEMENTATION_GUIDE.md")
        print(f"  3. Start benchmarking!")
        return True
    else:
        print(f"\n{RED}{'='*60}{RESET}")
        print(f"{RED}✗ Some checks failed. Please review the errors above.{RESET}")
        print(f"{RED}{'='*60}{RESET}")

        # Identify critical failures
        critical_failures = []
        if not results.get('Python Version'):
            critical_failures.append("Python 3.11+ required")
        if not results.get('Docker'):
            critical_failures.append("Docker required for execution")
        if not results.get('COBOL Docker Image'):
            critical_failures.append("COBOL Docker image must be built")
        if not results.get('LegacyCodeBench Modules'):
            critical_failures.append("Module import errors - check installation")

        if critical_failures:
            print(f"\n{RED}Critical issues:{RESET}")
            for issue in critical_failures:
                print(f"  - {issue}")

        print(f"\nRefer to V2_IMPLEMENTATION_GUIDE.md for setup instructions.")
        return False

def main():
    """Main verification flow"""
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}LegacyCodeBench v2.0 - Installation Verification{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    results = {}

    # Run all checks
    results['Python Version'] = check_python_version()
    results['Python Packages'] = check_python_packages()
    results['Docker'] = check_docker()
    results['COBOL Docker Image'] = check_docker_image()
    results['API Keys'] = check_api_keys()
    results['LegacyCodeBench Modules'] = check_legacycodebench_modules()
    results['Code Generator'] = test_code_generator()
    results['COBOL Executor'] = test_cobol_executor()
    results['Scoring System v2.0'] = test_scoring_system()

    # Print summary
    success = print_summary(results)

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
