#!/usr/bin/env python3
"""
Test script for AWS Transform (AWS Bedrock) integration

This script tests the AWS Bedrock integration for LegacyCodeBench.
It verifies:
1. AWS credentials are properly configured
2. boto3 is installed and working
3. The Bedrock API can be called successfully
4. Documentation and understanding tasks work correctly

Usage:
    python scripts/test_aws_transform.py

Environment Variables Required:
    AWS_ACCESS_KEY_ID - AWS access key
    AWS_SECRET_ACCESS_KEY - AWS secret access key
    AWS_REGION (optional) - AWS region (default: us-east-1)
    AWS_BEDROCK_MODEL_ID (optional) - Model ID to use
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from legacycodebench.ai_integration import get_ai_model
from legacycodebench.config import AI_MODELS

# Sample COBOL code for testing
SAMPLE_COBOL = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. CALC-INTEREST.

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  PRINCIPAL           PIC 9(7)V99.
       01  RATE                PIC 9V9999.
       01  YEARS               PIC 99.
       01  INTEREST            PIC 9(7)V99.

       PROCEDURE DIVISION.
       MAIN-LOGIC.
           MOVE 10000.00 TO PRINCIPAL
           MOVE 0.0525 TO RATE
           MOVE 5 TO YEARS

           COMPUTE INTEREST = PRINCIPAL * RATE * YEARS

           DISPLAY "PRINCIPAL: " PRINCIPAL
           DISPLAY "RATE: " RATE
           DISPLAY "YEARS: " YEARS
           DISPLAY "INTEREST: " INTEREST

           STOP RUN.
"""

class SimpleTask:
    """Mock task object for testing"""
    def __init__(self):
        self.task_id = "test-001"
        self.task_description = "Calculate interest on a principal amount"

def test_aws_credentials():
    """Test if AWS credentials are configured"""
    print("\n" + "="*60)
    print("TEST 1: AWS Credentials Check")
    print("="*60)

    required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    optional_vars = ["AWS_REGION", "AWS_BEDROCK_MODEL_ID"]

    all_configured = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✓ {var}: {'*' * 20} (configured)")
        else:
            print(f"✗ {var}: NOT CONFIGURED")
            all_configured = False

    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"✓ {var}: {value}")
        else:
            print(f"  {var}: (using default)")

    return all_configured

def test_boto3_import():
    """Test if boto3 is installed"""
    print("\n" + "="*60)
    print("TEST 2: boto3 Import Check")
    print("="*60)

    try:
        import boto3
        import botocore
        print(f"✓ boto3 version: {boto3.__version__}")
        print(f"✓ botocore version: {botocore.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("  Please install: pip install boto3 botocore")
        return False

def test_model_configuration():
    """Test if aws-transform model is configured"""
    print("\n" + "="*60)
    print("TEST 3: Model Configuration Check")
    print("="*60)

    if "aws-transform" not in AI_MODELS:
        print("✗ 'aws-transform' not found in AI_MODELS")
        return False

    config = AI_MODELS["aws-transform"]
    print(f"✓ Model ID: {config['model']}")
    print(f"✓ Provider: {config['provider']}")
    print(f"✓ Temperature: {config['temperature']}")
    print(f"✓ Max Tokens: {config['max_tokens']}")
    print(f"✓ Region: {config.get('region', 'N/A')}")

    return True

def test_documentation_generation():
    """Test documentation generation"""
    print("\n" + "="*60)
    print("TEST 4: Documentation Generation")
    print("="*60)

    try:
        # Create temporary COBOL file
        temp_file = Path("temp_test_cobol.cbl")
        with open(temp_file, 'w') as f:
            f.write(SAMPLE_COBOL)

        # Create AI model interface
        model = get_ai_model("aws-transform")
        task = SimpleTask()

        print("Calling AWS Bedrock to generate documentation...")
        print("(This may take 10-30 seconds)")

        # Generate documentation
        result = model.generate_documentation(task, [temp_file])

        # Clean up
        temp_file.unlink()

        # Check result
        if result and len(result) > 100:
            print(f"✓ Documentation generated: {len(result)} characters")
            print("\nFirst 500 characters of output:")
            print("-" * 60)
            print(result[:500])
            print("-" * 60)
            return True
        else:
            print(f"✗ Documentation too short: {len(result)} characters")
            print(f"Output: {result}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_understanding_generation():
    """Test understanding generation (JSON extraction)"""
    print("\n" + "="*60)
    print("TEST 5: Understanding Generation")
    print("="*60)

    try:
        # Create temporary COBOL file
        temp_file = Path("temp_test_cobol.cbl")
        with open(temp_file, 'w') as f:
            f.write(SAMPLE_COBOL)

        # Create AI model interface
        model = get_ai_model("aws-transform")
        task = SimpleTask()

        print("Calling AWS Bedrock to generate understanding output...")
        print("(This may take 10-30 seconds)")

        # Generate understanding
        result = model.generate_understanding(task, [temp_file])

        # Clean up
        temp_file.unlink()

        # Check result
        if result and len(result) > 50:
            print(f"✓ Understanding output generated: {len(result)} characters")
            print("\nOutput:")
            print("-" * 60)
            print(result[:500])
            print("-" * 60)

            # Try to parse as JSON
            import json
            try:
                json.loads(result)
                print("✓ Valid JSON structure")
            except json.JSONDecodeError:
                print("⚠ Warning: Output is not valid JSON (may need extraction)")

            return True
        else:
            print(f"✗ Understanding output too short: {len(result)} characters")
            print(f"Output: {result}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("AWS Transform Integration Test Suite")
    print("="*60)

    results = []

    # Run tests
    results.append(("AWS Credentials", test_aws_credentials()))
    results.append(("boto3 Import", test_boto3_import()))
    results.append(("Model Configuration", test_model_configuration()))

    # Only run API tests if credentials are configured
    if results[0][1]:
        results.append(("Documentation Generation", test_documentation_generation()))
        results.append(("Understanding Generation", test_understanding_generation()))
    else:
        print("\n⚠ Skipping API tests - AWS credentials not configured")
        print("\nTo configure AWS credentials:")
        print("  export AWS_ACCESS_KEY_ID='your-access-key'")
        print("  export AWS_SECRET_ACCESS_KEY='your-secret-key'")
        print("  export AWS_REGION='us-east-1'  # optional")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! AWS Transform integration is working.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
