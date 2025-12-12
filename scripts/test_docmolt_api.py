"""
DocMolt API Testing Script

This script tests the DocMolt API to understand its capabilities:
- Supported artefact types
- Response format
- COBOL language support
- Available models
- Error handling
"""

import requests
import json
import os
from pathlib import Path

DOCMOLT_API_ENDPOINT = "https://docmolt.hexaview.ai/api/docstream"

# Sample COBOL code for testing
SAMPLE_COBOL = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. SAMPLE-PROGRAM.

       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT CUSTOMER-FILE ASSIGN TO 'CUSTFILE.DAT'
               ORGANIZATION IS LINE SEQUENTIAL.

       DATA DIVISION.
       FILE SECTION.
       FD  CUSTOMER-FILE.
       01  CUSTOMER-RECORD.
           05  CUST-ID             PIC 9(10).
           05  CUST-NAME           PIC X(50).
           05  CUST-BALANCE        PIC 9(7)V99.
           05  ACCOUNT-TYPE        PIC X(1).

       WORKING-STORAGE SECTION.
       01  WS-EOF                  PIC X VALUE 'N'.
       01  WS-TOTAL-BALANCE        PIC 9(9)V99 VALUE ZEROS.
       01  MINIMUM-BALANCE         PIC 9(7)V99 VALUE 100.00.

       PROCEDURE DIVISION.
       MAIN-PROCEDURE.
           OPEN INPUT CUSTOMER-FILE
           PERFORM UNTIL WS-EOF = 'Y'
               READ CUSTOMER-FILE
                   AT END MOVE 'Y' TO WS-EOF
                   NOT AT END PERFORM PROCESS-RECORD
               END-READ
           END-PERFORM
           CLOSE CUSTOMER-FILE
           DISPLAY 'Total Balance: ' WS-TOTAL-BALANCE
           STOP RUN.

       PROCESS-RECORD.
           IF CUST-BALANCE < MINIMUM-BALANCE
               DISPLAY 'Low balance account: ' CUST-ID
           END-IF
           ADD CUST-BALANCE TO WS-TOTAL-BALANCE.

       END PROGRAM SAMPLE-PROGRAM.
"""


def test_docmolt_api(artefact_type: str, model: str, api_key: str = None):
    """Test DocMolt API with specific parameters"""

    print(f"\n{'='*80}")
    print(f"Testing: artefact='{artefact_type}', model='{model}'")
    print(f"{'='*80}")

    payload = {
        "code": SAMPLE_COBOL,
        "filename": "SAMPLE-PROGRAM.cbl",
        "language": "cobol",
        "artefact": artefact_type,
        "model": model,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        print(f"Sending request to {DOCMOLT_API_ENDPOINT}...")
        response = requests.post(
            DOCMOLT_API_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=120
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")

        if response.status_code == 200:
            try:
                result = response.json()
                print(f"\n✓ SUCCESS")
                print(f"Response Keys: {list(result.keys())}")

                # Try to extract documentation
                for key in ['documentation', 'output', 'content', 'result', 'data', 'text']:
                    if key in result:
                        content = result[key]
                        print(f"\nFound content in key: '{key}'")
                        print(f"Content length: {len(str(content))} chars")
                        print(f"Content preview (first 500 chars):")
                        print("-" * 80)
                        print(str(content)[:500])
                        print("-" * 80)
                        return result

                # If no known key, print full response
                print(f"\nFull Response:")
                print(json.dumps(result, indent=2)[:1000])
                return result

            except json.JSONDecodeError:
                print(f"Response is not JSON. Raw response:")
                print(response.text[:1000])
                return {"raw_text": response.text}
        else:
            print(f"\n✗ FAILED")
            print(f"Error: {response.text}")
            return None

    except requests.exceptions.Timeout:
        print(f"\n✗ TIMEOUT (120 seconds)")
        return None
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return None


def main():
    """Run comprehensive API tests"""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     DocMolt API Capability Discovery                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Check for API key
    api_key = os.getenv("DOCMOLT_API_KEY")
    if api_key:
        print(f"✓ Using API key from DOCMOLT_API_KEY environment variable")
    else:
        print(f"⚠ No DOCMOLT_API_KEY found (testing without authentication)")

    # Test different artefact types
    artefact_types = [
        "documentation",
        "technical-spec",
        "tdd",
        "api-docs",
        "readme",
        "user-guide",
    ]

    # Test different models
    models = [
        "gpt-4o",
        "gpt-4o-mini",
        "claude-sonnet-4",
        "claude-3-5-sonnet",
    ]

    results = {}

    # Test primary combinations
    print("\n" + "="*80)
    print("PHASE 1: Testing Primary Configurations")
    print("="*80)

    for artefact in ["documentation", "technical-spec"]:
        for model in ["gpt-4o", "gpt-4o-mini"]:
            key = f"{artefact}_{model}"
            result = test_docmolt_api(artefact, model, api_key)
            results[key] = {
                "success": result is not None,
                "artefact": artefact,
                "model": model,
                "result": result
            }

    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF API TESTS")
    print("="*80)

    successful = [k for k, v in results.items() if v["success"]]
    failed = [k for k, v in results.items() if not v["success"]]

    print(f"\n✓ Successful: {len(successful)}/{len(results)}")
    for key in successful:
        print(f"  - {key}")

    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(results)}")
        for key in failed:
            print(f"  - {key}")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR INTEGRATION")
    print("="*80)

    if successful:
        best = successful[0]
        result_data = results[best]
        print(f"\n✓ Recommended configuration:")
        print(f"  - artefact: '{result_data['artefact']}'")
        print(f"  - model: '{result_data['model']}'")

        if result_data['result']:
            print(f"\n✓ Response structure:")
            print(f"  - Keys: {list(result_data['result'].keys())}")
    else:
        print("\n⚠ No successful API calls. Possible issues:")
        print("  1. API key required (set DOCMOLT_API_KEY environment variable)")
        print("  2. API endpoint changed")
        print("  3. Network/connectivity issues")
        print("  4. Rate limiting")

    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("1. Review API test results above")
    print("2. Set DOCMOLT_API_KEY if authentication required")
    print("3. Update DOCMOLT_API_DOCS.md with findings")
    print("4. Proceed with integration using recommended config")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
