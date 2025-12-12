"""
Unit tests for DocMolt integration

Tests the DocMoltInterface class and its interaction with the API.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from legacycodebench.ai_integration import DocMoltInterface, get_ai_model
from legacycodebench.config import DOCMOLT_CONFIG


class TestDocMoltInterfaceInit:
    """Test DocMolt interface initialization"""

    def test_docmolt_interface_init_gpt4o(self):
        """Test DocMolt interface initialization with GPT-4o"""
        interface = DocMoltInterface("docmolt-gpt4o")

        assert interface.provider == "docmolt"
        assert interface.docmolt_model == "gpt-4o"
        assert interface.artefact == "documentation"
        assert interface.language == "cobol"
        assert interface.api_endpoint == "https://docmolt.hexaview.ai/api/docstream"

    def test_docmolt_interface_init_gpt4o_mini(self):
        """Test DocMolt interface initialization with GPT-4o-mini"""
        interface = DocMoltInterface("docmolt-gpt4o-mini")

        assert interface.provider == "docmolt"
        assert interface.docmolt_model == "gpt-4o-mini"
        assert interface.artefact == "documentation"

    def test_docmolt_interface_timeout_configured(self):
        """Test that timeout is configured correctly"""
        interface = DocMoltInterface("docmolt-gpt4o")

        assert interface.timeout == DOCMOLT_CONFIG["timeout_seconds"]
        assert interface.max_retries == DOCMOLT_CONFIG["max_retries"]


class TestDocMoltAPICall:
    """Test DocMolt API calls"""

    @patch('requests.post')
    def test_api_call_success_with_documentation_key(self, mock_post):
        """Test successful API call with 'documentation' key in response"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'test content'
        mock_response.json.return_value = {
            "documentation": "# Sample Documentation\n\n## Business Purpose\nThis program..."
        }
        mock_post.return_value = mock_response

        interface = DocMoltInterface("docmolt-gpt4o")
        result = interface._call_docmolt("COBOL CODE", "test.cbl")

        assert "Sample Documentation" in result
        assert "Business Purpose" in result
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_api_call_success_with_output_key(self, mock_post):
        """Test successful API call with 'output' key in response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'test content'
        mock_response.json.return_value = {
            "output": "# Documentation Output\n\nProgram details..."
        }
        mock_post.return_value = mock_response

        interface = DocMoltInterface("docmolt-gpt4o")
        result = interface._call_docmolt("COBOL CODE", "test.cbl")

        assert "Documentation Output" in result

    @patch('requests.post')
    def test_api_call_with_api_key(self, mock_post):
        """Test API call includes API key when available"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'test content'
        mock_response.json.return_value = {"documentation": "Test doc"}
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"DOCMOLT_API_KEY": "test-api-key"}):
            interface = DocMoltInterface("docmolt-gpt4o")
            interface._call_docmolt("COBOL CODE", "test.cbl")

            # Check that Authorization header was set
            call_kwargs = mock_post.call_args[1]
            interface._call_docmolt("COBOL CODE", "test.cbl")

    @patch('requests.post')
    def test_api_call_timeout(self, mock_post):
        """Test handling of timeout"""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()

        interface = DocMoltInterface("docmolt-gpt4o")

        with pytest.raises(Exception, match="seconds"):
            interface._call_docmolt("COBOL CODE", "test.cbl")


class TestDocMoltRetryLogic:
    """Test DocMolt retry logic"""

    @patch('requests.post')
    @patch('time.sleep')  # Mock sleep to speed up test
    def test_retry_on_connection_error(self, mock_sleep, mock_post):
        """Test that API call is retried on connection error"""
        import requests

        # First two calls fail, third succeeds
        mock_post.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            requests.exceptions.ConnectionError("Connection failed"),
            Mock(status_code=200, json=lambda: {"documentation": "Success"})
        ]

        interface = DocMoltInterface("docmolt-gpt4o")
        task = Mock()
        result = interface._call_docmolt_with_retry("COBOL CODE", "test.cbl", task)

        assert "Success" in result
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2  # Slept twice between retries

    @patch('requests.post')
    @patch('time.sleep')
    def test_fallback_after_max_retries(self, mock_sleep, mock_post):
        """Test fallback to mock documentation after max retries"""
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Always fails")

        interface = DocMoltInterface("docmolt-gpt4o")
        task = Mock(task_id="TEST-001")
        result = interface._call_docmolt_with_retry("COBOL CODE", "test.cbl", task)

        # Should return mock documentation
        assert "Business Purpose" in result or "Mock" in result
        assert mock_post.call_count == 4  # Initial + 3 retries


class TestDocMoltGenerateDocumentation:
    """Test generate_documentation method"""

    @patch('requests.post')
    def test_generate_documentation_with_file(self, mock_post, tmp_path):
        """Test documentation generation with actual file"""
        # Create temporary COBOL file
        cobol_file = tmp_path / "test.cbl"
        cobol_file.write_text("       IDENTIFICATION DIVISION.\n       PROGRAM-ID. TEST.")

        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'test content'
        mock_response.json.return_value = {
            "documentation": "# Test Program Documentation"
        }
        mock_post.return_value = mock_response

        interface = DocMoltInterface("docmolt-gpt4o")
        task = Mock()
        result = interface.generate_documentation(task, [cobol_file])

        assert "Test Program Documentation" in result
        # Verify API was called with COBOL code
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]
        assert "IDENTIFICATION DIVISION" in payload["code"]
        assert payload["filename"] == "test.cbl"
        assert payload["language"] == "cobol"
        assert payload["model"] == "gpt-4o"


class TestDocMoltExtractDocumentation:
    """Test documentation extraction from various response formats"""

    def test_extract_from_documentation_key(self):
        """Test extraction when documentation key exists"""
        interface = DocMoltInterface("docmolt-gpt4o")
        result = {"documentation": "Test content"}

        extracted = interface._extract_documentation_from_response(result)
        assert extracted == "Test content"

    def test_extract_from_output_key(self):
        """Test extraction when output key exists"""
        interface = DocMoltInterface("docmolt-gpt4o")
        result = {"output": "Output content"}

        extracted = interface._extract_documentation_from_response(result)
        assert extracted == "Output content"

    def test_extract_from_nested_dict(self):
        """Test extraction from nested dictionary"""
        interface = DocMoltInterface("docmolt-gpt4o")
        result = {
            "result": {
                "text": "Nested content"
            }
        }

        extracted = interface._extract_documentation_from_response(result)
        assert extracted == "Nested content"

    def test_extract_finds_large_string(self):
        """Test extraction finds large string value"""
        interface = DocMoltInterface("docmolt-gpt4o")
        result = {
            "status": "success",
            "unknown_key": "x" * 200  # Large string (>100 chars)
        }

        extracted = interface._extract_documentation_from_response(result)
        assert len(extracted) >= 100


class TestGetAIModelFactory:
    """Test get_ai_model factory function"""

    def test_factory_returns_docmolt_interface(self):
        """Test that factory returns DocMoltInterface for docmolt models"""
        interface = get_ai_model("docmolt-gpt4o")
        assert isinstance(interface, DocMoltInterface)

    def test_factory_returns_docmolt_for_all_variants(self):
        """Test that all docmolt variants return DocMoltInterface"""
        models = ["docmolt-gpt4o", "docmolt-gpt4o-mini", "docmolt-claude"]

        for model_id in models:
            interface = get_ai_model(model_id)
            assert isinstance(interface, DocMoltInterface)
            assert interface.provider == "docmolt"

    def test_factory_raises_for_unknown_model(self):
        """Test that factory raises error for unknown models"""
        with pytest.raises(ValueError, match="Unknown model"):
            get_ai_model("unknown-model-id")


class TestDocMoltPayloadFormat:
    """Test that payloads are formatted correctly"""

    @patch('requests.post')
    def test_payload_has_all_required_fields(self, mock_post):
        """Test that payload includes all required fields"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'test content'
        mock_response.json.return_value = {"documentation": "Test"}
        mock_post.return_value = mock_response

        interface = DocMoltInterface("docmolt-gpt4o")
        interface._call_docmolt("COBOL CODE", "program.cbl")

        # Extract payload from call
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]

        # Check all required fields
        assert "code" in payload
        assert "filename" in payload
        assert "language" in payload
        assert "artefact" in payload
        assert "model" in payload

        assert payload["code"] == "COBOL CODE"
        assert payload["filename"] == "program.cbl"
        assert payload["language"] == "cobol"
        assert payload["artefact"] == "documentation"
        assert payload["model"] == "gpt-4o"


# Integration test marker for tests requiring actual API
pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.getenv("DOCMOLT_API_KEY"), reason="No DOCMOLT_API_KEY set")
class TestDocMoltRealAPI:
    """
    Integration tests with real DocMolt API

    These tests are skipped unless DOCMOLT_API_KEY is set.
    Run with: DOCMOLT_API_KEY=your-key pytest tests/test_docmolt_integration.py -v -m integration
    """

    def test_real_api_call(self, tmp_path):
        """Test actual API call to DocMolt"""
        # Create sample COBOL file
        cobol_file = tmp_path / "sample.cbl"
        cobol_file.write_text("""       IDENTIFICATION DIVISION.
       PROGRAM-ID. SAMPLE-PROGRAM.

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  CUSTOMER-NAME    PIC X(50).
       01  ACCOUNT-BALANCE  PIC 9(7)V99.

       PROCEDURE DIVISION.
           DISPLAY 'Hello World'.
           STOP RUN.
""")

        interface = DocMoltInterface("docmolt-gpt4o")
        task = Mock()
        result = interface.generate_documentation(task, [cobol_file])

        # Basic validation
        assert len(result) > 100  # Should have substantial documentation
        assert isinstance(result, str)
        print(f"Documentation length: {len(result)} chars")
        print(f"Preview: {result[:200]}...")
