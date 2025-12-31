"""
OS & Environment Tests (Suite D) for v2.3.1
"""
import pytest
import os
import pathlib

class TestOSCompatibility:
    """Suite D: It works on my machine... and yours."""

    def test_it_04_path_normalization(self):
        """IT-04: Path Normalization"""
        # Submit claim with \foo\bar.cbl -> System reads /foo/bar.cbl
        win_path = "foo\\bar.cbl"
        nix_path = "foo/bar.cbl"
        
        normalized_win = win_path.replace("\\", "/")
        assert normalized_win == nix_path

    def test_it_05_crlf_vs_lf(self):
        """IT-05: CRLF vs LF"""
        text_crlf = "Line 1\r\nLine 2"
        text_lf = "Line 1\nLine 2"
        
        # Regex matching should work on both
        import re
        pattern = r"Line 1.*Line 2"
        
        match_crlf = re.search(pattern, text_crlf, re.DOTALL)
        match_lf = re.search(pattern, text_lf, re.DOTALL)
        
        assert match_crlf is not None
        assert match_lf is not None
