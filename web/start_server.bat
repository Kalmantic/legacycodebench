@echo off
echo Starting LegacyCodeBench Web Server...
echo.
echo Open your browser and navigate to:
echo   http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.
python -m http.server 8000
