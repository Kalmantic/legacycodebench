@echo off
REM LegacyCodeBench - One-Click Benchmark Runner (Windows Batch)

echo ================================================================================
echo LegacyCodeBench v2.0 - Automated Benchmark
echo ================================================================================
echo.

REM Check Docker
docker ps >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and run this script again.
    pause
    exit /b 1
)

REM Check API Key
if "%OPENAI_API_KEY%"=="" (
    if "%ANTHROPIC_API_KEY%"=="" (
        echo ERROR: No API keys found!
        echo.
        echo Please set your API key:
        echo   set OPENAI_API_KEY=sk-your-key-here
        echo   set ANTHROPIC_API_KEY=sk-ant-your-key-here
        echo.
        pause
        exit /b 1
    )
)

echo [1/8] Installing dependencies...
pip install -e . --quiet
echo   Done!
echo.

echo [2/8] Building Docker image...
docker images -q legacycodebench-cobol:latest >nul 2>&1
if errorlevel 1 (
    docker build -t legacycodebench-cobol:latest docker/cobol-sandbox/
)
echo   Done!
echo.

echo [3/8] Loading datasets...
if not exist datasets (
    legacycodebench load-datasets
)
echo   Done!
echo.

echo [4/8] Creating tasks...
if not exist tasks (
    legacycodebench create-tasks
)
echo   Done!
echo.

echo [5/8] Running quick test...
legacycodebench run-ai --model gpt-4o --task-id LCB-DOC-001
echo   Done!
echo.

echo [6/8] Running full benchmark (this takes 60-90 minutes)...
legacycodebench run-ai --model gpt-4o
echo   Done!
echo.

echo [7/8] Generating leaderboard...
legacycodebench leaderboard --print
echo   Done!
echo.

echo ================================================================================
echo BENCHMARK COMPLETE!
echo ================================================================================
echo.
echo Results are in: results/leaderboard.json
echo.
pause
