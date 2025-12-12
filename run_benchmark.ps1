# LegacyCodeBench - Complete Automated Benchmark Script
# This script does EVERYTHING needed to run the benchmark

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "LegacyCodeBench v2.0 - Automated Benchmark Setup & Execution" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Configuration
$SKIP_DOCKER_BUILD = $false  # Set to $true to skip Docker build if already done
$MODELS_TO_TEST = @("gpt-4o")  # Add "claude-sonnet-4", "docmolt-gpt4o" etc.

# ============================================================================
# STEP 1: Check Prerequisites
# ============================================================================
Write-Host "[1/8] Checking Prerequisites..." -ForegroundColor Yellow

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  ✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check pip
try {
    pip --version | Out-Null
    Write-Host "  ✓ pip found" -ForegroundColor Green
} catch {
    Write-Host "  ✗ pip not found" -ForegroundColor Red
    exit 1
}

# Check Docker
try {
    docker --version | Out-Null
    Write-Host "  ✓ Docker found" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Docker not found. Please install Docker Desktop" -ForegroundColor Red
    Write-Host "    Download from: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    exit 1
}

# Check Docker is running
try {
    docker ps | Out-Null
    Write-Host "  ✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Docker is not running. Please start Docker Desktop" -ForegroundColor Red
    Write-Host "    1. Open Docker Desktop from Start Menu" -ForegroundColor Yellow
    Write-Host "    2. Wait for whale icon in system tray" -ForegroundColor Yellow
    Write-Host "    3. Run this script again" -ForegroundColor Yellow
    exit 1
}

# Check API Keys
$apiKeysFound = $false

if ($env:OPENAI_API_KEY) {
    Write-Host "  ✓ OPENAI_API_KEY found" -ForegroundColor Green
    $apiKeysFound = $true
} else {
    Write-Host "  ⚠ OPENAI_API_KEY not set" -ForegroundColor Yellow
}

if ($env:ANTHROPIC_API_KEY) {
    Write-Host "  ✓ ANTHROPIC_API_KEY found" -ForegroundColor Green
    $apiKeysFound = $true
} else {
    Write-Host "  ⚠ ANTHROPIC_API_KEY not set" -ForegroundColor Yellow
}

if ($env:DOCMOLT_API_KEY) {
    Write-Host "  ✓ DOCMOLT_API_KEY found" -ForegroundColor Green
} else {
    Write-Host "  ⚠ DOCMOLT_API_KEY not set (optional)" -ForegroundColor Yellow
}

if (-not $apiKeysFound) {
    Write-Host ""
    Write-Host "ERROR: No API keys found!" -ForegroundColor Red
    Write-Host "Please set at least one API key:" -ForegroundColor Yellow
    Write-Host '  $env:OPENAI_API_KEY = "sk-your-key-here"' -ForegroundColor Yellow
    Write-Host '  $env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"' -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# ============================================================================
# STEP 2: Install Dependencies
# ============================================================================
Write-Host "[2/8] Installing Python Dependencies..." -ForegroundColor Yellow

pip install -e . --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "  ✗ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

Write-Host ""

# ============================================================================
# STEP 3: Build Docker Image
# ============================================================================
Write-Host "[3/8] Building COBOL Docker Image..." -ForegroundColor Yellow

# Check if image already exists
$imageExists = docker images -q legacycodebench-cobol:latest

if ($imageExists -and $SKIP_DOCKER_BUILD) {
    Write-Host "  ✓ Docker image already exists (skipping build)" -ForegroundColor Green
} else {
    Write-Host "  Building image (this takes 2-5 minutes)..." -ForegroundColor Cyan
    docker build -t legacycodebench-cobol:latest docker/cobol-sandbox/ --quiet

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Docker image built successfully" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Docker build failed" -ForegroundColor Red
        Write-Host "    Run manually: docker build -t legacycodebench-cobol:latest docker/cobol-sandbox/" -ForegroundColor Yellow
        exit 1
    }
}

# Verify image
docker run --rm legacycodebench-cobol:latest cobc --version | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ GnuCOBOL compiler working" -ForegroundColor Green
} else {
    Write-Host "  ✗ GnuCOBOL test failed" -ForegroundColor Red
    exit 1
}

Write-Host ""

# ============================================================================
# STEP 4: Load Datasets
# ============================================================================
Write-Host "[4/8] Loading COBOL Datasets..." -ForegroundColor Yellow

if (Test-Path "datasets") {
    $fileCount = (Get-ChildItem -Path datasets -Recurse -Filter *.cbl).Count
    if ($fileCount -gt 0) {
        Write-Host "  ✓ Datasets already loaded ($fileCount COBOL files)" -ForegroundColor Green
    } else {
        Write-Host "  Loading datasets from GitHub..." -ForegroundColor Cyan
        legacycodebench load-datasets
        Write-Host "  ✓ Datasets loaded" -ForegroundColor Green
    }
} else {
    Write-Host "  Loading datasets from GitHub..." -ForegroundColor Cyan
    legacycodebench load-datasets
    Write-Host "  ✓ Datasets loaded" -ForegroundColor Green
}

Write-Host ""

# ============================================================================
# STEP 5: Create Tasks
# ============================================================================
Write-Host "[5/8] Creating 200 Benchmark Tasks..." -ForegroundColor Yellow

if (Test-Path "tasks") {
    $taskCount = (Get-ChildItem -Path tasks -Filter *.json).Count
    if ($taskCount -eq 200) {
        Write-Host "  ✓ Tasks already created (200 tasks)" -ForegroundColor Green
    } else {
        Write-Host "  Creating tasks..." -ForegroundColor Cyan
        legacycodebench create-tasks
        Write-Host "  ✓ Tasks created" -ForegroundColor Green
    }
} else {
    Write-Host "  Creating tasks..." -ForegroundColor Cyan
    legacycodebench create-tasks
    Write-Host "  ✓ Tasks created" -ForegroundColor Green
}

Write-Host ""

# ============================================================================
# STEP 6: Run Quick Test (Single Task)
# ============================================================================
Write-Host "[6/8] Running Quick Test (Single Task)..." -ForegroundColor Yellow

Write-Host "  Testing with $($MODELS_TO_TEST[0]) on LCB-DOC-001..." -ForegroundColor Cyan
legacycodebench run-ai --model $MODELS_TO_TEST[0] --task-id LCB-DOC-001 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Test passed - system working correctly" -ForegroundColor Green

    # Show result
    $resultFile = "results/LCB-DOC-001_$($MODELS_TO_TEST[0])_$($MODELS_TO_TEST[0]).json"
    if (Test-Path $resultFile) {
        $result = Get-Content $resultFile | ConvertFrom-Json
        $score = [math]::Round($result.overall_score * 100, 2)
        Write-Host "  → Test Score: $score%" -ForegroundColor Cyan
    }
} else {
    Write-Host "  ✗ Test failed" -ForegroundColor Red
    Write-Host "    Check logs for errors" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# ============================================================================
# STEP 7: Run Full Benchmark
# ============================================================================
Write-Host "[7/8] Running Full Benchmark (200 tasks)..." -ForegroundColor Yellow
Write-Host "  This will take 60-120 minutes depending on models selected" -ForegroundColor Cyan
Write-Host ""

$startTime = Get-Date

foreach ($model in $MODELS_TO_TEST) {
    Write-Host "  Running benchmark with model: $model" -ForegroundColor Cyan
    Write-Host "  " -NoNewline

    $modelStartTime = Get-Date

    legacycodebench run-ai --model $model

    $modelEndTime = Get-Date
    $modelDuration = $modelEndTime - $modelStartTime

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ $model completed in $([math]::Round($modelDuration.TotalMinutes, 1)) minutes" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $model failed" -ForegroundColor Red
    }
    Write-Host ""
}

$endTime = Get-Date
$totalDuration = $endTime - $startTime

Write-Host "  ✓ All benchmarks completed in $([math]::Round($totalDuration.TotalMinutes, 1)) minutes" -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 8: Generate Leaderboard
# ============================================================================
Write-Host "[8/8] Generating Leaderboard..." -ForegroundColor Yellow

legacycodebench leaderboard --print

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "BENCHMARK COMPLETE!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to:" -ForegroundColor Cyan
Write-Host "  - Submissions: submissions/" -ForegroundColor White
Write-Host "  - Results: results/" -ForegroundColor White
Write-Host "  - Leaderboard: results/leaderboard.json" -ForegroundColor White
Write-Host ""
Write-Host "Run detailed analysis:" -ForegroundColor Cyan
Write-Host "  python scripts/analyze_results.py" -ForegroundColor White
Write-Host ""
Write-Host "Total execution time: $([math]::Round($totalDuration.TotalMinutes, 1)) minutes" -ForegroundColor Cyan
Write-Host ""
