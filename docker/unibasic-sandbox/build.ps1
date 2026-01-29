# Build script for ScarletDME Docker image (UniBasic execution)
# Run from repository root: .\docker\unibasic-sandbox\build.ps1

$ErrorActionPreference = "Stop"

$IMAGE_NAME = "legacycodebench-unibasic"
$IMAGE_TAG = "latest"
$FULL_TAG = "${IMAGE_NAME}:${IMAGE_TAG}"

Write-Host "Building ScarletDME Docker image for UniBasic execution..." -ForegroundColor Cyan
Write-Host "Image: $FULL_TAG" -ForegroundColor Gray

# Change to script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ScriptDir

try {
    # Build the image
    Write-Host "`nStep 1: Building Docker image..." -ForegroundColor Yellow
    docker build -t $FULL_TAG .

    if ($LASTEXITCODE -ne 0) {
        throw "Docker build failed with exit code $LASTEXITCODE"
    }

    Write-Host "`nStep 2: Verifying image..." -ForegroundColor Yellow
    docker images $FULL_TAG

    Write-Host "`nStep 3: Testing image..." -ForegroundColor Yellow
    $testResult = docker run --rm $FULL_TAG qm -version 2>&1
    if ($testResult -match "ScarletDME|QM") {
        Write-Host "ScarletDME is working!" -ForegroundColor Green
    } else {
        Write-Host "Warning: Could not verify ScarletDME installation" -ForegroundColor Yellow
        Write-Host "Output: $testResult" -ForegroundColor Gray
    }

    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "SUCCESS: Image built and ready!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "`nUsage:" -ForegroundColor Cyan
    Write-Host "  legacycodebench run-full-benchmark --language unibasic --enable-execution" -ForegroundColor White

} catch {
    Write-Host "`nERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}
