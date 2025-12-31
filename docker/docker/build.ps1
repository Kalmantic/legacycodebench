# LegacyCodeBench Docker Build Script
# Run this script to build the GnuCOBOL execution environment

Write-Host "Building LegacyCodeBench GnuCOBOL Docker Image..." -ForegroundColor Cyan

# Check if Docker is running
try {
    docker version | Out-Null
} catch {
    Write-Host "ERROR: Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Build the image
docker build -t legacycodebench-cobol:latest .

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "SUCCESS: Docker image 'legacycodebench-cobol:latest' built successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now run the benchmark with execution enabled:" -ForegroundColor Yellow
    Write-Host "  legacycodebench run-full-benchmark --enable-execution --task-limit 3"
    Write-Host ""
    Write-Host "To test the Docker image manually:" -ForegroundColor Yellow
    Write-Host "  docker run --rm legacycodebench-cobol:latest cobc --version"
} else {
    Write-Host "ERROR: Docker build failed." -ForegroundColor Red
    exit 1
}

