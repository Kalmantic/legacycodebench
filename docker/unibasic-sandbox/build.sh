#!/bin/bash
# Build script for ScarletDME Docker image (UniBasic execution)
# Run from repository root: ./docker/unibasic-sandbox/build.sh

set -e

IMAGE_NAME="legacycodebench-unibasic"
IMAGE_TAG="latest"
FULL_TAG="${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building ScarletDME Docker image for UniBasic execution..."
echo "Image: $FULL_TAG"

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Build the image
echo ""
echo "Step 1: Building Docker image..."
docker build -t "$FULL_TAG" .

# Step 2: Verify image
echo ""
echo "Step 2: Verifying image..."
docker images "$FULL_TAG"

# Step 3: Test image
echo ""
echo "Step 3: Testing image..."
if docker run --rm "$FULL_TAG" qm -version 2>&1 | grep -iE "ScarletDME|QM"; then
    echo "ScarletDME is working!"
else
    echo "Warning: Could not verify ScarletDME installation"
fi

echo ""
echo "========================================"
echo "SUCCESS: Image built and ready!"
echo "========================================"
echo ""
echo "Usage:"
echo "  legacycodebench run-full-benchmark --language unibasic --enable-execution"
