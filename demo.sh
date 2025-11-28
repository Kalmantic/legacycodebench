#!/bin/bash

# LegacyCodeBench Demo Script
# Run this for the Friday presentation

set -e

echo "================================================================"
echo "LegacyCodeBench Demo"
echo "================================================================"
echo ""

echo "[1/4] Loading datasets from GitHub..."
legacycodebench load-datasets
echo ""

echo "[2/4] Creating 10 tasks (5 doc + 5 understanding)..."
legacycodebench create-tasks
echo ""

echo "[3/4] Running AI models on tasks..."
echo "  - Claude Sonnet 4"
legacycodebench run-ai --model claude-sonnet-4 --task-id LCB-DOC-001 || echo "  (Using mock response - API key not set)"
echo "  - GPT-4o"
legacycodebench run-ai --model gpt-4o --task-id LCB-DOC-001 || echo "  (Using mock response - API key not set)"
echo "  - AWS Transform"
legacycodebench run-ai --model aws-transform --task-id LCB-DOC-001 || echo "  (Using mock response)"
echo ""

echo "[4/4] Generating leaderboard..."
legacycodebench leaderboard --print
echo ""

echo "================================================================"
echo "Demo Complete!"
echo "================================================================"
echo ""
echo "Results saved in:"
echo "  - Tasks: tasks/"
echo "  - Submissions: submissions/"
echo "  - Results: results/"
echo ""

