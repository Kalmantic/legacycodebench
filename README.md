# LegacyCodeBench

The first rigorous benchmark for AI systems on legacy code understanding and documentation.

**Version:** 2.0 (Ground Truth-based Evaluation)

## Quick Start

### Minimal Setup (Without Docker)

```bash
# Install package
pip install -e .

# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run interactive mode (easiest way to get started)
legacycodebench interactive

# Or run full benchmark (3 tasks, quick test)
legacycodebench run-full-benchmark
```

### Full Setup (With Docker for Behavioral Fidelity)

```bash
# Install package
pip install -e .

# Build Docker image for COBOL execution
cd docker/cobol-sandbox
docker build -t legacycodebench-cobol:latest .
cd ../..

# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run full benchmark with execution enabled (recommended)
legacycodebench run-full-benchmark --enable-execution --task-limit 200
```

**Note:** Without Docker, Behavioral Fidelity (BF) scores will be 0. Docker is required for the full 35% BF component of the LCB Score.

## Usage

### Interactive Mode (Recommended for First Time)

```bash
legacycodebench interactive
```

This will:
1. Prompt for API keys (OpenAI, Anthropic, AWS, DocMolt)
2. Load datasets from GitHub repositories
3. Create 200 tasks across 4 difficulty tiers
4. Ask you to select models to test
5. Run evaluation with v2.0 ground truth-based scoring
6. Generate leaderboard

### Command-Line Mode

```bash
# Load datasets
legacycodebench load-datasets

# Create tasks (200 tasks across T1-T4 tiers)
legacycodebench create-tasks

# Run specific models
legacycodebench run-full-benchmark \
  --models "claude-sonnet-4,gpt-4o,aws-transform" \
  --task-limit 200 \
  --enable-execution

# Generate leaderboard
legacycodebench leaderboard --detailed --export-csv results.csv
```

### Evaluate Single Submission

```bash
legacycodebench evaluate \
  --task-id LCB-T2-015 \
  --submission my_documentation.md \
  --submitter-model gpt-4o \
  --enable-execution
```

### Run AI Model on Tasks

```bash
legacycodebench run-ai --model claude-sonnet-4 --task-id LCB-T1-001
```

## Evaluation Framework (v2.0)

LegacyCodeBench uses a 4-pillar evaluation system:

| Component | Weight | Description |
|-----------|--------|-------------|
| **Structural Completeness (SC)** | 30% | Coverage of code elements vs. ground truth |
| **Behavioral Fidelity (BF)** | 35% | Execution-based testing (requires Docker) |
| **Semantic Quality (SQ)** | 25% | LLM-as-judge evaluation of clarity/accuracy |
| **Traceability (TR)** | 10% | Validation of code citations |

**LCB Score Formula:**
```
LCB_Score = (0.30 × SC) + (0.35 × BF) + (0.25 × SQ) + (0.10 × TR)
```

**Critical Failures:** Any critical failure → LCB_Score = 0

## Supported Models

LegacyCodeBench supports multiple AI models:

| Model | Provider | Environment Variables |
|-------|----------|----------------------|
| **claude-sonnet-4** | Anthropic | `ANTHROPIC_API_KEY` |
| **gpt-4o** | OpenAI | `OPENAI_API_KEY` |
| **gpt-4** | OpenAI | `OPENAI_API_KEY` |
| **aws-transform** | AWS Bedrock | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` |
| **docmolt-gpt4o** | DocMolt (Hexaview) | `DOCMOLT_API_KEY` |
| **docmolt-gpt4o-mini** | DocMolt (Hexaview) | `DOCMOLT_API_KEY` |
| **docmolt-claude** | DocMolt (Hexaview) | `DOCMOLT_API_KEY` |

### AWS Bedrock Setup

AWS Transform uses AWS Bedrock with Claude 3 Sonnet:

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"  # Optional, defaults to us-east-1

# Run with AWS Transform
legacycodebench run-full-benchmark --models "aws-transform"
```

**Note:** AWS SDK (`boto3`, `botocore`) is already included in requirements.

See [AWS Transform Integration Guide](AWS_TRANSFORM_INTEGRATION.md) for detailed setup and troubleshooting.

### DocMolt Setup

DocMolt provides AI-powered code understanding:

```bash
# Set DocMolt API key
export DOCMOLT_API_KEY="..."

# Run with DocMolt models
legacycodebench run-full-benchmark --models "docmolt-gpt4o,docmolt-claude"
```

## Docker Setup (For Behavioral Fidelity Testing)

To enable the Behavioral Fidelity (BF) component (35% of LCB Score), Docker is required:

### Installation

**Windows/Mac:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

**Linux:**
```bash
sudo apt-get update
sudo apt-get install docker.io
sudo usermod -aG docker $USER  # Add yourself to docker group
```

### Build COBOL Execution Image

```bash
cd docker/cobol-sandbox
docker build -t legacycodebench-cobol:latest .
```

### Verify Installation

```bash
docker run --rm legacycodebench-cobol:latest cobc --version
# Should output: cobc (GnuCOBOL) 3.2
```

### Run Benchmark with Execution

```bash
legacycodebench run-full-benchmark --enable-execution --task-limit 200
```

## Web UI

View results in the browser:

```bash
# Generate leaderboard
legacycodebench leaderboard --export-csv results/leaderboard.csv

# Serve web UI
cd web
python -m http.server 8080

# Open http://localhost:8080 in your browser
```

Available pages:
- `index.html` - Main leaderboard
- `datasets.html` - Dataset information
- `scoring.html` - Scoring methodology
- `docs.html` - Documentation

## Available Commands

```bash
legacycodebench --help                    # Show all commands
legacycodebench interactive               # Interactive mode with prompts
legacycodebench load-datasets             # Load COBOL from GitHub
legacycodebench create-tasks              # Create 200 tasks (T1-T4)
legacycodebench run-full-benchmark        # Run complete evaluation
legacycodebench run-ai                    # Run AI model on tasks
legacycodebench evaluate                  # Evaluate single submission
legacycodebench leaderboard               # Generate leaderboard
```

## Documentation

- **PRD (Product Requirements):** [PRD_FINAL_V2.md](PRD_FINAL_V2.md)
- **AWS Transform Integration:** [AWS_TRANSFORM_INTEGRATION.md](AWS_TRANSFORM_INTEGRATION.md)
- **Implementation Status:** [PRD_IMPLEMENTATION_GAP_ANALYSIS.md](PRD_IMPLEMENTATION_GAP_ANALYSIS.md)

## Project Structure

See `PRD_FINAL_V2.md` Section 2.2 for complete directory structure.

Key directories:
- `legacycodebench/` - Core package (CLI, evaluators, AI integration)
- `datasets/` - COBOL source files from GitHub
- `tasks/` - Generated task definitions (JSON)
- `results/` - Evaluation results and leaderboard
- `web/` - Static HTML UI
- `docker/` - GnuCOBOL Docker image

## License

Apache 2.0 (code), CC-BY 4.0 (data)

