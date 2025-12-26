# LegacyCodeBench

A benchmark for evaluating how well AI systems understand and document legacy COBOL code.

**Version 2.0** — Ground truth-based evaluation with execution testing.

---

## Why This Exists

220+ billion lines of COBOL still run critical infrastructure—banking, insurance, government. Modernization projects consistently fail, and the failure mode is almost always the same: the business logic was never properly understood before conversion began.

LegacyCodeBench tests whether AI can actually understand legacy code well enough to document it accurately.

---

## What It Measures

| Component | Weight | Description |
|-----------|--------|-------------|
| **Behavioral Fidelity (BF)** | 35% | Code generated from the AI's documentation must produce identical outputs to the original COBOL |
| **Structural Completeness (SC)** | 30% | All variables, control flow paths, and dependencies must be documented |
| **Semantic Quality (SQ)** | 25% | Documentation must be clear enough that a developer could understand the code without reading the original |
| **Traceability (TR)** | 10% | Line references and code citations must point to real code elements |

```
LCB_Score = (0.35 × BF) + (0.30 × SC) + (0.25 × SQ) + (0.10 × TR)
```

Critical failures (hallucinated functions, incorrect business rules) result in a score of 0 for that task.

---

## Quick Start

```bash
# Install
pip install -e .

# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run interactive mode
legacycodebench interactive
```

---

## Benchmark Details

- **200 tasks** across banking, insurance, retail, and government COBOL systems
- **4 difficulty tiers** (T1: single file → T4: multi-module enterprise)
- **Ground truth extraction** via static analysis
- **Execution-based testing** (requires Docker)

---

## Full Setup (With Execution Testing)

Behavioral Fidelity testing requires Docker to compile and run COBOL:

```bash
# Build COBOL execution environment
cd docker/cobol-sandbox
docker build -t legacycodebench-cobol:latest .
cd ../..

# Run with execution testing
legacycodebench run-full-benchmark --enable-execution --task-limit 200
```

Without Docker, BF scores will be 0.

---

## Supported Models

| Model | Provider | Environment Variables |
|-------|----------|----------------------|
| `claude-sonnet-4` | Anthropic | `ANTHROPIC_API_KEY` |
| `gpt-4o` | OpenAI | `OPENAI_API_KEY` |
| `gpt-4` | OpenAI | `OPENAI_API_KEY` |
| `aws-transform` | AWS Bedrock | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` |
| `docmolt-gpt4o` | DocMolt | `DOCMOLT_API_KEY` |
| `docmolt-claude` | DocMolt | `DOCMOLT_API_KEY` |

```bash
legacycodebench run-full-benchmark --models "claude-sonnet-4,gpt-4o"
```

---

## Commands

```bash
legacycodebench interactive         # Guided setup
legacycodebench run-full-benchmark  # Full evaluation
legacycodebench leaderboard         # Generate rankings
legacycodebench evaluate            # Score single submission
legacycodebench load-datasets       # Clone COBOL repositories
legacycodebench create-tasks        # Generate task definitions
```

---

## View Results

```bash
# Generate leaderboard
legacycodebench leaderboard --detailed --export-csv results.csv

# Web UI
cd web && python -m http.server 8080
# Open http://localhost:8080
```

---

## Project Structure

```
legacycodebench/
├── evaluators_v2/       # Ground-truth evaluation (SC, BF, SQ, TR)
├── static_analysis/     # COBOL parsing, ground truth extraction
├── execution/           # Docker-based COBOL compilation
└── cli.py

datasets/                # COBOL source files (auto-cloned from GitHub)
tasks/                   # 200 task definitions (JSON)
results/                 # Evaluation outputs
web/                     # Leaderboard UI
docker/                  # GnuCOBOL 3.2 execution environment
```

---

## Submit Results

Run the benchmark and open a PR with your `results/` directory. Results will be added to the public leaderboard.

---

## License

Apache 2.0 (code) · CC-BY 4.0 (data)

---

Built by [Kalmantic AI Labs](https://kalmantic.com) in partnership with [Hexaview Tech](https://hexaview.com)
