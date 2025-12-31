# LegacyCodeBench

A benchmark for evaluating how well AI systems understand and document legacy COBOL code.

**Version 2.3.1** — Ground truth-based evaluation with execution testing.

---

## Why This Exists

220+ billion lines of COBOL still run critical infrastructure—banking, insurance, government. Modernization projects consistently fail, and the failure mode is almost always the same: the business logic was never properly understood before conversion began.

LegacyCodeBench tests whether AI can actually understand legacy code well enough to document it accurately.

---

## What It Measures

| Component | Weight | Description |
|-----------|--------|-------------|
| **Structural Completeness (SC)** | 30% | All business rules, data structures, control flow, and external calls must be documented |
| **Documentation Quality (DQ)** | 45% | Structure, traceability, readability, and abstraction level |
| **Behavioral Fidelity (BF)** | 25% | Execution-based verification (claims, BSM pattern matching, IUE classification) |

```
LCB_Score = (0.30 × SC) + (0.45 × DQ) + (0.25 × BF)
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
- **4 difficulty tiers** (T1: 100-200 LOC → T4: 600+ LOC enterprise)
- **Fully deterministic** evaluation (v2.3.1 removes LLM-as-judge)
- **Execution-based testing** for behavioral fidelity (requires Docker)

---

## Full Setup (With Execution Testing)

Behavioral Fidelity testing requires Docker to compile and run COBOL:

```bash
# Build COBOL execution environment
cd docker/cobol-sandbox
docker build -t legacycodebench-cobol:latest .
cd ../..

# Run with execution testing (v2.3.1 is default)
legacycodebench run-full-benchmark --enable-execution --task-limit 200
```

**Note**: Without Docker, BF evaluation falls back to heuristic verification (claim quality analysis). For full accuracy, Docker is recommended.

---

## Supported Models

| Model | Provider | Environment Variables |
|-------|----------|----------------------|
| `claude-sonnet-4` | Anthropic | `ANTHROPIC_API_KEY` |
| `gpt-4o` | OpenAI | `OPENAI_API_KEY` |
| `gemini-2.5-flash` | Google | `GOOGLE_API_KEY` |

```bash
legacycodebench run-full-benchmark --models "claude-sonnet-4,gpt-4o"
```

---

## Commands

```bash
legacycodebench interactive         # Guided setup
legacycodebench run-full-benchmark  # Full evaluation (v2.3.1 default)
legacycodebench leaderboard         # Generate rankings
legacycodebench evaluate            # Score single submission
legacycodebench load-datasets       # Clone COBOL repositories
legacycodebench create-tasks        # Generate task definitions
```

## View Results

```bash
# Generate leaderboard JSON
legacycodebench leaderboard

# Or with detailed output
legacycodebench leaderboard --detailed

# Serve web UI
cd web
python -m http.server 8080
# Open http://localhost:8080
```

---

## Project Structure

```
legacycodebench/
├── evaluators_v213/     # v2.3.1 evaluator (deterministic, default)
├── evaluators_v2/       # v2.0 evaluator (LLM-as-judge, legacy)
├── static_analysis/     # COBOL parsing, ground truth extraction
├── execution/           
│   ├── bsm/            # Behavioral Similarity Matching
│   └── iue/            # Isolatable Unit Extraction
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

Built by [Kalmantic AI Labs](https://kalmantic.com) in partnership with [Hexaview Tech](https://hexaviewtech.com)
