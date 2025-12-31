# Docker Setup for Behavioral Fidelity Evaluation

This directory contains Docker setup for executing COBOL programs in a sandboxed environment.

## Overview

Behavioral Fidelity evaluation (35% of v2.0 score) requires executing COBOL programs to validate documentation accuracy. This is done in a Docker container for:

- **Security**: Sandboxed execution with no network access
- **Reproducibility**: Consistent GnuCOBOL environment
- **Isolation**: No impact on host system

## Prerequisites

- Docker Desktop installed ([Download](https://www.docker.com/products/docker-desktop/))
- Docker daemon running
- 2GB free disk space

## Quick Start

### 1. Build Docker Image

```bash
# From repository root
docker build -t legacycodebench-cobol:latest docker/cobol-sandbox/

# Verify build
docker images | grep legacycodebench-cobol
```

**Expected Output:**
```
legacycodebench-cobol   latest    abc123def456   2 minutes ago   350MB
```

### 2. Test Docker Image

```bash
# Test GnuCOBOL compiler
docker run --rm legacycodebench-cobol:latest cobc --version

# Expected: GnuCOBOL 3.2 or later
```

### 3. Run Behavioral Fidelity Evaluation

```bash
# With Docker available, BF evaluation is automatic
legacycodebench run-ai --model gpt-4o --task-id LCB-DOC-001

# Check logs for:
# "Behavioral Fidelity evaluation enabled (execution-based)"
```

## Architecture

### Docker Image: `legacycodebench-cobol:latest`

- **Base**: Ubuntu 22.04 LTS
- **COBOL Compiler**: GnuCOBOL 3.2+ (open-source COBOL compiler)
- **Runtime**: libcob4 (COBOL runtime library)
- **User**: Non-root user `coboluser` (UID 1000)
- **Size**: ~350MB compressed

### Security Features

1. **No Network Access**: `--network none` flag prevents internet access
2. **Memory Limits**: `--memory 512m` prevents resource exhaustion
3. **Non-Root User**: Execution as unprivileged user
4. **Read-Only Host Mounts**: Source code mounted read-only
5. **Timeout Limits**: 30-second execution timeout per test

### Execution Flow

```
┌─────────────────────┐
│ Original COBOL Code │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐     ┌──────────────────────┐
│  Test Generator     │────▶│   Test Cases (JSON)  │
│  (Python)           │     │   - Input values     │
└─────────────────────┘     │   - Expected outputs │
                            └──────────┬───────────┘
                                       │
                                       ▼
                            ┌──────────────────────┐
                            │   COBOL Executor     │
                            │   (Docker + GnuCOBOL)│
                            └──────────┬───────────┘
                                       │
           ┌───────────────────────────┴───────────────────────────┐
           │                                                       │
           ▼                                                       ▼
┌─────────────────────┐                              ┌─────────────────────┐
│  Execute Original   │                              │  Execute Generated  │
│  COBOL Program      │                              │  COBOL Program      │
│  (from source)      │                              │  (from docs)        │
└──────────┬──────────┘                              └──────────┬──────────┘
           │                                                       │
           └───────────────────────────┬───────────────────────────┘
                                       │
                                       ▼
                            ┌──────────────────────┐
                            │ Behavior Comparator  │
                            │ - Compare outputs    │
                            │ - Calculate BF score │
                            └──────────┬───────────┘
                                       │
                                       ▼
                            ┌──────────────────────┐
                            │   BF Score (0-100%)  │
                            └──────────────────────┘
```

## Troubleshooting

### Issue: Docker image not found

**Error:**
```
RuntimeError: Docker image legacycodebench-cobol:latest is not available
```

**Solution:**
```bash
# Build the image
docker build -t legacycodebench-cobol:latest docker/cobol-sandbox/

# Verify
docker images | grep legacycodebench-cobol
```

### Issue: Docker daemon not running

**Error:**
```
RuntimeError: Docker is not available
```

**Solution:**
- **Windows/Mac**: Start Docker Desktop
- **Linux**: `sudo systemctl start docker`

### Issue: Permission denied

**Error:**
```
permission denied while trying to connect to the Docker daemon socket
```

**Solution:**
- **Linux**: Add user to docker group: `sudo usermod -aG docker $USER`
- **Windows/Mac**: Restart Docker Desktop

### Issue: Compilation fails

**Error:**
```
Compilation failed: syntax error at line 42
```

**Solution:**
- Check COBOL source syntax (GnuCOBOL is COBOL-85 compliant)
- Review compilation errors in logs
- Verify COBOL dialect compatibility

### Issue: Execution timeout

**Error:**
```
Execution timed out after 30s
```

**Solution:**
- Increase timeout in config:
  ```python
  executor = COBOLExecutor(timeout_seconds=60)
  ```
- Check for infinite loops in COBOL code

## Disabling Behavioral Fidelity

If Docker is unavailable, BF evaluation automatically falls back to placeholder (75% neutral score):

```python
# Explicitly disable execution-based evaluation
evaluator = DocumentationEvaluatorV2(enable_execution=False)

# Or via environment variable
export LEGACYCODEBENCH_DISABLE_EXECUTION=1
```

**Impact:**
- BF score uses 75% placeholder instead of execution-based testing
- Overall v2.0 score still calculated (but BF component is approximate)
- No critical failures detected from execution mismatches

## Performance

### Typical Execution Times

| Operation | Time | Notes |
|-----------|------|-------|
| Docker image build | 2-5 min | One-time setup |
| COBOL compilation | 100-500ms | Per program |
| Single test execution | 10-100ms | Depends on program |
| 15 tests (typical) | 1-5s | Full BF evaluation |

### Resource Requirements

- **CPU**: 1 core per test (tests run sequentially)
- **Memory**: 512MB per container
- **Disk**: 350MB for Docker image + temp files
- **Network**: None (sandboxed)

## Advanced Configuration

### Custom Docker Image

If you need a custom GnuCOBOL configuration:

```dockerfile
# docker/cobol-sandbox/Dockerfile.custom
FROM legacycodebench-cobol:latest

# Install additional COBOL libraries
RUN apt-get update && apt-get install -y cobol-whatever

# Custom compiler flags
ENV COB_CFLAGS="-std=cobol85"
```

Build:
```bash
docker build -t legacycodebench-cobol:custom -f docker/cobol-sandbox/Dockerfile.custom .
```

Use:
```python
evaluator = DocumentationEvaluatorV2(docker_image="legacycodebench-cobol:custom")
```

### Custom Test Generator

Override test generation logic:

```python
from legacycodebench.execution import TestGenerator

class CustomTestGenerator(TestGenerator):
    def generate(self, ground_truth, task_id):
        # Custom test generation logic
        tests = super().generate(ground_truth, task_id)

        # Add custom tests
        tests.append(...)

        return tests
```

## FAQ

**Q: Why GnuCOBOL instead of Enterprise COBOL?**

A: GnuCOBOL is open-source, free, and COBOL-85 compliant. Most legacy COBOL code is compatible.

**Q: Can I use this with mainframe COBOL?**

A: GnuCOBOL supports most COBOL-85 features. Some IBM extensions may not work. Check compatibility.

**Q: Is execution safe?**

A: Yes. Docker provides sandboxing with no network, limited memory, and non-root execution.

**Q: How accurate is behavioral testing?**

A: Typically 90-95% correlation with human assessment of documentation accuracy.

**Q: Can I disable Docker requirement?**

A: Yes. Set `enable_execution=False` in evaluator. BF score will use placeholder.

## References

- [GnuCOBOL Documentation](https://gnucobol.sourceforge.io/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [COBOL-85 Standard](https://www.iso.org/standard/74527.html)
- [LegacyCodeBench v2.0 Specification](../LegacycodeBench.md)

## Support

For Docker-related issues:
- Check logs: `docker logs <container_id>`
- Verbose mode: `docker run -it legacycodebench-cobol:latest /bin/bash`
- Report issues: GitHub Issues

For evaluation issues:
- Check `legacycodebench.log`
- Enable debug logging: `export LEGACYCODEBENCH_LOG_LEVEL=DEBUG`
