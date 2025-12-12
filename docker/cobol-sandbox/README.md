# COBOL Execution Docker Environment

This Docker image provides a sandboxed GnuCOBOL environment for behavioral fidelity testing in LegacyCodeBench v2.0.

## Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- At least 2GB of free disk space

## Building the Image

From the root of the LegacyCodeBenchmark repository:

```bash
# Build the COBOL execution Docker image
docker build -t legacycodebench-cobol:latest docker/cobol-sandbox/
```

This will create a Docker image with:
- GnuCOBOL 3.2 compiler (`cobc`)
- COBOL runtime libraries
- Ubuntu 22.04 base

## Verifying the Installation

Test that the image works correctly:

```bash
# Check that GnuCOBOL is installed
docker run --rm legacycodebench-cobol:latest cobc --version
```

Expected output:
```
cobc (GnuCOBOL) 3.x.x
...
```

## Usage in LegacyCodeBench

The Docker image is automatically used by the `COBOLExecutor` class when behavioral fidelity evaluation is enabled.

### Manual Test

You can manually test COBOL compilation and execution:

```bash
# Create a test COBOL program
cat > test.cbl << 'EOF'
       IDENTIFICATION DIVISION.
       PROGRAM-ID. HELLO.
       PROCEDURE DIVISION.
           DISPLAY "Hello from COBOL!".
           STOP RUN.
EOF

# Compile and run using Docker
docker run --rm -v $(pwd):/workspace legacycodebench-cobol:latest /bin/bash -c "cobc -x test.cbl && ./test"
```

Expected output:
```
Hello from COBOL!
```

## Architecture

The COBOL Executor uses Docker to:

1. **Compile** COBOL source using `cobc -x program.cbl`
2. **Execute** the compiled program with test inputs
3. **Capture** stdout, stderr, and file outputs
4. **Enforce** timeouts and resource limits

### Volume Mounting

The executor mounts a temporary directory to `/workspace` inside the container:
- COBOL source files
- Compiled executables
- Input data files
- Output data files

### Security

The container runs with:
- `--network none` (no network access)
- `--memory 512m` (memory limit)
- Read-only filesystem (except /workspace)

## Troubleshooting

### Issue: "Docker is not available"

**Solution:** Install Docker Desktop or Docker Engine:
- Windows/Mac: https://www.docker.com/products/docker-desktop
- Linux: `sudo apt-get install docker.io`

### Issue: "Docker image not found"

**Solution:** Build the image:
```bash
docker build -t legacycodebench-cobol:latest docker/cobol-sandbox/
```

### Issue: "Permission denied" errors

**Windows (using Git Bash/MSYS):**
```bash
# Docker volume mounting may have issues with path conversion
# Use PowerShell or CMD instead, or set MSYS_NO_PATHCONV=1
export MSYS_NO_PATHCONV=1
```

**Linux:**
```bash
# Add your user to docker group
sudo usermod -aG docker $USER
# Log out and log back in
```

### Issue: Compilation errors in Docker

**Check COBOL syntax:**
```bash
# Run compiler with verbose output
docker run --rm -v $(pwd):/workspace legacycodebench-cobol:latest cobc -x -v program.cbl
```

## Advanced Configuration

### Custom COBOL Compiler Options

Edit `legacycodebench/execution/cobol_executor.py` and modify the `docker_cmd` in `_compile_cobol()`:

```python
docker_cmd = [
    "docker", "run", "--rm",
    "-v", f"{work_dir}:/workspace",
    "-w", "/workspace",
    self.docker_image,
    "cobc", "-x",
    "-std=cobol2014",  # Use COBOL 2014 standard
    "-Wall",           # Show all warnings
    source_file.name
]
```

### Resource Limits

Adjust in `COBOLExecutor.__init__()`:

```python
self.timeout_seconds = 60     # Increase timeout for complex programs
self.memory_limit = "1g"       # Increase memory limit
```

## For Development

### Interactive Shell

```bash
# Open an interactive shell in the container
docker run --rm -it -v $(pwd):/workspace legacycodebench-cobol:latest /bin/bash
```

### Testing Manually

```bash
# Inside the container
cd /workspace
cobc -x myprogram.cbl
./myprogram
```

## References

- GnuCOBOL Documentation: https://gnucobol.sourceforge.io/
- COBOL Standards: https://www.iso.org/standard/74527.html
- Docker Documentation: https://docs.docker.com/
