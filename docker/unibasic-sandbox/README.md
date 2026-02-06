# UniBasic Execution Environment (ScarletDME)

This Docker image provides a ScarletDME environment for executing UniBasic/Pick BASIC programs during behavioral fidelity evaluation.

## What is ScarletDME?

[ScarletDME](https://github.com/geneb/ScarletDME) is an open-source fork of OpenQM, providing a Pick/MultiValue BASIC implementation. It can execute programs written in:

- UniBasic (UniVerse BASIC)
- jBASIC (jBase)
- Pick BASIC
- QMBasic (native)

## Building the Image

### Windows (PowerShell)

```powershell
.\docker\unibasic-sandbox\build.ps1
```

### Linux/macOS

```bash
chmod +x ./docker/unibasic-sandbox/build.sh
./docker/unibasic-sandbox/build.sh
```

### Manual Build

```bash
docker build -t legacycodebench-unibasic:latest docker/unibasic-sandbox/
```

## Testing the Image

```bash
# Check ScarletDME version
docker run --rm legacycodebench-unibasic:latest qm -version

# Interactive shell
docker run -it --rm legacycodebench-unibasic:latest /bin/bash

# Compile and run a program
docker run --rm -v $(pwd)/test:/workspace legacycodebench-unibasic:latest \
    qm -c "BASIC BP MYPROGRAM && RUN BP MYPROGRAM"
```

## Usage with LegacyCodeBench

Once the image is built, UniBasic execution is automatically available:

```bash
# Run UniBasic benchmark with execution enabled
legacycodebench run-full-benchmark --language unibasic --enable-execution --task-limit 10

# Run without execution (static verification only)
legacycodebench run-full-benchmark --language unibasic --task-limit 10
```

## How It Works

1. **Compilation**: Programs are compiled using `BASIC BP <program>`
2. **Execution**: Programs are run using `RUN BP <program>`
3. **Output Capture**: stdout/stderr are captured for verification
4. **Fallback**: If execution fails, the system falls back to static verification (BF V3)

## Compatibility Notes

ScarletDME is highly compatible with most Pick BASIC dialects, but some programs may fail to compile due to:

- **Vendor-specific APIs**: UniVerse/UniData-specific functions may not be available
- **Missing includes**: Programs requiring external $INCLUDE files
- **Database operations**: Programs requiring actual MultiValue database connections

When compilation fails, LegacyCodeBench automatically falls back to static verification using the BF V3 evaluator.

## Troubleshooting

### Image build fails

Check that you have:
- Docker installed and running
- Sufficient disk space (~2GB for build)
- Network access to GitHub (for ScarletDME clone)

### Programs fail to compile

This is expected for many programs. The system will automatically fall back to static verification. Check the evaluation output for the failure reason:

- `VENDOR_API`: Program uses vendor-specific functions
- `INCLUDE_MISSING`: Missing $INCLUDE file
- `SYNTAX_ERROR`: General syntax incompatibility

## Image Details

- **Base**: Ubuntu 20.04 (Focal)
- **ScarletDME**: Latest from GitHub
- **Architecture**: 32-bit libraries for compatibility
- **Default user**: qmsys (QM system user)
