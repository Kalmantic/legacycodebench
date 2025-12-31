# LegacyCodeBench GnuCOBOL Execution Environment
# This Docker image provides a sandboxed environment for executing COBOL programs

FROM ubuntu:22.04

LABEL maintainer="LegacyCodeBench Team"
LABEL description="GnuCOBOL execution environment for behavioral fidelity testing"

# Install GnuCOBOL and dependencies
RUN apt-get update && apt-get install -y \
    gnucobol \
    build-essential \
    libcob4 \
    libgmp-dev \
    libdb-dev \
    && rm -rf /var/lib/apt/lists/*

# Verify GnuCOBOL installation
RUN cobc --version

# Create working directories
RUN mkdir -p /workspace/source \
    && mkdir -p /workspace/output \
    && mkdir -p /workspace/data

# Set working directory
WORKDIR /workspace

# Create execution script
COPY execute_cobol.sh /usr/local/bin/execute_cobol.sh
RUN chmod +x /usr/local/bin/execute_cobol.sh

# Default command shows help
CMD ["cobc", "--help"]

