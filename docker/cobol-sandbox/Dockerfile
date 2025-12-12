# LegacyCodeBench COBOL Execution Environment
# Based on GnuCOBOL 3.2 for behavioral fidelity testing

FROM ubuntu:22.04

LABEL maintainer="LegacyCodeBench Team"
LABEL description="GnuCOBOL 3.2 environment for COBOL program execution and testing"

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install GnuCOBOL and dependencies
RUN apt-get update && apt-get install -y \
    gnucobol \
    libcob4 \
    libcob-dev \
    && rm -rf /var/lib/apt/lists/*

# Verify GnuCOBOL installation
RUN cobc --version

# Set working directory
WORKDIR /workspace

# Default command (can be overridden)
CMD ["/bin/bash"]
