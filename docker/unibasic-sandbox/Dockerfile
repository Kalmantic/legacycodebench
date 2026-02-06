# LegacyCodeBench UniBasic Execution Environment
# Based on ScarletDME (Open Source Pick/MultiValue BASIC implementation)
#
# ScarletDME is a fork of OpenQM that provides a Pick-style BASIC interpreter
# compatible with most Pick/MultiValue BASIC dialects (UniBasic, jBASIC, etc.)
#
# Reference: https://github.com/geneb/ScarletDME

FROM ubuntu:focal

LABEL maintainer="LegacyCodeBench Team"
LABEL description="ScarletDME environment for UniBasic/Pick BASIC program execution"

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
# ScarletDME requires 32-bit libraries for compatibility
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y \
        build-essential \
        gcc-multilib \
        make \
        git \
        libncurses-dev \
        libncurses5-dev:i386 \
        libcrypt-dev:i386 \
        libc6-dev-i386 \
    && rm -rf /var/lib/apt/lists/*

# Create QM user and group (required by ScarletDME)
RUN groupadd qmusers && \
    useradd -G qmusers -d /usr/qmsys -m qmsys

# Clone ScarletDME from GitHub
RUN git clone --depth 1 https://github.com/geneb/ScarletDME.git /usr/src/ScarletDME

# Patch ScarletDME source to fix C99 bool type compatibility
# Modern GCC requires explicit #include <stdbool.h> for bool type
# Add it to ALL header files that use bool
RUN for f in /usr/src/ScarletDME/gplsrc/*.h; do \
        sed -i '1i #include <stdbool.h>' "$f"; \
    done && \
    for f in /usr/src/ScarletDME/gplsrc/*.c; do \
        sed -i '1i #include <stdbool.h>' "$f"; \
    done

# Set ownership and create required directories for install
# Touch placeholder files so 'rm dir/*' commands don't fail during make install
# Cache bust: v2
RUN chown -R qmsys:qmusers /usr/src/ScarletDME && \
    mkdir -p /usr/qmsys/NEWVOC /usr/qmsys/MESSAGES /usr/qmsys/bin /usr/qmsys/cat \
             /usr/qmsys/gcat /usr/qmsys/terminfo /usr/qmsys/pcode /usr/qmsys/GPL.BP && \
    for d in NEWVOC MESSAGES cat gcat pcode GPL.BP; do \
        touch /usr/qmsys/$d/placeholder; \
    done && \
    chown -R qmsys:qmusers /usr/qmsys

# Create QM config file
RUN cp /usr/src/ScarletDME/scarlet.conf /etc/scarlet.conf 2>/dev/null || \
    echo -e "QMSYS=/usr/qmsys\nQMUSERS=qmusers" > /etc/scarlet.conf && \
    chown qmsys:qmusers /etc/scarlet.conf

# Add root to qmusers group
RUN usermod -a -G qmusers root

# Build ScarletDME as qmsys user
USER qmsys:qmusers
WORKDIR /usr/src/ScarletDME
RUN make clean 2>/dev/null || true
RUN make

# Install as root (needs permission to create symlink in /usr/bin)
USER root:root
RUN make install datafiles && \
    chmod +x /usr/qmsys/bin/qm /usr/qmsys/bin/qm* 2>/dev/null || true

# Switch back to root for runtime flexibility
USER root:root

# Create workspace directory for program execution
RUN mkdir -p /workspace/BP && \
    chown -R qmsys:qmusers /workspace

# Set QM environment
ENV QMSYS=/usr/qmsys
ENV PATH="/usr/qmsys/bin:${PATH}"

WORKDIR /workspace

# Default command - can be overridden
CMD ["/bin/bash"]
