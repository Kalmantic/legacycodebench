#!/usr/bin/env python3
"""Script to create tasks from loaded datasets"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from legacycodebench.tasks import main

if __name__ == "__main__":
    main()

