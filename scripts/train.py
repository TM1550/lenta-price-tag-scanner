#!/usr/bin/env python3
"""Script to train the model."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variable
os.environ["PROJECT_ROOT"] = str(project_root)

from lenta_price_tag_scanner.commands import train  # noqa: E402


def main() -> None:
    """Run training."""
    train()


if __name__ == "__main__":
    main()
