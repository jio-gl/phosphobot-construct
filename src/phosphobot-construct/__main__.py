"""
Entry point for the Phosphobot Construct package.

This module allows the package to be run directly with 'python -m phosphobot_construct'.
"""

import sys
from phosphobot_construct.cli import main

if __name__ == "__main__":
    sys.exit(main())