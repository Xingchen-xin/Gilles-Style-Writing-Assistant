"""
GSWA Training CLI Entry Point.

Usage:
    python -m gswa.train --help
    python -m gswa.train info
    python -m gswa.train train --auto
"""

from gswa.training.cli import main

if __name__ == "__main__":
    main()
