#!/usr/bin/env python3
"""
Pipeline runner script for Heart Disease Prediction project.

This script orchestrates the complete data pipeline, including data acquisition,
preprocessing, model training, evaluation, and artifact generation.

Usage:
    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import pipeline module
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.pipeline import run_pipeline


def main():
    """Execute the full pipeline."""
    try:
        run_pipeline()
        return 0
    except Exception as e:
        print(f"\n ERROR: Pipeline execution failed!")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
