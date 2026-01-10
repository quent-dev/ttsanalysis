#!/usr/bin/env python3
"""
Quick test script to run the full cohort analysis
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from main import CohortAnalysisTool

if __name__ == "__main__":
    print("Starting cohort analysis test...")
    tool = CohortAnalysisTool()
    success = tool.run_full_analysis()

    if success:
        print("\n" + "="*80)
        print("SUCCESS! Analysis completed.")
        print("="*80)
        tool.display_summary()
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("FAILED! Analysis encountered errors.")
        print("="*80)
        sys.exit(1)
