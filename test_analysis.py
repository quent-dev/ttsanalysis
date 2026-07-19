#!/usr/bin/env python3
"""
Quick test script for cohort analysis
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from main import CohortAnalysisTool, setup_logging
import config

if __name__ == "__main__":
    # Get account ID from command line, default to MX
    account_id = sys.argv[1].upper() if len(sys.argv) > 1 else 'MX'

    if account_id not in config.ACCOUNTS:
        print(f"Error: Invalid account ID '{account_id}'")
        print(f"Valid accounts: {', '.join(config.ACCOUNTS.keys())}")
        sys.exit(1)

    account_name = config.ACCOUNTS[account_id]['name']

    # Set up logging
    setup_logging(account_id)

    print(f"Starting cohort analysis test for account: {account_name}")
    print("="*80)

    tool = CohortAnalysisTool(account_id)
    success = tool.run_full_analysis()

    if success:
        print("\n✓ Test completed successfully!")
    else:
        print("\n✗ Test failed - check logs for details")
        sys.exit(1)
