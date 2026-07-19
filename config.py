"""
Configuration settings for TikTok Shop Cohort Analysis Tool
"""
import os

# Project Root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Multi-Account Configuration
ACCOUNTS = {
    'MX': {
        'name': 'Mexico',
        'data_dir': os.path.join(PROJECT_ROOT, "data", "MX", "raw"),
        'processed_dir': os.path.join(PROJECT_ROOT, "data", "MX", "processed"),
        'csv_output_dir': os.path.join(PROJECT_ROOT, "output", "MX", "csv"),
        'viz_output_dir': os.path.join(PROJECT_ROOT, "output", "MX", "visualizations"),
        'log_file': os.path.join(PROJECT_ROOT, "cohort_analysis_mx.log"),
    },
    'US': {
        'name': 'United States',
        'data_dir': os.path.join(PROJECT_ROOT, "data", "US", "raw"),
        'processed_dir': os.path.join(PROJECT_ROOT, "data", "US", "processed"),
        'csv_output_dir': os.path.join(PROJECT_ROOT, "output", "US", "csv"),
        'viz_output_dir': os.path.join(PROJECT_ROOT, "output", "US", "visualizations"),
        'log_file': os.path.join(PROJECT_ROOT, "cohort_analysis_us.log"),
    }
}

# File Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CSV_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "csv")
VIZ_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# Data Files
CSV_FILENAME = "Todo pedido-2026-07-18-17_38.csv"
MARKETING_COSTS_FILENAME = "Marketing TTS.csv"
LIVE_SESSIONS_FILENAME = "Live Shopping Sessions.csv"
CLEANED_DATA_FILE = os.path.join(PROCESSED_DIR, "orders_cleaned.parquet")

# Analysis Parameters
ANALYSIS_END_DATE = '2026-06-30'  # Only include orders up to this date (inclusive). Set to None for no cutoff.
COMPLETED_STATUSES = ["Completado"]
DELIVERED_STATUS_PAIRS = [("Enviado", "Entregado")]  # (status, substatus) pairs
DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"

# Customer Identification
CUSTOMER_ID_FIELD = "Buyer Username"

# Cohort Settings
COHORT_FREQUENCY = "M"  # Monthly cohorts
MIN_COHORT_SIZE = 10  # Exclude cohorts smaller than this

# LTV Windows (in months)
LTV_WINDOWS = [3, 6, 12]

# Visualization Settings
FIGURE_SIZE = (12, 8)
DPI = 300
COLOR_PALETTE = "RdYlGn"  # Red-Yellow-Green for retention heatmap
HEATMAP_CMAP = "RdYlGn"
RETENTION_CURVE_COLORS = None  # Use default matplotlib colors
SAVE_FORMATS = ["png"]  # Add "html" for interactive charts

# Logging
LOG_FILE = os.path.join(PROJECT_ROOT, "cohort_analysis.log")
LOG_LEVEL = "INFO"

# CSV Export Settings
CSV_ENCODING = "utf-8-sig"  # UTF-8 with BOM for Excel compatibility
CSV_DECIMAL = "."
CSV_THOUSANDS = ","

# Live Shopping Analysis
ENABLE_LIVE_ANALYSIS = True  # Toggle live shopping analysis on/off
LIVE_SESSION_BUFFER_MINUTES = 30  # Buffer time after session ends to catch delayed orders
LIVE_COMPARISON_DATE_RANGE = None  # Auto-detect from sessions, or set ('2025-12-01', '2026-01-27')


# File Discovery Functions
def find_orders_csv(account_id: str) -> str:
    """Find the main orders CSV file in account's raw data directory.

    Looks for files containing 'Todo pedido' or 'All order' in the filename.
    If multiple files found, uses the most recently modified one.

    Args:
        account_id: Account identifier ('MX' or 'US')

    Returns:
        Absolute path to orders CSV file

    Raises:
        FileNotFoundError: If no matching CSV found
    """
    data_dir = ACCOUNTS[account_id]['data_dir']

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    # Look for orders CSV files
    csv_files = [
        f for f in os.listdir(data_dir)
        if f.endswith('.csv') and ('Todo pedido' in f or 'All order' in f)
    ]

    if not csv_files:
        raise FileNotFoundError(
            f"No orders CSV found in {data_dir}\n"
            f"Expected filename containing 'Todo pedido' or 'All order'"
        )

    if len(csv_files) > 1:
        # Sort by modification time, use most recent
        csv_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(data_dir, f)),
            reverse=True
        )
        import logging
        logging.getLogger(__name__).warning(
            f"Multiple order CSVs found in {data_dir}, using most recent: {csv_files[0]}"
        )

    return os.path.join(data_dir, csv_files[0])


def find_marketing_costs_csv(account_id: str) -> str:
    """Find marketing costs CSV file (optional).

    Args:
        account_id: Account identifier ('MX' or 'US')

    Returns:
        Absolute path to CSV file, or None if not found
    """
    data_dir = ACCOUNTS[account_id]['data_dir']

    if not os.path.exists(data_dir):
        return None

    matches = [
        f for f in os.listdir(data_dir)
        if f.endswith('.csv') and 'Marketing' in f
    ]

    return os.path.join(data_dir, matches[0]) if matches else None


def find_live_sessions_csv(account_id: str) -> str:
    """Find live shopping sessions CSV file (optional).

    Args:
        account_id: Account identifier ('MX' or 'US')

    Returns:
        Absolute path to CSV file, or None if not found
    """
    data_dir = ACCOUNTS[account_id]['data_dir']

    if not os.path.exists(data_dir):
        return None

    matches = [
        f for f in os.listdir(data_dir)
        if f.endswith('.csv') and 'Live Shopping' in f and 'Example' not in f
    ]

    return os.path.join(data_dir, matches[0]) if matches else None
