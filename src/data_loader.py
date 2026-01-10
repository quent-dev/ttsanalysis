"""
Data loading and cleaning module for TikTok Shop orders
"""
import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, Tuple
import os

logger = logging.getLogger(__name__)


def load_tiktok_orders(csv_path: str) -> pd.DataFrame:
    """
    Load TikTok Shop orders from CSV file with proper encoding and parsing.

    Args:
        csv_path: Path to the CSV file

    Returns:
        DataFrame with raw order data
    """
    logger.info(f"Loading orders from {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read CSV with UTF-8 BOM encoding
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    logger.info(f"Loaded {len(df):,} orders with {len(df.columns)} columns")

    return df


def clean_datetime_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and parse datetime fields, handling trailing tabs and whitespace.

    Args:
        df: DataFrame with raw datetime strings

    Returns:
        DataFrame with parsed datetime columns
    """
    logger.info("Cleaning and parsing datetime fields")

    date_columns = [
        'Created Time', 'Paid Time', 'RTS Time',
        'Shipped Time', 'Delivered Time', 'Cancelled Time'
    ]

    for col in date_columns:
        if col in df.columns:
            # Strip whitespace and tabs
            df[col] = df[col].astype(str).str.strip()

            # Replace empty strings with NaN
            df[col] = df[col].replace(['', 'nan', 'None'], np.nan)

            # Parse datetime with error handling
            df[col] = pd.to_datetime(df[col], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

            null_count = df[col].isna().sum()
            if null_count > 0:
                logger.warning(f"{col}: {null_count:,} invalid/missing dates")

    return df


def parse_currency_amount(series: pd.Series) -> pd.Series:
    """
    Parse currency strings (e.g., "MXN 790.00" or "MXN 1,234.56") to float.

    Args:
        series: Series with currency strings

    Returns:
        Series with numeric values
    """
    # Convert to string and handle NaN
    series_str = series.astype(str).str.strip()

    # Remove "MXN" prefix and commas
    series_str = series_str.str.replace('MXN', '', regex=False)
    series_str = series_str.str.replace(',', '', regex=False)
    series_str = series_str.str.strip()

    # Convert to float, coerce errors to NaN
    numeric_series = pd.to_numeric(series_str, errors='coerce')

    return numeric_series


def filter_completed_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to only completed/delivered orders.

    Includes:
    - Order Status = "Completado"
    - Order Status = "Enviado" AND Order Substatus = "Entregado"

    Args:
        df: DataFrame with all orders

    Returns:
        DataFrame with only completed/delivered orders
    """
    logger.info("Filtering to completed/delivered orders")

    initial_count = len(df)

    # Create filter conditions
    is_completado = df['Order Status'] == 'Completado'
    is_enviado_entregado = (df['Order Status'] == 'Enviado') & (df['Order Substatus'] == 'Entregado')

    # Apply filter
    df_filtered = df[is_completado | is_enviado_entregado].copy()

    filtered_count = len(df_filtered)
    logger.info(f"Kept {filtered_count:,} completed orders ({filtered_count/initial_count*100:.1f}%)")
    logger.info(f"Excluded {initial_count - filtered_count:,} orders")

    return df_filtered


def validate_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate data quality and return summary statistics.

    Args:
        df: DataFrame to validate

    Returns:
        Dictionary with validation results
    """
    logger.info("Validating data quality")

    validation = {
        'total_orders': len(df),
        'unique_order_ids': df['Order ID'].nunique(),
        'duplicate_orders': len(df) - df['Order ID'].nunique(),
        'missing_buyer_username': df['Buyer Username'].isna().sum(),
        'missing_order_amount': df['Order Amount'].isna().sum(),
        'missing_created_time': df['Created Time'].isna().sum(),
        'zero_value_orders': (df['Order Amount'] == 0).sum(),
        'negative_value_orders': (df['Order Amount'] < 0).sum(),
    }

    # Log validation results
    logger.info(f"Validation results:")
    for key, value in validation.items():
        if value > 0 and key != 'total_orders' and key != 'unique_order_ids':
            logger.warning(f"  {key}: {value:,}")
        else:
            logger.info(f"  {key}: {value:,}")

    return validation


def load_marketing_costs(csv_path: str) -> pd.DataFrame:
    """
    Load marketing costs by cohort/month from CSV file.

    Args:
        csv_path: Path to the marketing costs CSV file

    Returns:
        DataFrame with 'Cohort' and 'Marketing Cost' columns
    """
    logger.info(f"Loading marketing costs from {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Marketing costs file not found: {csv_path}")

    # Read CSV with UTF-8 BOM encoding
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # Expected columns: Month, Marketing
    if 'Month' not in df.columns or 'Marketing' not in df.columns:
        raise ValueError(f"Marketing CSV must have 'Month' and 'Marketing' columns. Found: {df.columns.tolist()}")

    # Clean and rename
    df_clean = df[['Month', 'Marketing']].copy()
    df_clean.columns = ['Cohort', 'Marketing Cost']

    # Parse marketing cost (format: "$65,987.98 ")
    df_clean['Marketing Cost'] = df_clean['Marketing Cost'].astype(str).str.strip()
    df_clean['Marketing Cost'] = df_clean['Marketing Cost'].str.replace('$', '', regex=False)
    df_clean['Marketing Cost'] = df_clean['Marketing Cost'].str.replace(',', '', regex=False)
    df_clean['Marketing Cost'] = pd.to_numeric(df_clean['Marketing Cost'], errors='coerce')

    # Remove any rows with missing data
    df_clean = df_clean.dropna()

    logger.info(f"Loaded marketing costs for {len(df_clean)} cohorts")

    return df_clean


def load_live_sessions(csv_path: str) -> pd.DataFrame:
    """
    Load Live Shopping session data from CSV file.

    Args:
        csv_path: Path to the Live Shopping Sessions CSV file

    Returns:
        DataFrame with session data including Session ID, Start Time, End Time, and metadata
    """
    logger.info(f"Loading live shopping sessions from {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Live shopping sessions file not found: {csv_path}")

    # Read CSV with UTF-8 BOM encoding
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # Required columns
    required_cols = ['Session ID', 'Start Time', 'End Time']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Live Shopping Sessions CSV must have columns: {required_cols}. Missing: {missing_cols}")

    # Parse datetime fields
    for col in ['Start Time', 'End Time']:
        df[col] = df[col].astype(str).str.strip()
        df[col] = pd.to_datetime(df[col], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

        null_count = df[col].isna().sum()
        if null_count > 0:
            logger.warning(f"{col}: {null_count} invalid/missing dates")

    # Remove rows with missing critical data
    initial_count = len(df)
    df = df.dropna(subset=['Session ID', 'Start Time', 'End Time'])
    dropped = initial_count - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} sessions with missing critical data")

    # Validate End Time > Start Time
    invalid_times = (df['End Time'] <= df['Start Time']).sum()
    if invalid_times > 0:
        logger.error(f"Found {invalid_times} sessions where End Time <= Start Time")
        df = df[df['End Time'] > df['Start Time']]
        logger.warning(f"Removed {invalid_times} invalid sessions")

    # Calculate Duration Minutes if not provided
    if 'Duration Minutes' not in df.columns or df['Duration Minutes'].isna().all():
        df['Duration Minutes'] = (df['End Time'] - df['Start Time']).dt.total_seconds() / 60
        df['Duration Minutes'] = df['Duration Minutes'].round(0).astype(int)
        logger.info("Calculated Duration Minutes from Start/End times")

    # Parse numeric fields if present
    numeric_fields = ['Peak Viewers', 'Total Viewers']
    for col in numeric_fields:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    logger.info(f"Loaded {len(df)} live shopping sessions")
    logger.info(f"Date range: {df['Start Time'].min()} to {df['End Time'].max()}")

    return df


def load_and_clean_data(csv_path: str, force_reload: bool = False) -> pd.DataFrame:
    """
    Load and clean TikTok Shop order data with caching.

    Args:
        csv_path: Path to CSV file
        force_reload: If True, ignore cached parquet file

    Returns:
        Cleaned DataFrame ready for analysis
    """
    from config import CLEANED_DATA_FILE, PROCESSED_DIR

    # Check if cached cleaned data exists
    if os.path.exists(CLEANED_DATA_FILE) and not force_reload:
        logger.info(f"Loading cached cleaned data from {CLEANED_DATA_FILE}")
        return pd.read_parquet(CLEANED_DATA_FILE)

    # Load raw data
    df = load_tiktok_orders(csv_path)

    # Clean datetime fields
    df = clean_datetime_fields(df)

    # Parse currency fields
    currency_columns = [
        'SKU Unit Original Price', 'SKU Subtotal Before Discount',
        'SKU Platform Discount', 'SKU Seller Discount',
        'SKU Subtotal After Discount', 'Shipping Fee After Discount',
        'Original Shipping Fee', 'Shipping Fee Seller Discount',
        'Shipping Fee Platform Discount', 'Payment platform discount',
        'Retail Delivery Fee', 'Order Amount', 'Order Refund Amount'
    ]

    for col in currency_columns:
        if col in df.columns:
            df[col] = parse_currency_amount(df[col])

    # Filter to completed orders
    df = filter_completed_orders(df)

    # Drop duplicates (keep first occurrence)
    initial_count = len(df)
    df = df.drop_duplicates(subset=['Order ID'], keep='first')
    duplicates_removed = initial_count - len(df)
    if duplicates_removed > 0:
        logger.warning(f"Removed {duplicates_removed:,} duplicate order IDs")

    # Drop rows with missing critical fields
    df = df.dropna(subset=['Buyer Username', 'Created Time', 'Order Amount'])

    # Validate data quality
    validation = validate_data_quality(df)

    # Save cleaned data to parquet for faster loading
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_parquet(CLEANED_DATA_FILE, index=False)
    logger.info(f"Saved cleaned data to {CLEANED_DATA_FILE}")

    return df
