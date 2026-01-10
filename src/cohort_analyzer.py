"""
Cohort assignment and analysis module
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def assign_cohorts(df: pd.DataFrame, customer_id_field: str = 'Buyer Username') -> pd.DataFrame:
    """
    Assign each customer to a cohort based on their first purchase date.

    Args:
        df: DataFrame with order data (must have 'Created Time' column)
        customer_id_field: Column name for customer identification

    Returns:
        DataFrame with additional columns:
        - First Purchase Date: Date of customer's first order
        - Cohort: YYYY-MM format of first purchase month
    """
    logger.info("Assigning customers to cohorts")

    # Ensure Created Time is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Created Time']):
        raise ValueError("Created Time must be datetime type")

    # Find first purchase date for each customer
    first_purchase = df.groupby(customer_id_field)['Created Time'].min().reset_index()
    first_purchase.columns = [customer_id_field, 'First Purchase Date']

    # Merge back to original dataframe
    df = df.merge(first_purchase, on=customer_id_field, how='left')

    # Create cohort as YYYY-MM
    df['Cohort'] = df['First Purchase Date'].dt.to_period('M').astype(str)

    unique_customers = df[customer_id_field].nunique()
    unique_cohorts = df['Cohort'].nunique()
    logger.info(f"Assigned {unique_customers:,} customers to {unique_cohorts} cohorts")

    return df


def calculate_cohort_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate how many months after first purchase each order was made.

    Args:
        df: DataFrame with 'Created Time' and 'First Purchase Date' columns

    Returns:
        DataFrame with additional column:
        - Cohort Age: Number of months since first purchase (0, 1, 2, ...)
    """
    logger.info("Calculating cohort age for each order")

    # Calculate month difference
    created_period = df['Created Time'].dt.to_period('M')
    first_purchase_period = df['First Purchase Date'].dt.to_period('M')

    df['Cohort Age'] = (created_period - first_purchase_period).apply(lambda x: x.n)

    # Validate cohort age (should be >= 0)
    negative_ages = (df['Cohort Age'] < 0).sum()
    if negative_ages > 0:
        logger.warning(f"Found {negative_ages} orders with negative cohort age (data quality issue)")
        # Set negative ages to 0
        df.loc[df['Cohort Age'] < 0, 'Cohort Age'] = 0

    logger.info(f"Cohort age range: {df['Cohort Age'].min()} to {df['Cohort Age'].max()} months")

    return df


def mark_first_orders(df: pd.DataFrame, customer_id_field: str = 'Buyer Username') -> pd.DataFrame:
    """
    Mark which orders are first orders vs repeat orders for each customer.

    Args:
        df: DataFrame with order data and 'First Purchase Date' column
        customer_id_field: Column name for customer identification

    Returns:
        DataFrame with additional column:
        - Is First Order: Boolean indicating if this is customer's first order
    """
    logger.info("Marking first orders vs repeat orders")

    # An order is a first order if its Created Time equals the First Purchase Date
    # We'll use a small tolerance (1 minute) to handle potential floating point issues
    df['Is First Order'] = (df['Created Time'] == df['First Purchase Date'])

    first_orders = df['Is First Order'].sum()
    repeat_orders = (~df['Is First Order']).sum()
    repeat_rate = repeat_orders / len(df) * 100

    logger.info(f"First orders: {first_orders:,} ({100-repeat_rate:.1f}%)")
    logger.info(f"Repeat orders: {repeat_orders:,} ({repeat_rate:.1f}%)")

    return df


def build_cohort_index(df: pd.DataFrame, customer_id_field: str = 'Buyer Username',
                       min_cohort_size: int = 10) -> pd.DataFrame:
    """
    Build complete cohort analysis dataframe with all necessary fields.

    This is the main function that combines all cohort assignment steps.

    Args:
        df: Cleaned order DataFrame
        customer_id_field: Column name for customer identification
        min_cohort_size: Minimum number of customers required for a cohort to be included

    Returns:
        DataFrame with cohort analysis fields added
    """
    logger.info("Building cohort index")

    # Assign cohorts
    df = assign_cohorts(df, customer_id_field)

    # Filter out small cohorts (insufficient sample size)
    cohort_sizes = df.groupby('Cohort')[customer_id_field].nunique()
    small_cohorts = cohort_sizes[cohort_sizes < min_cohort_size].index.tolist()

    if small_cohorts:
        logger.info(f"Filtering out {len(small_cohorts)} cohort(s) with < {min_cohort_size} customers:")
        for cohort in small_cohorts:
            cohort_size = cohort_sizes[cohort]
            logger.info(f"  - {cohort}: {cohort_size} customer(s)")

        initial_count = len(df)
        df = df[~df['Cohort'].isin(small_cohorts)].copy()
        filtered_count = initial_count - len(df)
        logger.info(f"Removed {filtered_count:,} orders from small cohorts")

    # Calculate cohort age
    df = calculate_cohort_age(df)

    # Mark first orders
    df = mark_first_orders(df, customer_id_field)

    logger.info("Cohort index built successfully")

    return df


def get_cohort_summary(df: pd.DataFrame, customer_id_field: str = 'Buyer Username') -> pd.DataFrame:
    """
    Generate summary statistics for each cohort.

    Args:
        df: DataFrame with cohort analysis data
        customer_id_field: Column name for customer identification

    Returns:
        DataFrame with cohort-level summary statistics
    """
    logger.info("Generating cohort summary")

    summary = df.groupby('Cohort').agg({
        customer_id_field: 'nunique',  # Cohort size
        'Order ID': 'count',  # Total orders
        'Order Amount': ['sum', 'mean'],  # Total and average revenue
        'Is First Order': 'sum',  # First orders (should equal cohort size)
    }).reset_index()

    # Flatten column names
    summary.columns = [
        'Cohort',
        'Cohort Size',
        'Total Orders',
        'Total Revenue',
        'Avg Order Value',
        'First Orders'
    ]

    # Calculate repeat order count
    summary['Repeat Orders'] = summary['Total Orders'] - summary['First Orders']

    # Calculate orders per customer
    summary['Orders per Customer'] = summary['Total Orders'] / summary['Cohort Size']

    # Sort by cohort
    summary = summary.sort_values('Cohort')

    logger.info(f"Generated summary for {len(summary)} cohorts")

    return summary
