"""
Live Shopping session analysis module
"""
import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from typing import Tuple

logger = logging.getLogger(__name__)


def match_orders_to_sessions(orders_df: pd.DataFrame, sessions_df: pd.DataFrame,
                             buffer_minutes: int = 30) -> pd.DataFrame:
    """
    Match orders to Live Shopping sessions based on Created Time.

    Orders are matched if Created Time falls between Session Start Time and
    (End Time + buffer_minutes). This accounts for delayed order placement.

    Args:
        orders_df: DataFrame with order data (must have 'Created Time' column)
        sessions_df: DataFrame with session data (must have 'Session ID', 'Start Time', 'End Time')
        buffer_minutes: Minutes to add after session ends for attribution window

    Returns:
        DataFrame with additional columns:
        - Is Live Order: Boolean indicating if order was during a live session
        - Session ID: Session identifier (null for non-live orders)
        - Order Source: "Live Shopping" or "Regular"
        - Host: Session host name (if available)
        - Peak Viewers: Session peak viewers (if available)
        - Total Viewers: Session total viewers (if available)
    """
    logger.info("Matching orders to live shopping sessions")
    logger.info(f"Using {buffer_minutes} minute buffer after session end")

    if sessions_df is None or len(sessions_df) == 0:
        logger.warning("No sessions provided, marking all orders as 'Regular'")
        orders_df['Is Live Order'] = False
        orders_df['Session ID'] = None
        orders_df['Order Source'] = 'Regular'
        return orders_df

    # Ensure Created Time is datetime
    if not pd.api.types.is_datetime64_any_dtype(orders_df['Created Time']):
        raise ValueError("orders_df 'Created Time' must be datetime type")

    # Initialize new columns
    orders_df = orders_df.copy()
    orders_df['Is Live Order'] = False
    orders_df['Session ID'] = None
    orders_df['Order Source'] = 'Regular'

    # Add session metadata columns if they exist
    session_metadata_cols = ['Host', 'Peak Viewers', 'Total Viewers']
    for col in session_metadata_cols:
        if col in sessions_df.columns:
            orders_df[col] = None

    # Match each order to sessions
    matched_count = 0
    for idx, session in sessions_df.iterrows():
        session_start = session['Start Time']
        session_end = session['End Time'] + timedelta(minutes=buffer_minutes)

        # Find orders within this session's time window
        in_session = (
            (orders_df['Created Time'] >= session_start) &
            (orders_df['Created Time'] <= session_end)
        )

        # Only match orders that haven't been matched yet (first session wins)
        in_session_unmatched = in_session & (~orders_df['Is Live Order'])

        # Update matched orders
        if in_session_unmatched.any():
            orders_df.loc[in_session_unmatched, 'Is Live Order'] = True
            orders_df.loc[in_session_unmatched, 'Session ID'] = session['Session ID']
            orders_df.loc[in_session_unmatched, 'Order Source'] = 'Live Shopping'

            # Add metadata if available
            for col in session_metadata_cols:
                if col in session.index and col in orders_df.columns:
                    orders_df.loc[in_session_unmatched, col] = session[col]

            matched_count += in_session_unmatched.sum()

    live_orders = orders_df['Is Live Order'].sum()
    live_pct = (live_orders / len(orders_df)) * 100

    logger.info(f"Matched {live_orders:,} orders to {sessions_df['Session ID'].nunique()} live sessions")
    logger.info(f"Live orders: {live_pct:.1f}%, Regular orders: {100-live_pct:.1f}%")

    return orders_df


def get_session_summary(orders_df: pd.DataFrame, sessions_df: pd.DataFrame,
                       customer_id_field: str = 'Buyer Username') -> pd.DataFrame:
    """
    Generate performance summary for each live shopping session.

    Args:
        orders_df: DataFrame with order data (must be matched to sessions first)
        sessions_df: DataFrame with session data
        customer_id_field: Column name for customer identification

    Returns:
        DataFrame with session-level metrics including:
        - Total Orders, Total Revenue, Unique Customers
        - AOV, New vs Returning customers
        - Conversion Rate (if viewer data available)
    """
    logger.info("Calculating session-level performance metrics")

    # Filter to only live orders
    live_orders = orders_df[orders_df['Is Live Order']].copy()

    if len(live_orders) == 0:
        logger.warning("No live orders found, returning empty summary")
        return pd.DataFrame()

    # Aggregate by session
    session_agg = live_orders.groupby('Session ID').agg({
        'Order ID': 'count',
        'Order Amount': ['sum', 'mean'],
        customer_id_field: 'nunique'
    }).reset_index()

    session_agg.columns = ['Session ID', 'Total Orders', 'Total Revenue', 'AOV', 'Unique Customers']

    # Determine new vs returning customers for each session
    # A customer is "new" if this is their first order
    first_order_customers = orders_df[orders_df['Is First Order']].groupby('Session ID')[customer_id_field].nunique().reset_index()
    first_order_customers.columns = ['Session ID', 'New Customers']

    session_agg = session_agg.merge(first_order_customers, on='Session ID', how='left')
    session_agg['New Customers'] = session_agg['New Customers'].fillna(0).astype(int)
    session_agg['Returning Customers'] = session_agg['Unique Customers'] - session_agg['New Customers']

    # Merge with session metadata
    session_metadata = sessions_df.copy()
    session_agg = session_agg.merge(session_metadata, on='Session ID', how='left')

    # Calculate conversion rate if viewer data available
    if 'Total Viewers' in session_agg.columns:
        session_agg['Conversion Rate %'] = (
            (session_agg['Unique Customers'] / session_agg['Total Viewers']) * 100
        ).round(2)
        # Handle division by zero or missing viewer data
        session_agg['Conversion Rate %'] = session_agg['Conversion Rate %'].fillna(0)

    # Calculate revenue per minute if duration available
    if 'Duration Minutes' in session_agg.columns:
        session_agg['Revenue per Minute'] = (
            session_agg['Total Revenue'] / session_agg['Duration Minutes']
        ).round(2)

    # Sort by session start time
    if 'Start Time' in session_agg.columns:
        session_agg = session_agg.sort_values('Start Time')

    # Round numeric columns
    numeric_cols = ['Total Revenue', 'AOV']
    for col in numeric_cols:
        if col in session_agg.columns:
            session_agg[col] = session_agg[col].round(2)

    logger.info(f"Generated summary for {len(session_agg)} sessions")
    logger.info(f"Average orders per session: {session_agg['Total Orders'].mean():.1f}")
    logger.info(f"Average revenue per session: MXN {session_agg['Total Revenue'].mean():,.2f}")

    return session_agg


def get_host_performance(session_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate session performance by host/presenter.

    Args:
        session_summary: DataFrame from get_session_summary()

    Returns:
        DataFrame with host-level aggregated metrics
    """
    if 'Host' not in session_summary.columns:
        logger.warning("No 'Host' column found in session summary")
        return pd.DataFrame()

    logger.info("Calculating host performance metrics")

    # Group by host
    host_perf = session_summary.groupby('Host').agg({
        'Session ID': 'count',
        'Total Orders': 'sum',
        'Total Revenue': 'sum',
        'Unique Customers': 'sum',
        'AOV': 'mean',
        'New Customers': 'sum',
        'Returning Customers': 'sum'
    }).reset_index()

    host_perf.columns = [
        'Host', 'Total Sessions', 'Total Orders', 'Total Revenue',
        'Total Customers', 'Avg AOV', 'New Customers', 'Returning Customers'
    ]

    # Calculate averages per session
    host_perf['Orders per Session'] = (host_perf['Total Orders'] / host_perf['Total Sessions']).round(1)
    host_perf['Revenue per Session'] = (host_perf['Total Revenue'] / host_perf['Total Sessions']).round(2)

    # Sort by total revenue descending
    host_perf = host_perf.sort_values('Total Revenue', ascending=False)

    # Round numeric columns
    host_perf['Total Revenue'] = host_perf['Total Revenue'].round(2)
    host_perf['Avg AOV'] = host_perf['Avg AOV'].round(2)

    logger.info(f"Analyzed performance for {len(host_perf)} hosts")

    return host_perf
