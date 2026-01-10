"""
Business metrics calculation module for cohort analysis
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def calculate_retention_matrix(df: pd.DataFrame, customer_id_field: str = 'Buyer Username') -> pd.DataFrame:
    """
    Calculate cohort retention matrix showing unique customers active in each cohort-month.

    Args:
        df: DataFrame with cohort analysis data (must have 'Cohort' and 'Cohort Age')
        customer_id_field: Column name for customer identification

    Returns:
        DataFrame with cohorts as rows, cohort ages as columns, values are customer counts
    """
    logger.info("Calculating retention matrix")

    # Create pivot table: cohort x cohort_age, count unique customers
    retention_matrix = df.groupby(['Cohort', 'Cohort Age'])[customer_id_field].nunique().reset_index()
    retention_matrix = retention_matrix.pivot(index='Cohort', columns='Cohort Age', values=customer_id_field)

    # Fill NaN with 0
    retention_matrix = retention_matrix.fillna(0).astype(int)

    # Sort by cohort
    retention_matrix = retention_matrix.sort_index()

    # Mask impossible future months (cohorts can't have data for months that haven't occurred yet)
    max_date = df['Created Time'].max()
    for cohort in retention_matrix.index:
        # Parse cohort (YYYY-MM format)
        cohort_date = pd.Period(cohort, freq='M')
        # Calculate max possible cohort age for this cohort
        max_possible_age = (pd.Period(max_date, freq='M') - cohort_date).n
        # Set future months to NaN
        for col in retention_matrix.columns:
            if col > max_possible_age:
                retention_matrix.loc[cohort, col] = np.nan

    logger.info(f"Retention matrix shape: {retention_matrix.shape[0]} cohorts x {retention_matrix.shape[1]} months")

    return retention_matrix


def calculate_retention_rates(df: pd.DataFrame, customer_id_field: str = 'Buyer Username') -> pd.DataFrame:
    """
    Calculate cohort retention rates as percentages.

    Args:
        df: DataFrame with cohort analysis data
        customer_id_field: Column name for customer identification

    Returns:
        DataFrame with cohorts as rows, cohort ages as columns, values are retention percentages
    """
    logger.info("Calculating retention rates")

    # Get retention matrix (raw counts)
    retention_matrix = calculate_retention_matrix(df, customer_id_field)

    # Get cohort sizes (month 0 counts)
    cohort_sizes = retention_matrix[0]

    # Calculate retention rates as percentages
    retention_rates = retention_matrix.div(cohort_sizes, axis=0) * 100

    # Round to 2 decimal places
    retention_rates = retention_rates.round(2)

    logger.info("Retention rates calculated")

    return retention_rates


def calculate_aov_overall(df: pd.DataFrame) -> float:
    """
    Calculate overall Average Order Value.

    Args:
        df: DataFrame with 'Order Amount' column

    Returns:
        Overall AOV as float
    """
    aov = df['Order Amount'].mean()
    logger.info(f"Overall AOV: MXN {aov:,.2f}")
    return aov


def calculate_aov_by_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Average Order Value by cohort.

    Args:
        df: DataFrame with cohort analysis data

    Returns:
        DataFrame with Cohort and AOV columns
    """
    logger.info("Calculating AOV by cohort")

    aov_by_cohort = df.groupby('Cohort')['Order Amount'].mean().reset_index()
    aov_by_cohort.columns = ['Cohort', 'AOV']
    aov_by_cohort = aov_by_cohort.sort_values('Cohort')

    return aov_by_cohort


def calculate_aov_first_vs_repeat(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate AOV for first orders vs repeat orders.

    Args:
        df: DataFrame with 'Is First Order' and 'Order Amount' columns

    Returns:
        Dictionary with 'first_order_aov' and 'repeat_order_aov' keys
    """
    logger.info("Calculating AOV for first vs repeat orders")

    first_order_aov = df[df['Is First Order']]['Order Amount'].mean()
    repeat_order_aov = df[~df['Is First Order']]['Order Amount'].mean()

    result = {
        'first_order_aov': first_order_aov,
        'repeat_order_aov': repeat_order_aov,
        'difference': repeat_order_aov - first_order_aov,
        'difference_pct': (repeat_order_aov / first_order_aov - 1) * 100
    }

    logger.info(f"First Order AOV: MXN {first_order_aov:,.2f}")
    logger.info(f"Repeat Order AOV: MXN {repeat_order_aov:,.2f}")
    logger.info(f"Difference: MXN {result['difference']:,.2f} ({result['difference_pct']:+.1f}%)")

    return result


def calculate_customer_ltv(df: pd.DataFrame, customer_id_field: str = 'Buyer Username',
                           ltv_windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
    """
    Calculate Lifetime Value for each customer with multiple time windows.

    Args:
        df: DataFrame with cohort analysis data
        customer_id_field: Column name for customer identification
        ltv_windows: List of months to calculate LTV for (e.g., [3, 6, 12])

    Returns:
        DataFrame with customer-level LTV metrics
    """
    logger.info(f"Calculating customer LTV for windows: {ltv_windows}")

    # Group by customer
    customer_data = []

    for customer_id, customer_df in df.groupby(customer_id_field):
        customer_df = customer_df.sort_values('Created Time')

        customer_record = {
            customer_id_field: customer_id,
            'First Purchase Date': customer_df['First Purchase Date'].iloc[0],
            'Cohort': customer_df['Cohort'].iloc[0],
            'Order Count': len(customer_df),
            'Total Spend': customer_df['Order Amount'].sum(),
            'AOV': customer_df['Order Amount'].mean(),
        }

        # Calculate LTV for each window
        for months in ltv_windows:
            # Filter orders within the window
            orders_in_window = customer_df[customer_df['Cohort Age'] < months]
            ltv = orders_in_window['Order Amount'].sum()
            customer_record[f'LTV_{months}Mo'] = ltv

        # All-time LTV (same as Total Spend)
        customer_record['LTV_AllTime'] = customer_record['Total Spend']

        customer_data.append(customer_record)

    customer_ltv_df = pd.DataFrame(customer_data)

    logger.info(f"Calculated LTV for {len(customer_ltv_df):,} customers")

    return customer_ltv_df


def calculate_ltv_by_cohort(df: pd.DataFrame, customer_id_field: str = 'Buyer Username',
                            ltv_windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
    """
    Calculate average LTV by cohort for multiple time windows.

    Args:
        df: DataFrame with cohort analysis data
        customer_id_field: Column name for customer identification
        ltv_windows: List of months to calculate LTV for

    Returns:
        DataFrame with cohort-level average LTV metrics
    """
    logger.info("Calculating average LTV by cohort")

    # First get customer-level LTV
    customer_ltv = calculate_customer_ltv(df, customer_id_field, ltv_windows)

    # Aggregate by cohort
    ltv_columns = [f'LTV_{m}Mo' for m in ltv_windows] + ['LTV_AllTime']
    agg_dict = {col: 'mean' for col in ltv_columns}
    agg_dict['Order Count'] = 'mean'

    cohort_ltv = customer_ltv.groupby('Cohort').agg(agg_dict).reset_index()

    # Round to 2 decimal places
    for col in ltv_columns + ['Order Count']:
        cohort_ltv[col] = cohort_ltv[col].round(2)

    cohort_ltv = cohort_ltv.sort_values('Cohort')

    return cohort_ltv


def calculate_repeat_purchase_rate(df: pd.DataFrame, customer_id_field: str = 'Buyer Username') -> pd.DataFrame:
    """
    Calculate repeat purchase rate by cohort.

    Args:
        df: DataFrame with cohort analysis data
        customer_id_field: Column name for customer identification

    Returns:
        DataFrame with cohort and repeat purchase rate
    """
    logger.info("Calculating repeat purchase rate by cohort")

    # Count customers who made repeat purchases (Cohort Age > 0)
    cohort_stats = df.groupby('Cohort').agg({
        customer_id_field: 'nunique',  # Total customers in cohort
    }).reset_index()
    cohort_stats.columns = ['Cohort', 'Total Customers']

    # Count customers who made at least one repeat purchase
    repeat_customers = df[df['Cohort Age'] > 0].groupby('Cohort')[customer_id_field].nunique().reset_index()
    repeat_customers.columns = ['Cohort', 'Repeat Customers']

    # Merge
    cohort_stats = cohort_stats.merge(repeat_customers, on='Cohort', how='left')
    cohort_stats['Repeat Customers'] = cohort_stats['Repeat Customers'].fillna(0).astype(int)

    # Calculate repeat rate
    cohort_stats['Repeat Rate %'] = (cohort_stats['Repeat Customers'] / cohort_stats['Total Customers'] * 100).round(2)

    cohort_stats = cohort_stats.sort_values('Cohort')

    return cohort_stats


def calculate_cac(df: pd.DataFrame, marketing_costs_df: pd.DataFrame,
                  customer_id_field: str = 'Buyer Username') -> pd.DataFrame:
    """
    Calculate Customer Acquisition Cost (CAC) by cohort.

    CAC = Marketing Cost / Number of New Customers in Cohort

    Args:
        df: DataFrame with cohort analysis data
        marketing_costs_df: DataFrame with 'Cohort' and 'Marketing Cost' columns
        customer_id_field: Column name for customer identification

    Returns:
        DataFrame with Cohort, Marketing Cost, Cohort Size, and CAC columns
    """
    logger.info("Calculating Customer Acquisition Cost (CAC) by cohort")

    # Get cohort sizes (number of unique customers in each cohort)
    cohort_sizes = df.groupby('Cohort')[customer_id_field].nunique().reset_index()
    cohort_sizes.columns = ['Cohort', 'Cohort Size']

    # Merge with marketing costs
    cac_df = cohort_sizes.merge(marketing_costs_df, on='Cohort', how='left')

    # Calculate CAC
    cac_df['CAC'] = cac_df['Marketing Cost'] / cac_df['Cohort Size']

    # Round to 2 decimal places
    cac_df['CAC'] = cac_df['CAC'].round(2)
    cac_df['Marketing Cost'] = cac_df['Marketing Cost'].round(2)

    # Log summary
    avg_cac = cac_df['CAC'].mean()
    logger.info(f"Average CAC across all cohorts: MXN {avg_cac:,.2f}")

    return cac_df


def generate_cohort_summary(df: pd.DataFrame, customer_id_field: str = 'Buyer Username',
                            ltv_windows: List[int] = [3, 6, 12],
                            marketing_costs_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generate comprehensive cohort-level summary with all key metrics.

    Args:
        df: DataFrame with cohort analysis data
        customer_id_field: Column name for customer identification
        ltv_windows: List of months for LTV calculation
        marketing_costs_df: Optional DataFrame with marketing costs by cohort

    Returns:
        DataFrame with comprehensive cohort summary
    """
    logger.info("Generating comprehensive cohort summary")

    # Get cohort sizes and basic stats
    cohort_summary = df.groupby('Cohort').agg({
        customer_id_field: 'nunique',
        'Order ID': 'count',
        'Order Amount': ['sum', 'mean'],
    }).reset_index()

    cohort_summary.columns = ['Cohort', 'Cohort Size', 'Total Orders', 'Total Revenue', 'AOV']

    # Add first order AOV and repeat order AOV
    first_order_aov = df[df['Is First Order']].groupby('Cohort')['Order Amount'].mean().reset_index()
    first_order_aov.columns = ['Cohort', 'First Order AOV']

    repeat_order_aov = df[~df['Is First Order']].groupby('Cohort')['Order Amount'].mean().reset_index()
    repeat_order_aov.columns = ['Cohort', 'Repeat Order AOV']

    cohort_summary = cohort_summary.merge(first_order_aov, on='Cohort', how='left')
    cohort_summary = cohort_summary.merge(repeat_order_aov, on='Cohort', how='left')

    # Add repeat purchase rate
    repeat_rate = calculate_repeat_purchase_rate(df, customer_id_field)
    cohort_summary = cohort_summary.merge(
        repeat_rate[['Cohort', 'Repeat Customers', 'Repeat Rate %']],
        on='Cohort',
        how='left'
    )

    # Add LTV metrics
    ltv_by_cohort = calculate_ltv_by_cohort(df, customer_id_field, ltv_windows)
    ltv_columns = [f'LTV_{m}Mo' for m in ltv_windows] + ['LTV_AllTime']
    cohort_summary = cohort_summary.merge(
        ltv_by_cohort[['Cohort'] + ltv_columns],
        on='Cohort',
        how='left'
    )

    # Add CAC metrics if marketing costs provided
    if marketing_costs_df is not None:
        cac_df = calculate_cac(df, marketing_costs_df, customer_id_field)
        cohort_summary = cohort_summary.merge(
            cac_df[['Cohort', 'Marketing Cost', 'CAC']],
            on='Cohort',
            how='left'
        )

        # Calculate LTV:CAC ratio if we have both metrics
        for ltv_col in ltv_columns:
            if ltv_col in cohort_summary.columns and 'CAC' in cohort_summary.columns:
                ratio_col = f'{ltv_col}_CAC_Ratio'
                cohort_summary[ratio_col] = (cohort_summary[ltv_col] / cohort_summary['CAC']).round(2)

    # Add retention rates for key months
    retention_rates = calculate_retention_rates(df, customer_id_field)
    key_months = [1, 3, 6]
    for month in key_months:
        if month in retention_rates.columns:
            retention_col = retention_rates[month].reset_index()
            retention_col.columns = ['Cohort', f'Retention_Month_{month}']
            cohort_summary = cohort_summary.merge(retention_col, on='Cohort', how='left')

    # Round numeric columns
    numeric_cols = cohort_summary.select_dtypes(include=[np.number]).columns
    cohort_summary[numeric_cols] = cohort_summary[numeric_cols].round(2)

    cohort_summary = cohort_summary.sort_values('Cohort')

    logger.info(f"Generated comprehensive summary for {len(cohort_summary)} cohorts")

    return cohort_summary
