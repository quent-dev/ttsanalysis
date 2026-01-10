"""
Visualization module for cohort analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Tuple
import os

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_retention_heatmap(retention_rates: pd.DataFrame, save_path: str, dpi: int = 300) -> None:
    """
    Create a heatmap visualization of cohort retention rates.

    Args:
        retention_rates: DataFrame with cohorts as rows, cohort ages as columns, values as percentages
        save_path: Path to save the figure
        dpi: Resolution for saved figure
    """
    logger.info("Generating retention heatmap")

    fig, ax = plt.subplots(figsize=(14, 10))

    # Create a mask for NaN values (future months that haven't occurred)
    mask = retention_rates.isna()

    # Create custom annotations that show empty string for NaN and 0 values
    annot_data = retention_rates.copy()
    # Replace NaN with empty string for annotation
    annot_array = np.where(mask, '', annot_data.values)
    # Replace 0 with empty string for annotation (except month 0 which is always 100%)
    for i in range(len(annot_data)):
        for j in range(len(annot_data.columns)):
            if j == 0:  # Month 0 is always 100%, keep it
                annot_array[i, j] = f'{annot_data.iloc[i, j]:.0f}'
            elif not mask.iloc[i, j] and annot_data.iloc[i, j] == 0:
                annot_array[i, j] = ''
            elif not mask.iloc[i, j]:
                annot_array[i, j] = f'{annot_data.iloc[i, j]:.1f}'

    # Create heatmap with mask for NaN values
    # Exclude 0 from color scale by setting vmin=0.1 (so only actual retention gets colored)
    sns.heatmap(
        retention_rates,
        annot=annot_array,
        fmt='',
        mask=mask,
        cmap='RdYlGn',
        center=20,  # Adjusted center for better color distribution
        vmin=0.1,   # Exclude 0 from color scale
        vmax=100,
        cbar_kws={'label': 'Retention Rate (%)'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray',
        cbar=True
    )

    ax.set_title('Cohort Retention Heatmap (%)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Months Since First Purchase', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cohort (First Purchase Month)', fontsize=12, fontweight='bold')

    # Rotate x-axis labels
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved retention heatmap to {save_path}")


def plot_retention_curves(retention_rates: pd.DataFrame, save_path: str, dpi: int = 300,
                          highlight_cohorts: List[str] = None) -> None:
    """
    Create line charts showing retention curves for each cohort.

    Args:
        retention_rates: DataFrame with retention rates by cohort
        save_path: Path to save the figure
        dpi: Resolution for saved figure
        highlight_cohorts: Optional list of cohorts to highlight
    """
    logger.info("Generating retention curves")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each cohort as a line
    for cohort in retention_rates.index:
        # Get data for this cohort (exclude month 0 as it's always 100%)
        data = retention_rates.loc[cohort][1:]

        # Plot with different styling based on highlight
        if highlight_cohorts and cohort in highlight_cohorts:
            ax.plot(data.index, data.values, marker='o', linewidth=2.5, label=cohort, alpha=0.9)
        else:
            ax.plot(data.index, data.values, marker='o', linewidth=1, alpha=0.5)

    # Calculate and plot average retention curve
    avg_retention = retention_rates.mean(axis=0)[1:]
    ax.plot(avg_retention.index, avg_retention.values, marker='s', linewidth=3,
            color='black', label='Average', linestyle='--', alpha=0.8)

    ax.set_title('Cohort Retention Curves', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Months Since First Purchase', fontsize=12, fontweight='bold')
    ax.set_ylabel('Retention Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Add legend (only for highlighted cohorts + average)
    if highlight_cohorts:
        ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved retention curves to {save_path}")


def plot_ltv_curves(cohort_ltv: pd.DataFrame, save_path: str, dpi: int = 300) -> None:
    """
    Create visualization of LTV progression by cohort.

    Args:
        cohort_ltv: DataFrame with LTV metrics by cohort
        save_path: Path to save the figure
        dpi: Resolution for saved figure
    """
    logger.info("Generating LTV curves")

    # Identify LTV columns
    ltv_columns = [col for col in cohort_ltv.columns if col.startswith('LTV_') and col.endswith('Mo')]

    if not ltv_columns:
        logger.warning("No LTV columns found, skipping LTV curve plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: LTV by cohort for different time windows
    x_positions = range(len(cohort_ltv))
    width = 0.8 / len(ltv_columns)

    for i, ltv_col in enumerate(ltv_columns):
        offset = (i - len(ltv_columns)/2) * width + width/2
        window = ltv_col.replace('LTV_', '').replace('Mo', '')
        ax1.bar([x + offset for x in x_positions], cohort_ltv[ltv_col],
                width=width, label=f'{window} Month LTV', alpha=0.8)

    ax1.set_xlabel('Cohort', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average LTV (MXN)', fontsize=12, fontweight='bold')
    ax1.set_title('Average Lifetime Value by Cohort and Time Window', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(cohort_ltv['Cohort'], rotation=45, ha='right')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: All-time LTV trend
    ax2.plot(cohort_ltv['Cohort'], cohort_ltv['LTV_AllTime'], marker='o',
             linewidth=2, markersize=8, color='darkgreen', label='All-Time LTV')

    # Add trend line
    x_numeric = range(len(cohort_ltv))
    z = np.polyfit(x_numeric, cohort_ltv['LTV_AllTime'], 1)
    p = np.poly1d(z)
    ax2.plot(cohort_ltv['Cohort'], p(x_numeric), linestyle='--', color='red',
             alpha=0.6, linewidth=2, label='Trend')

    ax2.set_xlabel('Cohort', fontsize=12, fontweight='bold')
    ax2.set_ylabel('All-Time LTV (MXN)', fontsize=12, fontweight='bold')
    ax2.set_title('All-Time Lifetime Value Trend by Cohort', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticklabels(cohort_ltv['Cohort'], rotation=45, ha='right')
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved LTV curves to {save_path}")


def plot_aov_distribution(df: pd.DataFrame, save_path: str, dpi: int = 300) -> None:
    """
    Create visualization comparing AOV for first orders vs repeat orders.

    Args:
        df: DataFrame with 'Is First Order' and 'Order Amount' columns
        save_path: Path to save the figure
        dpi: Resolution for saved figure
    """
    logger.info("Generating AOV distribution plot")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Box plot comparison
    data_to_plot = [
        df[df['Is First Order']]['Order Amount'],
        df[~df['Is First Order']]['Order Amount']
    ]

    bp = ax1.boxplot(data_to_plot, labels=['First Order', 'Repeat Order'],
                     patch_artist=True, showmeans=True, meanline=True)

    # Color the boxes
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_ylabel('Order Amount (MXN)', fontsize=12, fontweight='bold')
    ax1.set_title('AOV Distribution: First vs Repeat Orders', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add median values as text
    for i, data in enumerate(data_to_plot):
        median = data.median()
        mean = data.mean()
        ax1.text(i + 1, median, f'Median: MXN {median:.2f}',
                horizontalalignment='center', verticalalignment='bottom', fontsize=9)

    # Plot 2: Violin plot for distribution shape
    first_order_data = df[df['Is First Order']]['Order Amount']
    repeat_order_data = df[~df['Is First Order']]['Order Amount']

    parts = ax2.violinplot([first_order_data, repeat_order_data], positions=[1, 2],
                           showmeans=True, showmedians=True)

    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['First Order', 'Repeat Order'])
    ax2.set_ylabel('Order Amount (MXN)', fontsize=12, fontweight='bold')
    ax2.set_title('AOV Distribution Shape', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved AOV distribution to {save_path}")


def plot_cohort_sizes(cohort_summary: pd.DataFrame, save_path: str, dpi: int = 300) -> None:
    """
    Create bar chart showing cohort sizes over time.

    Args:
        cohort_summary: DataFrame with Cohort and Cohort Size columns
        save_path: Path to save the figure
        dpi: Resolution for saved figure
    """
    logger.info("Generating cohort size chart")

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(cohort_summary['Cohort'], cohort_summary['Cohort Size'],
                  color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Cohort', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of New Customers', fontsize=12, fontweight='bold')
    ax.set_title('Cohort Sizes (New Customers by Month)', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved cohort sizes chart to {save_path}")


def plot_repeat_rate_by_cohort(cohort_summary: pd.DataFrame, save_path: str, dpi: int = 300) -> None:
    """
    Create visualization of repeat purchase rate by cohort.

    Args:
        cohort_summary: DataFrame with Cohort and Repeat Rate % columns
        save_path: Path to save the figure
        dpi: Resolution for saved figure
    """
    logger.info("Generating repeat rate chart")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Line chart with markers
    ax.plot(cohort_summary['Cohort'], cohort_summary['Repeat Rate %'],
            marker='o', linewidth=2, markersize=8, color='darkblue', label='Repeat Rate %')

    # Add average line
    avg_repeat_rate = cohort_summary['Repeat Rate %'].mean()
    ax.axhline(y=avg_repeat_rate, color='red', linestyle='--', linewidth=2,
               label=f'Average: {avg_repeat_rate:.1f}%', alpha=0.7)

    ax.set_xlabel('Cohort', fontsize=12, fontweight='bold')
    ax.set_ylabel('Repeat Purchase Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Repeat Purchase Rate by Cohort', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved repeat rate chart to {save_path}")


def generate_all_visualizations(df: pd.DataFrame, retention_rates: pd.DataFrame,
                                cohort_ltv: pd.DataFrame, cohort_summary: pd.DataFrame,
                                output_dir: str, dpi: int = 300) -> None:
    """
    Generate all visualization charts and save them.

    Args:
        df: DataFrame with order-level cohort analysis data
        retention_rates: DataFrame with retention rates
        cohort_ltv: DataFrame with LTV metrics by cohort
        cohort_summary: DataFrame with cohort summary statistics
        output_dir: Directory to save visualizations
        dpi: Resolution for saved figures
    """
    logger.info("Generating all visualizations")

    os.makedirs(output_dir, exist_ok=True)

    # Generate each visualization
    plot_retention_heatmap(
        retention_rates,
        os.path.join(output_dir, 'retention_heatmap.png'),
        dpi=dpi
    )

    plot_retention_curves(
        retention_rates,
        os.path.join(output_dir, 'retention_curves.png'),
        dpi=dpi
    )

    plot_ltv_curves(
        cohort_ltv,
        os.path.join(output_dir, 'ltv_curves.png'),
        dpi=dpi
    )

    plot_aov_distribution(
        df,
        os.path.join(output_dir, 'aov_distribution.png'),
        dpi=dpi
    )

    plot_cohort_sizes(
        cohort_summary,
        os.path.join(output_dir, 'cohort_sizes.png'),
        dpi=dpi
    )

    plot_repeat_rate_by_cohort(
        cohort_summary,
        os.path.join(output_dir, 'repeat_rate_by_cohort.png'),
        dpi=dpi
    )

    logger.info(f"All visualizations saved to {output_dir}")
