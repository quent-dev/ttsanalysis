#!/usr/bin/env python3
"""
TikTok Shop Cohort Analysis Tool
Main execution script with menu-driven interface
"""
import sys
import os
import logging
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

import config
from src.data_loader import load_and_clean_data, load_marketing_costs, load_live_sessions
from src.cohort_analyzer import build_cohort_index, get_cohort_summary
from src.metrics_calculator import (
    calculate_retention_rates,
    calculate_aov_overall,
    calculate_aov_by_cohort,
    calculate_aov_first_vs_repeat,
    calculate_ltv_by_cohort,
    calculate_customer_ltv,
    generate_cohort_summary,
    compare_live_vs_regular,
    calculate_live_cohort_retention
)
from src.session_analyzer import match_orders_to_sessions, get_session_summary, get_host_performance
from src.visualizer import generate_all_visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CohortAnalysisTool:
    """Main class for TikTok Shop cohort analysis."""

    def __init__(self):
        self.df = None
        self.marketing_costs_df = None
        self.sessions_df = None
        self.retention_rates = None
        self.cohort_ltv = None
        self.cohort_summary = None
        self.customer_ltv = None
        self.live_comparison = None
        self.session_summary = None
        self.host_performance = None
        self.live_retention = None

    def load_data(self, force_reload: bool = False):
        """Load and clean order data."""
        logger.info("="*80)
        logger.info("LOADING AND CLEANING DATA")
        logger.info("="*80)

        csv_path = os.path.join(config.DATA_DIR, config.CSV_FILENAME)

        try:
            self.df = load_and_clean_data(csv_path, force_reload=force_reload)
            logger.info(f"✓ Successfully loaded {len(self.df):,} orders")
            logger.info(f"✓ Date range: {self.df['Created Time'].min()} to {self.df['Created Time'].max()}")
            logger.info(f"✓ Unique customers: {self.df[config.CUSTOMER_ID_FIELD].nunique():,}")

            # Load marketing costs if available
            self.load_marketing_costs()

            return True
        except Exception as e:
            logger.error(f"✗ Error loading data: {e}")
            return False

    def load_marketing_costs(self):
        """Load marketing costs data for CAC calculation."""
        marketing_costs_path = os.path.join(config.DATA_DIR, config.MARKETING_COSTS_FILENAME)

        try:
            if os.path.exists(marketing_costs_path):
                self.marketing_costs_df = load_marketing_costs(marketing_costs_path)
                logger.info(f"✓ Successfully loaded marketing costs for {len(self.marketing_costs_df):,} cohorts")
            else:
                logger.warning(f"⚠ Marketing costs file not found at {marketing_costs_path}")
                logger.warning("⚠ CAC metrics will not be calculated")
                self.marketing_costs_df = None
        except Exception as e:
            logger.error(f"✗ Error loading marketing costs: {e}")
            self.marketing_costs_df = None

    def load_sessions(self):
        """Load Live Shopping sessions data."""
        if not config.ENABLE_LIVE_ANALYSIS:
            logger.info("Live Shopping analysis is disabled in config")
            return False

        sessions_path = os.path.join(config.DATA_DIR, config.LIVE_SESSIONS_FILENAME)

        try:
            if os.path.exists(sessions_path):
                self.sessions_df = load_live_sessions(sessions_path)
                logger.info(f"✓ Successfully loaded {len(self.sessions_df):,} live shopping sessions")
                return True
            else:
                logger.warning(f"⚠ Live shopping sessions file not found at {sessions_path}")
                logger.warning("⚠ Live shopping analysis will be skipped")
                self.sessions_df = None
                return False
        except Exception as e:
            logger.error(f"✗ Error loading live shopping sessions: {e}")
            self.sessions_df = None
            return False

    def analyze_live_sessions(self):
        """Analyze Live Shopping session performance."""
        if self.df is None:
            logger.error("✗ No data loaded. Please load data first.")
            return False

        if self.sessions_df is None or len(self.sessions_df) == 0:
            logger.warning("⚠ No sessions loaded. Skipping live shopping analysis.")
            return False

        logger.info("="*80)
        logger.info("ANALYZING LIVE SHOPPING SESSIONS")
        logger.info("="*80)

        try:
            # Match orders to sessions
            logger.info("Matching orders to live shopping sessions...")
            self.df = match_orders_to_sessions(
                self.df,
                self.sessions_df,
                buffer_minutes=config.LIVE_SESSION_BUFFER_MINUTES
            )

            # Calculate live vs regular comparison
            logger.info("Comparing live vs regular performance...")
            self.live_comparison = compare_live_vs_regular(
                self.df,
                config.CUSTOMER_ID_FIELD,
                config.LTV_WINDOWS
            )

            # Calculate session-level metrics
            logger.info("Calculating session performance...")
            self.session_summary = get_session_summary(
                self.df,
                self.sessions_df,
                config.CUSTOMER_ID_FIELD
            )

            # Calculate host performance if host data available
            if 'Host' in self.sessions_df.columns:
                logger.info("Calculating host performance...")
                self.host_performance = get_host_performance(self.session_summary)

            # Calculate retention for live-acquired customers
            logger.info("Calculating retention for live-acquired customers...")
            self.live_retention = calculate_live_cohort_retention(
                self.df,
                config.CUSTOMER_ID_FIELD
            )

            logger.info("✓ Live shopping analysis completed successfully")
            return True
        except Exception as e:
            logger.error(f"✗ Error analyzing live shopping sessions: {e}")
            import traceback
            traceback.print_exc()
            return False

    def assign_cohorts(self):
        """Assign customers to cohorts."""
        if self.df is None:
            logger.error("✗ No data loaded. Please load data first.")
            return False

        logger.info("="*80)
        logger.info("ASSIGNING COHORTS")
        logger.info("="*80)

        try:
            self.df = build_cohort_index(self.df, config.CUSTOMER_ID_FIELD, config.MIN_COHORT_SIZE)
            logger.info(f"✓ Successfully assigned cohorts")
            return True
        except Exception as e:
            logger.error(f"✗ Error assigning cohorts: {e}")
            return False

    def calculate_metrics(self):
        """Calculate all metrics."""
        if self.df is None:
            logger.error("✗ No data loaded. Please load data first.")
            return False

        logger.info("="*80)
        logger.info("CALCULATING METRICS")
        logger.info("="*80)

        try:
            # Retention rates
            logger.info("Calculating retention rates...")
            self.retention_rates = calculate_retention_rates(self.df, config.CUSTOMER_ID_FIELD)

            # AOV metrics
            logger.info("Calculating AOV metrics...")
            overall_aov = calculate_aov_overall(self.df)
            aov_first_vs_repeat = calculate_aov_first_vs_repeat(self.df)

            # LTV metrics
            logger.info("Calculating LTV metrics...")
            self.cohort_ltv = calculate_ltv_by_cohort(
                self.df,
                config.CUSTOMER_ID_FIELD,
                config.LTV_WINDOWS
            )
            self.customer_ltv = calculate_customer_ltv(
                self.df,
                config.CUSTOMER_ID_FIELD,
                config.LTV_WINDOWS
            )

            # Comprehensive cohort summary
            logger.info("Generating cohort summary...")
            self.cohort_summary = generate_cohort_summary(
                self.df,
                config.CUSTOMER_ID_FIELD,
                config.LTV_WINDOWS,
                self.marketing_costs_df
            )

            logger.info("✓ All metrics calculated successfully")
            return True
        except Exception as e:
            logger.error(f"✗ Error calculating metrics: {e}")
            return False

    def export_csv_reports(self):
        """Export all reports to CSV."""
        if self.retention_rates is None or self.cohort_summary is None:
            logger.error("✗ Metrics not calculated. Please calculate metrics first.")
            return False

        logger.info("="*80)
        logger.info("EXPORTING CSV REPORTS")
        logger.info("="*80)

        try:
            os.makedirs(config.CSV_OUTPUT_DIR, exist_ok=True)

            # Export retention rates
            retention_file = os.path.join(config.CSV_OUTPUT_DIR, 'cohort_retention.csv')
            self.retention_rates.to_csv(retention_file, encoding=config.CSV_ENCODING)
            logger.info(f"✓ Exported retention rates to {retention_file}")

            # Export customer metrics
            customer_file = os.path.join(config.CSV_OUTPUT_DIR, 'customer_metrics.csv')
            self.customer_ltv.to_csv(customer_file, index=False, encoding=config.CSV_ENCODING)
            logger.info(f"✓ Exported customer metrics to {customer_file}")

            # Export cohort summary
            summary_file = os.path.join(config.CSV_OUTPUT_DIR, 'cohort_summary.csv')
            self.cohort_summary.to_csv(summary_file, index=False, encoding=config.CSV_ENCODING)
            logger.info(f"✓ Exported cohort summary to {summary_file}")

            # Export live shopping reports if available
            if self.live_comparison is not None:
                logger.info("Exporting live shopping analysis reports...")

                # Export comparison summary
                comparison_file = os.path.join(config.CSV_OUTPUT_DIR, 'live_vs_regular_comparison.csv')
                comparison_data = []
                for source, metrics in self.live_comparison.items():
                    if source != 'Difference':
                        row = {'Order Source': source}
                        row.update(metrics)
                        comparison_data.append(row)
                pd.DataFrame(comparison_data).to_csv(comparison_file, index=False, encoding=config.CSV_ENCODING)
                logger.info(f"✓ Exported live vs regular comparison to {comparison_file}")

                # Export session performance
                if self.session_summary is not None and len(self.session_summary) > 0:
                    session_file = os.path.join(config.CSV_OUTPUT_DIR, 'session_performance.csv')
                    self.session_summary.to_csv(session_file, index=False, encoding=config.CSV_ENCODING)
                    logger.info(f"✓ Exported session performance to {session_file}")

                # Export host performance
                if self.host_performance is not None and len(self.host_performance) > 0:
                    host_file = os.path.join(config.CSV_OUTPUT_DIR, 'host_performance.csv')
                    self.host_performance.to_csv(host_file, index=False, encoding=config.CSV_ENCODING)
                    logger.info(f"✓ Exported host performance to {host_file}")

                # Export live cohort retention
                if self.live_retention is not None and len(self.live_retention) > 0:
                    live_retention_file = os.path.join(config.CSV_OUTPUT_DIR, 'live_cohort_retention.csv')
                    self.live_retention.to_csv(live_retention_file, encoding=config.CSV_ENCODING)
                    logger.info(f"✓ Exported live cohort retention to {live_retention_file}")

            logger.info("✓ All CSV reports exported successfully")
            return True
        except Exception as e:
            logger.error(f"✗ Error exporting CSV reports: {e}")
            return False

    def generate_visualizations(self):
        """Generate all visualizations."""
        if self.df is None or self.retention_rates is None or self.cohort_summary is None:
            logger.error("✗ Data and metrics required. Please run full analysis first.")
            return False

        logger.info("="*80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*80)

        try:
            generate_all_visualizations(
                self.df,
                self.retention_rates,
                self.cohort_ltv,
                self.cohort_summary,
                config.VIZ_OUTPUT_DIR,
                dpi=config.DPI
            )
            logger.info("✓ All visualizations generated successfully")
            return True
        except Exception as e:
            logger.error(f"✗ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        logger.info("="*80)
        logger.info("RUNNING FULL COHORT ANALYSIS")
        logger.info("="*80)

        start_time = datetime.now()

        # Load data
        if not self.load_data():
            return False

        # Assign cohorts
        if not self.assign_cohorts():
            return False

        # Calculate metrics
        if not self.calculate_metrics():
            return False

        # Load and analyze live shopping sessions if available
        if config.ENABLE_LIVE_ANALYSIS:
            if self.load_sessions():
                self.analyze_live_sessions()

        # Export CSV reports
        if not self.export_csv_reports():
            return False

        # Generate visualizations
        if not self.generate_visualizations():
            return False

        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"Total processing time: {duration:.2f} seconds")
        logger.info(f"CSV reports saved to: {config.CSV_OUTPUT_DIR}")
        logger.info(f"Visualizations saved to: {config.VIZ_OUTPUT_DIR}")

        return True

    def display_summary(self):
        """Display summary statistics."""
        if self.cohort_summary is None:
            logger.error("✗ No summary available. Please run analysis first.")
            return False

        logger.info("="*80)
        logger.info("COHORT ANALYSIS SUMMARY")
        logger.info("="*80)

        print("\n" + "="*80)
        print("COHORT SUMMARY")
        print("="*80)
        print(self.cohort_summary.to_string(index=False))
        print("\n")

        # Overall statistics
        total_customers = self.cohort_summary['Cohort Size'].sum()
        total_revenue = self.cohort_summary['Total Revenue'].sum()
        avg_aov = self.cohort_summary['AOV'].mean()
        avg_repeat_rate = self.cohort_summary['Repeat Rate %'].mean()

        print("OVERALL STATISTICS")
        print("-"*80)
        print(f"Total Customers: {total_customers:,}")
        print(f"Total Revenue: MXN {total_revenue:,.2f}")
        print(f"Average AOV: MXN {avg_aov:,.2f}")
        print(f"Average Repeat Rate: {avg_repeat_rate:.2f}%")

        # Add CAC statistics if available
        if 'CAC' in self.cohort_summary.columns:
            avg_cac = self.cohort_summary['CAC'].mean()
            total_marketing = self.cohort_summary['Marketing Cost'].sum()
            print(f"Total Marketing Spend: MXN {total_marketing:,.2f}")
            print(f"Average CAC: MXN {avg_cac:,.2f}")

            # Show LTV:CAC ratios if available
            if 'LTV_12Mo_CAC_Ratio' in self.cohort_summary.columns:
                avg_ltv_cac_ratio = self.cohort_summary['LTV_12Mo_CAC_Ratio'].mean()
                print(f"Average LTV:CAC Ratio (12Mo): {avg_ltv_cac_ratio:.2f}x")

        # Add Live Shopping statistics if available
        if self.live_comparison is not None and len(self.live_comparison) > 0:
            print("\nLIVE SHOPPING PERFORMANCE")
            print("-"*80)

            if 'Live Shopping' in self.live_comparison:
                live = self.live_comparison['Live Shopping']
                print(f"Live Shopping Orders: {live.get('Total Orders', 0):,}")
                print(f"Live Shopping AOV: MXN {live.get('AOV', 0):,.2f}")
                if 'LTV_12Mo' in live:
                    print(f"Live Shopping LTV (12Mo): MXN {live.get('LTV_12Mo', 0):,.2f}")

            if 'Regular' in self.live_comparison:
                regular = self.live_comparison['Regular']
                print(f"\nRegular Orders: {regular.get('Total Orders', 0):,}")
                print(f"Regular AOV: MXN {regular.get('AOV', 0):,.2f}")
                if 'LTV_12Mo' in regular:
                    print(f"Regular LTV (12Mo): MXN {regular.get('LTV_12Mo', 0):,.2f}")

            if self.session_summary is not None and len(self.session_summary) > 0:
                print(f"\nTotal Live Sessions: {len(self.session_summary)}")
                print(f"Avg Orders per Session: {self.session_summary['Total Orders'].mean():.1f}")
                print(f"Avg Revenue per Session: MXN {self.session_summary['Total Revenue'].mean():,.2f}")

        print("="*80 + "\n")

        return True


def display_menu():
    """Display main menu."""
    print("\n" + "="*80)
    print("TikTok Shop Cohort Analysis Tool")
    print("="*80)
    print("1. Run Full Analysis (recommended)")
    print("2. Load & Clean Data Only")
    print("3. Calculate Metrics")
    print("4. Analyze Live Shopping Sessions")
    print("5. Export CSV Reports")
    print("6. Generate Visualizations")
    print("7. Display Summary Statistics")
    print("8. Exit")
    print("="*80)


def main():
    """Main execution function."""
    tool = CohortAnalysisTool()

    while True:
        display_menu()

        try:
            choice = input("\nSelect an option (1-8): ").strip()

            if choice == '1':
                tool.run_full_analysis()
            elif choice == '2':
                tool.load_data(force_reload=True)
                tool.assign_cohorts()
            elif choice == '3':
                if tool.df is None:
                    tool.load_data()
                    tool.assign_cohorts()
                tool.calculate_metrics()
            elif choice == '4':
                if tool.df is None:
                    tool.load_data()
                    tool.assign_cohorts()
                if tool.load_sessions():
                    tool.analyze_live_sessions()
            elif choice == '5':
                if tool.cohort_summary is None:
                    print("\n⚠ Running full analysis first...")
                    tool.run_full_analysis()
                else:
                    tool.export_csv_reports()
            elif choice == '6':
                if tool.cohort_summary is None:
                    print("\n⚠ Running full analysis first...")
                    tool.run_full_analysis()
                else:
                    tool.generate_visualizations()
            elif choice == '7':
                if tool.cohort_summary is None:
                    print("\n⚠ Running full analysis first...")
                    tool.run_full_analysis()
                else:
                    tool.display_summary()
            elif choice == '8':
                print("\n👋 Exiting. Thank you for using the Cohort Analysis Tool!")
                break
            else:
                print("\n✗ Invalid choice. Please select 1-8.")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Exiting...")
            break
        except Exception as e:
            logger.error(f"✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
