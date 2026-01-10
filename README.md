# TikTok Shop Cohort Analysis Tool

A comprehensive Python-based tool for analyzing customer cohorts, retention rates, and lifetime value (LTV) from TikTok Shop order data.

## Overview

This tool processes TikTok Shop order exports to generate detailed cohort analysis including:
- Monthly cohort assignments based on first purchase date
- Retention rate calculations and visualizations
- Average Order Value (AOV) analysis (first orders vs repeat orders)
- Lifetime Value (LTV) metrics (3-month, 6-month, 12-month, all-time)
- Comprehensive CSV reports and publication-ready visualizations

## Project Structure

```
cohorts-tts/
├── data/
│   ├── raw/                          # Original CSV exports from TikTok Shop
│   └── processed/                    # Cleaned data (parquet format for fast loading)
│
├── output/
│   ├── csv/                          # CSV reports
│   │   ├── cohort_retention.csv     # Retention matrix (cohort × month)
│   │   ├── customer_metrics.csv     # Customer-level metrics
│   │   └── cohort_summary.csv       # Cohort-level summary statistics
│   └── visualizations/              # Charts and graphs (PNG)
│       ├── retention_heatmap.png
│       ├── retention_curves.png
│       ├── ltv_curves.png
│       ├── aov_distribution.png
│       ├── cohort_sizes.png
│       └── repeat_rate_by_cohort.png
│
├── src/
│   ├── data_loader.py               # CSV loading and cleaning
│   ├── cohort_analyzer.py           # Cohort assignment logic
│   ├── metrics_calculator.py        # Retention, AOV, LTV calculations
│   └── visualizer.py                # Chart generation
│
├── main.py                           # Main execution script (menu-driven)
├── test_analysis.py                  # Quick test script
├── config.py                         # Configuration settings
└── requirements.txt                  # Python dependencies
```

## Installation

### Prerequisites

- Python 3.12+ (or Python 3.8+)
- Virtual environment (recommended)

### Setup

1. **Activate your virtual environment:**
   ```bash
   source /home/quent/sarelly-venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your TikTok Shop CSV export in the `data/raw/` directory**

## Usage

### Quick Start (Recommended)

Run the full analysis with a single command:

```bash
python test_analysis.py
```

This will:
1. Load and clean the data
2. Assign customers to cohorts
3. Calculate all metrics (retention, AOV, LTV)
4. Export CSV reports
5. Generate all visualizations
6. Display summary statistics

**Processing time:** ~3-4 minutes for 65,000+ orders

### Interactive Menu

For more control, use the menu-driven interface:

```bash
python main.py
```

**Menu Options:**

1. **Run Full Analysis** - Complete pipeline (recommended for first-time users)
2. **Load & Clean Data Only** - Load CSV and prepare data (cached for faster subsequent runs)
3. **Calculate Metrics** - Compute retention, AOV, and LTV metrics
4. **Export CSV Reports** - Generate CSV files with analysis results
5. **Generate Visualizations** - Create all charts and graphs
6. **Display Summary Statistics** - View cohort summary in terminal
7. **Exit** - Close the application

## Output Files

### CSV Reports

#### 1. `cohort_retention.csv`
Retention matrix showing what percentage of each cohort made purchases in subsequent months.

| Cohort | Month_0 | Month_1 | Month_2 | Month_3 | ... |
|--------|---------|---------|---------|---------|-----|
| 2025-02 | 100.0% | 4.89% | 5.17% | 4.89% | ... |
| 2025-03 | 100.0% | 4.16% | 3.44% | 3.98% | ... |

#### 2. `customer_metrics.csv`
Customer-level data with order counts, spend, and LTV metrics.

| Buyer Username | Cohort | Order Count | Total Spend | AOV | LTV_3Mo | LTV_6Mo | LTV_12Mo | LTV_AllTime |
|----------------|--------|-------------|-------------|-----|---------|---------|----------|-------------|
| user123 | 2025-02 | 5 | 1,245.00 | 249.00 | 1,245.00 | 1,245.00 | 1,245.00 | 1,245.00 |

#### 3. `cohort_summary.csv`
Cohort-level aggregated statistics.

Includes:
- Cohort size (new customers)
- Total orders and revenue
- Average Order Value (overall, first orders, repeat orders)
- Repeat purchase rate
- LTV metrics (3mo, 6mo, 12mo, all-time)
- Retention rates for months 1, 3, and 6

### Visualizations

All charts are saved as high-resolution PNG files (300 DPI, publication-ready).

1. **retention_heatmap.png** - Color-coded heatmap showing retention % across all cohorts
2. **retention_curves.png** - Line charts comparing retention curves between cohorts
3. **ltv_curves.png** - LTV progression by cohort and time window
4. **aov_distribution.png** - Box plots and violin plots comparing first vs repeat order values
5. **cohort_sizes.png** - Bar chart showing new customer acquisition by month
6. **repeat_rate_by_cohort.png** - Trend line showing repeat purchase rates over time

## Key Metrics Explained

### Retention Rate
Percentage of customers from a cohort who made at least one purchase in a given month after their first order.

**Example:** If 1,778 customers joined in Feb 2025 and 87 made purchases in March 2025, the Month 1 retention is 4.89%.

### Average Order Value (AOV)
Mean value of completed orders. Calculated separately for:
- **Overall AOV**: All orders
- **First Order AOV**: Customer's initial purchase
- **Repeat Order AOV**: All subsequent purchases

**Insight:** Repeat orders typically have 18-20% higher AOV than first orders.

### Lifetime Value (LTV)
Total revenue generated by a customer over a specified time window.

- **3-Month LTV**: Sum of all orders within 3 months of first purchase
- **6-Month LTV**: Sum of all orders within 6 months
- **12-Month LTV**: Sum of all orders within 12 months
- **All-Time LTV**: Total customer spend (same as "Total Spend")

### Repeat Purchase Rate
Percentage of customers in a cohort who made at least one additional purchase after their first order.

**Example:** If a cohort has 1,778 customers and 564 made repeat purchases, the repeat rate is 31.72%.

## Configuration

Edit `config.py` to customize:

- **Data file paths**: Change CSV filename or directory structure
- **Analysis parameters**: Modify cohort frequency, LTV windows, minimum cohort size
- **Visualization settings**: Adjust figure size, DPI, color schemes
- **Customer identification**: Change from "Buyer Username" to another field if needed

## Data Requirements

### Input CSV Format

The tool expects TikTok Shop order exports with the following columns:

**Required:**
- Order ID
- Order Status
- Order Substatus
- Buyer Username
- Created Time (format: `MM/DD/YYYY HH:MM:SS AM/PM`)
- Order Amount (format: `MXN XXX.XX`)

**Optional but recommended:**
- Paid Time, Shipped Time, Delivered Time
- Product Name, SKU ID, Variation
- Recipient, Phone #
- Payment Method

### Data Filtering

The tool automatically filters to **completed/delivered orders only**:
- Order Status = "Completado"
- Order Status = "Enviado" AND Order Substatus = "Entregado"

Unpaid orders, cancelled orders, and orders in-transit are excluded from analysis.

## Performance

**Tested with:**
- 65,509 total orders
- 49,472 completed/delivered orders
- 36,070 unique customers
- 13 monthly cohorts

**Processing time:** 3-4 minutes on standard hardware

**Optimizations:**
- Parquet caching for faster subsequent runs (10x speedup)
- Vectorized pandas operations
- Efficient memory management for large datasets

## Troubleshooting

### "File not found" error
Ensure your CSV file is in `data/raw/` and the filename matches `config.py` setting.

### Memory issues with large files
Reduce the dataset size or increase available RAM. Consider processing cohorts separately.

### Date parsing errors
Check that your CSV uses the expected date format: `MM/DD/YYYY HH:MM:SS AM/PM`

### Missing visualizations
Ensure matplotlib and seaborn are installed: `pip install matplotlib seaborn`

## Extending the Tool

### Adding New Metrics

1. Add calculation function to `src/metrics_calculator.py`
2. Call function in `main.py`'s `calculate_metrics()` method
3. Update CSV export if needed

### Custom Visualizations

1. Add plotting function to `src/visualizer.py`
2. Call function in `generate_all_visualizations()`
3. Update output directory structure if needed

### Different Cohort Definitions

Edit `cohort_analyzer.py` to:
- Change from monthly to weekly cohorts
- Segment by product category, geographic region, etc.
- Use different customer identification fields

## Example Results

Based on the most recent analysis:

- **Total Customers:** 36,070
- **Total Revenue:** MXN 14,753,364.62
- **Overall AOV:** MXN 298.22
- **Average Repeat Rate:** 18.09%
- **First Order AOV:** MXN 283.99
- **Repeat Order AOV:** MXN 336.63 (+18.5% higher)

**Top Performing Cohorts:**
- Nov 2025: 7,366 new customers (largest cohort)
- Dec 2025: MXN 361.87 AOV (highest average order value)
- Feb 2025: 31.72% repeat rate (highest retention)

## License

Internal tool for Sarelly business analytics.

## Support

For questions or issues, check the log file: `cohort_analysis.log`
