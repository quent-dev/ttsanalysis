# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project is a comprehensive Python-based cohort analysis tool for TikTok Shop (TTS) order data. It processes CSV exports from TikTok Shop's seller center to generate detailed customer retention metrics, lifetime value calculations, and business intelligence visualizations.

## Development Commands

```bash
# Activate virtual environment
source /home/quent/sarelly-venv/bin/activate

# Run full cohort analysis (non-interactive)
python test_analysis.py

# Run interactive menu-driven interface
python main.py

# Install/update dependencies
pip install -r requirements.txt
```

## Architecture

### Project Structure
```
cohorts-tts/
├── src/                    # Core analysis modules
│   ├── data_loader.py      # CSV loading, cleaning, validation
│   ├── cohort_analyzer.py  # Cohort assignment logic
│   ├── metrics_calculator.py # Retention, AOV, LTV calculations
│   └── visualizer.py       # Chart generation (matplotlib/seaborn)
├── data/
│   ├── raw/                # TikTok Shop CSV exports
│   └── processed/          # Cached parquet files (10x faster loading)
├── output/
│   ├── csv/                # Analysis reports
│   └── visualizations/     # PNG charts (300 DPI)
├── main.py                 # Menu-driven execution script
├── test_analysis.py        # Quick test runner
└── config.py               # Centralized configuration
```

### Tech Stack
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **matplotlib/seaborn**: Static visualizations
- **pyarrow**: Parquet file support for caching

### Key Metrics Generated
1. **Retention Rates**: Cohort-based retention matrix (monthly)
2. **Average Order Value (AOV)**: First orders vs repeat orders
3. **Lifetime Value (LTV)**: 3-month, 6-month, 12-month, all-time
4. **Repeat Purchase Rate**: By cohort and overall

## Data Format

### CSV Structure
The project works with TikTok Shop order exports with the following key fields:
- **Order Information**: Order ID, Order Status, Order Substatus, Fulfillment Type
- **Product Details**: SKU ID, Seller SKU, Product Name, Variation, Quantity
- **Pricing**: SKU Unit Price, Discounts (Platform/Seller), Shipping Fees, Order Amount
- **Timestamps**: Created Time, Paid Time, Shipped Time, Delivered Time, Cancelled Time
- **Customer Data**: Buyer Username, Recipient, Phone, Location (Country, State, City, Zipcode)
- **Logistics**: Tracking ID, Shipping Provider, Warehouse Name, Delivery Option

### Data Characteristics
- **Encoding**: UTF-8 with BOM (handled automatically by data_loader.py)
- **Language**: Spanish (México market)
- **Date Format**: MM/DD/YYYY HH:MM:SS AM/PM (automatically parsed)
- **Currency**: MXN (Mexican Peso) - stripped and converted to float
- **Privacy**: Customer PII is masked with asterisks in exports
- **Typical Size**: 50,000-70,000 orders, ~50MB CSV file
- **Processing Time**: 3-4 minutes for full analysis

## Code Patterns

### Data Loading and Cleaning (data_loader.py)
```python
# Use the centralized loader with automatic caching
from src.data_loader import load_and_clean_data
df = load_and_clean_data(csv_path)  # Returns cleaned DataFrame
```

Key features:
- Automatic UTF-8 BOM handling
- Date parsing with error handling
- Currency string to float conversion
- Filters to completed/delivered orders only
- Parquet caching for 10x faster subsequent loads
- Comprehensive data quality validation

### Cohort Assignment (cohort_analyzer.py)
```python
from src.cohort_analyzer import build_cohort_index
df = build_cohort_index(df, customer_id_field='Buyer Username')
```

Adds columns:
- `First Purchase Date`: Customer's initial order date
- `Cohort`: YYYY-MM format (e.g., "2025-02")
- `Cohort Age`: Months since first purchase (0, 1, 2, ...)
- `Is First Order`: Boolean flag

### Metrics Calculation (metrics_calculator.py)
```python
from src.metrics_calculator import (
    calculate_retention_rates,
    calculate_customer_ltv,
    generate_cohort_summary
)

retention_rates = calculate_retention_rates(df)  # Cohort × Month matrix
customer_ltv = calculate_customer_ltv(df, ltv_windows=[3, 6, 12])
cohort_summary = generate_cohort_summary(df)  # All metrics in one DataFrame
```

### Visualization (visualizer.py)
```python
from src.visualizer import generate_all_visualizations
generate_all_visualizations(df, retention_rates, cohort_ltv,
                            cohort_summary, output_dir)
```

Generates 6 publication-ready charts at 300 DPI.

## Data Quality Considerations

The tool automatically handles:
- Trailing tabs and whitespace in date fields
- Duplicate Order IDs (keeps first occurrence)
- Missing critical fields (drops rows)
- Zero-value orders (included in retention, flagged in AOV/LTV)
- Cancelled/unpaid orders (excluded from analysis)

**Order Status Filtering:**
- Includes: "Completado" and "Enviado" + "Entregado"
- Excludes: "No pagado", "Cancelado", and in-transit orders

## Extending the Tool

### Adding New Metrics
1. Add calculation function to `src/metrics_calculator.py`
2. Follow existing patterns (use groupby operations, not loops)
3. Add to `generate_cohort_summary()` if cohort-level metric
4. Update CSV exports in `main.py`

### Custom Visualizations
1. Add plotting function to `src/visualizer.py`
2. Use seaborn for statistical plots, matplotlib for custom charts
3. Call from `generate_all_visualizations()`
4. Always save with `dpi=300` for publication quality

### Performance Optimization
- Use vectorized pandas operations (avoid Python loops)
- Leverage parquet caching for repeated analysis
- Use categorical dtypes for repetitive string columns
- Process cohorts separately if memory constrained
