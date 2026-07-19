# Updating the Cohort Analysis with New Order Data

Steps to refresh the analysis when new TikTok Shop order data becomes available.

## 1. Export fresh order data

From TikTok Shop Seller Center, export the orders report for each account you want to update
(MX and/or US). Export the full order history available — the tool filters and caps dates
itself (see step 3), so there's no need to hand-pick a date range.

## 2. Drop the CSV into the account's raw folder

- Mexico: `data/MX/raw/`
- US: `data/US/raw/`

Filename must contain `Todo pedido` (MX) or `All order` (US) — anything else in the name is
fine. **Remove or archive the old orders CSV** from that folder before adding the new one.
`config.find_orders_csv()` picks the most-recently-modified matching file if more than one is
present (and logs a warning), so leaving stale files around risks the wrong one being picked
silently.

Optional files, same folders:
- Marketing costs: filename contains `Marketing`
- Live shopping sessions: filename contains `Live Shopping` (not `... Example`)

## 3. Update the analysis cutoff (if needed)

`config.py`:
```python
ANALYSIS_END_DATE = '2026-06-30'  # orders after this date are excluded, inclusive
```

Set this to the last date you want included, or `None` for no cutoff. Check the latest order
date actually present in your new CSV before setting this — the export may contain more recent
orders than you intend to analyze yet.

To check the latest order date in a raw file:
```bash
source /home/quent/sarelly-venv/bin/activate
python3 -c "
import pandas as pd
df = pd.read_csv('data/US/raw/<your_file>.csv', usecols=['Created Time'], encoding='utf-8-sig')
print(pd.to_datetime(df['Created Time'], format='%m/%d/%Y %I:%M:%S %p').max())
"
```

## 4. Clear the cached parquet files

The tool caches cleaned data per account and won't re-read the CSV unless the cache is cleared
or `force_reload=True` is used:

```bash
rm -f data/MX/processed/orders_cleaned.parquet
rm -f data/US/processed/orders_cleaned.parquet
```

## 5. Run the analysis

```bash
source /home/quent/sarelly-venv/bin/activate
cd /home/quent/projects/cohorts-tts
python test_analysis.py MX
python test_analysis.py US
```

Or use `python main.py` for the interactive menu (option 1 = full analysis).

## 6. Check the outputs

- CSV reports: `output/MX/csv/`, `output/US/csv/`
- Visualizations: `output/MX/visualizations/`, `output/US/visualizations/`
- Logs: `cohort_analysis_mx.log`, `cohort_analysis_us.log`

## Troubleshooting

**`ArrowTypeError: Expected bytes, got a 'float' object` when saving to parquet** — caused by
pandas' chunked CSV reading inferring inconsistent per-chunk dtypes for a column (you'll see a
`DtypeWarning: Columns (...) have mixed types` just before it). Already fixed in
`src/data_loader.py` (`load_tiktok_orders` reads with `low_memory=False`), but if it recurs on a
new export with different problem columns, that's the fix to check first.
