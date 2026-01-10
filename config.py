"""
Configuration settings for TikTok Shop Cohort Analysis Tool
"""
import os

# Project Root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# File Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CSV_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "csv")
VIZ_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# Data Files
CSV_FILENAME = "Todo pedido-2026-01-09-16_57.csv"
MARKETING_COSTS_FILENAME = "Marketing TTS.csv"
LIVE_SESSIONS_FILENAME = "Live Shopping Sessions.csv"
CLEANED_DATA_FILE = os.path.join(PROCESSED_DIR, "orders_cleaned.parquet")

# Analysis Parameters
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
