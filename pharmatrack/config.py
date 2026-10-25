"""
PharmaTrack Configuration Module

Centralized configuration for paths, expiry warning thresholds,
and application settings.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent
BASE_DIR = PROJECT_ROOT

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"

# Source directories
SRC_DIR = BASE_DIR / "src"

# CSV file names
MEDICINES_CSV = "medicines.csv"
BATCHES_CSV = "batches.csv"
SALES_CSV = "sales.csv"

# Expiry warning thresholds (in days)
EXPIRY_CRITICAL_THRESHOLD = 7    # Critical - expires within 7 days
EXPIRY_WARNING_THRESHOLD = 30   # Warning - expires within 30 days
EXPIRY_CAUTION_THRESHOLD = 90   # Caution - expires within 90 days

# Inventory thresholds
DEFAULT_REORDER_LEVEL = 50
LOW_STOCK_THRESHOLD_RATIO = 0.5  # Below 50% of reorder level

# Dashboard settings
PAGE_TITLE = "PharmaTrack - Smart Pharmacy Inventory"
PAGE_ICON = "💊"
LAYOUT = "wide"

# Date format
DATE_FORMAT = "%Y-%m-%d"


def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        SRC_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_raw_data_path(filename: str) -> Path:
    """Get full path for a raw data file."""
    return RAW_DATA_DIR / filename


def get_processed_data_path(filename: str) -> Path:
    """Get full path for a processed data file."""
    return PROCESSED_DATA_DIR / filename


def get_report_path(filename: str) -> Path:
    """Get full path for a report file."""
    return REPORTS_DIR / filename


# Ensure directories exist on import
ensure_directories()
