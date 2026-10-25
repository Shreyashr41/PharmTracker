# PharmaTrack - Smart Pharmacy Inventory, Sales & Expiry Analytics System

A comprehensive, beginner-friendly pharmacy management analytics system built with Python, Pandas, NumPy, Matplotlib, Seaborn, and Streamlit. This is a **demo project** for educational purposes (college viva) - not production-ready or connected to a live pharmacy system.

## Features

### Phase 0: Project Setup
- Clean modular project structure
- Configuration management
- Streamlit application entry point
- Smoke tests

### Phase 1: Data Ingestion & Validation
- Synthetic data generation (reproducible with seed)
- CSV schemas for medicines, inventory, and sales
- Comprehensive validation (unique IDs, foreign keys, ranges, dates)
- DataLoader with cleaning and validation reports

### Phase 2: Inventory Management
- Batch-level inventory tracking with expiry dates and purchase costs
- Medicine catalogue with stock summaries
- Current quantity and purchase-value calculations
- Configurable low-stock alerts
- Usable stock calculations (excludes expired batches)
- Filtering by medicine, category, supplier
- Replenishment suggestions using historical demand or fallback rules
- Exportable inventory reports

### Phase 3: Sales & Revenue Analytics
- Daily, weekly, and monthly sales summaries
- Revenue by medicine, category, and time period
- Top-selling medicines ranked by quantity and revenue
- Slow-moving medicine analysis
- Gross profit estimates using actual batch purchase costs
- Sales statistics (average price, quantities)
- Trend analysis and demand estimates with disclosed limitations
- CSV export of analytical summaries

### Phase 4: Expiry & Wastage Risk Analysis
- Days remaining calculation until batch expiry
- Batch classification (expired, near-expiry, OK)
- Configurable near-expiry threshold (default 30 days)
- High-value batches approaching expiry identification
- Value-at-risk calculation using purchase costs (estimates, not losses)
- Batch-level expiry reports
- Historical wastage analysis (only if records exist)
- FEFO (First-Expiry-First-Out) stock rotation recommendations

### Phase 5: Visualizations & Interactive Dashboard
- **Overview**: Key metrics, alerts
- **Inventory**: Searchable tables, stock distribution, low-stock reports
- **Sales Analytics**: Trends, top sellers, category comparisons, date filters
- **Expiry Analysis**: Expired/near-expiry batches, value at risk, FEFO priority
- **Reports**: Downloadable CSV summaries
- Matplotlib for time-series and bar charts
- Seaborn for heatmaps, box plots, distributions
- CSV upload support for custom data

## Architecture

```
pharmatrack/
├── app.py                  # Main Streamlit entry point
├── config.py              # Configuration and paths
├── requirements.txt       # Dependencies
├── src/                   # Core business logic
│   ├── data/             # Data generation, ingestion, validation
│   ├── inventory/        # Inventory management
│   ├── analytics/        # Sales analytics
│   ├── expiry/           # Expiry and wastage analysis
│   └── visualization/    # Chart functions (Matplotlib/Seaborn)
├── pages/                # Streamlit multi-page dashboard
│   ├── 01_Inventory.py
│   ├── 02_Sales.py
│   ├── 03_Expiry.py
│   └── 04_Reports.py
├── tests/                # Test suite (pytest)
├── data/raw/             # Generated/uploaded CSV data
├── reports/              # Exported reports
└── logs/                 # Development logs
```

## Setup Instructions

### Prerequisites
- Python 3.9+ (tested with Python 3.14.3)
- pip

### 1. Create Virtual Environment

**Windows (Command Prompt):**
```cmd
cd "D:\IDEs\Visual Studio Code\PharmaTrack (Krishuu)\pharmatrack"
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
cd "D:\IDEs\Visual Studio Code\PharmaTrack (Krishuu)\pharmatrack"
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
cd pharmatrack
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Sample Data
```bash
python -c "from src.data import generate_sample_data; generate_sample_data(seed=42)"
```

### 4. Run Tests
```bash
python -m pytest -v
```

### 5. Start Streamlit Dashboard
```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

## Sample Data

The system generates synthetic data with:
- **Medicines**: 15 medicines across different categories (Analgesic, Antibiotic, etc.)
- **Inventory**: 41 batches with varying expiry dates, purchase/selling prices, suppliers
- **Sales**: 200 transactions over a 90-day period

All data is synthetic with no personal or patient information. Use fixed seed (42) for reproducibility.

## Usage

1. Launch the dashboard with `streamlit run app.py`
2. Use the sidebar to switch between synthetic data or upload custom CSV files
3. Navigate through pages: Overview, Inventory, Sales, Expiry, Reports
4. Apply filters and download reports as CSV

## Limitations

- **Demo only**: Not production-ready, no authentication, no database
- **Synthetic data**: Uses generated sample data by default
- **Estimates**: Gross profit uses batch costs, demand estimates are based on historical data only
- **No refunds**: Returns/refunds only handled if explicitly in schema
- **Value at risk**: Estimates based on purchase costs, not actual financial losses
- **Wastage**: Only analyzed if wastage records exist in data
- **No ML**: Uses simple statistical methods, no machine learning models

## Troubleshooting

**Import errors:**
```bash
# Ensure you're in the pharmatrack directory and venv is activated
cd pharmatrack
pip install -r requirements.txt
```

**Streamlit not found:**
```bash
pip install streamlit
```

**Permission errors on Windows PowerShell:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Data not loading:**
```bash
# Regenerate sample data
python -c "from src.data import generate_sample_data; generate_sample_data(seed=42)"
```

## Testing

All tests pass (66/66). Run with:
```bash
python -m pytest -v
```

## License

This project is for educational purposes.
"""
