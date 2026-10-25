"""
Data schemas and validation for PharmaTrack Phase 1 datasets.

Defines expected columns and validation rules for:
- medicines.csv
- inventory.csv
- sales.csv
"""

from typing import Dict, List, Any
import pandas as pd


# Schema definitions for Phase 1 datasets
MEDICINES_SCHEMA = {
    "medicine_id": "string",
    "medicine_name": "string", 
    "category": "string",
    "reorder_level": "int64"
}

INVENTORY_SCHEMA = {
    "batch_id": "string",
    "medicine_id": "string",
    "quantity": "int64",
    "purchase_price": "float64",
    "selling_price": "float64",
    "expiry_date": "datetime64[ns]",
    "supplier": "string"
}

SALES_SCHEMA = {
    "sale_id": "string",
    "date": "datetime64[ns]",
    "batch_id": "string",
    "quantity_sold": "int64",
    "unit_price": "float64"
}

REQUIRED_COLUMNS = {
    "medicines": ["medicine_id", "medicine_name", "category", "reorder_level"],
    "inventory": ["batch_id", "medicine_id", "quantity", "purchase_price", "selling_price", "expiry_date", "supplier"],
    "sales": ["sale_id", "date", "batch_id", "quantity_sold", "unit_price"]
}


class ValidationError(Exception):
    """Raised when data validation fails."""
    def __init__(self, message: str, errors: List[str] = None):
        super().__init__(message)
        self.errors = errors or []


class ValidationReport:
    """Validation report with errors, warnings, and statistics."""
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.valid = True
        self.stats: Dict[str, Any] = {}
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.valid = False
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)
    
    def __str__(self):
        lines = []
        lines.append("=" * 60)
        lines.append("PHARMATRACK VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Status: {'VALID' if self.valid else 'INVALID'}")
        lines.append("")
        
        if self.errors:
            lines.append("ERRORS:")
            for i, err in enumerate(self.errors, 1):
                lines.append(f"  {i}. {err}")
            lines.append("")
        
        if self.warnings:
            lines.append("WARNINGS:")
            for i, warn in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warn}")
            lines.append("")
        
        if self.stats:
            lines.append("STATISTICS:")
            for key, value in self.stats.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def validate_medicines(df: pd.DataFrame) -> ValidationReport:
    """Validate medicines DataFrame."""
    report = ValidationReport()
    
    # Check required columns
    missing_cols = set(REQUIRED_COLUMNS["medicines"]) - set(df.columns)
    if missing_cols:
        report.add_error(f"Missing required columns: {missing_cols}")
    
    if not report.errors:
        # Check for duplicates in medicine_id
        if df['medicine_id'].duplicated().any():
            dupes = df[df['medicine_id'].duplicated(keep=False)]['medicine_id'].unique()
            report.add_error(f"Duplicate medicine_id found: {list(dupes)}")
        
        # Check for missing values
        for col in REQUIRED_COLUMNS["medicines"]:
            if df[col].isna().any():
                report.add_error(f"Missing values in column: {col}")
        
        # Check numeric range for reorder_level
        if 'reorder_level' in df.columns:
            if (df['reorder_level'] < 0).any():
                report.add_error("reorder_level cannot be negative")
            if (df['reorder_level'] == 0).any():
                report.add_warning("Some reorder_level values are 0")
        
        # Check string fields not empty
        for col in ['medicine_id', 'medicine_name', 'category']:
            if col in df.columns:
                if (df[col].astype(str).str.strip() == '').any():
                    report.add_error(f"Empty strings found in column: {col}")
    
    report.stats = {
        "Total records": len(df),
        "Unique medicines": df['medicine_id'].nunique() if 'medicine_id' in df.columns else 0
    }
    
    return report


def validate_inventory(df: pd.DataFrame) -> ValidationReport:
    """Validate inventory DataFrame."""
    report = ValidationReport()
    
    # Check required columns
    missing_cols = set(REQUIRED_COLUMNS["inventory"]) - set(df.columns)
    if missing_cols:
        report.add_error(f"Missing required columns: {missing_cols}")
    
    if not report.errors:
        # Check for duplicates in batch_id
        if df['batch_id'].duplicated().any():
            dupes = df[df['batch_id'].duplicated(keep=False)]['batch_id'].unique()
            report.add_error(f"Duplicate batch_id found: {list(dupes)}")
        
        # Check for missing values
        for col in REQUIRED_COLUMNS["inventory"]:
            if df[col].isna().any():
                report.add_error(f"Missing values in column: {col}")
        
        # Check numeric ranges
        if 'quantity' in df.columns:
            if (df['quantity'] < 0).any():
                report.add_error("quantity cannot be negative")
            if (df['quantity'] == 0).any():
                report.add_warning("Some quantity values are 0")
        
        if 'purchase_price' in df.columns:
            if (df['purchase_price'] <= 0).any():
                report.add_error("purchase_price must be greater than 0")
        
        if 'selling_price' in df.columns:
            if (df['selling_price'] <= 0).any():
                report.add_error("selling_price must be greater than 0")
        
        # Check if selling price >= purchase price
        if 'purchase_price' in df.columns and 'selling_price' in df.columns:
            if (df['selling_price'] < df['purchase_price']).any():
                report.add_warning("Some selling_price is less than purchase_price")
        
        # Check date format
        if 'expiry_date' in df.columns:
            try:
                pd.to_datetime(df['expiry_date'])
            except Exception as e:
                report.add_error(f"Invalid date format in expiry_date: {e}")
        
        # Check string fields not empty
        for col in ['batch_id', 'medicine_id', 'supplier']:
            if col in df.columns:
                if (df[col].astype(str).str.strip() == '').any():
                    report.add_error(f"Empty strings found in column: {col}")
    
    report.stats = {
        "Total records": len(df),
        "Unique batches": df['batch_id'].nunique() if 'batch_id' in df.columns else 0,
        "Total stock": df['quantity'].sum() if 'quantity' in df.columns else 0
    }
    
    return report


def validate_sales(df: pd.DataFrame, inventory_df: pd.DataFrame = None) -> ValidationReport:
    """Validate sales DataFrame."""
    report = ValidationReport()
    
    # Check required columns
    missing_cols = set(REQUIRED_COLUMNS["sales"]) - set(df.columns)
    if missing_cols:
        report.add_error(f"Missing required columns: {missing_cols}")
    
    if not report.errors:
        # Check for duplicates in sale_id
        if df['sale_id'].duplicated().any():
            dupes = df[df['sale_id'].duplicated(keep=False)]['sale_id'].unique()
            report.add_error(f"Duplicate sale_id found: {list(dupes)}")
        
        # Check for missing values
        for col in REQUIRED_COLUMNS["sales"]:
            if df[col].isna().any():
                report.add_error(f"Missing values in column: {col}")
        
        # Check numeric ranges
        if 'quantity_sold' in df.columns:
            if (df['quantity_sold'] <= 0).any():
                report.add_error("quantity_sold must be greater than 0")
        
        if 'unit_price' in df.columns:
            if (df['unit_price'] <= 0).any():
                report.add_error("unit_price must be greater than 0")
        
        # Check date format
        if 'date' in df.columns:
            try:
                pd.to_datetime(df['date'])
            except Exception as e:
                report.add_error(f"Invalid date format in date: {e}")
        
        # Check string fields not empty
        for col in ['sale_id', 'batch_id']:
            if col in df.columns:
                if (df[col].astype(str).str.strip() == '').any():
                    report.add_error(f"Empty strings found in column: {col}")
        
        # Check foreign key relationship to inventory
        if inventory_df is not None and 'batch_id' in df.columns and 'batch_id' in inventory_df.columns:
            invalid_batches = set(df['batch_id']) - set(inventory_df['batch_id'])
            if invalid_batches:
                report.add_error(f"Sales reference invalid batch_ids: {list(invalid_batches)}")
        
        # Check sales consistency with available stock (basic check)
        if inventory_df is not None and 'batch_id' in df.columns and 'quantity_sold' in df.columns:
            # Group sales by batch
            sales_by_batch = df.groupby('batch_id')['quantity_sold'].sum()
            for batch_id, total_sold in sales_by_batch.items():
                batch_stock = inventory_df[inventory_df['batch_id'] == batch_id]['quantity']
                if not batch_stock.empty and total_sold > batch_stock.iloc[0]:
                    report.add_warning(f"Batch {batch_id}: total sales ({total_sold}) exceed available stock ({batch_stock.iloc[0]})")
    
    report.stats = {
        "Total records": len(df),
        "Total quantity sold": df['quantity_sold'].sum() if 'quantity_sold' in df.columns else 0,
        "Total revenue": (df['quantity_sold'] * df['unit_price']).sum() if 'quantity_sold' in df.columns and 'unit_price' in df.columns else 0
    }
    
    return report


def validate_all_data(medicines_df: pd.DataFrame, inventory_df: pd.DataFrame, sales_df: pd.DataFrame) -> Dict[str, ValidationReport]:
    """Validate all datasets and check cross-references."""
    reports = {}
    
    reports['medicines'] = validate_medicines(medicines_df)
    reports['inventory'] = validate_inventory(inventory_df)
    reports['sales'] = validate_sales(sales_df, inventory_df)
    
    # Cross-reference checks
    cross_report = ValidationReport()
    
    # Check inventory references valid medicines
    if not reports['medicines'].errors and not reports['inventory'].errors:
        invalid_meds = set(inventory_df['medicine_id']) - set(medicines_df['medicine_id'])
        if invalid_meds:
            cross_report.add_error(f"Inventory references invalid medicine_ids: {list(invalid_meds)}")
        
        cross_report.stats = {
            "Medicines referenced in inventory": len(set(inventory_df['medicine_id'])),
            "Total medicines in catalog": len(medicines_df)
        }
    
    reports['cross_reference'] = cross_report
    
    return reports
