"""
Data ingestion and loading for PharmaTrack Phase 1.

Provides functions to read, clean, and validate datasets.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

from .schemas import (
    validate_all_data, 
    ValidationReport, 
    ValidationError,
    REQUIRED_COLUMNS
)


class DataLoader:
    """Load and process PharmaTrack data."""
    
    def __init__(self, data_dir: Path = None):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing CSV files
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        self.data_dir = Path(data_dir)
    
    def load_medicines(self, filename: str = "medicines.csv") -> pd.DataFrame:
        """
        Load medicines dataset.
        
        Args:
            filename: CSV filename
            
        Returns:
            DataFrame with medicines data
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Medicines file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        return df
    
    def load_inventory(self, filename: str = "inventory.csv") -> pd.DataFrame:
        """
        Load inventory dataset.
        
        Args:
            filename: CSV filename
            
        Returns:
            DataFrame with inventory data
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Inventory file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        # Parse dates
        if 'expiry_date' in df.columns:
            df['expiry_date'] = pd.to_datetime(df['expiry_date'], errors='coerce')
        return df
    
    def load_sales(self, filename: str = "sales.csv") -> pd.DataFrame:
        """
        Load sales dataset.
        
        Args:
            filename: CSV filename
            
        Returns:
            DataFrame with sales data
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Sales file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        # Parse dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    
    def load_all(self, 
                 medicines_file: str = "medicines.csv",
                 inventory_file: str = "inventory.csv",
                 sales_file: str = "sales.csv") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all three datasets.
        
        Returns:
            Tuple of (medicines_df, inventory_df, sales_df)
        """
        medicines_df = self.load_medicines(medicines_file)
        inventory_df = self.load_inventory(inventory_file)
        sales_df = self.load_sales(sales_file)
        
        return medicines_df, inventory_df, sales_df
    
    def clean_data(self, medicines_df: pd.DataFrame, inventory_df: pd.DataFrame, 
                   sales_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Clean and standardize data.
        
        Returns:
            Tuple of cleaned DataFrames
        """
        # Create copies
        med_clean = medicines_df.copy()
        inv_clean = inventory_df.copy()
        sales_clean = sales_df.copy()
        
        # Standardize column names and data types
        if 'medicine_id' in med_clean.columns:
            med_clean['medicine_id'] = med_clean['medicine_id'].astype(str).str.strip()
        if 'medicine_name' in med_clean.columns:
            med_clean['medicine_name'] = med_clean['medicine_name'].astype(str).str.strip()
        if 'category' in med_clean.columns:
            med_clean['category'] = med_clean['category'].astype(str).str.strip()
        
        if 'batch_id' in inv_clean.columns:
            inv_clean['batch_id'] = inv_clean['batch_id'].astype(str).str.strip()
        if 'medicine_id' in inv_clean.columns:
            inv_clean['medicine_id'] = inv_clean['medicine_id'].astype(str).str.strip()
        if 'supplier' in inv_clean.columns:
            inv_clean['supplier'] = inv_clean['supplier'].astype(str).str.strip()
        
        if 'sale_id' in sales_clean.columns:
            sales_clean['sale_id'] = sales_clean['sale_id'].astype(str).str.strip()
        if 'batch_id' in sales_clean.columns:
            sales_clean['batch_id'] = sales_clean['batch_id'].astype(str).str.strip()
        
        return med_clean, inv_clean, sales_clean
    
    def load_and_validate(self, 
                         medicines_file: str = "medicines.csv",
                         inventory_file: str = "inventory.csv",
                         sales_file: str = "sales.csv",
                         raise_on_error: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, ValidationReport]]:
        """
        Load, clean, and validate all data.
        
        Args:
            raise_on_error: If True, raise ValidationError on validation failure
            
        Returns:
            Tuple of (medicines_df, inventory_df, sales_df, validation_reports)
        """
        # Load data
        medicines_df, inventory_df, sales_df = self.load_all(medicines_file, inventory_file, sales_file)
        
        # Clean data
        medicines_df, inventory_df, sales_df = self.clean_data(medicines_df, inventory_df, sales_df)
        
        # Validate data
        reports = validate_all_data(medicines_df, inventory_df, sales_df)
        
        # Check if any errors
        has_errors = any(not report.valid for report in reports.values())
        if has_errors and raise_on_error:
            all_errors = []
            for name, report in reports.items():
                all_errors.extend([f"{name}: {err}" for err in report.errors])
            raise ValidationError("Data validation failed", all_errors)
        
        return medicines_df, inventory_df, sales_df, reports
    
    def generate_summary(self, medicines_df: pd.DataFrame, inventory_df: pd.DataFrame, 
                        sales_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics using Pandas and NumPy.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        
        # Medicines summary
        summary['medicines'] = {
            'count': len(medicines_df),
            'categories': medicines_df['category'].value_counts().to_dict() if 'category' in medicines_df.columns else {},
            'avg_reorder_level': float(np.mean(medicines_df['reorder_level'])) if 'reorder_level' in medicines_df.columns else 0
        }
        
        # Inventory summary
        if not inventory_df.empty:
            summary['inventory'] = {
                'total_batches': len(inventory_df),
                'total_quantity': int(inventory_df['quantity'].sum()),
                'avg_purchase_price': float(np.mean(inventory_df['purchase_price'])),
                'avg_selling_price': float(np.mean(inventory_df['selling_price'])),
                'total_inventory_value_cost': float(np.sum(inventory_df['quantity'] * inventory_df['purchase_price'])),
                'total_inventory_value_retail': float(np.sum(inventory_df['quantity'] * inventory_df['selling_price']))
            }
        
        # Sales summary
        if not sales_df.empty and not inventory_df.empty:
            # Join sales with inventory for analysis
            sales_merged = sales_df.merge(inventory_df[['batch_id', 'medicine_id', 'purchase_price']], on='batch_id', how='left')
            sales_merged['revenue'] = sales_merged['quantity_sold'] * sales_merged['unit_price']
            sales_merged['cost'] = sales_merged['quantity_sold'] * sales_merged['purchase_price']
            sales_merged['profit'] = sales_merged['revenue'] - sales_merged['cost']
            
            summary['sales'] = {
                'total_transactions': len(sales_df),
                'total_quantity_sold': int(sales_df['quantity_sold'].sum()),
                'total_revenue': float(np.sum(sales_merged['revenue'])),
                'total_cost': float(np.sum(sales_merged['cost'])),
                'total_profit': float(np.sum(sales_merged['profit'])),
                'avg_transaction_value': float(np.mean(sales_merged['revenue'])),
                'unique_batches_sold': int(sales_df['batch_id'].nunique())
            }
        
        return summary


def print_validation_report(reports: Dict[str, ValidationReport]):
    """Print all validation reports."""
    for name, report in reports.items():
        print(f"\n{report}")
        print()
