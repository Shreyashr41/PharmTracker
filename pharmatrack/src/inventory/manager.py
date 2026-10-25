"""
Inventory Management Module

Provides functionality for:
- Medicine catalogue and batch-level inventory summaries
- Current quantity and purchase-value calculations
- Configurable low-stock alerts
- Excluding expired/unused stock from usable calculations
- Filtering by medicine, category, supplier
- Replenishment suggestions using historical demand
- Exportable reports
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class InventoryManager:
    """Manage inventory with batch-level tracking."""
    
    def __init__(self, medicines_df: pd.DataFrame, inventory_df: pd.DataFrame, sales_df: pd.DataFrame = None):
        """
        Initialize inventory manager.
        
        Args:
            medicines_df: DataFrame with medicine catalogue
            inventory_df: DataFrame with batch-level inventory
            sales_df: DataFrame with sales history (optional)
        """
        self.medicines_df = medicines_df.copy()
        self.inventory_df = inventory_df.copy()
        self.sales_df = sales_df.copy() if sales_df is not None else pd.DataFrame()
        
        # Ensure dates are datetime
        if 'expiry_date' in self.inventory_df.columns:
            self.inventory_df['expiry_date'] = pd.to_datetime(self.inventory_df['expiry_date'])
        if not self.sales_df.empty and 'date' in self.sales_df.columns:
            self.sales_df['date'] = pd.to_datetime(self.sales_df['date'])
        
        # Add computed fields
        self._prepare_inventory()
    
    def _prepare_inventory(self):
        """Prepare inventory data with computed fields."""
        today = pd.Timestamp.now()
        
        # Calculate if expired
        if 'expiry_date' in self.inventory_df.columns:
            self.inventory_df['is_expired'] = self.inventory_df['expiry_date'] < today
            self.inventory_df['days_to_expiry'] = (self.inventory_df['expiry_date'] - today).dt.days
        else:
            self.inventory_df['is_expired'] = False
            self.inventory_df['days_to_expiry'] = 9999
        
        # Calculate batch value
        if 'quantity' in self.inventory_df.columns and 'purchase_price' in self.inventory_df.columns:
            self.inventory_df['batch_value_purchase'] = self.inventory_df['quantity'] * self.inventory_df['purchase_price']
            self.inventory_df['batch_value_retail'] = self.inventory_df['quantity'] * self.inventory_df.get('selling_price', 0)
        
        # Usable quantity (exclude expired)
        self.inventory_df['usable_quantity'] = self.inventory_df.apply(
            lambda row: 0 if row['is_expired'] else row['quantity'], axis=1
        )
        
        # Usable value
        if 'purchase_price' in self.inventory_df.columns:
            self.inventory_df['usable_value'] = self.inventory_df['usable_quantity'] * self.inventory_df['purchase_price']
    
    def get_medicine_catalogue(self) -> pd.DataFrame:
        """Get medicine catalogue with summary information."""
        catalogue = self.medicines_df.copy()
        
        # Add inventory summaries per medicine
        if not self.inventory_df.empty:
            # Join with inventory
            inv_summary = self.inventory_df.groupby('medicine_id').agg(
                total_batches=('batch_id', 'count'),
                total_quantity=('quantity', 'sum'),
                usable_quantity=('usable_quantity', 'sum'),
                expired_quantity=('quantity', lambda x: x[self.inventory_df.loc[x.index, 'is_expired']].sum()),
                total_purchase_value=('batch_value_purchase', 'sum'),
                usable_purchase_value=('usable_value', 'sum'),
                suppliers=('supplier', lambda x: ', '.join(sorted(set(x)))),
                earliest_expiry=('expiry_date', 'min'),
                latest_expiry=('expiry_date', 'max')
            ).reset_index()
            
            catalogue = catalogue.merge(inv_summary, on='medicine_id', how='left')
            catalogue = catalogue.fillna({
                'total_batches': 0,
                'total_quantity': 0,
                'usable_quantity': 0,
                'expired_quantity': 0,
                'total_purchase_value': 0.0,
                'usable_purchase_value': 0.0,
                'suppliers': ''
            })
        else:
            catalogue = catalogue.assign(
                total_batches=0,
                total_quantity=0,
                usable_quantity=0,
                expired_quantity=0,
                total_purchase_value=0.0,
                usable_purchase_value=0.0,
                suppliers='',
                earliest_expiry=pd.NaT,
                latest_expiry=pd.NaT
            )
        
        return catalogue
    
    def get_batch_inventory(self, medicine_id: str = None, category: str = None, 
                           supplier: str = None) -> pd.DataFrame:
        """
        Get batch-level inventory with filtering.
        
        Args:
            medicine_id: Filter by medicine ID
            category: Filter by medicine category
            supplier: Filter by supplier
            
        Returns:
            Filtered batch inventory DataFrame
        """
        df = self.inventory_df.copy()
        
        # Join with medicine info
        if not self.medicines_df.empty:
            df = df.merge(self.medicines_df[['medicine_id', 'medicine_name', 'category']], 
                         on='medicine_id', how='left')
        
        # Apply filters
        if medicine_id:
            df = df[df['medicine_id'] == medicine_id]
        if category and 'category' in df.columns:
            df = df[df['category'] == category]
        if supplier:
            df = df[df['supplier'] == supplier]
        
        return df
    
    def get_current_stock(self, include_expired: bool = False) -> pd.DataFrame:
        """
        Get current stock summary.
        
        Args:
            include_expired: Include expired stock in calculations
            
        Returns:
            DataFrame with current stock per medicine
        """
        df = self.inventory_df.copy()
        
        if not include_expired:
            df = df[~df['is_expired']]
        
        # Group by medicine
        stock = df.groupby('medicine_id').agg(
            usable_quantity=('usable_quantity' if not include_expired else 'quantity', 'sum'),
            total_purchase_value=('usable_value' if not include_expired else 'batch_value_purchase', 'sum'),
            batch_count=('batch_id', 'nunique')
        ).reset_index()
        
        # Add medicine details
        stock = stock.merge(self.medicines_df, on='medicine_id', how='left')
        return stock
    
    def get_low_stock_alerts(self, low_stock_threshold: int = None) -> pd.DataFrame:
        """
        Identify medicines below low stock threshold.
        
        Args:
            low_stock_threshold: Override default threshold (uses reorder_level if None)
            
        Returns:
            DataFrame with low stock medicines
        """
        alerts = []
        catalogue = self.get_medicine_catalogue()
        
        for _, med in catalogue.iterrows():
            reorder_level = med.get('reorder_level', 50)
            threshold = low_stock_threshold if low_stock_threshold is not None else reorder_level
            usable_qty = med.get('usable_quantity', 0)
            
            if usable_qty < threshold:
                shortage = threshold - usable_qty
                alerts.append({
                    'medicine_id': med['medicine_id'],
                    'medicine_name': med.get('medicine_name', ''),
                    'category': med.get('category', ''),
                    'usable_quantity': int(usable_qty),
                    'reorder_level': int(reorder_level),
                    'threshold_used': int(threshold),
                    'shortage': int(shortage),
                    'status': 'CRITICAL' if usable_qty == 0 else 'LOW',
                    'suppliers': med.get('suppliers', '')
                })
        
        return pd.DataFrame(alerts)
    
    def get_expired_stock(self) -> pd.DataFrame:
        """Get expired stock details."""
        expired = self.inventory_df[self.inventory_df['is_expired']].copy()
        if not expired.empty and not self.medicines_df.empty:
            expired = expired.merge(self.medicines_df[['medicine_id', 'medicine_name', 'category']], 
                                   on='medicine_id', how='left')
        return expired
    
    def calculate_inventory_valuation(self) -> Dict[str, float]:
        """
        Calculate total inventory valuation.
        
        Returns:
            Dictionary with valuation metrics
        """
        today = pd.Timestamp.now()
        df = self.inventory_df.copy()
        
        # Current usable stock (not expired)
        usable_df = df[df['expiry_date'] >= today] if 'expiry_date' in df.columns else df
        
        valuation = {
            'total_purchase_value_all': float(df['batch_value_purchase'].sum()) if 'batch_value_purchase' in df.columns else 0.0,
            'total_retail_value_all': float(df['batch_value_retail'].sum()) if 'batch_value_retail' in df.columns else 0.0,
            'usable_purchase_value': float(usable_df['batch_value_purchase'].sum()) if not usable_df.empty and 'batch_value_purchase' in usable_df.columns else 0.0,
            'usable_retail_value': float(usable_df['batch_value_retail'].sum()) if not usable_df.empty and 'batch_value_retail' in usable_df.columns else 0.0,
            'expired_purchase_value': float(df[~(df['expiry_date'] >= today)]['batch_value_purchase'].sum()) if 'expiry_date' in df.columns and len(df[~(df['expiry_date'] >= today)]) > 0 else 0.0,
            'total_quantity_all': int(df['quantity'].sum()) if 'quantity' in df.columns else 0,
            'usable_quantity': int(usable_df['quantity'].sum()) if not usable_df.empty and 'quantity' in usable_df.columns else 0,
            'expired_quantity': int(df[~(df['expiry_date'] >= today)]['quantity'].sum()) if 'expiry_date' in df.columns and len(df[~(df['expiry_date'] >= today)]) > 0 else 0
        }
        
        return valuation
    
    def get_historical_demand(self, medicine_id: str, days: int = 90) -> Dict[str, float]:
        """
        Calculate historical demand for a medicine.
        
        Args:
            medicine_id: Medicine ID
            days: Number of days to look back
            
        Returns:
            Dictionary with demand metrics
        """
        if self.sales_df.empty:
            return {'avg_daily_demand': 0.0, 'total_demand': 0, 'days_with_sales': 0, 'has_data': False}
        
        cutoff_date = pd.Timestamp.now() - timedelta(days=days)
        sales_period = self.sales_df[
            (self.sales_df['medicine_id'] == medicine_id) & 
            (self.sales_df['date'] >= cutoff_date)
        ] if 'medicine_id' in self.sales_df.columns else pd.DataFrame()
        
        if sales_period.empty and 'batch_id' in self.sales_df.columns:
            # Try to join via inventory
            sales_with_med = self.sales_df.merge(
                self.inventory_df[['batch_id', 'medicine_id']], 
                on='batch_id', 
                how='left'
            )
            sales_period = sales_with_med[
                (sales_with_med['medicine_id'] == medicine_id) & 
                (sales_with_med['date'] >= cutoff_date)
            ]
        
        if sales_period.empty:
            return {'avg_daily_demand': 0.0, 'total_demand': 0, 'days_with_sales': 0, 'has_data': False}
        
        total_demand = sales_period['quantity_sold'].sum()
        days_with_sales = sales_period['date'].dt.date.nunique()
        avg_daily_demand = total_demand / days if days > 0 else 0.0
        
        return {
            'avg_daily_demand': float(avg_daily_demand),
            'total_demand': int(total_demand),
            'days_with_sales': int(days_with_sales),
            'has_data': True,
            'period_days': days
        }
    
    def get_replenishment_suggestions(self, lead_time_days: int = 7, safety_factor: float = 1.2) -> pd.DataFrame:
        """
        Generate replenishment suggestions.
        Uses historical demand when available, fallback to reorder rule otherwise.
        
        Args:
            lead_time_days: Expected lead time for delivery
            safety_factor: Safety stock multiplier
            
        Returns:
            DataFrame with replenishment suggestions (estimates, not orders)
        """
        suggestions = []
        catalogue = self.get_medicine_catalogue()
        
        for _, med in catalogue.iterrows():
            medicine_id = med['medicine_id']
            usable_qty = med.get('usable_quantity', 0)
            reorder_level = med.get('reorder_level', 50)
            
            # Get historical demand
            demand_info = self.get_historical_demand(medicine_id)
            
            if demand_info['has_data'] and demand_info['avg_daily_demand'] > 0:
                # Use historical demand
                avg_daily = demand_info['avg_daily_demand']
                days_of_stock = usable_qty / avg_daily if avg_daily > 0 else float('inf')
                lead_time_demand = avg_daily * lead_time_days
                safety_stock = lead_time_demand * (safety_factor - 1)
                suggested_qty = max(0, int(np.ceil(lead_time_demand + safety_stock - usable_qty)))
                
                method = "historical_demand"
                confidence = "medium" if demand_info['days_with_sales'] >= 10 else "low"
            else:
                # Fallback reorder rule
                if usable_qty < reorder_level:
                    suggested_qty = max(0, reorder_level * 2 - usable_qty)  # Order 2x reorder level
                else:
                    suggested_qty = 0
                avg_daily = 0.0
                days_of_stock = float('inf')
                method = "fallback_reorder_rule"
                confidence = "low"
            
            if suggested_qty > 0 or usable_qty < reorder_level:
                priority = 'HIGH' if usable_qty == 0 else ('MEDIUM' if usable_qty < reorder_level else 'LOW')
                suggestions.append({
                    'medicine_id': medicine_id,
                    'medicine_name': med.get('medicine_name', ''),
                    'category': med.get('category', ''),
                    'current_usable_stock': int(usable_qty),
                    'reorder_level': int(reorder_level),
                    'avg_daily_demand': round(avg_daily, 2),
                    'days_of_stock': round(days_of_stock, 1) if days_of_stock != float('inf') else 'N/A',
                    'suggested_order_qty': int(suggested_qty),
                    'method': method,
                    'confidence': confidence,
                    'priority': priority,
                    'note': 'ESTIMATE - NOT AN AUTOMATIC PURCHASE ORDER'
                })
        
        return pd.DataFrame(suggestions)
    
    def export_inventory_report(self, output_path: Path = None) -> str:
        """
        Export inventory summary report.
        
        Args:
            output_path: Path to save report (optional)
            
        Returns:
            Report content as string
        """
        catalogue = self.get_medicine_catalogue()
        valuation = self.calculate_inventory_valuation()
        low_stock = self.get_low_stock_alerts()
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("PHARMATRACK - INVENTORY SUMMARY REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Valuation summary
        report_lines.append("INVENTORY VALUATION:")
        report_lines.append(f"  Total Purchase Value (All): ${valuation['total_purchase_value_all']:.2f}")
        report_lines.append(f"  Usable Purchase Value: ${valuation['usable_purchase_value']:.2f}")
        report_lines.append(f"  Expired Purchase Value: ${valuation['expired_purchase_value']:.2f}")
        report_lines.append(f"  Total Quantity (All): {valuation['total_quantity_all']}")
        report_lines.append(f"  Usable Quantity: {valuation['usable_quantity']}")
        report_lines.append(f"  Expired Quantity: {valuation['expired_quantity']}")
        report_lines.append("")
        
        # Low stock alerts
        report_lines.append("LOW STOCK ALERTS:")
        if low_stock.empty:
            report_lines.append("  No low stock items")
        else:
            for _, alert in low_stock.iterrows():
                report_lines.append(f"  [{alert['status']}] {alert['medicine_name']} ({alert['medicine_id']}): "
                                  f"Usable: {alert['usable_quantity']}, Threshold: {alert['threshold_used']}, "
                                  f"Shortage: {alert['shortage']}")
        report_lines.append("")
        
        # Medicine catalogue
        report_lines.append("MEDICINE CATALOGUE SUMMARY:")
        for _, med in catalogue.iterrows():
            report_lines.append(f"  {med['medicine_name']} ({med['medicine_id']}): "
                              f"Usable: {med.get('usable_quantity', 0)}, "
                              f"Batches: {med.get('total_batches', 0)}, "
                              f"Categories: {med.get('category', '')}")
        report_lines.append("")
        
        report_lines.append("=" * 60)
        report_content = "\n".join(report_lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_content)
            print(f"Report exported to {output_path}")
        
        return report_content
    
    def export_low_stock_report(self, output_path: Path = None) -> str:
        """
        Export low-stock report.
        
        Args:
            output_path: Path to save report (optional)
            
        Returns:
            Report content as string
        """
        low_stock = self.get_low_stock_alerts()
        suggestions = self.get_replenishment_suggestions()
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("PHARMATRACK - LOW STOCK REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        report_lines.append("LOW STOCK ITEMS:")
        if low_stock.empty:
            report_lines.append("  No low stock items found")
        else:
            for _, alert in low_stock.iterrows():
                report_lines.append(f"  Medicine: {alert['medicine_name']} ({alert['medicine_id']})")
                report_lines.append(f"    Category: {alert['category']}")
                report_lines.append(f"    Status: {alert['status']}")
                report_lines.append(f"    Usable Quantity: {alert['usable_quantity']}")
                report_lines.append(f"    Reorder Level: {alert['reorder_level']}")
                report_lines.append(f"    Shortage: {alert['shortage']}")
                report_lines.append(f"    Suppliers: {alert['suppliers']}")
                report_lines.append("")
        
        report_lines.append("REPLENISHMENT SUGGESTIONS (ESTIMATES):")
        if suggestions.empty:
            report_lines.append("  No replenishment suggestions")
        else:
            for _, sug in suggestions.iterrows():
                report_lines.append(f"  Medicine: {sug['medicine_name']} ({sug['medicine_id']})")
                report_lines.append(f"    Priority: {sug['priority']} | Method: {sug['method']} | Confidence: {sug['confidence']}")
                report_lines.append(f"    Current Stock: {sug['current_usable_stock']} | Suggested Order: {sug['suggested_order_qty']}")
                report_lines.append(f"    {sug['note']}")
                report_lines.append("")
        
        report_lines.append("=" * 60)
        report_content = "\n".join(report_lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_content)
            print(f"Low stock report exported to {output_path}")
        
        return report_content
