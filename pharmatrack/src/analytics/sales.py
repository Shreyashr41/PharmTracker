"""
Sales and Revenue Analytics Module

Provides functionality for:
- Daily, weekly, and monthly sales summaries
- Revenue by medicine, category, and time period
- Top-selling medicines ranked by quantity and revenue
- Slow-moving medicine analysis
- Gross profit estimates using actual batch purchase costs
- Average selling price and sales quantity statistics
- Trend analysis and demand estimates
- CSV export of analytical summaries
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class SalesAnalytics:
    """Sales and revenue analytics for PharmaTrack."""
    
    def __init__(self, medicines_df: pd.DataFrame, inventory_df: pd.DataFrame, sales_df: pd.DataFrame):
        """
        Initialize sales analytics.
        
        Args:
            medicines_df: DataFrame with medicine catalogue
            inventory_df: DataFrame with batch inventory (contains purchase costs)
            sales_df: DataFrame with sales transactions
        """
        self.medicines_df = medicines_df.copy()
        self.inventory_df = inventory_df.copy()
        self.sales_df = sales_df.copy()
        
        # Ensure dates are datetime
        if not self.sales_df.empty and 'date' in self.sales_df.columns:
            self.sales_df['date'] = pd.to_datetime(self.sales_df['date'])
        
        # Prepare enriched sales data
        self._prepare_sales_data()
    
    def _prepare_sales_data(self):
        """Prepare sales data with batch and medicine information."""
        if self.sales_df.empty:
            self.enriched_sales = pd.DataFrame()
            return
        
        # Join sales with inventory to get purchase costs (actual batch costs)
        enriched = self.sales_df.merge(
            self.inventory_df[['batch_id', 'medicine_id', 'purchase_price', 'selling_price', 'supplier']],
            on='batch_id',
            how='left'
        )
        
        # Join with medicine details
        enriched = enriched.merge(
            self.medicines_df[['medicine_id', 'medicine_name', 'category', 'reorder_level']],
            on='medicine_id',
            how='left'
        )
        
        # Check for missing batch info (data quality)
        missing_batch = enriched['purchase_price'].isna().sum()
        if missing_batch > 0:
            print(f"Warning: {missing_batch} sales records have missing batch/purchase cost information")
        
        # Calculate revenue and profit using actual batch costs
        enriched['revenue'] = enriched['quantity_sold'] * enriched['unit_price']
        enriched['cost_of_goods_sold'] = enriched['quantity_sold'] * enriched['purchase_price']
        enriched['gross_profit'] = enriched['revenue'] - enriched['cost_of_goods_sold']
        enriched['gross_margin_pct'] = np.where(
            enriched['revenue'] > 0,
            (enriched['gross_profit'] / enriched['revenue']) * 100,
            0.0
        )
        
        # Add time-based columns
        enriched['year'] = enriched['date'].dt.year
        enriched['month'] = enriched['date'].dt.month
        enriched['week'] = enriched['date'].dt.isocalendar().week
        enriched['day_of_week'] = enriched['date'].dt.day_name()
        enriched['month_name'] = enriched['date'].dt.strftime('%Y-%m')
        enriched['week_name'] = enriched['date'].dt.strftime('%Y-W%U')
        enriched['date_only'] = enriched['date'].dt.date
        
        self.enriched_sales = enriched
    
    def get_sales_summary_daily(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get daily sales summary."""
        if self.enriched_sales.empty:
            return pd.DataFrame()
        
        df = self.enriched_sales.copy()
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        daily = df.groupby('date_only').agg(
            transactions=('sale_id', 'count'),
            quantity_sold=('quantity_sold', 'sum'),
            revenue=('revenue', 'sum'),
            cost_of_goods_sold=('cost_of_goods_sold', 'sum'),
            gross_profit=('gross_profit', 'sum'),
            unique_medicines=('medicine_id', 'nunique'),
            unique_batches=('batch_id', 'nunique'),
            avg_unit_price=('unit_price', 'mean')
        ).reset_index()
        
        daily['gross_margin_pct'] = np.where(
            daily['revenue'] > 0,
            (daily['gross_profit'] / daily['revenue']) * 100,
            0.0
        )
        
        return daily.sort_values('date_only')
    
    def get_sales_summary_weekly(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get weekly sales summary."""
        if self.enriched_sales.empty:
            return pd.DataFrame()
        
        df = self.enriched_sales.copy()
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        weekly = df.groupby(['year', 'week_name']).agg(
            transactions=('sale_id', 'count'),
            quantity_sold=('quantity_sold', 'sum'),
            revenue=('revenue', 'sum'),
            cost_of_goods_sold=('cost_of_goods_sold', 'sum'),
            gross_profit=('gross_profit', 'sum'),
            unique_medicines=('medicine_id', 'nunique'),
            start_date=('date', 'min'),
            end_date=('date', 'max')
        ).reset_index()
        
        weekly['gross_margin_pct'] = np.where(
            weekly['revenue'] > 0,
            (weekly['gross_profit'] / weekly['revenue']) * 100,
            0.0
        )
        
        return weekly.sort_values('start_date')
    
    def get_sales_summary_monthly(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get monthly sales summary."""
        if self.enriched_sales.empty:
            return pd.DataFrame()
        
        df = self.enriched_sales.copy()
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        monthly = df.groupby(['year', 'month_name']).agg(
            transactions=('sale_id', 'count'),
            quantity_sold=('quantity_sold', 'sum'),
            revenue=('revenue', 'sum'),
            cost_of_goods_sold=('cost_of_goods_sold', 'sum'),
            gross_profit=('gross_profit', 'sum'),
            unique_medicines=('medicine_id', 'nunique')
        ).reset_index()
        
        monthly['gross_margin_pct'] = np.where(
            monthly['revenue'] > 0,
            (monthly['gross_profit'] / monthly['revenue']) * 100,
            0.0
        )
        
        return monthly.sort_values('month_name')
    
    def get_revenue_by_medicine(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get revenue breakdown by medicine."""
        if self.enriched_sales.empty:
            return pd.DataFrame()
        
        df = self.enriched_sales.copy()
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        by_medicine = df.groupby(['medicine_id', 'medicine_name', 'category']).agg(
            transactions=('sale_id', 'count'),
            quantity_sold=('quantity_sold', 'sum'),
            revenue=('revenue', 'sum'),
            cost_of_goods_sold=('cost_of_goods_sold', 'sum'),
            gross_profit=('gross_profit', 'sum'),
            avg_unit_price=('unit_price', 'mean'),
            avg_purchase_cost=('purchase_price', 'mean'),
            batches_used=('batch_id', 'nunique')
        ).reset_index()
        
        by_medicine['gross_margin_pct'] = np.where(
            by_medicine['revenue'] > 0,
            (by_medicine['gross_profit'] / by_medicine['revenue']) * 100,
            0.0
        )
        
        return by_medicine.sort_values('revenue', ascending=False)
    
    def get_revenue_by_category(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get revenue breakdown by category."""
        if self.enriched_sales.empty:
            return pd.DataFrame()
        
        df = self.enriched_sales.copy()
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        by_category = df.groupby('category').agg(
            medicines_count=('medicine_id', 'nunique'),
            transactions=('sale_id', 'count'),
            quantity_sold=('quantity_sold', 'sum'),
            revenue=('revenue', 'sum'),
            cost_of_goods_sold=('cost_of_goods_sold', 'sum'),
            gross_profit=('gross_profit', 'sum'),
            avg_unit_price=('unit_price', 'mean')
        ).reset_index()
        
        by_category['gross_margin_pct'] = np.where(
            by_category['revenue'] > 0,
            (by_category['gross_profit'] / by_category['revenue']) * 100,
            0.0
        )
        
        return by_category.sort_values('revenue', ascending=False)
    
    def get_revenue_by_time_period(self, period: str = 'month', start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get revenue by time period."""
        if period == 'day':
            return self.get_sales_summary_daily(start_date, end_date)
        elif period == 'week':
            return self.get_sales_summary_weekly(start_date, end_date)
        elif period == 'month':
            return self.get_sales_summary_monthly(start_date, end_date)
        else:
            raise ValueError("Period must be 'day', 'week', or 'month'")
    
    def get_top_selling_by_quantity(self, n: int = 10, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get top selling medicines by quantity."""
        by_medicine = self.get_revenue_by_medicine(start_date, end_date)
        if by_medicine.empty:
            return pd.DataFrame()
        by_quantity = by_medicine.sort_values('quantity_sold', ascending=False).head(n)
        return by_quantity[['medicine_id', 'medicine_name', 'category', 'quantity_sold', 'revenue', 'gross_profit']]
    
    def get_top_selling_by_revenue(self, n: int = 10, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get top selling medicines by revenue."""
        by_medicine = self.get_revenue_by_medicine(start_date, end_date)
        if by_medicine.empty:
            return pd.DataFrame()
        by_revenue = by_medicine.sort_values('revenue', ascending=False).head(n)
        return by_revenue[['medicine_id', 'medicine_name', 'category', 'revenue', 'quantity_sold', 'gross_profit']]
    
    def get_slow_moving_medicines(self, days_threshold: int = 30, min_quantity_threshold: int = 5) -> pd.DataFrame:
        """
        Identify slow-moving medicines.
        
        Definition: Medicines with no sales or very low sales in the specified period
        relative to their stock levels.
        """
        if self.enriched_sales.empty:
            # All medicines with no sales are slow moving
            result = self.medicines_df.copy()
            result['quantity_sold_period'] = 0
            result['revenue_period'] = 0.0
            result['days_since_last_sale'] = 'N/A'
            result['status'] = 'NO_SALES'
            result['reason'] = f'No sales history available'
            return result
        
        cutoff_date = pd.Timestamp.now() - timedelta(days=days_threshold)
        recent_sales = self.enriched_sales[self.enriched_sales['date'] >= cutoff_date]
        
        # Sales in recent period
        recent_summary = recent_sales.groupby('medicine_id').agg(
            quantity_sold_period=('quantity_sold', 'sum'),
            revenue_period=('revenue', 'sum'),
            last_sale_date=('date', 'max'),
            sales_count=('sale_id', 'count')
        ).reset_index()
        
        # All medicines
        all_medicines = self.medicines_df.copy()
        all_medicines = all_medicines.merge(recent_summary, on='medicine_id', how='left')
        all_medicines = all_medicines.fillna({
            'quantity_sold_period': 0,
            'revenue_period': 0.0,
            'sales_count': 0
        })
        
        # Calculate days since last sale
        all_medicines['days_since_last_sale'] = (pd.Timestamp.now() - pd.to_datetime(all_medicines['last_sale_date'])).dt.days
        all_medicines['days_since_last_sale'] = all_medicines['days_since_last_sale'].fillna('N/A')
        
        # Identify slow moving
        slow_moving = []
        for _, med in all_medicines.iterrows():
            qty_period = med['quantity_sold_period']
            days_since = med['days_since_last_sale']
            
            if qty_period == 0:
                status = 'NO_SALES'
                reason = f'No sales in last {days_threshold} days'
                slow_moving.append({**med.to_dict(), 'status': status, 'reason': reason})
            elif qty_period < min_quantity_threshold:
                status = 'SLOW_MOVING'
                reason = f'Only {qty_period} units sold in last {days_threshold} days'
                slow_moving.append({**med.to_dict(), 'status': status, 'reason': reason})
        
        return pd.DataFrame(slow_moving)
    
    def get_gross_profit_analysis(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get gross profit analysis using actual batch purchase costs.
        NOTE: Uses historical batch costs, not current inventory costs.
        """
        if self.enriched_sales.empty:
            return pd.DataFrame()
        
        df = self.enriched_sales.copy()
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        profit_analysis = df.groupby(['medicine_id', 'medicine_name', 'batch_id']).agg(
            quantity_sold=('quantity_sold', 'sum'),
            revenue=('revenue', 'sum'),
            purchase_cost_per_unit=('purchase_price', 'first'),  # Actual batch cost
            total_cost=('cost_of_goods_sold', 'sum'),
            gross_profit=('gross_profit', 'sum'),
            avg_selling_price=('unit_price', 'mean')
        ).reset_index()
        
        profit_analysis['gross_margin_pct'] = np.where(
            profit_analysis['revenue'] > 0,
            (profit_analysis['gross_profit'] / profit_analysis['revenue']) * 100,
            0.0
        )
        
        return profit_analysis.sort_values('gross_profit', ascending=False)
    
    def get_sales_statistics(self) -> Dict[str, Any]:
        """Get average selling price and sales quantity statistics."""
        if self.enriched_sales.empty:
            return {
                'avg_transaction_quantity': 0.0,
                'avg_unit_price': 0.0,
                'avg_transaction_value': 0.0,
                'median_unit_price': 0.0,
                'total_transactions': 0,
                'note': 'No sales data available'
            }
        
        stats = {
            'avg_transaction_quantity': float(np.mean(self.enriched_sales['quantity_sold'])),
            'median_transaction_quantity': float(np.median(self.enriched_sales['quantity_sold'])),
            'avg_unit_price': float(np.mean(self.enriched_sales['unit_price'])),
            'median_unit_price': float(np.median(self.enriched_sales['unit_price'])),
            'avg_transaction_value': float(np.mean(self.enriched_sales['revenue'])),
            'median_transaction_value': float(np.median(self.enriched_sales['revenue'])),
            'std_unit_price': float(np.std(self.enriched_sales['unit_price'])),
            'total_transactions': len(self.enriched_sales),
            'total_quantity_sold': int(self.enriched_sales['quantity_sold'].sum()),
            'total_revenue': float(self.enriched_sales['revenue'].sum()),
            'total_gross_profit': float(self.enriched_sales['gross_profit'].sum())
        }
        
        return stats
    
    def get_trend_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Get trend analysis and basic demand estimates."""
        if self.enriched_sales.empty:
            return {
                'trend': 'no_data',
                'daily_avg': 0.0,
                'weekly_avg': 0.0,
                'monthly_avg': 0.0,
                'demand_estimate': 0.0,
                'limitations': ['No historical sales data available']
            }
        
        cutoff_date = pd.Timestamp.now() - timedelta(days=days)
        recent = self.enriched_sales[self.enriched_sales['date'] >= cutoff_date]
        
        if recent.empty:
            return {
                'trend': 'no_recent_data',
                'daily_avg': 0.0,
                'weekly_avg': 0.0,
                'monthly_avg': 0.0,
                'demand_estimate': 0.0,
                'limitations': [f'No sales in last {days} days']
            }
        
        daily_totals = recent.groupby('date_only')['quantity_sold'].sum()
        daily_avg = float(np.mean(daily_totals))
        weekly_avg = daily_avg * 7
        monthly_avg = daily_avg * 30
        
        # Simple demand estimate (linear extrapolation)
        demand_estimate = daily_avg * 30  # Next 30 days estimate
        
        # Trend direction
        if len(daily_totals) >= 7:
            first_week_avg = daily_totals.head(7).mean()
            last_week_avg = daily_totals.tail(7).mean()
            if last_week_avg > first_week_avg * 1.1:
                trend = 'increasing'
            elif last_week_avg < first_week_avg * 0.9:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'period_days': days,
            'daily_avg': round(daily_avg, 2),
            'weekly_avg': round(weekly_avg, 2),
            'monthly_avg': round(monthly_avg, 2),
            'demand_estimate_next_30_days': round(demand_estimate, 2),
            'total_quantity_period': int(recent['quantity_sold'].sum()),
            'limitations': [
                'Estimates based on historical data only - may not predict future accurately',
                'Does not account for seasonality, promotions, or market changes',
                'Assumes linear trends which may not reflect actual demand patterns',
                'External factors (stockouts, expiry) not considered'
            ]
        }
    
    def export_summary_csv(self, output_dir: Path = None) -> Dict[str, str]:
        """
        Export analytical summaries to CSV.
        
        Returns:
            Dictionary with export paths
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "reports"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exports = {}
        
        # Revenue by medicine
        rev_medicine = self.get_revenue_by_medicine()
        if not rev_medicine.empty:
            path = output_dir / f'revenue_by_medicine_{timestamp}.csv'
            rev_medicine.to_csv(path, index=False)
            exports['revenue_by_medicine'] = str(path)
        
        # Revenue by category
        rev_category = self.get_revenue_by_category()
        if not rev_category.empty:
            path = output_dir / f'revenue_by_category_{timestamp}.csv'
            rev_category.to_csv(path, index=False)
            exports['revenue_by_category'] = str(path)
        
        # Monthly summary
        monthly = self.get_sales_summary_monthly()
        if not monthly.empty:
            path = output_dir / f'sales_monthly_{timestamp}.csv'
            monthly.to_csv(path, index=False)
            exports['sales_monthly'] = str(path)
        
        # Top sellers by quantity
        top_qty = self.get_top_selling_by_quantity()
        if not top_qty.empty:
            path = output_dir / f'top_selling_quantity_{timestamp}.csv'
            top_qty.to_csv(path, index=False)
            exports['top_selling_quantity'] = str(path)
        
        # Top sellers by revenue
        top_rev = self.get_top_selling_by_revenue()
        if not top_rev.empty:
            path = output_dir / f'top_selling_revenue_{timestamp}.csv'
            top_rev.to_csv(path, index=False)
            exports['top_selling_revenue'] = str(path)
        
        # Gross profit analysis
        profit = self.get_gross_profit_analysis()
        if not profit.empty:
            path = output_dir / f'gross_profit_analysis_{timestamp}.csv'
            profit.to_csv(path, index=False)
            exports['gross_profit_analysis'] = str(path)
        
        # Slow moving
        slow = self.get_slow_moving_medicines()
        if not slow.empty:
            path = output_dir / f'slow_moving_{timestamp}.csv'
            slow.to_csv(path, index=False)
            exports['slow_moving'] = str(path)
        
        print(f"Exported {len(exports)} reports to {output_dir}")
        return exports
