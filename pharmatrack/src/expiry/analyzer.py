"""
Expiry and Wastage Risk Analysis Module

Provides functionality for:
- Calculating days remaining until batch expiry
- Classifying batches as expired, near expiry, or outside warning period
- Identifying high-value batches approaching expiry
- Calculating purchase value of remaining stock at risk
- Batch-level expiry reports
- Analyzing historical wastage if records are available
- FEFO (First-Expiry-First-Out) stock rotation recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class ExpiryAnalyzer:
    """Expiry and wastage risk analysis for PharmaTrack."""
    
    def __init__(self, 
                 medicines_df: pd.DataFrame, 
                 inventory_df: pd.DataFrame, 
                 wastage_df: pd.DataFrame = None,
                 reference_date: datetime = None,
                 near_expiry_threshold: int = 30):
        """
        Initialize expiry analyzer.
        
        Args:
            medicines_df: DataFrame with medicine catalogue
            inventory_df: DataFrame with batch inventory
            wastage_df: DataFrame with wastage records (optional)
            reference_date: Reference date for calculations (injectable for testing)
            near_expiry_threshold: Days threshold for near-expiry classification (default 30)
        """
        self.medicines_df = medicines_df.copy()
        self.inventory_df = inventory_df.copy()
        self.wastage_df = wastage_df.copy() if wastage_df is not None else pd.DataFrame()
        self.near_expiry_threshold = near_expiry_threshold
        
        # Use injectable reference date for deterministic tests
        if reference_date is None:
            self.reference_date = pd.Timestamp.now()
        else:
            self.reference_date = pd.Timestamp(reference_date)
        
        # Ensure dates are datetime
        if 'expiry_date' in self.inventory_df.columns:
            self.inventory_df['expiry_date'] = pd.to_datetime(self.inventory_df['expiry_date'])
        if not self.wastage_df.empty and 'date' in self.wastage_df.columns:
            self.wastage_df['date'] = pd.to_datetime(self.wastage_df['date'])
        
        # Prepare expiry analysis data
        self._prepare_expiry_data()
    
    def _prepare_expiry_data(self):
        """Prepare inventory data with expiry calculations."""
        df = self.inventory_df.copy()
        
        if df.empty or 'expiry_date' not in df.columns:
            self.expiry_df = df
            return
        
        # Calculate days remaining until expiry
        df['days_remaining'] = (df['expiry_date'] - self.reference_date).dt.days
        
        # Classify batches - expired takes precedence over near expiry
        def classify_batch(days_remain):
            if pd.isna(days_remain):
                return 'unknown'
            elif days_remain < 0:
                return 'expired'
            elif days_remain <= self.near_expiry_threshold:
                # Batch expiring today (days_remaining == 0) is classified as near_expiry
                return 'near_expiry'
            else:
                return 'ok'
        
        df['expiry_status'] = df['days_remaining'].apply(classify_batch)
        
        # Calculate risk value (purchase value of remaining stock at risk)
        # Only non-expired batches count as "at risk" for near-expiry classification
        # Estimate risk value using remaining quantity * purchase cost
        if 'quantity' in df.columns and 'purchase_price' in df.columns:
            df['purchase_value_total'] = df['quantity'] * df['purchase_price']
            # At-risk value only for near-expiry non-expired batches
            df['risk_value'] = 0.0
            mask_near = (df['expiry_status'] == 'near_expiry') & (df['days_remaining'] >= 0)
            df.loc[mask_near, 'risk_value'] = df.loc[mask_near, 'quantity'] * df.loc[mask_near, 'purchase_price']
        else:
            df['purchase_value_total'] = 0.0
            df['risk_value'] = 0.0
        
        # Join with medicine details
        if not self.medicines_df.empty:
            df = df.merge(
                self.medicines_df[['medicine_id', 'medicine_name', 'category']], 
                on='medicine_id', 
                how='left'
            )
        
        self.expiry_df = df
    
    def get_days_remaining(self, batch_id: str = None) -> pd.DataFrame:
        """Get days remaining until expiry for batches."""
        df = self.expiry_df.copy()
        if batch_id:
            df = df[df['batch_id'] == batch_id]
        if 'days_remaining' in df.columns:
            return df[['batch_id', 'medicine_id', 'expiry_date', 'days_remaining', 'expiry_status']]
        return df
    
    def classify_batches(self, threshold: int = None) -> pd.DataFrame:
        """
        Classify batches as expired, near expiry, or outside warning period.
        
        Args:
            threshold: Override default near-expiry threshold
        """
        if threshold is not None and threshold != self.near_expiry_threshold:
            # Reclassify with different threshold
            df = self.inventory_df.copy()
            if df.empty or 'expiry_date' not in df.columns:
                return df
            
            df['expiry_date'] = pd.to_datetime(df['expiry_date'])
            df['days_remaining'] = (df['expiry_date'] - self.reference_date).dt.days
            
            def classify_batch(days_remain):
                if pd.isna(days_remain):
                    return 'unknown'
                elif days_remain < 0:
                    return 'expired'
                elif days_remain <= threshold:
                    return 'near_expiry'
                else:
                    return 'ok'
            
            df['expiry_status'] = df['days_remaining'].apply(classify_batch)
            if not self.medicines_df.empty:
                df = df.merge(self.medicines_df[['medicine_id', 'medicine_name', 'category']], 
                             on='medicine_id', how='left')
            return df
        
        return self.expiry_df.copy()
    
    def get_expired_batches(self) -> pd.DataFrame:
        """Get all expired batches."""
        expired = self.expiry_df[self.expiry_df['expiry_status'] == 'expired'].copy()
        return expired
    
    def get_near_expiry_batches(self, threshold: int = None) -> pd.DataFrame:
        """
        Get batches near expiry (within threshold days).
        Expired batches are never included.
        """
        if threshold is not None and threshold != self.near_expiry_threshold:
            classified = self.classify_batches(threshold)
            near = classified[(classified['expiry_status'] == 'near_expiry') & 
                             (classified['days_remaining'] >= 0)].copy()
        else:
            near = self.expiry_df[(self.expiry_df['expiry_status'] == 'near_expiry') & 
                                 (self.expiry_df['days_remaining'] >= 0)].copy()
        return near
    
    def get_high_value_batches_at_risk(self, min_value: float = 100.0) -> pd.DataFrame:
        """Identify high-value batches approaching expiry."""
        near_expiry = self.get_near_expiry_batches()
        if near_expiry.empty:
            return pd.DataFrame()
        
        high_value = near_expiry[near_expiry['risk_value'] >= min_value].copy()
        high_value = high_value.sort_values('risk_value', ascending=False)
        return high_value
    
    def calculate_value_at_risk(self) -> Dict[str, float]:
        """
        Calculate purchase value of remaining stock at risk.
        NOTE: This is an estimated risk value, not an actual financial loss.
        """
        near_expiry = self.get_near_expiry_batches()
        expired = self.get_expired_batches()
        
        value_at_risk = float(near_expiry['risk_value'].sum()) if not near_expiry.empty and 'risk_value' in near_expiry.columns else 0.0
        expired_value = float(expired['purchase_value_total'].sum()) if not expired.empty and 'purchase_value_total' in expired.columns else 0.0
        
        return {
            'near_expiry_value_at_risk': value_at_risk,
            'expired_value': expired_value,
            'total_at_risk_estimate': value_at_risk,  # Only near-expiry counted as "at risk"
            'near_expiry_batches_count': len(near_expiry),
            'expired_batches_count': len(expired),
            'disclaimer': 'Values are estimates based on purchase costs of remaining stock, not actual financial losses'
        }
    
    def generate_batch_expiry_report(self) -> pd.DataFrame:
        """Generate batch-level expiry report."""
        report = self.expiry_df.copy()
        
        if report.empty:
            return report
        
        # Select relevant columns
        cols = ['batch_id', 'medicine_id']
        if 'medicine_name' in report.columns:
            cols.append('medicine_name')
        if 'category' in report.columns:
            cols.append('category')
        cols.extend(['quantity', 'purchase_price', 'purchase_value_total', 
                     'expiry_date', 'days_remaining', 'expiry_status', 'risk_value'])
        
        # Only include columns that exist
        cols = [c for c in cols if c in report.columns]
        report = report[cols].copy()
        
        # Sort by days remaining (expired first, then near expiry ascending)
        report = report.sort_values(['expiry_status', 'days_remaining'])
        
        return report
    
    def analyze_wastage(self) -> Dict[str, Any]:
        """
        Analyze recorded historical wastage if wastage records are available.
        Does NOT claim wastage occurred unless dataset contains recorded wastage transactions.
        """
        if self.wastage_df.empty:
            return {
                'wastage_records_found': False,
                'total_wastage_quantity': 0,
                'total_wastage_value': 0.0,
                'message': 'No wastage records found in dataset. Wastage analysis not available.',
                'disclaimer': 'No wastage transactions recorded in the provided data'
            }
        
        # Check if wastage has required columns
        wastage_analysis = {
            'wastage_records_found': True,
            'total_records': len(self.wastage_df),
            'disclaimer': 'Analysis based solely on recorded wastage transactions in dataset'
        }
        
        if 'quantity' in self.wastage_df.columns:
            wastage_analysis['total_wastage_quantity'] = int(self.wastage_df['quantity'].sum())
        if 'quantity' in self.wastage_df.columns and 'cost' in self.wastage_df.columns:
            wastage_analysis['total_wastage_value'] = float((self.wastage_df['quantity'] * self.wastage_df['cost']).sum())
        elif 'value' in self.wastage_df.columns:
            wastage_analysis['total_wastage_value'] = float(self.wastage_df['value'].sum())
        
        # Join with medicine info if possible
        if 'medicine_id' in self.wastage_df.columns and not self.medicines_df.empty:
            wastage_with_med = self.wastage_df.merge(
                self.medicines_df[['medicine_id', 'medicine_name', 'category']], 
                on='medicine_id', 
                how='left'
            )
            if 'quantity' in wastage_with_med.columns:
                by_medicine = wastage_with_med.groupby(['medicine_id', 'medicine_name']).agg(
                    wastage_quantity=('quantity', 'sum')
                ).reset_index()
                wastage_analysis['wastage_by_medicine'] = by_medicine.to_dict('records')
        
        # By reason if available
        if 'reason' in self.wastage_df.columns:
            by_reason = self.wastage_df.groupby('reason').agg(
                count=('reason', 'count'),
                quantity=('quantity', 'sum') if 'quantity' in self.wastage_df.columns else ('reason', 'count')
            ).reset_index()
            wastage_analysis['wastage_by_reason'] = by_reason.to_dict('records')
        
        return wastage_analysis
    
    def get_fefo_recommendations(self) -> pd.DataFrame:
        """
        Provide FEFO (First-Expiry-First-Out) stock rotation recommendations.
        Prioritize batches expiring earliest first.
        Excludes expired batches from active rotation recommendations.
        """
        df = self.expiry_df.copy()
        if df.empty:
            return pd.DataFrame()
        
        # Only consider non-expired batches for rotation
        active = df[df['expiry_status'] != 'expired'].copy()
        if active.empty:
            return pd.DataFrame()
        
        # Sort by expiry date (earliest first) - FEFO principle
        recommendations = active.sort_values('expiry_date').copy()
        
        # Add recommendation priority
        recommendations['rotation_priority'] = range(1, len(recommendations) + 1)
        recommendations['recommendation'] = recommendations.apply(
            lambda row: f"Use first - expires in {row['days_remaining']} days" 
            if row['days_remaining'] >= 0 else "EXPIRED - remove",
            axis=1
        )
        
        # Select relevant columns
        cols = ['rotation_priority', 'batch_id', 'medicine_id']
        if 'medicine_name' in recommendations.columns:
            cols.append('medicine_name')
        cols.extend(['quantity', 'expiry_date', 'days_remaining', 'expiry_status', 'recommendation'])
        
        cols = [c for c in cols if c in recommendations.columns]
        return recommendations[cols]
    
    def get_expiry_summary(self) -> Dict[str, Any]:
        """Get overall expiry summary."""
        total_batches = len(self.expiry_df)
        expired_count = len(self.get_expired_batches())
        near_expiry_count = len(self.get_near_expiry_batches())
        ok_count = total_batches - expired_count - near_expiry_count
        
        value_risk = self.calculate_value_at_risk()
        
        return {
            'reference_date': self.reference_date.strftime('%Y-%m-%d'),
            'near_expiry_threshold_days': self.near_expiry_threshold,
            'total_batches': total_batches,
            'expired_batches': expired_count,
            'near_expiry_batches': near_expiry_count,
            'ok_batches': ok_count,
            'value_at_risk': value_risk,
            'wastage_analysis': self.analyze_wastage()
        }
    
    def export_expiry_report(self, output_path: Path = None) -> str:
        """Export batch-level expiry report."""
        report_df = self.generate_batch_expiry_report()
        summary = self.get_expiry_summary()
        fefo = self.get_fefo_recommendations()
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("PHARMATRACK - EXPIRY & WASTAGE RISK ANALYSIS REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Reference Date: {summary['reference_date']}")
        report_lines.append(f"Near-Expiry Threshold: {summary['near_expiry_threshold_days']} days")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Summary
        report_lines.append("SUMMARY:")
        report_lines.append(f"  Total Batches: {summary['total_batches']}")
        report_lines.append(f"  Expired: {summary['expired_batches']}")
        report_lines.append(f"  Near Expiry: {summary['near_expiry_batches']}")
        report_lines.append(f"  OK: {summary['ok_batches']}")
        report_lines.append("")
        report_lines.append("VALUE AT RISK (ESTIMATES):")
        report_lines.append(f"  Near-Expiry Value at Risk: ${summary['value_at_risk']['near_expiry_value_at_risk']:.2f}")
        report_lines.append(f"  Expired Value: ${summary['value_at_risk']['expired_value']:.2f}")
        report_lines.append(f"  {summary['value_at_risk']['disclaimer']}")
        report_lines.append("")
        
        # Wastage
        wastage = summary['wastage_analysis']
        report_lines.append("WASTAGE ANALYSIS:")
        if wastage['wastage_records_found']:
            report_lines.append(f"  Total Wastage Records: {wastage['total_records']}")
            if 'total_wastage_quantity' in wastage:
                report_lines.append(f"  Total Wastage Quantity: {wastage['total_wastage_quantity']}")
            if 'total_wastage_value' in wastage:
                report_lines.append(f"  Total Wastage Value: ${wastage['total_wastage_value']:.2f}")
        else:
            report_lines.append(f"  {wastage['message']}")
        report_lines.append("")
        
        # Batch details
        report_lines.append("BATCH-LEVEL EXPIRY REPORT:")
        if report_df.empty:
            report_lines.append("  No batches to report")
        else:
            for _, row in report_df.iterrows():
                report_lines.append(f"  Batch: {row['batch_id']} | Medicine: {row.get('medicine_name', row['medicine_id'])}")
                report_lines.append(f"    Qty: {row.get('quantity', 0)} | Purchase Value: ${row.get('purchase_value_total', 0):.2f} | Status: {row['expiry_status']}")
                report_lines.append(f"    Expiry: {row['expiry_date']} | Days Remaining: {row['days_remaining']}")
                if row['expiry_status'] == 'near_expiry':
                    report_lines.append(f"    Risk Value: ${row.get('risk_value', 0):.2f}")
                report_lines.append("")
        
        # FEFO recommendations
        report_lines.append("FEFO (FIRST-EXPIRY-FIRST-OUT) RECOMMENDATIONS:")
        if fefo.empty:
            report_lines.append("  No active batches for rotation")
        else:
            for _, row in fefo.head(10).iterrows():
                report_lines.append(f"  #{row['rotation_priority']}: {row['batch_id']} ({row.get('medicine_name', row['medicine_id'])}) - {row['recommendation']}")
        report_lines.append("")
        report_lines.append("=" * 70)
        
        report_content = "\n".join(report_lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_content)
            print(f"Expiry report exported to {output_path}")
        
        return report_content
