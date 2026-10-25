"""
Test suite for PharmaTrack expiry analysis.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from pharmatrack.expiry.analysis import (
    ExpiryAnalyzer,
    compute_financial_exposure,
    RiskLevel,
    assess_batch_risk,
)
from pharmatrack.expiry.risk import BatchRiskAssessment, assess_batch, get_priority_action_list
from pharmatrack.utils.dates import is_expired, is_near_expiry, days_until_expiry, get_expiry_status


class TestDateUtils:
    """Test date utility functions."""

    def test_days_until_expiry_future(self):
        future = pd.Timestamp.now().normalize() + timedelta(days=30)
        assert days_until_expiry(future) == 30

    def test_days_until_expiry_today(self):
        today = pd.Timestamp.now().normalize()
        assert days_until_expiry(today) == 0

    def test_days_until_expiry_past(self):
        past = pd.Timestamp.now().normalize() - timedelta(days=10)
        assert days_until_expiry(past) == -10

    def test_is_expired_true(self):
        past = pd.Timestamp.now().normalize() - timedelta(days=1)
        assert is_expired(past) is True

    def test_is_expired_false(self):
        future = pd.Timestamp.now().normalize() + timedelta(days=1)
        assert is_expired(future) is False

    def test_is_near_expiry(self):
        today = pd.Timestamp.now().normalize()
        assert is_near_expiry(today + timedelta(days=15), threshold_days=30) is True
        assert is_near_expiry(today + timedelta(days=45), threshold_days=30) is False
        assert is_near_expiry(today - timedelta(days=5), threshold_days=30) is False  # Expired

    def test_get_expiry_status(self):
        today = pd.Timestamp.now().normalize()
        assert get_expiry_status(today - timedelta(days=5)) == "Expired"
        assert get_expiry_status(today + timedelta(days=15), near_threshold=30) == "Near Expiry"
        assert get_expiry_status(today + timedelta(days=60), near_threshold=30) == "Valid"


class TestRiskAssessment:
    """Test risk level assessment."""

    def test_assess_batch_risk_expired(self):
        past = pd.Timestamp.now().normalize() - timedelta(days=10)
        assert assess_batch_risk(past) == RiskLevel.EXPIRED

    def test_assess_batch_risk_critical(self):
        today = pd.Timestamp.now().normalize()
        assert assess_batch_risk(today + timedelta(days=3)) == RiskLevel.CRITICAL
        assert assess_batch_risk(today + timedelta(days=7)) == RiskLevel.CRITICAL

    def test_assess_batch_risk_high(self):
        today = pd.Timestamp.now().normalize()
        assert assess_batch_risk(today + timedelta(days=15)) == RiskLevel.HIGH
        assert assess_batch_risk(today + timedelta(days=30)) == RiskLevel.HIGH

    def test_assess_batch_risk_medium(self):
        today = pd.Timestamp.now().normalize()
        assert assess_batch_risk(today + timedelta(days=45)) == RiskLevel.MEDIUM

    def test_assess_batch_risk_low(self):
        today = pd.Timestamp.now().normalize()
        assert assess_batch_risk(today + timedelta(days=75)) == RiskLevel.LOW

    def test_assess_batch_risk_safe(self):
        today = pd.Timestamp.now().normalize()
        assert assess_batch_risk(today + timedelta(days=120)) == RiskLevel.SAFE


class TestFinancialExposure:
    """Test financial exposure computation."""

    def test_compute_financial_exposure(self):
        today = pd.Timestamp.now().normalize()
        batches = pd.DataFrame([
            # Expired batch
            {"batch_id": "B1", "quantity_remaining": 10, "purchase_cost_per_unit": 5.0,
             "expiry_date": today - timedelta(days=10), "is_expired": True,
             "days_to_expiry": -10, "unit_selling_price": 10.0},
            # Near expiry batch
            {"batch_id": "B2", "quantity_remaining": 20, "purchase_cost_per_unit": 8.0,
             "expiry_date": today + timedelta(days=15), "is_expired": False,
             "days_to_expiry": 15, "unit_selling_price": 15.0},
            # Healthy batch
            {"batch_id": "B3", "quantity_remaining": 50, "purchase_cost_per_unit": 6.0,
             "expiry_date": today + timedelta(days=180), "is_expired": False,
             "days_to_expiry": 180, "unit_selling_price": 12.0},
        ])

        exposure = compute_financial_exposure(batches, near_expiry_threshold=30)

        assert exposure["expired"]["batches"] == 1
        assert exposure["expired"]["quantity"] == 10
        assert exposure["expired"]["value_at_cost"] == 50.0
        assert exposure["expired"]["value_at_price"] == 100.0

        assert exposure["near_expiry"]["batches"] == 1
        assert exposure["near_expiry"]["quantity"] == 20
        assert exposure["near_expiry"]["value_at_cost"] == 160.0
        assert exposure["near_expiry"]["value_at_price"] == 300.0

        assert exposure["healthy"]["batches"] == 1
        assert exposure["healthy"]["quantity"] == 50

        assert exposure["total_exposure_at_cost"] == 210.0  # 50 + 160
        assert exposure["total_exposure_at_price"] == 400.0  # 100 + 300


class TestExpiryAnalyzer:
    """Test ExpiryAnalyzer class."""

    @pytest.fixture
    def sample_data(self):
        today = pd.Timestamp.now().normalize()
        medicines = pd.DataFrame([
            {"medicine_id": "M1", "name": "Drug A", "category": "Antibiotic", "unit_selling_price": 20.0},
            {"medicine_id": "M2", "name": "Drug B", "category": "Analgesic", "unit_selling_price": 10.0},
        ])
        batches = pd.DataFrame([
            # M1 - expired
            {"batch_id": "B1", "medicine_id": "M1", "quantity_received": 100,
             "quantity_remaining": 10, "purchase_cost_per_unit": 8.0,
             "manufacture_date": today - timedelta(days=200),
             "expiry_date": today - timedelta(days=10),
             "supplier_id": "SUP001", "supplier_name": "Supplier"},
            # M1 - near expiry
            {"batch_id": "B2", "medicine_id": "M1", "quantity_received": 50,
             "quantity_remaining": 20, "purchase_cost_per_unit": 9.0,
             "manufacture_date": today - timedelta(days=100),
             "expiry_date": today + timedelta(days=15),
             "supplier_id": "SUP001", "supplier_name": "Supplier"},
            # M1 - healthy
            {"batch_id": "B3", "medicine_id": "M1", "quantity_received": 80,
             "quantity_remaining": 40, "purchase_cost_per_unit": 8.5,
             "manufacture_date": today - timedelta(days=30),
             "expiry_date": today + timedelta(days=180),
             "supplier_id": "SUP001", "supplier_name": "Supplier"},
            # M2 - healthy
            {"batch_id": "B4", "medicine_id": "M2", "quantity_received": 60,
             "quantity_remaining": 30, "purchase_cost_per_unit": 4.0,
             "manufacture_date": today - timedelta(days=10),
             "expiry_date": today + timedelta(days=365),
             "supplier_id": "SUP002", "supplier_name": "Supplier"},
        ])
        return medicines, batches

    def test_get_expired_batches(self, sample_data):
        meds, batches = sample_data
        ea = ExpiryAnalyzer(batches, meds)
        expired = ea.get_expired_batches()
        assert len(expired) == 1
        assert expired.iloc[0]["batch_id"] == "B1"

    def test_get_near_expiry_batches(self, sample_data):
        meds, batches = sample_data
        ea = ExpiryAnalyzer(batches, meds, near_expiry_threshold=30)
        near = ea.get_near_expiry_batches()
        assert len(near) == 1
        assert near.iloc[0]["batch_id"] == "B2"

    def test_get_risk_breakdown(self, sample_data):
        meds, batches = sample_data
        ea = ExpiryAnalyzer(batches, meds)
        risk = ea.get_risk_breakdown()
        assert len(risk) > 0
        assert "risk_level" in risk.columns
        assert "batches" in risk.columns
        assert "total_qty" in risk.columns
        assert "value_at_cost" in risk.columns

    def test_get_expiry_timeline(self, sample_data):
        meds, batches = sample_data
        ea = ExpiryAnalyzer(batches, meds)
        timeline = ea.get_expiry_timeline(60)
        assert len(timeline) == 61  # today + 60 days
        assert "date" in timeline.columns
        assert "expiring_qty" in timeline.columns

    def test_get_wastage_report(self, sample_data):
        meds, batches = sample_data
        ea = ExpiryAnalyzer(batches, meds)
        report = ea.get_wastage_report()

        assert "summary" in report
        assert "expired_batches_detail" in report
        assert "near_expiry_batches_detail" in report
        assert "risk_distribution" in report
        assert "category_breakdown" in report

        assert len(report["expired_batches_detail"]) == 1
        assert len(report["near_expiry_batches_detail"]) == 1


class TestBatchRiskAssessment:
    """Test batch risk assessment utilities."""

    def test_assess_batch(self):
        today = pd.Timestamp.now().normalize()
        batch_row = pd.Series({
            "batch_id": "B1",
            "medicine_id": "M1",
            "name": "Drug A",
            "quantity_remaining": 10,
            "purchase_cost_per_unit": 5.0,
            "expiry_date": today + timedelta(days=5),
            "days_to_expiry": 5,
        })
        assessment = assess_batch(batch_row, near_threshold=30)
        assert isinstance(assessment, BatchRiskAssessment)
        assert assessment.batch_id == "B1"
        assert assessment.risk_level == RiskLevel.CRITICAL
        assert assessment.value_at_risk == 50.0
        assert "Urgent" in assessment.recommended_action

    def test_get_priority_action_list(self):
        today = pd.Timestamp.now().normalize()
        batches = pd.DataFrame([
            {"batch_id": "B1", "medicine_id": "M1", "name": "Drug A",
             "quantity_remaining": 10, "purchase_cost_per_unit": 5.0,
             "expiry_date": today - timedelta(days=5), "days_to_expiry": -5, "is_expired": True},
            {"batch_id": "B2", "medicine_id": "M1", "name": "Drug A",
             "quantity_remaining": 20, "purchase_cost_per_unit": 8.0,
             "expiry_date": today + timedelta(days=3), "days_to_expiry": 3, "is_expired": False},
            {"batch_id": "B3", "medicine_id": "M1", "name": "Drug A",
             "quantity_remaining": 50, "purchase_cost_per_unit": 6.0,
             "expiry_date": today + timedelta(days=100), "days_to_expiry": 100, "is_expired": False},
        ])

        priority = get_priority_action_list(batches, near_threshold=30, top_n=10)
        assert len(priority) == 3
        # Expired should be first
        assert priority[0].risk_level == RiskLevel.EXPIRED
        # Critical should be second
        assert priority[1].risk_level == RiskLevel.CRITICAL
        # Safe should be last
        assert priority[2].risk_level == RiskLevel.SAFE