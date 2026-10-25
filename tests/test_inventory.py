"""
Test suite for PharmaTrack inventory management.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from pharmatrack.inventory.models import (
    Medicine,
    Batch,
    InventorySnapshot,
    create_inventory_snapshot,
)
from pharmatrack.inventory.manager import InventoryManager
from pharmatrack.utils.dates import is_expired, days_until_expiry


class TestModels:
    """Test inventory data models."""

    def test_medicine_from_series(self):
        series = pd.Series({
            "medicine_id": "MED001",
            "name": "Test Drug",
            "category": "Antibiotic",
            "manufacturer": "TestMfr",
            "unit_selling_price": 15.50,
            "reorder_level": 40,
            "unit_of_measure": "tablet",
        })
        med = Medicine.from_series(series)
        assert med.medicine_id == "MED001"
        assert med.unit_selling_price == 15.50
        assert med.reorder_level == 40

    def test_batch_from_series(self):
        series = pd.Series({
            "batch_id": "BATCH001",
            "medicine_id": "MED001",
            "quantity_received": 100,
            "quantity_remaining": 50,
            "purchase_cost_per_unit": 8.00,
            "manufacture_date": pd.Timestamp("2024-01-01"),
            "expiry_date": pd.Timestamp("2025-01-01"),
            "supplier_id": "SUP001",
            "supplier_name": "Test Supplier",
        })
        batch = Batch.from_series(series)
        assert batch.batch_id == "BATCH001"
        assert batch.quantity_received == 100
        assert batch.quantity_remaining == 50

    def test_batch_update_computed_fields(self):
        today = pd.Timestamp.now().normalize()
        batch = Batch(
            batch_id="BATCH001",
            medicine_id="MED001",
            quantity_received=100,
            quantity_remaining=50,
            purchase_cost_per_unit=8.00,
            manufacture_date=today - timedelta(days=30),
            expiry_date=today + timedelta(days=180),
            supplier_id="SUP001",
            supplier_name="Test Supplier",
        )
        batch.update_computed_fields()
        assert batch.saleable_quantity == 50
        assert batch.batch_value_at_cost == 400.0
        assert batch.is_expired is False

    def test_create_inventory_snapshot(self):
        today = pd.Timestamp.now().normalize()
        med = Medicine(
            medicine_id="MED001",
            name="Test Drug",
            category="Antibiotic",
            manufacturer="TestMfr",
            unit_selling_price=15.00,
            reorder_level=30,
            unit_of_measure="tablet",
        )
        batch1 = Batch(
            batch_id="BATCH001", medicine_id="MED001",
            quantity_received=100, quantity_remaining=50,
            purchase_cost_per_unit=8.00,
            manufacture_date=today - timedelta(days=30),
            expiry_date=today + timedelta(days=180),
            supplier_id="SUP001", supplier_name="Supplier",
        )
        batch1.update_computed_fields()
        batch2 = Batch(
            batch_id="BATCH002", medicine_id="MED001",
            quantity_received=50, quantity_remaining=10,
            purchase_cost_per_unit=7.50,
            manufacture_date=today - timedelta(days=10),
            expiry_date=today + timedelta(days=60),
            supplier_id="SUP001", supplier_name="Supplier",
        )
        batch2.update_computed_fields()

        snap = create_inventory_snapshot(med, [batch1, batch2], near_expiry_threshold=30)
        assert snap.medicine_id == "MED001"
        assert snap.total_received == 150
        assert snap.total_remaining == 60
        assert snap.total_saleable == 60
        assert snap.total_value_at_cost == 50*8.00 + 10*7.50
        assert snap.potential_revenue == 60 * 15.00
        assert snap.is_low_stock is False  # 60 > 30


class TestInventoryManager:
    """Test InventoryManager analytical queries."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        today = pd.Timestamp.now().normalize()
        medicines = pd.DataFrame([
            {"medicine_id": "MED001", "name": "Drug A", "category": "Antibiotic",
             "manufacturer": "Mfr1", "unit_selling_price": 20.0, "reorder_level": 30, "unit_of_measure": "tablet"},
            {"medicine_id": "MED002", "name": "Drug B", "category": "Analgesic",
             "manufacturer": "Mfr2", "unit_selling_price": 10.0, "reorder_level": 50, "unit_of_measure": "tablet"},
        ])

        batches = pd.DataFrame([
            # MED001 - healthy batch
            {"batch_id": "BATCH001", "medicine_id": "MED001", "quantity_received": 100,
             "quantity_remaining": 50, "purchase_cost_per_unit": 10.0,
             "manufacture_date": today - timedelta(days=30),
             "expiry_date": today + timedelta(days=180),
             "supplier_id": "SUP001", "supplier_name": "Supplier 1"},
            # MED001 - near expiry batch
            {"batch_id": "BATCH002", "medicine_id": "MED001", "quantity_received": 50,
             "quantity_remaining": 20, "purchase_cost_per_unit": 9.0,
             "manufacture_date": today - timedelta(days=100),
             "expiry_date": today + timedelta(days=15),
             "supplier_id": "SUP001", "supplier_name": "Supplier 1"},
            # MED001 - expired batch
            {"batch_id": "BATCH003", "medicine_id": "MED001", "quantity_received": 30,
             "quantity_remaining": 15, "purchase_cost_per_unit": 11.0,
             "manufacture_date": today - timedelta(days=200),
             "expiry_date": today - timedelta(days=10),
             "supplier_id": "SUP002", "supplier_name": "Supplier 2"},
            # MED002 - low stock
            {"batch_id": "BATCH004", "medicine_id": "MED002", "quantity_received": 100,
             "quantity_remaining": 10, "purchase_cost_per_unit": 5.0,
             "manufacture_date": today - timedelta(days=10),
             "expiry_date": today + timedelta(days=365),
             "supplier_id": "SUP001", "supplier_name": "Supplier 1"},
        ])

        return medicines, batches

    def test_get_medicine(self, sample_data):
        meds, batches = sample_data
        inv = InventoryManager(meds, batches)
        med = inv.get_medicine("MED001")
        assert med is not None
        assert med.name == "Drug A"
        assert inv.get_medicine("NONEXISTENT") is None

    def test_get_saleable_batches(self, sample_data):
        meds, batches = sample_data
        inv = InventoryManager(meds, batches)
        saleable = inv.get_saleable_batches("MED001")
        # Should only return non-expired batches with remaining qty
        assert len(saleable) == 2  # BATCH001 and BATCH002
        assert (saleable["is_expired"] == False).all()
        assert (saleable["saleable_quantity"] > 0).all()

    def test_get_expired_batches(self, sample_data):
        meds, batches = sample_data
        inv = InventoryManager(meds, batches)
        expired = inv.get_expired_batches("MED001")
        assert len(expired) == 1
        assert expired.iloc[0]["batch_id"] == "BATCH003"

    def test_get_near_expiry_batches(self, sample_data):
        meds, batches = sample_data
        inv = InventoryManager(meds, batches, near_expiry_threshold=30)
        near = inv.get_near_expiry_batches("MED001")
        assert len(near) == 1
        assert near.iloc[0]["batch_id"] == "BATCH002"

    def test_get_inventory_snapshot(self, sample_data):
        meds, batches = sample_data
        inv = InventoryManager(meds, batches)
        snap = inv.get_inventory_snapshot("MED001")
        assert snap is not None
        assert snap.total_remaining == 85  # 50 + 20 + 15
        assert snap.total_saleable == 70  # 50 + 20 (expired excluded)
        assert snap.expired_quantity == 15
        assert snap.near_expiry_quantity == 20
        assert snap.total_value_at_cost == 50*10 + 20*9  # expired excluded

    def test_get_low_stock_medicines(self, sample_data):
        meds, batches = sample_data
        inv = InventoryManager(meds, batches)
        low_stock = inv.get_low_stock_medicines()
        # MED002 has 10 saleable, reorder_level=50
        assert len(low_stock) == 1
        assert low_stock[0].medicine_id == "MED002"

    def test_get_total_inventory_value(self, sample_data):
        meds, batches = sample_data
        inv = InventoryManager(meds, batches)
        total = inv.get_total_inventory_value()
        assert total["saleable_quantity"] == 80  # 70 + 10
        assert total["expired_quantity"] == 15
        assert total["saleable_batches"] == 3
        assert total["expired_batches"] == 1

    def test_get_category_summary(self, sample_data):
        meds, batches = sample_data
        inv = InventoryManager(meds, batches)
        cat = inv.get_category_summary()
        assert len(cat) == 2
        assert "category" in cat.columns
        assert "total_saleable_qty" in cat.columns

    def test_get_expiry_timeline(self, sample_data):
        meds, batches = sample_data
        inv = InventoryManager(meds, batches)
        timeline = inv.get_expiry_timeline(60)
        assert len(timeline) > 0
        assert "period_start" in timeline.columns
        assert "expiring_qty" in timeline.columns