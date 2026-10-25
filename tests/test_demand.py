"""
Test suite for PharmaTrack demand forecasting and replenishment.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from pharmatrack.demand.forecasting import (
    DemandForecaster,
    compute_moving_average,
    compute_exponential_smoothing,
    compute_weekly_seasonality,
)
from pharmatrack.demand.replenishment import (
    ReplenishmentEngine,
    ReplenishmentSuggestion,
)
from pharmatrack.inventory.manager import InventoryManager
from pharmatrack.inventory.models import Medicine, Batch, InventorySnapshot, create_inventory_snapshot


class TestForecasting:
    """Test demand forecasting functions."""

    def test_compute_moving_average(self):
        series = pd.Series([10, 12, 8, 15, 11, 9, 14, 13], index=pd.date_range("2024-01-01", periods=8))
        ma = compute_moving_average(series, window=3)
        assert len(ma) == 8
        assert ma.iloc[2] == pytest.approx((10 + 12 + 8) / 3)
        assert ma.iloc[3] == pytest.approx((12 + 8 + 15) / 3)

    def test_compute_exponential_smoothing(self):
        series = pd.Series([10, 12, 8, 15, 11], index=pd.date_range("2024-01-01", periods=5))
        ses = compute_exponential_smoothing(series, alpha=0.5)
        assert len(ses) == 5
        assert ses.iloc[0] == 10  # First value unchanged
        assert ses.iloc[1] == pytest.approx(0.5 * 12 + 0.5 * 10)

    def test_compute_weekly_seasonality(self):
        dates = pd.date_range("2024-01-01", periods=14)  # 2 weeks
        values = [10, 12, 8, 15, 11, 20, 22,  # Week 1
                  10, 12, 8, 15, 11, 20, 22]  # Week 2 (same pattern)
        series = pd.Series(values, index=dates)
        seasonality = compute_weekly_seasonality(series)
        assert len(seasonality) == 7
        # Each day should have average of the two weeks
        assert seasonality[5] == pytest.approx(20.0)  # Saturday
        assert seasonality[6] == pytest.approx(22.0)  # Sunday


class TestDemandForecaster:
    """Test DemandForecaster class."""

    @pytest.fixture
    def sample_sales(self):
        """Create sample sales data with daily pattern."""
        today = pd.Timestamp.now().normalize()
        dates = pd.date_range(today - timedelta(days=59), today, freq="D")  # 60 days inclusive
        # Base demand with weekly seasonality (higher on weekends)
        base_demand = 10
        weekend_boost = 5
        data = []
        for i, d in enumerate(dates):
            dow = d.dayofweek
            qty = base_demand + (weekend_boost if dow >= 5 else 0) + np.random.randint(-2, 3)
            qty = max(0, qty)
            data.append({
                "sale_id": f"S{i}",
                "medicine_id": "M1",
                "batch_id": "B1",
                "quantity_sold": qty,
                "sale_date": d,
                "selling_price_per_unit": 20.0,
                "payment_method": "Cash",
            })
        return pd.DataFrame(data)

    def test_get_daily_demand(self, sample_sales):
        forecaster = DemandForecaster(sample_sales)
        daily = forecaster.get_daily_demand("M1")
        assert len(daily) == 60
        assert daily.index.freq == "D"
        assert (daily >= 0).all()

    def test_forecast_moving_average(self, sample_sales):
        forecaster = DemandForecaster(sample_sales)
        result = forecaster.forecast_moving_average("M1", window=7, forecast_days=7)
        assert "forecast" in result
        assert len(result["forecast"]) == 7
        assert result["method"] == "moving_average_7d"
        assert result["last_ma"] > 0

    def test_forecast_exponential_smoothing(self, sample_sales):
        forecaster = DemandForecaster(sample_sales)
        result = forecaster.forecast_exponential_smoothing("M1", alpha=0.3, forecast_days=7)
        assert len(result["forecast"]) == 7
        assert result["method"].startswith("exponential_smoothing")

    def test_forecast_seasonal_naive(self, sample_sales):
        forecaster = DemandForecaster(sample_sales)
        result = forecaster.forecast_seasonal_naive("M1", forecast_days=7)
        assert len(result["forecast"]) == 7
        assert result["method"] == "seasonal_naive_weekly"

    def test_get_demand_summary(self, sample_sales):
        forecaster = DemandForecaster(sample_sales)
        summary = forecaster.get_demand_summary("M1", days=30)
        assert summary["medicine_id"] == "M1"
        assert summary["period_days"] == 30
        assert summary["total_sold"] > 0
        assert summary["avg_daily"] > 0
        assert summary["coefficient_of_variation"] >= 0
        assert "weekly_pattern" in summary


class TestReplenishmentEngine:
    """Test ReplenishmentEngine class."""

    @pytest.fixture
    def sample_inventory(self):
        """Create sample inventory and sales for testing."""
        today = pd.Timestamp.now().normalize()

        # Medicines
        medicines = pd.DataFrame([
            {"medicine_id": "M1", "name": "Drug A", "category": "Antibiotic",
             "manufacturer": "Mfr1", "unit_selling_price": 20.0, "reorder_level": 30, "unit_of_measure": "tablet"},
            {"medicine_id": "M2", "name": "Drug B", "category": "Analgesic",
             "manufacturer": "Mfr2", "unit_selling_price": 10.0, "reorder_level": 50, "unit_of_measure": "tablet"},
            {"medicine_id": "M3", "name": "Drug C", "category": "Antibiotic",
             "manufacturer": "Mfr1", "unit_selling_price": 15.0, "reorder_level": 20, "unit_of_measure": "capsule"},
        ])

        # Batches - M1 has 80 saleable, M2 has 10 (low), M3 has 50
        batches = pd.DataFrame([
            {"batch_id": "B1", "medicine_id": "M1", "quantity_received": 100,
             "quantity_remaining": 50, "purchase_cost_per_unit": 10.0,
             "manufacture_date": today - timedelta(days=30),
             "expiry_date": today + timedelta(days=180),
             "supplier_id": "SUP001", "supplier_name": "Supplier",
             "is_expired": False, "days_to_expiry": 180, "saleable_quantity": 50},
            {"batch_id": "B2", "medicine_id": "M1", "quantity_received": 50,
             "quantity_remaining": 30, "purchase_cost_per_unit": 9.0,
             "manufacture_date": today - timedelta(days=10),
             "expiry_date": today + timedelta(days=365),
             "supplier_id": "SUP001", "supplier_name": "Supplier",
             "is_expired": False, "days_to_expiry": 365, "saleable_quantity": 30},
            {"batch_id": "B3", "medicine_id": "M2", "quantity_received": 50,
             "quantity_remaining": 10, "purchase_cost_per_unit": 4.0,
             "manufacture_date": today - timedelta(days=5),
             "expiry_date": today + timedelta(days=365),
             "supplier_id": "SUP001", "supplier_name": "Supplier",
             "is_expired": False, "days_to_expiry": 365, "saleable_quantity": 10},
            {"batch_id": "B4", "medicine_id": "M3", "quantity_received": 60,
             "quantity_remaining": 50, "purchase_cost_per_unit": 7.0,
             "manufacture_date": today - timedelta(days=1),
             "expiry_date": today + timedelta(days=365),
             "supplier_id": "SUP001", "supplier_name": "Supplier",
             "is_expired": False, "days_to_expiry": 365, "saleable_quantity": 50},
        ])

        # Sales - M1 sells ~10/day, M2 sells ~5/day, M3 sells ~2/day
        dates = pd.date_range(today - timedelta(days=30), today, freq="D")
        sales_data = []
        sale_id = 0
        for d in dates:
            # M1
            for _ in range(np.random.poisson(10)):
                sales_data.append({"sale_id": f"S{sale_id}", "batch_id": "B1", "medicine_id": "M1",
                                  "quantity_sold": 1, "sale_date": d, "selling_price_per_unit": 20.0, "payment_method": "Cash"})
                sale_id += 1
            # M2
            for _ in range(np.random.poisson(5)):
                sales_data.append({"sale_id": f"S{sale_id}", "batch_id": "B3", "medicine_id": "M2",
                                  "quantity_sold": 1, "sale_date": d, "selling_price_per_unit": 10.0, "payment_method": "Cash"})
                sale_id += 1
            # M3
            for _ in range(np.random.poisson(2)):
                sales_data.append({"sale_id": f"S{sale_id}", "batch_id": "B4", "medicine_id": "M3",
                                  "quantity_sold": 1, "sale_date": d, "selling_price_per_unit": 15.0, "payment_method": "Cash"})
                sale_id += 1

        sales = pd.DataFrame(sales_data)

        return medicines, batches, sales

    def test_replenishment_suggestions(self, sample_inventory):
        meds, batches, sales = sample_inventory
        inv = InventoryManager(meds, batches)
        demand = DemandForecaster(sales)
        engine = ReplenishmentEngine(inv, demand, lead_time_days=7, safety_factor=1.5)

        suggestions = engine.generate_all_suggestions()
        assert len(suggestions) == 3

        # M2 should be URGENT (10 saleable vs reorder 50)
        m2_sug = next(s for s in suggestions if s.medicine_id == "M2")
        assert m2_sug.priority == "URGENT"
        assert m2_sug.current_saleable_qty == 10
        assert m2_sug.suggested_order_qty > 0

        # M1 should have adequate stock (80 saleable vs reorder 30)
        m1_sug = next(s for s in suggestions if s.medicine_id == "M1")
        assert m1_sug.priority in ["OK", "MEDIUM"]

        # M3 should be OK (50 saleable vs reorder 20)
        m3_sug = next(s for s in suggestions if s.medicine_id == "M3")
        assert m3_sug.priority == "OK"

    def test_replenishment_summary(self, sample_inventory):
        meds, batches, sales = sample_inventory
        inv = InventoryManager(meds, batches)
        demand = DemandForecaster(sales)
        engine = ReplenishmentEngine(inv, demand)
        summary = engine.get_replenishment_summary()

        assert summary["total_medicines"] == 3
        assert summary["urgent_count"] == 1  # M2
        assert summary["total_suggested_order_qty"] > 0
        assert summary["estimated_order_value"] > 0

    def test_replenishment_suggestion_to_dict(self):
        sug = ReplenishmentSuggestion(
            medicine_id="M1",
            name="Drug A",
            category="Antibiotic",
            current_saleable_qty=10,
            reorder_level=30,
            avg_daily_demand=5.0,
            forecast_daily_demand=5.5,
            days_of_stock_remaining=1.8,
            suggested_order_qty=100,
            priority="URGENT",
            reason="Below reorder level",
            estimated_stockout_date="2024-07-01",
            near_expiry_qty=0,
            expired_qty=0,
        )
        d = sug.to_dict()
        assert d["medicine_id"] == "M1"
        assert d["priority"] == "URGENT"
        assert d["suggested_order_qty"] == 100