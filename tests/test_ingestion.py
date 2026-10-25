"""
Test suite for PharmaTrack data ingestion and validation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from pharmatrack.data.ingestion import (
    load_medicines,
    load_batches,
    load_sales,
    load_all,
    clean_batches_for_analysis,
)
from pharmatrack.data.schemas import (
    validate_dataframe,
    validate_foreign_keys,
    validate_business_rules,
    ValidationError,
    MEDICINES_SCHEMA,
    BATCHES_SCHEMA,
    SALES_SCHEMA,
)
from pharmatrack.data.sample_gen import generate_sample_data


class TestSchemas:
    """Test schema validation functions."""

    def test_validate_dataframe_valid(self):
        df = pd.DataFrame({
            "medicine_id": ["MED001", "MED002"],
            "name": ["Drug A", "Drug B"],
            "category": ["Antibiotic", "Analgesic"],
            "manufacturer": ["Mfr1", "Mfr2"],
            "unit_selling_price": [10.0, 15.0],
            "reorder_level": [50, 30],
            "unit_of_measure": ["tablet", "capsule"],
        })
        errors = validate_dataframe(df, MEDICINES_SCHEMA, list(MEDICINES_SCHEMA.keys()), name="medicines")
        assert errors == []

    def test_validate_dataframe_missing_columns(self):
        df = pd.DataFrame({"medicine_id": ["MED001"]})
        errors = validate_dataframe(df, MEDICINES_SCHEMA, list(MEDICINES_SCHEMA.keys()), name="medicines")
        assert len(errors) > 0
        assert "missing required columns" in errors[0]

    def test_validate_dataframe_duplicate_pk(self):
        df = pd.DataFrame({
            "medicine_id": ["MED001", "MED001"],
            "name": ["Drug A", "Drug B"],
            "category": ["Antibiotic", "Analgesic"],
            "manufacturer": ["Mfr1", "Mfr2"],
            "unit_selling_price": [10.0, 15.0],
            "reorder_level": [50, 30],
            "unit_of_measure": ["tablet", "capsule"],
        })
        errors = validate_dataframe(df, MEDICINES_SCHEMA, list(MEDICINES_SCHEMA.keys()), name="medicines")
        assert any("duplicate medicine_id" in e for e in errors)

    def test_validate_dataframe_null_values(self):
        df = pd.DataFrame({
            "medicine_id": ["MED001", None],
            "name": ["Drug A", "Drug B"],
            "category": ["Antibiotic", "Analgesic"],
            "manufacturer": ["Mfr1", "Mfr2"],
            "unit_selling_price": [10.0, 15.0],
            "reorder_level": [50, 30],
            "unit_of_measure": ["tablet", "capsule"],
        })
        errors = validate_dataframe(df, MEDICINES_SCHEMA, list(MEDICINES_SCHEMA.keys()), name="medicines")
        assert any("null values" in e for e in errors)

    def test_validate_foreign_keys_valid(self):
        meds = pd.DataFrame({"medicine_id": ["MED001", "MED002"]})
        batches = pd.DataFrame({"medicine_id": ["MED001", "MED002"]})
        sales = pd.DataFrame({"medicine_id": ["MED001"], "batch_id": ["BATCH001"]})
        batches = batches.assign(batch_id=["BATCH001", "BATCH002"])
        errors = validate_foreign_keys(meds, batches, sales)
        assert errors == []

    def test_validate_foreign_keys_invalid(self):
        meds = pd.DataFrame({"medicine_id": ["MED001"]})
        batches = pd.DataFrame({"medicine_id": ["MED999"], "batch_id": ["BATCH001"]})
        sales = pd.DataFrame({"medicine_id": ["MED001"], "batch_id": ["BATCH999"]})
        errors = validate_foreign_keys(meds, batches, sales)
        assert len(errors) >= 2

    def test_validate_business_rules_negative_qty(self):
        meds = pd.DataFrame({
            "medicine_id": ["MED001"],
            "name": ["Drug A"],
            "category": ["Antibiotic"],
            "manufacturer": ["Mfr1"],
            "unit_selling_price": [10.0],
            "reorder_level": [50],
            "unit_of_measure": ["tablet"],
        })
        batches = pd.DataFrame({
            "batch_id": ["BATCH001"],
            "medicine_id": ["MED001"],
            "quantity_received": [100],
            "quantity_remaining": [-10],  # Invalid
            "purchase_cost_per_unit": [5.0],
            "manufacture_date": pd.Timestamp("2024-01-01"),
            "expiry_date": pd.Timestamp("2025-01-01"),
            "supplier_id": ["SUP001"],
            "supplier_name": ["Supplier"],
        })
        sales = pd.DataFrame({
            "sale_id": ["SALE001"],
            "batch_id": ["BATCH001"],
            "medicine_id": ["MED001"],
            "quantity_sold": [5],
            "sale_date": pd.Timestamp("2024-06-01"),
            "selling_price_per_unit": [10.0],
            "payment_method": ["Cash"],
        })
        errors = validate_business_rules(meds, batches, sales)
        assert any("negative" in e for e in errors)

    def test_validate_business_rules_qty_remaining_exceeds_received(self):
        batches = pd.DataFrame({
            "batch_id": ["BATCH001"],
            "medicine_id": ["MED001"],
            "quantity_received": [50],
            "quantity_remaining": [100],  # Invalid
            "purchase_cost_per_unit": [5.0],
            "manufacture_date": pd.Timestamp("2024-01-01"),
            "expiry_date": pd.Timestamp("2025-01-01"),
            "supplier_id": ["SUP001"],
            "supplier_name": ["Supplier"],
        })
        meds = pd.DataFrame({"medicine_id": ["MED001"], "unit_selling_price": [10.0]})
        sales = pd.DataFrame()
        errors = validate_business_rules(meds, batches, sales)
        assert any("quantity_remaining > quantity_received" in e for e in errors)

    def test_validate_business_rules_expiry_before_manufacture(self):
        batches = pd.DataFrame({
            "batch_id": ["BATCH001"],
            "medicine_id": ["MED001"],
            "quantity_received": [100],
            "quantity_remaining": [50],
            "purchase_cost_per_unit": [5.0],
            "manufacture_date": pd.Timestamp("2025-01-01"),
            "expiry_date": pd.Timestamp("2024-01-01"),  # Invalid
            "supplier_id": ["SUP001"],
            "supplier_name": ["Supplier"],
        })
        meds = pd.DataFrame({"medicine_id": ["MED001"], "unit_selling_price": [10.0]})
        sales = pd.DataFrame()
        errors = validate_business_rules(meds, batches, sales)
        assert any("expiry_date < manufacture_date" in e for e in errors)


class TestIngestion:
    """Test data ingestion and cleaning."""

    @pytest.fixture
    def sample_data_dir(self):
        """Create temporary directory with sample data."""
        tmpdir = Path(tempfile.mkdtemp())
        generate_sample_data(tmpdir, n_medicines=10, n_batches=30, n_sales=100)
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_load_medicines(self, sample_data_dir):
        df = load_medicines(sample_data_dir / "medicines.csv")
        assert len(df) == 10
        assert "medicine_id" in df.columns
        assert df["medicine_id"].is_unique

    def test_load_batches(self, sample_data_dir):
        df = load_batches(sample_data_dir / "batches.csv")
        assert len(df) == 30
        assert pd.api.types.is_datetime64_any_dtype(df["manufacture_date"])
        assert pd.api.types.is_datetime64_any_dtype(df["expiry_date"])

    def test_load_sales(self, sample_data_dir):
        df = load_sales(sample_data_dir / "sales.csv")
        assert len(df) == 100
        assert pd.api.types.is_datetime64_any_dtype(df["sale_date"])

    def test_load_all(self, sample_data_dir):
        meds, batches, sales = load_all(sample_data_dir)
        assert len(meds) == 10
        assert len(batches) == 30
        assert len(sales) == 100

    def test_load_all_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_all(Path("/nonexistent"))

    def test_clean_batches_for_analysis(self, sample_data_dir):
        batches = load_batches(sample_data_dir / "batches.csv")
        cleaned = clean_batches_for_analysis(batches)
        assert "is_expired" in cleaned.columns
        assert "days_to_expiry" in cleaned.columns
        assert "saleable_quantity" in cleaned.columns
        assert "batch_value_at_cost" in cleaned.columns
        # Saleable qty should be 0 for expired batches
        expired = cleaned[cleaned["is_expired"]]
        assert (expired["saleable_quantity"] == 0).all()


class TestSampleGeneration:
    """Test synthetic data generation."""

    def test_generate_medicines(self):
        from pharmatrack.data.sample_gen import generate_medicines
        df = generate_medicines(5)
        assert len(df) == 5
        assert all(c in df.columns for c in MEDICINES_SCHEMA.keys())

    def test_generate_batches(self):
        from pharmatrack.data.sample_gen import generate_medicines, generate_batches
        meds = generate_medicines(5)
        batches = generate_batches(meds, 20)
        assert len(batches) == 20
        assert all(c in batches.columns for c in BATCHES_SCHEMA.keys())
        assert batches["medicine_id"].isin(meds["medicine_id"]).all()

    def test_generate_sales(self):
        from pharmatrack.data.sample_gen import generate_medicines, generate_batches, generate_sales
        meds = generate_medicines(5)
        batches = generate_batches(meds, 20)
        sales = generate_sales(meds, batches, 50)
        assert len(sales) <= 50  # May be fewer if no saleable batches
        assert all(c in sales.columns for c in SALES_SCHEMA.keys())
        assert sales["batch_id"].isin(batches["batch_id"]).all()