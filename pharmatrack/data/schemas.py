"""
Data schemas and validation for PharmaTrack CSV files.

Defines expected columns, dtypes, and validation rules for:
- medicines.csv
- batches.csv
- sales.csv
"""

from typing import Any
import pandas as pd


# Expected columns and dtypes for each dataset
MEDICINES_SCHEMA = {
    "medicine_id": "string",
    "name": "string",
    "category": "string",
    "manufacturer": "string",
    "unit_selling_price": "float64",
    "reorder_level": "int64",
    "unit_of_measure": "string",
}

BATCHES_SCHEMA = {
    "batch_id": "string",
    "medicine_id": "string",
    "quantity_received": "int64",
    "quantity_remaining": "int64",
    "purchase_cost_per_unit": "float64",
    "manufacture_date": "datetime64[ns]",
    "expiry_date": "datetime64[ns]",
    "supplier_id": "string",
    "supplier_name": "string",
}

SALES_SCHEMA = {
    "sale_id": "string",
    "batch_id": "string",
    "medicine_id": "string",
    "quantity_sold": "int64",
    "sale_date": "datetime64[ns]",
    "selling_price_per_unit": "float64",
    "payment_method": "string",
}


REQUIRED_COLUMNS = {
    "medicines": list(MEDICINES_SCHEMA.keys()),
    "batches": list(BATCHES_SCHEMA.keys()),
    "sales": list(SALES_SCHEMA.keys()),
}

DATE_COLUMNS = {
    "batches": ["manufacture_date", "expiry_date"],
    "sales": ["sale_date"],
}


class ValidationError(Exception):
    """Raised when data validation fails."""

    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or []


def validate_dataframe(
    df: pd.DataFrame,
    schema: dict[str, str],
    required_cols: list[str],
    date_cols: list[str] | None = None,
    name: str = "DataFrame",
) -> list[str]:
    """
    Validate a DataFrame against expected schema.

    Returns list of error messages (empty if valid).
    """
    errors = []

    # Check required columns
    missing = set(required_cols) - set(df.columns)
    if missing:
        errors.append(f"{name}: missing required columns: {sorted(missing)}")

    # Check for extra columns (warning only)
    extra = set(df.columns) - set(required_cols)
    if extra:
        errors.append(f"{name}: unexpected columns (ignored): {sorted(extra)}")

    # Check dtypes for present columns
    for col, expected_dtype in schema.items():
        if col not in df.columns:
            continue
        actual_dtype = str(df[col].dtype)
        if expected_dtype == "datetime64[ns]":
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                errors.append(
                    f"{name}.{col}: expected datetime, got {actual_dtype}"
                )
        elif expected_dtype in ("int64", "float64"):
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(
                    f"{name}.{col}: expected numeric, got {actual_dtype}"
                )
        elif expected_dtype == "string":
            if not pd.api.types.is_string_dtype(df[col]) and df[col].dtype != "object":
                errors.append(
                    f"{name}.{col}: expected string, got {actual_dtype}"
                )

    # Check for nulls in required columns
    for col in required_cols:
        if col in df.columns and df[col].isna().any():
            errors.append(f"{name}.{col}: contains null values")

    # Check for duplicate primary keys
    if name == "medicines" and "medicine_id" in df.columns:
        dupes = df[df.duplicated(subset=["medicine_id"], keep=False)]
        if not dupes.empty:
            errors.append(
                f"{name}: duplicate medicine_id values: {dupes['medicine_id'].unique()[:10]}"
            )
    if name == "batches" and "batch_id" in df.columns:
        dupes = df[df.duplicated(subset=["batch_id"], keep=False)]
        if not dupes.empty:
            errors.append(
                f"{name}: duplicate batch_id values: {dupes['batch_id'].unique()[:10]}"
            )
    if name == "sales" and "sale_id" in df.columns:
        dupes = df[df.duplicated(subset=["sale_id"], keep=False)]
        if not dupes.empty:
            errors.append(
                f"{name}: duplicate sale_id values: {dupes['sale_id'].unique()[:10]}"
            )

    # Date logical checks
    if date_cols:
        for col in date_cols:
            if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
                if df[col].isna().any():
                    errors.append(f"{name}.{col}: contains unparseable dates")

    return errors


def validate_foreign_keys(
    medicines: pd.DataFrame,
    batches: pd.DataFrame,
    sales: pd.DataFrame,
) -> list[str]:
    """
    Validate foreign key relationships between datasets.

    Returns list of error messages.
    """
    errors = []

    # batches.medicine_id -> medicines.medicine_id
    if "medicine_id" in batches.columns and "medicine_id" in medicines.columns:
        invalid = batches[~batches["medicine_id"].isin(medicines["medicine_id"])]
        if not invalid.empty:
            errors.append(
                f"batches: {len(invalid)} rows reference unknown medicine_id: "
                f"{invalid['medicine_id'].unique()[:10]}"
            )

    # sales.medicine_id -> medicines.medicine_id
    if "medicine_id" in sales.columns and "medicine_id" in medicines.columns:
        invalid = sales[~sales["medicine_id"].isin(medicines["medicine_id"])]
        if not invalid.empty:
            errors.append(
                f"sales: {len(invalid)} rows reference unknown medicine_id: "
                f"{invalid['medicine_id'].unique()[:10]}"
            )

    # sales.batch_id -> batches.batch_id
    if "batch_id" in sales.columns and "batch_id" in batches.columns:
        invalid = sales[~sales["batch_id"].isin(batches["batch_id"])]
        if not invalid.empty:
            errors.append(
                f"sales: {len(invalid)} rows reference unknown batch_id: "
                f"{invalid['batch_id'].unique()[:10]}"
            )

    return errors


def validate_business_rules(
    medicines: pd.DataFrame,
    batches: pd.DataFrame,
    sales: pd.DataFrame,
) -> list[str]:
    """
    Validate business logic rules.

    Returns list of error messages.
    """
    errors = []

    # Non-negative quantities
    for df, cols, name in [
        (batches, ["quantity_received", "quantity_remaining"], "batches"),
        (sales, ["quantity_sold"], "sales"),
    ]:
        for col in cols:
            if col in df.columns:
                neg = df[df[col] < 0]
                if not neg.empty:
                    errors.append(f"{name}.{col}: {len(neg)} rows have negative values")

    # quantity_remaining <= quantity_received
    if "quantity_remaining" in batches.columns and "quantity_received" in batches.columns:
        invalid = batches[batches["quantity_remaining"] > batches["quantity_received"]]
        if not invalid.empty:
            errors.append(
                f"batches: {len(invalid)} rows have quantity_remaining > quantity_received"
            )

    # Expiry date >= manufacture date
    if "expiry_date" in batches.columns and "manufacture_date" in batches.columns:
        invalid = batches[batches["expiry_date"] < batches["manufacture_date"]]
        if not invalid.empty:
            errors.append(
                f"batches: {len(invalid)} rows have expiry_date < manufacture_date"
            )

    # Sale date within reasonable range (not future, not too old)
    if "sale_date" in sales.columns and pd.api.types.is_datetime64_any_dtype(sales["sale_date"]):
        now = pd.Timestamp.now()
        future = sales[sales["sale_date"] > now]
        if not future.empty:
            errors.append(f"sales: {len(future)} rows have future sale_date")

    # Positive prices
    for df, cols, name in [
        (medicines, ["unit_selling_price"], "medicines"),
        (batches, ["purchase_cost_per_unit"], "batches"),
        (sales, ["selling_price_per_unit"], "sales"),
    ]:
        for col in cols:
            if col in df.columns:
                invalid = df[df[col] <= 0]
                if not invalid.empty:
                    errors.append(f"{name}.{col}: {len(invalid)} rows have non-positive values")

    return errors