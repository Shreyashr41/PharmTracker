"""
Data ingestion, cleaning, and validation for PharmaTrack.

Loads CSV files, parses dates, validates schema and business rules,
and returns clean DataFrames ready for analysis.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from .schemas import (
    MEDICINES_SCHEMA,
    BATCHES_SCHEMA,
    SALES_SCHEMA,
    REQUIRED_COLUMNS,
    DATE_COLUMNS,
    validate_dataframe,
    validate_foreign_keys,
    validate_business_rules,
    ValidationError,
)


DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def load_medicines(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and validate medicines CSV.

    Args:
        path: Path to medicines.csv. Defaults to data/raw/medicines.csv

    Returns:
        Validated DataFrame with medicine master data.

    Raises:
        ValidationError: If data fails validation.
    """
    if path is None:
        path = DEFAULT_DATA_DIR / "medicines.csv"

    if not path.exists():
        raise FileNotFoundError(f"Medicines file not found: {path}")

    df = pd.read_csv(path, dtype={"medicine_id": "string"})

    # Parse dates if any (none expected for medicines)
    errors = validate_dataframe(
        df, MEDICINES_SCHEMA, REQUIRED_COLUMNS["medicines"], name="medicines"
    )
    if errors:
        raise ValidationError("Medicines validation failed", errors)

    return df


def load_batches(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and validate batches CSV.

    Args:
        path: Path to batches.csv. Defaults to data/raw/batches.csv

    Returns:
        Validated DataFrame with batch-level inventory data.

    Raises:
        ValidationError: If data fails validation.
    """
    if path is None:
        path = DEFAULT_DATA_DIR / "batches.csv"

    if not path.exists():
        raise FileNotFoundError(f"Batches file not found: {path}")

    df = pd.read_csv(
        path,
        dtype={"batch_id": "string", "medicine_id": "string", "supplier_id": "string"},
        parse_dates=DATE_COLUMNS["batches"],
    )

    errors = validate_dataframe(
        df, BATCHES_SCHEMA, REQUIRED_COLUMNS["batches"], DATE_COLUMNS["batches"], name="batches"
    )
    if errors:
        raise ValidationError("Batches validation failed", errors)

    return df


def load_sales(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and validate sales CSV.

    Args:
        path: Path to sales.csv. Defaults to data/raw/sales.csv

    Returns:
        Validated DataFrame with sales transactions.

    Raises:
        ValidationError: If data fails validation.
    """
    if path is None:
        path = DEFAULT_DATA_DIR / "sales.csv"

    if not path.exists():
        raise FileNotFoundError(f"Sales file not found: {path}")

    df = pd.read_csv(
        path,
        dtype={"sale_id": "string", "batch_id": "string", "medicine_id": "string"},
        parse_dates=DATE_COLUMNS["sales"],
    )

    errors = validate_dataframe(
        df, SALES_SCHEMA, REQUIRED_COLUMNS["sales"], DATE_COLUMNS["sales"], name="sales"
    )
    if errors:
        raise ValidationError("Sales validation failed", errors)

    return df


def validate_all(
    medicines: pd.DataFrame,
    batches: pd.DataFrame,
    sales: pd.DataFrame,
) -> list[str]:
    """
    Run all cross-dataset validations.

    Returns:
        List of error messages (empty if all valid).
    """
    errors = []
    errors.extend(validate_foreign_keys(medicines, batches, sales))
    errors.extend(validate_business_rules(medicines, batches, sales))
    return errors


def load_all(
    data_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three datasets and run cross-validation.

    Args:
        data_dir: Directory containing medicines.csv, batches.csv, sales.csv

    Returns:
        Tuple of (medicines, batches, sales) DataFrames.

    Raises:
        ValidationError: If any validation fails.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    medicines = load_medicines(data_dir / "medicines.csv")
    batches = load_batches(data_dir / "batches.csv")
    sales = load_sales(data_dir / "sales.csv")

    errors = validate_all(medicines, batches, sales)
    if errors:
        raise ValidationError("Cross-dataset validation failed", errors)

    return medicines, batches, sales


def clean_batches_for_analysis(batches: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare batches DataFrame for analysis by adding computed fields.

    - saleable_quantity: quantity_remaining for non-expired batches, 0 for expired
    - is_expired: boolean flag
    - days_to_expiry: days until expiry (negative if expired)
    """
    from ..utils.dates import is_expired, days_until_expiry

    df = batches.copy()
    df["is_expired"] = df["expiry_date"].apply(is_expired)
    df["days_to_expiry"] = df["expiry_date"].apply(days_until_expiry)
    df["saleable_quantity"] = df.apply(
        lambda row: 0 if row["is_expired"] else row["quantity_remaining"], axis=1
    )
    df["batch_value_at_cost"] = df["saleable_quantity"] * df["purchase_cost_per_unit"]
    df["batch_value_at_price"] = df["saleable_quantity"] * df.get(
        "unit_selling_price", df["purchase_cost_per_unit"]
    )
    return df