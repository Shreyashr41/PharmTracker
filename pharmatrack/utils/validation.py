"""
Validation utilities and exceptions for PharmaTrack.
"""

from typing import Any


class ValidationError(Exception):
    """Base exception for validation errors."""

    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or []


def validate_non_negative(value: Any, field_name: str) -> None:
    """
    Validate that a numeric value is non-negative.

    Raises:
        ValidationError: If value is negative.
    """
    if value is not None and value < 0:
        raise ValidationError(f"{field_name} must be non-negative, got {value}")


def validate_positive(value: Any, field_name: str) -> None:
    """
    Validate that a numeric value is positive (> 0).

    Raises:
        ValidationError: If value is not positive.
    """
    if value is not None and value <= 0:
        raise ValidationError(f"{field_name} must be positive, got {value}")


def validate_foreign_keys(
    df: Any,
    reference_df: Any,
    fk_column: str,
    ref_column: str,
    df_name: str = "DataFrame",
    ref_name: str = "reference",
) -> list[str]:
    """
    Validate foreign key references.

    Returns list of error messages.
    """
    errors = []
    if fk_column not in df.columns:
        errors.append(f"{df_name}: missing foreign key column '{fk_column}'")
        return errors
    if ref_column not in reference_df.columns:
        errors.append(f"{ref_name}: missing reference column '{ref_column}'")
        return errors

    invalid = df[~df[fk_column].isin(reference_df[ref_column])]
    if not invalid.empty:
        errors.append(
            f"{df_name}: {len(invalid)} rows have invalid {fk_column} "
            f"(not found in {ref_name}.{ref_column})"
        )
    return errors