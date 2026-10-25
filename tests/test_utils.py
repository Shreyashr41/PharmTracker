"""
Test suite for PharmaTrack utility functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from pharmatrack.utils.dates import (
    parse_date,
    days_until_expiry,
    is_expired,
    is_near_expiry,
    get_expiry_status,
)
from pharmatrack.utils.validation import (
    ValidationError,
    validate_non_negative,
    validate_positive,
    validate_foreign_keys,
)


class TestDateUtils:
    """Test date utility functions."""

    def test_parse_date_string(self):
        result = parse_date("2024-06-15")
        assert isinstance(result, pd.Timestamp)
        assert result.date() == date(2024, 6, 15)

    def test_parse_date_timestamp(self):
        ts = pd.Timestamp("2024-06-15")
        result = parse_date(ts)
        assert result == ts.normalize()

    def test_parse_date_datetime(self):
        from datetime import datetime
        dt = datetime(2024, 6, 15, 14, 30)
        result = parse_date(dt)
        assert result == pd.Timestamp("2024-06-15")

    def test_parse_date_date(self):
        result = parse_date(date(2024, 6, 15))
        assert result == pd.Timestamp("2024-06-15")

    def test_parse_date_invalid_type(self):
        with pytest.raises(TypeError):
            parse_date(12345)

    def test_days_until_expiry(self):
        today = pd.Timestamp.now().normalize()
        future = today + timedelta(days=30)
        past = today - timedelta(days=10)
        assert days_until_expiry(future) == 30
        assert days_until_expiry(today) == 0
        assert days_until_expiry(past) == -10

    def test_is_expired(self):
        today = pd.Timestamp.now().normalize()
        assert is_expired(today - timedelta(days=1)) is True
        assert is_expired(today) is False
        assert is_expired(today + timedelta(days=1)) is False

    def test_is_near_expiry(self):
        today = pd.Timestamp.now().normalize()
        assert is_near_expiry(today + timedelta(days=15), threshold_days=30) is True
        assert is_near_expiry(today + timedelta(days=45), threshold_days=30) is False
        assert is_near_expiry(today - timedelta(days=5), threshold_days=30) is False

    def test_get_expiry_status(self):
        today = pd.Timestamp.now().normalize()
        assert get_expiry_status(today - timedelta(days=5)) == "Expired"
        assert get_expiry_status(today + timedelta(days=15), near_threshold=30) == "Near Expiry"
        assert get_expiry_status(today + timedelta(days=60), near_threshold=30) == "Valid"


class TestValidationUtils:
    """Test validation utility functions."""

    def test_validate_non_negative_pass(self):
        validate_non_negative(0, "test_field")
        validate_non_negative(10, "test_field")
        validate_non_negative(0.0, "test_field")

    def test_validate_non_negative_fail(self):
        with pytest.raises(ValidationError):
            validate_non_negative(-1, "test_field")
        with pytest.raises(ValidationError):
            validate_non_negative(-0.5, "test_field")

    def test_validate_positive_pass(self):
        validate_positive(1, "test_field")
        validate_positive(0.1, "test_field")

    def test_validate_positive_fail(self):
        with pytest.raises(ValidationError):
            validate_positive(0, "test_field")
        with pytest.raises(ValidationError):
            validate_positive(-1, "test_field")

    def test_validate_foreign_keys_valid(self):
        df = pd.DataFrame({"fk": ["A", "B", "C"]})
        ref = pd.DataFrame({"pk": ["A", "B", "C", "D"]})
        errors = validate_foreign_keys(df, ref, "fk", "pk")
        assert errors == []

    def test_validate_foreign_keys_invalid(self):
        df = pd.DataFrame({"fk": ["A", "B", "X"]})
        ref = pd.DataFrame({"pk": ["A", "B", "C"]})
        errors = validate_foreign_keys(df, ref, "fk", "pk")
        assert len(errors) == 1
        assert "invalid fk" in errors[0]

    def test_validate_foreign_keys_missing_columns(self):
        df = pd.DataFrame({"other": [1, 2]})
        ref = pd.DataFrame({"pk": ["A"]})
        errors = validate_foreign_keys(df, ref, "fk", "pk")
        assert any("missing foreign key column" in e for e in errors)

        errors = validate_foreign_keys(df, ref, "other", "missing")
        assert any("missing reference column" in e for e in errors)

    def test_validation_error_contains_errors(self):
        try:
            raise ValidationError("Test error", ["error1", "error2"])
        except ValidationError as e:
            assert e.errors == ["error1", "error2"]
            assert str(e) == "Test error"