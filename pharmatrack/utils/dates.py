"""
Date utility functions for PharmaTrack.
"""

from datetime import date, datetime
from typing import Union
import pandas as pd


def parse_date(value: Union[str, date, datetime, pd.Timestamp]) -> pd.Timestamp:
    """
    Parse various date formats to pandas Timestamp.

    Args:
        value: Date as string, date, datetime, or Timestamp.

    Returns:
        pandas Timestamp (normalized to midnight).
    """
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    if isinstance(value, datetime):
        return pd.Timestamp(value).normalize()
    if isinstance(value, date):
        return pd.Timestamp(value)
    if isinstance(value, str):
        return pd.Timestamp(value).normalize()
    raise TypeError(f"Cannot parse date from type {type(value)}")


def days_until_expiry(expiry_date: Union[str, date, datetime, pd.Timestamp]) -> int:
    """
    Calculate days until expiry from today.

    Args:
        expiry_date: Expiry date.

    Returns:
        Positive if future, negative if expired, 0 if today.
    """
    today = pd.Timestamp.now().normalize()
    exp = parse_date(expiry_date)
    return (exp - today).days


def is_expired(expiry_date: Union[str, date, datetime, pd.Timestamp]) -> bool:
    """
    Check if a batch is expired as of today.

    Args:
        expiry_date: Expiry date.

    Returns:
        True if expiry date is before today.
    """
    return days_until_expiry(expiry_date) < 0


def is_near_expiry(
    expiry_date: Union[str, date, datetime, pd.Timestamp],
    threshold_days: int = 30,
) -> bool:
    """
    Check if a batch is near expiry (within threshold days).

    Args:
        expiry_date: Expiry date.
        threshold_days: Days ahead to consider "near expiry" (default 30).

    Returns:
        True if expires within threshold_days (but not yet expired).
    """
    days = days_until_expiry(expiry_date)
    return 0 <= days <= threshold_days


def get_expiry_status(
    expiry_date: Union[str, date, datetime, pd.Timestamp],
    near_threshold: int = 30,
) -> str:
    """
    Get human-readable expiry status.

    Returns: "Expired", "Near Expiry", or "Valid"
    """
    if is_expired(expiry_date):
        return "Expired"
    if is_near_expiry(expiry_date, near_threshold):
        return "Near Expiry"
    return "Valid"