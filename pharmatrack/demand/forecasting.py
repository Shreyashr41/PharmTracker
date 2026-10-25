"""
Demand forecasting for PharmaTrack.

Provides simple statistical forecasting methods for demand analysis.
"""

from typing import Optional
import pandas as pd
import numpy as np


def compute_moving_average(
    series: pd.Series,
    window: int = 7,
    min_periods: int = 1,
) -> pd.Series:
    """
    Compute simple moving average.

    Args:
        series: Time series data.
        window: Rolling window size.
        min_periods: Minimum observations required.

    Returns:
        Series with moving average.
    """
    return series.rolling(window=window, min_periods=min_periods).mean()


def compute_exponential_smoothing(
    series: pd.Series,
    alpha: float = 0.3,
) -> pd.Series:
    """
    Compute simple exponential smoothing.

    Args:
        series: Time series data.
        alpha: Smoothing factor (0 < alpha <= 1).

    Returns:
        Series with smoothed values.
    """
    return series.ewm(alpha=alpha, adjust=False).mean()


def compute_weekly_seasonality(
    daily_series: pd.Series,
) -> pd.Series:
    """
    Compute average demand by day of week.

    Args:
        daily_series: Daily time series with DatetimeIndex.

    Returns:
        Series indexed by day of week (0=Monday) with average demand.
    """
    if not isinstance(daily_series.index, pd.DatetimeIndex):
        raise ValueError("Series must have DatetimeIndex")
    return daily_series.groupby(daily_series.index.dayofweek).mean()


class DemandForecaster:
    """
    Demand forecasting using simple statistical methods.

    Supports:
    - Moving average
    - Exponential smoothing
    - Seasonal naive (weekly pattern)
    """

    def __init__(
        self,
        sales: pd.DataFrame,
        date_col: str = "sale_date",
        qty_col: str = "quantity_sold",
        medicine_col: str = "medicine_id",
    ):
        """
        Initialize with sales data.

        Args:
            sales: Sales transactions DataFrame.
            date_col: Column name for sale date.
            qty_col: Column name for quantity sold.
            medicine_col: Column name for medicine ID.
        """
        self.sales = sales.copy()
        self.date_col = date_col
        self.qty_col = qty_col
        self.medicine_col = medicine_col

        # Ensure date column is datetime
        self.sales[date_col] = pd.to_datetime(self.sales[date_col])

    def get_daily_demand(
        self,
        medicine_id: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        """
        Get daily demand time series for a medicine.

        Returns:
            Series indexed by date with daily quantity sold.
        """
        df = self.sales[self.sales[self.medicine_col] == medicine_id].copy()

        if start_date:
            df = df[df[self.date_col] >= start_date]
        if end_date:
            df = df[df[self.date_col] <= end_date]

        if df.empty:
            return pd.Series(dtype=float)

        daily = df.groupby(self.date_col)[self.qty_col].sum()

        # Reindex to fill missing dates with 0
        if not daily.empty:
            full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
            daily = daily.reindex(full_range, fill_value=0)

        return daily

    def forecast_moving_average(
        self,
        medicine_id: str,
        window: int = 7,
        forecast_days: int = 7,
    ) -> dict:
        """
        Forecast using moving average.

        Returns:
            Dict with forecast values, model info.
        """
        daily = self.get_daily_demand(medicine_id)
        if daily.empty:
            return {"forecast": [], "method": "moving_average", "error": "No data"}

        ma = compute_moving_average(daily, window=window)
        last_ma = ma.iloc[-1] if not ma.empty else 0

        forecast = [float(last_ma)] * forecast_days
        future_dates = pd.date_range(
            daily.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D",
        )

        return {
            "forecast": list(zip(future_dates.strftime("%Y-%m-%d"), forecast)),
            "method": f"moving_average_{window}d",
            "last_observed": float(daily.iloc[-1]),
            "last_ma": float(last_ma),
            "historical_mean": float(daily.mean()),
            "historical_std": float(daily.std()),
        }

    def forecast_exponential_smoothing(
        self,
        medicine_id: str,
        alpha: float = 0.3,
        forecast_days: int = 7,
    ) -> dict:
        """
        Forecast using exponential smoothing.
        """
        daily = self.get_daily_demand(medicine_id)
        if daily.empty:
            return {"forecast": [], "method": "exponential_smoothing", "error": "No data"}

        ses = compute_exponential_smoothing(daily, alpha=alpha)
        last_ses = ses.iloc[-1] if not ses.empty else 0

        forecast = [float(last_ses)] * forecast_days
        future_dates = pd.date_range(
            daily.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D",
        )

        return {
            "forecast": list(zip(future_dates.strftime("%Y-%m-%d"), forecast)),
            "method": f"exponential_smoothing_alpha_{alpha}",
            "last_observed": float(daily.iloc[-1]),
            "last_ses": float(last_ses),
        }

    def forecast_seasonal_naive(
        self,
        medicine_id: str,
        forecast_days: int = 7,
    ) -> dict:
        """
        Forecast using weekly seasonal naive (same day last week).
        """
        daily = self.get_daily_demand(medicine_id)
        if daily.empty:
            return {"forecast": [], "method": "seasonal_naive", "error": "No data"}

        if len(daily) < 7:
            # Fallback to mean
            mean_val = daily.mean()
            forecast = [float(mean_val)] * forecast_days
        else:
            # Use same weekday from previous week
            last_date = daily.index[-1]
            forecast = []
            for i in range(1, forecast_days + 1):
                target_date = last_date + pd.Timedelta(days=i)
                same_weekday_last_week = target_date - pd.Timedelta(days=7)
                if same_weekday_last_week in daily.index:
                    forecast.append(float(daily[same_weekday_last_week]))
                else:
                    forecast.append(float(daily.mean()))

        future_dates = pd.date_range(
            daily.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D",
        )

        return {
            "forecast": list(zip(future_dates.strftime("%Y-%m-%d"), forecast)),
            "method": "seasonal_naive_weekly",
        }

    def get_demand_summary(self, medicine_id: str, days: int = 30) -> dict:
        """
        Get demand summary statistics for a medicine.

        Args:
            medicine_id: Medicine to analyze.
            days: Lookback period in days.

        Returns:
            Dict with demand statistics.
        """
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
        daily = self.get_daily_demand(medicine_id, start_date=cutoff)

        if daily.empty:
            return {
                "medicine_id": medicine_id,
                "period_days": days,
                "total_sold": 0,
                "avg_daily": 0,
                "max_daily": 0,
                "zero_demand_days": days,
                "coefficient_of_variation": 0,
            }

        return {
            "medicine_id": medicine_id,
            "period_days": days,
            "total_sold": int(daily.sum()),
            "avg_daily": float(daily.mean()),
            "max_daily": float(daily.max()),
            "min_daily": float(daily.min()),
            "zero_demand_days": int((daily == 0).sum()),
            "coefficient_of_variation": (
                float(daily.std() / daily.mean()) if daily.mean() > 0 else 0
            ),
            "weekly_pattern": compute_weekly_seasonality(daily).to_dict(),
        }