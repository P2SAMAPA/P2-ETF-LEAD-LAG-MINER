"""
US market calendar utilities.
"""
import pandas_market_calendars as mcal
from datetime import datetime, timedelta


def get_us_calendar():
    """Return NYSE calendar."""
    return mcal.get_calendar("NYSE")


def next_trading_day(date: datetime = None) -> datetime:
    """
    Return the next trading day after given date.
    If date is None, use today.
    """
    if date is None:
        date = datetime.now()
    nyse = get_us_calendar()
    schedule = nyse.schedule(start_date=date, end_date=date + timedelta(days=10))
    if schedule.empty:
        return date + timedelta(days=1)
    # Find first trading day after date
    future_dates = schedule.index[schedule.index > date]
    if len(future_dates) > 0:
        return future_dates[0].to_pydatetime()
    return date + timedelta(days=1)


def is_trading_day(date: datetime) -> bool:
    """Check if date is a US trading day."""
    nyse = get_us_calendar()
    schedule = nyse.schedule(start_date=date - timedelta(days=5), end_date=date + timedelta(days=5))
    return date in schedule.index
