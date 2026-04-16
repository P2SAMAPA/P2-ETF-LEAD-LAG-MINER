"""
US market calendar utilities.
"""
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import pytz


def get_us_calendar():
    """Return NYSE calendar."""
    return mcal.get_calendar("NYSE")


def next_trading_day(date: datetime = None) -> datetime:
    """
    Return the next trading day after given date.
    If date is None, use current UTC time.
    Ensures the returned date is the next market open day in US/Eastern time.
    """
    if date is None:
        date = datetime.utcnow()

    nyse = get_us_calendar()
    # Get schedule for a window around the date
    start = date - timedelta(days=5)
    end = date + timedelta(days=10)
    schedule = nyse.schedule(start_date=start, end_date=end)

    # Convert date to timezone-aware (UTC)
    utc = pytz.UTC
    date_aware = utc.localize(date) if date.tzinfo is None else date

    # Find first trading day strictly after date
    future = schedule.index[schedule.index > date_aware]
    if len(future) > 0:
        return future[0].to_pydatetime()
    # Fallback: add one day and try again (shouldn't happen)
    return date + timedelta(days=1)


def is_trading_day(date: datetime) -> bool:
    """Check if date is a US trading day."""
    nyse = get_us_calendar()
    schedule = nyse.schedule(start_date=date - timedelta(days=5), end_date=date + timedelta(days=5))
    return date in schedule.index
