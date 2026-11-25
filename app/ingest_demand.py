import csv
import datetime as dt
import io
import logging
import math
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from app.db import upsert_demand

URL = "https://api.eia.gov/v2/electricity/rto/daily-region-sub-ba-data/data/"
PAGE_SIZE = 5000
NYISO_DAILY_URL = "https://mis.nyiso.com/public/csv/eiadaily/NYISdaily{date:%j%Y}.csv"
NYISO_HOUR_COLUMNS = tuple(f"HR{hour}" for hour in range(1, 26))
NY_TZ = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


def fetch_demand(
    start: dt.date,
    end: dt.date,
    subba: str = "ZONJ",
    api_key: str | None = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch finite EIA Zone J daily demand values in Eastern time."""
    api_key = api_key or os.environ["EIA_API_KEY"]

    params = {
        "api_key": api_key,
        "frequency": "daily",
        "data[0]": "value",
        "facets[subba][]": subba,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": PAGE_SIZE,
    }

    response = requests.get(URL, params=params, timeout=timeout)
    response.raise_for_status()

    # df = pd.DataFrame(response.json()["response"]["data"])
    records = response.json()["response"]["data"]
    if records == []:
        return pd.DataFrame({
            "date": pd.Series(dtype="datetime64[ns]"),
            "demand": pd.Series(dtype="float64"),
        })
    
    df = pd.DataFrame(records)

    df = df[df["timezone"] == "Eastern"].copy()

    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")

    df = df.loc[np.isfinite(df["value"]), ["period", "value"]].sort_values("period")
    df = df.rename(columns={"period": "date", "value": "demand"})

    df = df.reset_index(drop=True)

    return df


def fetch_nyiso_demand(anchor_date: dt.date, *, timeout: int = 15) -> float | None:
    """Read a complete Zone J day from NYISO's next-day EIA-930 posting."""
    publication_date = anchor_date + dt.timedelta(days=1)
    response = requests.get(NYISO_DAILY_URL.format(date=publication_date), timeout=timeout)
    if response.status_code == 404:
        return None
    response.raise_for_status()

    report = csv.DictReader(io.StringIO(response.content.decode("utf-8-sig")))
    required_columns = {
        "Product_Name", "Posting_Type", "Balancing_Authority_Code",
        "Data_Type", "Data_Code", "Data_Date", *NYISO_HOUR_COLUMNS,
    }
    if not required_columns.issubset(report.fieldnames or []):
        raise ValueError(f"Unexpected NYISO daily report columns for {anchor_date}")

    anchor_code = anchor_date.strftime("%j%Y")
    rows = [
        row for row in report
        if row["Product_Name"] == "EIA-930"
        and row["Posting_Type"] == "Daily"
        and row["Balancing_Authority_Code"] == "NYIS"
        and row["Data_Date"] == anchor_code
    ]
    demand_rows = [row for row in rows if row["Data_Type"] == "D" and row["Data_Code"] == "ZONJ"]
    timestamp_rows = [row for row in rows if row["Data_Type"] == "UTCD"]
    if len(demand_rows) > 1 or len(timestamp_rows) > 1:
        raise ValueError(f"Duplicate NYISO demand or timestamp rows for {anchor_date}")
    if not demand_rows or not timestamp_rows:
        return None

    start = dt.datetime.combine(anchor_date, dt.time(), NY_TZ).astimezone(dt.UTC)
    end = dt.datetime.combine(publication_date, dt.time(), NY_TZ).astimezone(dt.UTC)
    hour = dt.timedelta(hours=1)
    hour_count = (end - start) // hour
    timestamps = [(timestamp_rows[0][column] or "").strip() for column in NYISO_HOUR_COLUMNS]
    loads = [(demand_rows[0][column] or "").strip() for column in NYISO_HOUR_COLUMNS]
    if any(timestamps[hour_count:]) or any(loads[hour_count:]):
        raise ValueError(f"Unexpected extra hours in NYISO demand for {anchor_date}")
    if not all(timestamps[:hour_count]) or not all(loads[:hour_count]):
        return None

    try:
        actual_hours = [dt.datetime.fromisoformat(value) for value in timestamps[:hour_count]]
        hourly_demand = [float(value) for value in loads[:hour_count]]
    except ValueError as exc:
        raise ValueError(f"Malformed NYISO hourly data for {anchor_date}") from exc

    expected_hours = [start + index * hour for index in range(1, hour_count + 1)]
    if actual_hours != expected_hours or not all(map(math.isfinite, hourly_demand)):
        raise ValueError(f"Invalid NYISO hourly data for {anchor_date}")
    return math.fsum(hourly_demand)


def ingest_recent_demand(
    start: dt.date,
    end: dt.date,
    subba: str = "ZONJ",
    api_key: str | None = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch recent EIA demand rows and upsert Eastern-time values."""
    logger.info("Fetching EIA demand: start=%s end=%s", start, end)
    df = fetch_demand(
        start=start,
        end=end,
        subba=subba,
        api_key=api_key,
        timeout=timeout,
    )
    logger.info("Fetched EIA demand: rows=%d", len(df))

    for record in df.itertuples(index=False):
        upsert_demand(
            date=record.date.date(),
            demand=float(record.demand),
            is_imputed=False,
        )

    return df


if __name__ == "__main__":
    today = dt.date.today()
    demand = ingest_recent_demand(today - dt.timedelta(days=10), today)
    print(demand.tail())

