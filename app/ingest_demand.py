import datetime as dt
import os

import pandas as pd
import requests

from app.db import upsert_demand

URL = "https://api.eia.gov/v2/electricity/rto/daily-region-sub-ba-data/data/"
PAGE_SIZE = 5000


def fetch_demand(
    start: dt.date,
    end: dt.date,
    subba: str = "ZONJ",
    api_key: str | None = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch EIA Zone J daily demand and return Eastern-time rows."""
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

    df = pd.DataFrame(response.json()["response"]["data"])

    df = df[df["timezone"] == "Eastern"].copy()

    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"])

    df = df[["period", "value"]].sort_values("period")
    df = df.rename(columns={"period": "date", "value": "demand"})

    df = df.reset_index(drop=True)

    return df


def ingest_recent_demand(
    start: dt.date,
    end: dt.date,
    subba: str = "ZONJ",
    api_key: str | None = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch recent EIA demand rows and upsert Eastern-time values."""
    df = fetch_demand(
        start=start,
        end=end,
        subba=subba,
        api_key=api_key,
        timeout=timeout,
    )

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

