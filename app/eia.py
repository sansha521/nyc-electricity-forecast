import os

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["EIA_API_KEY"]

url = "https://api.eia.gov/v2/electricity/rto/daily-region-sub-ba-data/data/"

PAGE_SIZE = 5000  # EIA v2 caps a single request at 5000 rows


def fetch_eia_demand(start_date, end_date):
    """
    Fetch daily electricity demand for ZONJ from the EIA API.

    Returns a DataFrame with:
        date
        demand
    """
    params = {
        "api_key": API_KEY,
        "frequency": "daily",
        "data[0]": "value",
        "facets[subba][]": "ZONJ",
        "start": start_date,
        "end": end_date,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": PAGE_SIZE
    }

    rows = []
    offset = 0
    total = None

    while True:
        response = requests.get(url, params={**params, "offset": offset}, timeout=30)
        response.raise_for_status()

        payload = response.json()["response"]
        page = payload["data"]

        if total is None:
            total = int(payload["total"])
            print(f"total rows available: {total}")

        rows.extend(page)
        print(f"fetched {len(rows)} / {total}")

        if not page or len(rows) >= total:
            break

        offset += PAGE_SIZE

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["period", "value"])

    df = df[["period", "value"]]

    return df

