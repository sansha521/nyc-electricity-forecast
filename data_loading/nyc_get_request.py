import os

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["EIA_API_KEY"]

url = "https://api.eia.gov/v2/electricity/rto/daily-region-sub-ba-data/data/"

PAGE_SIZE = 5000  # EIA v2 caps a single request at 5000 rows

params = {
    "api_key": API_KEY,
    "frequency": "daily",
    "data[0]": "value",
    "facets[subba][]": "ZONJ",
    "start": "2019-01-01",
    "end": "2026-07-22",
    "sort[0][column]": "period",
    "sort[0][direction]": "desc",
    "offset": 0,
    "length": 5000
}

rows = []
offset = 0
total = None

while True:
    response = requests.get(url, params={**params, "offset": offset})
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

df = df[df["timezone"] == "Eastern"]
df.to_csv("daily_nyc_demand.csv", index=False)

df = df[["period","value",]]
df.to_csv("daily_nyc_demand_cleaned.csv", index=False)

print(df.columns)
print(df.head)
print(df.tail)