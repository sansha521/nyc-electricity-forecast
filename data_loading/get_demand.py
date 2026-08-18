import os

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["EIA_API_KEY"]

url = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"

PAGE_SIZE = 5000  # EIA v2 caps a single request at 5000 rows

params = {
    "api_key": API_KEY,
    "frequency": "daily",
    "data[0]": "value",
    "facets[respondent][]": "NY",
    "start": "2025-12-06",
    "end": "2025-12-31",
    "sort[0][column]": "period",
    "sort[0][direction]": "desc",
    "offset": 0,
    "length": PAGE_SIZE,
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
# print(df.head(10))
print(df.shape)

df = df[df["timezone"] == "Eastern"]
# print(df.head(10))
# print(df.shape)
# print(df["type"].unique())
# df.to_csv("daily_eastern_data.csv", index=False)

df = df[df["type"] == "D"]
# print(df.head(10))
# print(df.shape)
# df.to_csv("daily_eastern_demand.csv", index=False)

print(df.head(10))
print(df.tail())

df = pd.read_csv("daily_eastern_demand.csv")
df = df[["period","value",]]
df.to_csv("daily_demand_fill_missing.csv", index=False)