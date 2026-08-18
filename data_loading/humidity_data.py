import pandas as pd
import requests

# Central Park, matching the NOAA station the rest of the weather comes from
URL = "https://archive-api.open-meteo.com/v1/archive"
PARAMS = {
    "latitude": 40.7794,
    "longitude": -73.9692,
    "start_date": "2019-01-01",
    "end_date": "2026-07-22",
    "daily": (
        "relative_humidity_2m_mean,relative_humidity_2m_max,dew_point_2m_mean,"
        "shortwave_radiation_sum,cloud_cover_mean,daylight_duration,sunshine_duration"
    ),
    "temperature_unit": "fahrenheit",
    "timezone": "America/New_York",
}

response = requests.get(URL, params=PARAMS, timeout=60)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])
df = df.rename(columns={
    "time": "date",
    "relative_humidity_2m_mean": "rh_mean",
    "relative_humidity_2m_max": "rh_max",
    "dew_point_2m_mean": "dew_point",
    "shortwave_radiation_sum": "solar_rad",
    "cloud_cover_mean": "cloud_cover",
})

# The API reports both durations in seconds -- hours are easier to read
df["daylight_hours"] = (df["daylight_duration"] / 3600).round(2)
df["sunshine_hours"] = (df["sunshine_duration"] / 3600).round(2)
df = df.drop(columns=["daylight_duration", "sunshine_duration"])

print(df.head)
print(len(df), "rows from", df["date"].min(), "to", df["date"].max())
df.to_csv("humidity.csv", index=False)

# rh_mean, rh_max (%), dew_point (°F)
"""
daylight_hours — sunrise to sunset.
sunshine_hours — the portion of those daylight hours when direct sun
  actually reached the ground rather than being blocked by cloud.
cloud_cover — average share of sky covered by cloud over the day, 0–100%.
solar_rad — total solar energy that actually landed on a horizontal square
  meter over the whole day, in MJ/m². 
"""