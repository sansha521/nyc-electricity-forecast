import datetime as dt
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd


VC_API_KEY = os.environ["VISUAL_CROSSING_API_KEY"]

# VC weather data constants
LOCATION = "40.7794,-73.9692"               # Central Park -- matches training data
BASE_URL = ("https://weather.visualcrossing.com/VisualCrossingWebServices"
            "/rest/services/timeline/")


DEMAND_CSV = Path("../all_features_and_target.csv")
SCHEMA_CSV = Path("../fusion/inference_feature_splits/X_train.csv")

VC_ELEMENTS = (
    "datetime,"
    "tempmax,tempmin,dew,humidity,precip,snow,"
    "snowdepth,solarenergy,cloudcover,"
    "sunriseEpoch,sunsetEpoch"
)

def fetch_weather(start: dt.date, end: dt.date) -> pd.DataFrame:
    url = (
        f"{BASE_URL}{urllib.parse.quote(LOCATION)}/{start}/{end}"
        f"?unitGroup=us"
        f"&contentType=json"
        f"&include=days"
        f"&elements={urllib.parse.quote(VC_ELEMENTS)}"
        f"&key={VC_API_KEY}"
    )

    with urllib.request.urlopen(url, timeout=60) as r:
        payload = json.loads(r.read())

    w = pd.DataFrame(payload["days"])
    w["date"] = pd.to_datetime(w["datetime"]).dt.date
    w = w.set_index("date")

    out = pd.DataFrame(index=w.index)

    out["dew"] = w["dew"]
    out["precip"] = w["precip"]
    out["snow"] = w["snow"]
    out["tempmax"] = w["tempmax"]
    out["tempmin"] = w["tempmin"]
    out["humidity"] = w["humidity"]
    out["snowdepth"] = w["snowdepth"].fillna(0.0)
    out["solarenergy"] = w["solarenergy"]
    out["cloudcover"] = w["cloudcover"]
    out["sunrise"] = w["sunriseEpoch"]
    out["sunset"] = w["sunsetEpoch"]

    return out

def fetch_tomorrow_forecast(today: dt.date) -> pd.Series:
    """Fetch tomorrow's weather forecast from Visual Crossing."""

    tomorrow = today + dt.timedelta(days=1)

    url = (
        f"{BASE_URL}{urllib.parse.quote(LOCATION)}/{tomorrow}"
        f"?unitGroup=us"
        f"&contentType=json"
        f"&include=days"
        f"&elements={urllib.parse.quote(VC_ELEMENTS)}"
        f"&key={VC_API_KEY}"
    )

    with urllib.request.urlopen(url, timeout=60) as r:
        payload = json.loads(r.read())

    days = payload["days"]

    if not days:
        raise ValueError("Visual Crossing returned no forecast data.")

    w = days[0]

    # Make sure VC actually returned tomorrow
    returned_date = dt.date.fromisoformat(w["datetime"])

    if returned_date != tomorrow:
        raise ValueError(
            f"Expected forecast for {tomorrow}, got {returned_date}"
        )

    return pd.Series({
        "target_date": tomorrow,
        "tmin_fc": w["tempmin"],
        "tmax_fc": w["tempmax"],
        "dew_point_fc": w["dew"],
        "rh_mean_fc": w["humidity"],
        "prcp_fc": w["precip"],
        "snow_fc": w["snow"],
        "snwd_fc": w["snowdepth"] if w["snowdepth"] is not None else 0.0,
        "solar_rad_fc": w["solarenergy"],
        "cloud_cover_fc": w["cloudcover"],
        "daylight_hours_next": (
            w["sunset"] - w["sunrise"]
        ) / 3600.0,
    })