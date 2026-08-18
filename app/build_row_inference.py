import datetime as dt
import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path

import holidays
import pandas as pd
from dotenv import load_dotenv

from app.db import upsert_forecast
from app.ingest_demand import fetch_demand

load_dotenv()

VC_API_KEY = os.environ["VISUAL_CROSSING_API_KEY"]

logger = logging.getLogger(__name__)


class DemandDataUnavailable(RuntimeError):
    """Raised when EIA has not published the demand needed for inference."""


# Visual Crossing weather data constants.
LOCATION = "40.7794,-73.9692"  # Central Park, matching the training data.
BASE_URL = ("https://weather.visualcrossing.com/VisualCrossingWebServices"
            "/rest/services/timeline/")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_META_PATH = PROJECT_ROOT / "fusion" / "models" / "lgbm_v1.meta.json"

with open(MODEL_META_PATH) as f:
    meta = json.load(f)

FEATURES = meta["features"]

HDD_BASE, CDD_BASE = 50.0, 65.0  # Fitted bases, not the conventional 65/65.

VC_ELEMENTS = (
    "datetime,tempmax,tempmin,dew,humidity,precip,snow,snowdepth,solarenergy,solarradiation,cloudcover,sunriseEpoch,sunsetEpoch"
)


def _fetch_visual_crossing_days(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Fetch daily Visual Crossing weather rows over an inclusive date range."""
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
    out["tmin"] = w["tempmin"]
    out["tmax"] = w["tempmax"]
    out["tavg"] = (out["tmin"] + out["tmax"]) / 2

    return out


def fetch_target_forecast(target_date: dt.date) -> pd.Series:
    url = (
        f"{BASE_URL}{urllib.parse.quote(LOCATION)}/{target_date}"
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

    forecast = w.loc[target_date]

    tmin = forecast["tempmin"]
    tmax = forecast["tempmax"]
    tavg_fc = (tmin + tmax) / 2
    hdd_fc = max(HDD_BASE - tavg_fc, 0)
    cdd_fc = max(tavg_fc - CDD_BASE, 0)

    # round values
    tavg_fc = round(tavg_fc, 1)
    hdd_fc = round(hdd_fc, 1)
    cdd_fc = round(cdd_fc, 1)

    return pd.Series({
        "tmin_fc": tmin,
        "tmax_fc": tmax,
        "tavg_fc": tavg_fc,
        "dew_point_fc": forecast["dew"],
        "rh_mean_fc": forecast["humidity"],
        "prcp_fc": forecast["precip"],
        "snow_fc": forecast["snow"],
        "snwd_fc": forecast["snowdepth"] if pd.notna(forecast["snowdepth"]) else 0.0,
        "solar_rad_fc": forecast["solarenergy"],
        "cloud_cover_fc": forecast["cloudcover"],
        "hdd_fc": hdd_fc,
        "cdd_fc": cdd_fc,
        "daylight_hours_next": round(
            (forecast["sunsetEpoch"] - forecast["sunriseEpoch"]) / 3600.0,
            2,
        ),
    })


def build_3day_forecast_features(
    target_date: dt.date,
    forecast_tavg: float,
) -> dict:
    d_minus_2 = target_date - dt.timedelta(days=2)
    d_minus_1 = target_date - dt.timedelta(days=1)
    observed = _fetch_visual_crossing_days(d_minus_2, d_minus_1)

    # D-2 + D-1 + forecast(D)
    tavg_3day_fc = (
        observed.loc[d_minus_2, "tavg"]
        + observed.loc[d_minus_1, "tavg"]
        + forecast_tavg
    ) / 3

    cdd_3day_fc = max(tavg_3day_fc - CDD_BASE, 0)
    hdd_3day_fc = max(HDD_BASE - tavg_3day_fc, 0)

    # Round values after deriving degree-day features.
    tavg_3day_fc = round(tavg_3day_fc, 1)
    cdd_3day_fc = round(cdd_3day_fc, 1)
    hdd_3day_fc = round(hdd_3day_fc, 1)

    return {
        "tavg_3day_fc": tavg_3day_fc,
        "cdd_3day_fc": cdd_3day_fc,
        "hdd_3day_fc": hdd_3day_fc,
    }


def fetch_recent_demand(target_date: dt.date) -> pd.Series:
    """Fetch the demand history needed to build lag and rolling features."""
    target_date = pd.Timestamp(target_date)
    anchor_date = target_date - dt.timedelta(days=1)

    start = anchor_date - dt.timedelta(days=400)
    end = anchor_date

    df = fetch_demand(
        start.date(),
        end.date(),
    )

    demand = (
        df.assign(date=pd.to_datetime(df["date"]).dt.normalize())
        .set_index("date")["demand"]
        .sort_index()
    )

    latest = demand.index.max()
    logger.info("Latest EIA demand date available for inference: %s", latest.date())

    if anchor_date not in demand.index:
        latest_text = latest.date() if pd.notna(latest) else "none"
        raise DemandDataUnavailable(
            f"EIA demand for {anchor_date.date()} is not available. "
            f"Latest available demand: {latest_text}"
        )

    return demand


def demand_features(demand: pd.Series, target_date: dt.date) -> dict:
    target_date = pd.Timestamp(target_date)
    anchor_date = target_date - dt.timedelta(days=1)

    features = {}

    for lag in [0, 1, 2, 3, 7, 14, 21, 28, 365, 366]:
        date = anchor_date - pd.Timedelta(days=lag)
        features[f"lag_{lag}"] = demand.loc[date]

    features["rolling_7"] = demand.loc[
        anchor_date - pd.Timedelta(days=7):
        anchor_date - pd.Timedelta(days=1)
    ].mean()

    features["rolling_30"] = demand.loc[
        anchor_date - pd.Timedelta(days=30):
        anchor_date - pd.Timedelta(days=1)
    ].mean()

    features["rolling_7"] = round(features["rolling_7"], 1)
    features["rolling_30"] = round(features["rolling_30"], 1)

    return features


def build_calendar_features(target_date: dt.date) -> dict:
    """Build calendar flags for the target date."""
    target_timestamp = pd.Timestamp(target_date)

    return {
        "is_weekend_next": int(target_timestamp.weekday() >= 5),
        "is_holiday_next": int(
            target_timestamp in holidays.US(years=[target_timestamp.year])
        ),
    }


def build_row(target_date: dt.date) -> pd.DataFrame:
    """Build one inference row in the exact feature order expected by the model."""
    recent_demand = fetch_recent_demand(target_date)
    demand_feats = demand_features(recent_demand, target_date)

    forecast_feats = fetch_target_forecast(target_date)

    upsert_forecast(
        target_date=target_date,
        dew=forecast_feats["dew_point_fc"],
        precip=forecast_feats["prcp_fc"],
        snow=forecast_feats["snow_fc"],
        tempmax=forecast_feats["tmax_fc"],
        tempmin=forecast_feats["tmin_fc"],
        humidity=forecast_feats["rh_mean_fc"],
        snowdepth=forecast_feats["snwd_fc"],
        solarenergy=forecast_feats["solar_rad_fc"],
        cloudcover=forecast_feats["cloud_cover_fc"],
    )

    three_day_feats = build_3day_forecast_features(
        target_date,
        forecast_feats["tavg_fc"],
    )

    calendar_feats = build_calendar_features(target_date)

    features = {
        **demand_feats,
        **forecast_feats,
        **three_day_feats,
        **calendar_feats,
    }

    row = pd.DataFrame([features], columns=FEATURES)

    assert list(row.columns) == FEATURES, (
        f"Feature order mismatch:\n"
        f"Expected: {FEATURES}\n"
        f"Got: {list(row.columns)}"
    )

    assert not row.isna().any().any(), (
        f"Missing inference features: "
        f"{row.columns[row.isna().any()].tolist()}"
    )

    return row


def main():
    target_date = dt.datetime.now().date()
    row = build_row(target_date)
    pd.set_option("display.max_columns", None)
    print(row)
    print("\nFeature count:", len(row.columns))


if __name__ == "__main__":
    main()
