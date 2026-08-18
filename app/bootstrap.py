from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from db import upsert_demand, upsert_weather
from eia import fetch_eia_demand
from vc import fetch_weather

load_dotenv()


DATA_PATH = Path("../all_features_and_target.csv")

EIA_BACKFILL_START = "2026-08-12"
WEATHER_BACKFILL_START = "2026-07-23"

### DEMAND
def load_historical_demand():
    """Load historical demand from all_features_and_target.csv and populate db"""
    df = pd.read_csv(DATA_PATH, parse_dates=["period"])

    # required_columns = {"period", "value", "is_imputed"}

    print(f"Loading {len(df)} historical demand rows...")
    for _, row in df.iterrows():
        upsert_demand(
            date=row["period"].date(),
            demand=float(row["value"]),
            is_imputed=bool(row["is_imputed"]),
        )
    print("Historical demand loaded.")

def backfill_eia():
    end_date = date.today().isoformat()

    df = fetch_eia_demand(
        start_date=EIA_BACKFILL_START,
        end_date=end_date,
    )

    for _, row in df.iterrows():
        upsert_demand(
            date=row["period"],
            demand=float(row["value"]),
            is_imputed=False,
        )

    print(f"Loaded {len(df)} of EIA demad rows.")


### WEATHER
def load_historical_weather():
    """Load historical observed weather from all_features_and_target.csv."""
    df = pd.read_csv(DATA_PATH, parse_dates=["period"])

    print(f"Loading {len(df)} historical weather rows...")

    for _, row in df.iterrows():
        upsert_weather(
            date=row["period"].date(),
            dew=float(row["dew_point"]),
            precip=float(row["prcp"]),
            snow=float(row["snow"]),
            tempmax=float(row["tmax"]),
            tempmin=float(row["tmin"]),
            humidity=float(row["rh_mean"]),
            snowdepth=float(row["snwd"]),
            solarenergy=float(row["solar_rad"]),
            cloudcover=float(row["cloud_cover"]),
        )

    print("Historical weather loaded.")


REQUIRED_WEATHER_COLUMNS = [
    "tempmin",
    "tempmax",
    "dew",
    "humidity",
    "precip",
    "snow",
    "solarenergy",
    "cloudcover",
]

def validate_weather(df: pd.DataFrame):
    missing = df[REQUIRED_WEATHER_COLUMNS].isna()

    if missing.any().any():
        bad = df.index[missing.any(axis=1)].tolist()

        raise ValueError(
            f"Missing required weather values for dates: {bad}"
        )

def backfill_weather():
    """Backfill observed weather after the historical CSV ends."""

    start_date = date.fromisoformat(WEATHER_BACKFILL_START)
    end_date = date.today()

    df = fetch_weather(start_date, end_date)

    print(f"Backfilling {len(df)} weather rows...")

    validate_weather(df)

    for date_, row in df.iterrows():
        upsert_weather(
            date=date_,
            dew=float(row["dew"]),
            precip=float(row["precip"]),
            snow=float(row["snow"]),
            tempmax=float(row["tempmax"]),
            tempmin=float(row["tempmin"]),
            humidity=float(row["humidity"]),
            snowdepth=float(row["snowdepth"]),
            solarenergy=float(row["solarenergy"]),
            cloudcover=float(row["cloudcover"]),
        )

    print(f"Backfilled {len(df)} weather rows.")


def main():
    print("Starting database bootstrap...\n")
    # load_historical_demand()
    # backfill_eia()
    # load_historical_weather()
    # backfill_weather()
    print("\nDatabase bootstrap complete.")


if __name__ == "__main__":
    main()