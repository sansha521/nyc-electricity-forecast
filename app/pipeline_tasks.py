"""Independently callable operations for operational forecasting.

No Airflow imports: these functions can be reused in local commands and tests.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

from app.build_row_inference import DemandDataUnavailable

NY_TZ = ZoneInfo("America/New_York")
HISTORY_DAYS = 400
MODEL_VERSION = "lgbm_v1"


def default_target_date() -> dt.date:
    return dt.datetime.now(NY_TZ).date()


def ingest_latest_eia(target_date: dt.date) -> int:
    """Upsert a full lag window. Idempotent on demand_daily.date."""
    from app.ingest_demand import ingest_recent_demand

    anchor = target_date - dt.timedelta(days=1)
    rows = ingest_recent_demand(anchor - dt.timedelta(days=HISTORY_DAYS), anchor)
    return len(rows)


def fetch_missing_anchor(target_date: dt.date) -> bool:
    """Sensor poke: refresh yesterday's demand if EIA has now published it."""
    from app.db import upsert_demand
    from app.ingest_demand import fetch_demand

    anchor = target_date - dt.timedelta(days=1)
    rows = fetch_demand(anchor, anchor)
    rows = rows[pd.to_datetime(rows["date"]).dt.date == anchor]
    if rows.empty:
        return False
    for rec in rows.itertuples(index=False):
        upsert_demand(date=anchor, demand=float(rec.demand), is_imputed=False)
    return True


def load_demand_history(target_date: dt.date) -> pd.Series:
    """Read the already-ingested demand instead of making a second EIA request."""
    from app.db import get_connection

    anchor = target_date - dt.timedelta(days=1)
    
    start = anchor - dt.timedelta(days=HISTORY_DAYS)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT date, demand, is_imputed FROM demand_daily "
                "WHERE date BETWEEN %s AND %s ORDER BY date",
                (start, anchor),
            )
            rows = cur.fetchall()
    
    data = pd.DataFrame(rows, columns=["date", "demand", "is_imputed"])
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data = data.set_index("date").sort_index()

    anchor_key = pd.Timestamp(anchor)
    if anchor_key not in data.index or data.at[anchor_key, "is_imputed"]:
        raise DemandDataUnavailable(f"EIA demand for {anchor} has not been published")

    demand = data["demand"].astype("float64")
    required_dates = [
        anchor_key - pd.Timedelta(days=lag) for lag in (*range(31), 365, 366)
    ]
    required = demand.reindex(required_dates)
    invalid_dates = required.index[~np.isfinite(required)]
    if len(invalid_dates):
        dates = invalid_dates.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Missing or invalid EIA demand for {dates}")

    return demand

    # data = pd.DataFrame(rows, columns=["date", "demand", "is_imputed"])
    
    # if data.empty:
    #     raise DemandDataUnavailable("No demand rows in Postgres for inference")
    
    # data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    # data = data.set_index("date").sort_index()
    
    # required_dates = {
    #     anchor - dt.timedelta(days=lag)
    #     for lag in (0, 1, 2, 3, 7, 14, 21, 28, 365, 366)
    # }
    
    # required_dates.update(anchor - dt.timedelta(days=lag) for lag in range(1, 31))
    # missing = sorted(required_dates - set(data.index.date))
    
    # if missing:
    #     raise DemandDataUnavailable(f"Missing EIA demand dates: {missing[:8]}")
    
    # # Never use an imputed anchor as the supposedly published D-1 actual.
    # if bool(data.loc[pd.Timestamp(anchor), "is_imputed"]):
    #     raise DemandDataUnavailable(f"EIA anchor {anchor} is still imputed")
    
    # if data.loc[pd.to_datetime(sorted(required_dates)), "demand"].isna().any():
    #     raise DemandDataUnavailable("Required EIA demand values contain NULLs")
    
    # return data["demand"].astype(float)


def validate_data_availability(target_date: dt.date) -> bool:
    """Check Postgres first, then poke EIA for a missing D-1 anchor.

    Other historical gaps are surfaced as failures; a transient D-1 gap yields
    False so the Airflow sensor reschedules instead of marking it successful.
    """
    try:
        load_demand_history(target_date)
        return True
    except DemandDataUnavailable:
        if not fetch_missing_anchor(target_date):
            return False
        load_demand_history(target_date)
        return True


def build_feature_row(target_date: dt.date) -> dict[str, float]:
    """Build features and persist the weather snapshot through existing code."""
    from app.build_row_inference import build_row

    demand = load_demand_history(target_date)
    row = build_row(target_date, demand=demand)
    return {name: float(value) for name, value in row.iloc[0].items()}


def generate_and_store_forecast(target_date: dt.date, features: dict[str, float]) -> float:
    """Validate the saved model's feature order, predict, and upsert."""
    from app.build_row_inference import FEATURES
    from app.db import upsert_prediction
    from app.predict import load_model, predict

    row = pd.DataFrame([features]).reindex(columns=FEATURES)
    if set(features) != set(FEATURES) or row.isna().any().any():
        raise ValueError("Prediction feature schema is incomplete or contains NaN")
    model = load_model()
    forecast = float(predict(model, row))
    upsert_prediction(
        target_date=target_date,
        predicted_demand=forecast,
        model_version=MODEL_VERSION,
        features=features,
    )
    return forecast


def update_actuals_and_scores(as_of_date: dt.date) -> dict[str, object]:
    """Independently callable, refreshes EIA actuals and revisable scores."""
    from app.evaluate_predictions import run_evaluation

    return run_evaluation(today=as_of_date)
