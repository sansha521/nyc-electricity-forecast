"""Independently callable operations for operational forecasting.

No Airflow imports: these functions can be reused in local commands and tests.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

from app.build_row_inference import DemandDataUnavailable

NY_TZ = ZoneInfo("America/New_York")
HISTORY_DAYS = 400
MODEL_VERSION = "lgbm_v1"
logger = logging.getLogger(__name__)


def default_target_date() -> dt.date:
    return dt.datetime.now(NY_TZ).date()


def ingest_latest_eia(target_date: dt.date) -> int:
    """Upsert a full lag window. Idempotent on demand_daily.date."""
    from app.ingest_demand import ingest_recent_demand

    anchor = target_date - dt.timedelta(days=1)
    rows = ingest_recent_demand(anchor - dt.timedelta(days=HISTORY_DAYS), anchor)
    return len(rows)


def nyiso_fallback_enabled() -> bool:
    return os.environ.get("ENABLE_NYISO_DEMAND_FALLBACK") == "1"


def fetch_missing_anchor(target_date: dt.date) -> bool:
    """Refresh yesterday's demand from EIA."""
    from app.db import upsert_demand
    from app.ingest_demand import fetch_demand

    anchor = target_date - dt.timedelta(days=1)
    rows = fetch_demand(anchor, anchor)
    rows = rows[pd.to_datetime(rows["date"]).dt.date == anchor]
    for rec in rows.itertuples(index=False):
        upsert_demand(date=anchor, demand=float(rec.demand), is_imputed=False)
    return not rows.empty


def _read_demand_history(target_date: dt.date) -> pd.DataFrame:
    from app.db import get_connection

    anchor = target_date - dt.timedelta(days=1)
    start = anchor - dt.timedelta(days=HISTORY_DAYS)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT date, demand, is_imputed
                FROM demand_daily
                WHERE date BETWEEN %s AND %s AND (%s OR source = 'eia')
                ORDER BY date
                """,
                (start, anchor, nyiso_fallback_enabled()),
            )
            rows = cur.fetchall()

    data = pd.DataFrame(rows, columns=["date", "demand", "is_imputed"])
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    return data.set_index("date").sort_index()


def _demand_gaps(data: pd.DataFrame, anchor: pd.Timestamp) -> pd.DataFrame:
    required_dates = [
        anchor - pd.Timedelta(days=lag) for lag in (*range(31), 365, 366)
    ]
    required = data.reindex(required_dates)
    missing = ~np.isfinite(required["demand"].astype("float64"))
    if required["is_imputed"].eq(True).at[anchor]:
        missing.at[anchor] = True
    return required.loc[missing]


def _missing_demand_error(gaps: pd.DataFrame) -> ValueError:
    dates = ", ".join(gaps.index.sort_values().strftime("%Y-%m-%d"))
    return ValueError(f"Missing or invalid demand for required dates: {dates}")


def load_demand_history(target_date: dt.date) -> pd.Series:
    """Read and validate already-ingested demand from Postgres."""
    data = _read_demand_history(target_date)
    anchor = pd.Timestamp(target_date) - pd.Timedelta(days=1)
    gaps = _demand_gaps(data, anchor)
    if anchor in gaps.index:
        raise DemandDataUnavailable(f"Observed demand for {anchor.date()} is not available")
    if not gaps.empty:
        raise _missing_demand_error(gaps)
    return data["demand"].astype("float64")


def validate_data_availability(target_date: dt.date) -> bool:
    """Recover required demand gaps and return False for a pending EIA-only anchor."""
    from app.db import upsert_demand
    from app.ingest_demand import fetch_nyiso_demand

    anchor = pd.Timestamp(target_date) - pd.Timedelta(days=1)
    gaps = _demand_gaps(_read_demand_history(target_date), anchor)
    if anchor in gaps.index and fetch_missing_anchor(target_date):
        gaps = _demand_gaps(_read_demand_history(target_date), anchor)

    if not gaps.empty and nyiso_fallback_enabled():
        historical_imputed = gaps["is_imputed"].eq(True) & (gaps.index != anchor)
        eligible = gaps.index[~historical_imputed]
        for date in eligible:
            day = date.date()
            demand = fetch_nyiso_demand(day)
            if demand is None:
                continue
            upsert_demand(
                date=day, demand=demand, source="nyiso", replace_imputed=date == anchor,
            )
            logger.info("Fetched NYISO Zone J demand: date=%s demand_mwh=%s", day, demand)
        if len(eligible):
            gaps = _demand_gaps(_read_demand_history(target_date), anchor)
    elif anchor in gaps.index:
        return False

    if not gaps.empty:
        raise _missing_demand_error(gaps)
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
