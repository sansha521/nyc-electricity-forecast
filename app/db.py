import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv

try:
    import psycopg
    from psycopg.types.json import Jsonb
except ImportError:
    psycopg = None

    def Jsonb(value: Any) -> Any:
        return value


load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]


def get_connection():
    if psycopg is None:
        raise RuntimeError("psycopg is required for database operations")
    return psycopg.connect(DATABASE_URL)


def upsert_demand(
    date,
    demand,
    is_imputed=False,
):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO demand_daily
                    (date, demand, is_imputed)
                VALUES
                    (%s, %s, %s)
                ON CONFLICT (date)
                DO UPDATE SET
                    demand = EXCLUDED.demand,
                    is_imputed = EXCLUDED.is_imputed,
                    updated_at = now()
                """,
                (date, demand, is_imputed),
            )


def upsert_forecast(
    target_date,
    dew=None,
    precip=None,
    snow=None,
    tempmax=None,
    tempmin=None,
    humidity=None,
    snowdepth=None,
    solarenergy=None,
    cloudcover=None,
):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO weather_forecast (
                    target_date,
                    dew,
                    precip,
                    snow,
                    tempmax,
                    tempmin,
                    humidity,
                    snowdepth,
                    solarenergy,
                    cloudcover
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                ON CONFLICT (target_date) DO UPDATE SET
                    updated_at = now(),
                    dew = EXCLUDED.dew,
                    precip = EXCLUDED.precip,
                    snow = EXCLUDED.snow,
                    tempmax = EXCLUDED.tempmax,
                    tempmin = EXCLUDED.tempmin,
                    humidity = EXCLUDED.humidity,
                    snowdepth = EXCLUDED.snowdepth,
                    solarenergy = EXCLUDED.solarenergy,
                    cloudcover = EXCLUDED.cloudcover
                """,
                (
                    target_date,
                    dew,
                    precip,
                    snow,
                    tempmax,
                    tempmin,
                    humidity,
                    snowdepth,
                    solarenergy,
                    cloudcover,
                ),
            )


def upsert_prediction(
    target_date,
    predicted_demand,
    model_version,
    features,
):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions (
                    target_date,
                    predicted_demand,
                    model_version,
                    features
                )
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (target_date)
                DO UPDATE SET
                    predicted_demand = EXCLUDED.predicted_demand,
                    model_version = EXCLUDED.model_version,
                    features = EXCLUDED.features,
                    created_at = now()
                """,
                (
                    target_date,
                    predicted_demand,
                    model_version,
                    Jsonb(features),
                ),
            )


def load_prediction_actuals(start, end):
    with get_connection() as conn:
        return pd.read_sql(
            """
            SELECT
                p.target_date,
                p.predicted_demand,
                d.demand AS actual_demand,
                p.model_version,
                d.is_imputed
            FROM predictions p
            JOIN demand_daily d
                ON d.date = p.target_date
            WHERE p.target_date BETWEEN %s AND %s
            ORDER BY p.target_date
            """,
            conn,
            params=(start, end),
        )


def upsert_prediction_score(
    target_date,
    predicted_demand,
    actual_demand,
    error,
    abs_error,
    pct_error,
    abs_pct_error,
    model_version,
    is_imputed=False,
):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO prediction_scores (
                    target_date,
                    predicted_demand,
                    actual_demand,
                    error,
                    abs_error,
                    pct_error,
                    abs_pct_error,
                    model_version,
                    is_imputed
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (target_date)
                DO UPDATE SET
                    predicted_demand = EXCLUDED.predicted_demand,
                    actual_demand = EXCLUDED.actual_demand,
                    error = EXCLUDED.error,
                    abs_error = EXCLUDED.abs_error,
                    pct_error = EXCLUDED.pct_error,
                    abs_pct_error = EXCLUDED.abs_pct_error,
                    model_version = EXCLUDED.model_version,
                    is_imputed = EXCLUDED.is_imputed,
                    scored_at = now()
                """,
                (
                    target_date,
                    predicted_demand,
                    actual_demand,
                    error,
                    abs_error,
                    pct_error,
                    abs_pct_error,
                    model_version,
                    is_imputed,
                ),
            )


def upsert_weather(
    date,
    dew,
    precip,
    snow,
    tempmax,
    tempmin,
    humidity,
    snowdepth,
    solarenergy,
    cloudcover,
):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO weather_daily (
                    date,
                    dew,
                    precip,
                    snow,
                    tempmax,
                    tempmin,
                    humidity,
                    snowdepth,
                    solarenergy,
                    cloudcover
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                ON CONFLICT (date) DO UPDATE SET
                    dew = EXCLUDED.dew,
                    precip = EXCLUDED.precip,
                    snow = EXCLUDED.snow,
                    tempmax = EXCLUDED.tempmax,
                    tempmin = EXCLUDED.tempmin,
                    humidity = EXCLUDED.humidity,
                    snowdepth = EXCLUDED.snowdepth,
                    solarenergy = EXCLUDED.solarenergy,
                    cloudcover = EXCLUDED.cloudcover
                """,
                (
                    date,
                    dew,
                    precip,
                    snow,
                    tempmax,
                    tempmin,
                    humidity,
                    snowdepth,
                    solarenergy,
                    cloudcover,
                ),
            )


def load_demand_series():
    with get_connection() as conn:
        return pd.read_sql(
            """
            SELECT
                date,
                demand,
                is_imputed
            FROM demand_daily
            ORDER BY date
            """,
            conn,
        )
