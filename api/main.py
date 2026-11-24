from fastapi import FastAPI, HTTPException
from datetime import date
from pydantic import BaseModel
from math import isfinite

import pandas as pd
import uvicorn

from app.db import get_connection
from app.evaluation import summarize_scores

app = FastAPI(
    title="NYC Electricity Forecast API",
    version="1.0.0"
)

class Forecast(BaseModel):
    target_date: date
    predicted_mwh: float

class MetricsWindow(BaseModel):
    count: int
    mae: float | None
    mape: float | None
    rmse: float | None


class MetricsResponse(BaseModel):
    latest_target_date: date
    last_7: MetricsWindow
    last_30: MetricsWindow


def get_latest_forecast() -> Forecast | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT target_date, predicted_demand
            FROM predictions
            ORDER BY target_date DESC
            LIMIT 1
            """
        ).fetchone()

    if row is None:
        return None

    target_date, predicted_mwh = row
    return Forecast(target_date=target_date, predicted_mwh=predicted_mwh)


def _finite_or_none(value: float) -> float | None:
    return value if isfinite(value) else None


def _summarize_window(scores: pd.DataFrame) -> MetricsWindow:
    summary = summarize_scores(scores)
    return MetricsWindow(
        count=int(summary["count"]),
        mae=_finite_or_none(summary["mae"]),
        mape=_finite_or_none(summary["mape"]),
        rmse=_finite_or_none(summary["rmse"]),
    )


def get_latest_metrics() -> MetricsResponse | None:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT target_date, error, abs_error,
                   CASE WHEN actual_demand <> 0 THEN abs_pct_error END
                       AS abs_pct_error
            FROM prediction_scores
            WHERE is_imputed = FALSE
            ORDER BY target_date DESC
            LIMIT 30
            """
        ).fetchall()

    if not rows:
        return None

    scores = pd.DataFrame.from_records(
        rows,
        columns=["target_date", "error", "abs_error", "abs_pct_error"],
    )
    return MetricsResponse(
        latest_target_date=scores["target_date"].iloc[0],
        last_7=_summarize_window(scores.head(7)),
        last_30=_summarize_window(scores),
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/forecasts/latest", response_model=Forecast)
def latest_forecast():
    # Implement a read-only Postgres query here.
    # Adapt to the existing prediction table schema.
    forecast = get_latest_forecast()

    if forecast is None:
        raise HTTPException(
            status_code=404,
            detail="No forecast available"
        )

    return forecast

@app.get("/metrics/latest", response_model=MetricsResponse)
def latest_metrics():
    # Implement a read-only Postgres query here.
    # Adapt to the existing metrics table schema.
    metrics = get_latest_metrics()

    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail="No metrics available"
        )

    return metrics

@app.get("/forecasts", response_model=list[Forecast])
def list_forecasts(start: date, end: date) -> list[Forecast]:
    if start > end:
        raise HTTPException(
            status_code=400,
            detail="Start date must be before or equal to end date"
        )
    
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT target_date, predicted_demand
            FROM predictions
            WHERE target_date BETWEEN %s AND %s
            ORDER BY target_date ASC
            """,
            (start, end)
        ).fetchall()
    
    return [
        Forecast(target_date=target_date, predicted_mwh=predicted_mwh)
        for target_date, predicted_mwh in rows
    ]

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)