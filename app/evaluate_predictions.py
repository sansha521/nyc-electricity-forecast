from __future__ import annotations

import datetime as dt
import logging
import math
import sys
from zoneinfo import ZoneInfo

import pandas as pd

from app.db import load_prediction_actuals, upsert_prediction_score
from app.evaluation import compute_score_rows, summarize_scores
from app.ingest_demand import ingest_recent_demand


NY_TZ = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _format_metric(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.2f}"


def _log_summary(label: str, summary: dict[str, float | int]) -> None:
    logger.info(
        "%s: count=%d MAE=%s RMSE=%s MAPE=%s%% bias=%s",
        label,
        summary["count"],
        _format_metric(summary["mae"]),
        _format_metric(summary["rmse"]),
        _format_metric(summary["mape"]),
        _format_metric(summary["bias"]),
    )


def _upsert_scores(scores: pd.DataFrame) -> None:
    for record in scores.itertuples(index=False):
        upsert_prediction_score(
            target_date=record.target_date,
            predicted_demand=float(record.predicted_demand),
            actual_demand=float(record.actual_demand),
            error=float(record.error),
            abs_error=float(record.abs_error),
            pct_error=float(record.pct_error),
            abs_pct_error=float(record.abs_pct_error),
            model_version=record.model_version,
            is_imputed=bool(record.is_imputed),
        )


def run_evaluation(
    today: dt.date | None = None,
    days_back: int = 30,
) -> dict[str, object]:
    today = today or dt.datetime.now(NY_TZ).date()
    start = today - dt.timedelta(days=days_back)
    end = today - dt.timedelta(days=1)

    logger.info("Refreshing EIA demand for %s through %s", start, end)
    ingest_recent_demand(start, end)

    logger.info("Loading predictions with actuals for %s through %s", start, end)
    joined = load_prediction_actuals(start, end)
    scores = compute_score_rows(joined)
    _upsert_scores(scores)

    last_7 = summarize_scores(scores.tail(7))
    last_30 = summarize_scores(scores.tail(30))

    logger.info("Scored/updated %d prediction(s)", len(scores))
    if not scores.empty:
        latest = scores.iloc[-1]
        logger.info(
            "Latest scored: target_date=%s predicted=%.2f actual=%.2f "
            "error=%.2f abs_pct_error=%.2f%%",
            latest["target_date"],
            latest["predicted_demand"],
            latest["actual_demand"],
            latest["error"],
            latest["abs_pct_error"],
        )
    _log_summary("Last 7 scored non-imputed days", last_7)
    _log_summary("Last 30 scored non-imputed days", last_30)

    return {
        "scored_count": len(scores),
        "last_7": last_7,
        "last_30": last_30,
    }


def main() -> int:
    setup_logging()

    try:
        run_evaluation()
        return 0
    except Exception:
        logger.exception("Prediction evaluation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
