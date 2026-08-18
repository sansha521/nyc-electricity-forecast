import datetime as dt
import logging
import sys
from zoneinfo import ZoneInfo

from app.build_row_inference import DemandDataUnavailable, build_row
from app.db import upsert_prediction
from app.ingest_demand import ingest_recent_demand
from app.predict import load_model, predict

NY_TZ = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_daily() -> None:
    target_date = dt.datetime.now(NY_TZ).date()
    logger.info("Starting daily inference for target_date=%s", target_date)

    # EIA typically publishes yesterday's final daily demand today.
    anchor_date = target_date - dt.timedelta(days=1)
    start = anchor_date - dt.timedelta(days=35)
    end = anchor_date

    logger.info("Step 1/4: ingesting demand")
    ingest_recent_demand(start, end)
    logger.info(
        "Ingested EIA demand: start=%s end=%s",
        start,
        end,
    )

    logger.info("Step 2/4: building inference row")
    row = build_row(target_date)
    logger.info(
        "Feature row built: target_date=%s feature_count=%d",
        target_date,
        len(row.columns),
    )

    logger.info("Step 3/4: generating prediction")
    model = load_model()
    prediction = predict(model, row)
    logger.info(
        "Prediction generated: target_date=%s prediction=%.2f",
        target_date,
        prediction,
    )

    logger.info("Step 4/4: saving prediction")
    upsert_prediction(
        target_date=target_date,
        predicted_demand=prediction,
        model_version="lgbm_v1",
        features=row.iloc[0].to_dict(),
    )
    logger.info(
        "Daily inference completed successfully: target_date=%s prediction=%.2f",
        target_date,
        prediction,
    )


def main() -> int:
    setup_logging()

    try:
        run_daily()
        return 0

    except DemandDataUnavailable as exc:
        logger.warning("Daily inference delayed: %s", exc)
        return 75

    except Exception:
        logger.exception("Daily inference failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
