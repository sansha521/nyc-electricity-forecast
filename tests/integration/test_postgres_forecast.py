import datetime as dt
from io import BytesIO
import json
import math
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit
from uuid import uuid4

import psycopg
from psycopg import sql

with patch.dict(os.environ, {"PYTHON_DOTENV_DISABLED": "1"}):
    from app import db, evaluate_predictions, run_daily
    from app.predict import load_metadata


SCHEMA_FILE = Path(__file__).resolve().parents[2] / "app" / "db" / "schema.sql"
TARGET_DATE = dt.date(2026, 11, 23)
ANCHOR_DATE = TARGET_DATE - dt.timedelta(days=1)


@unittest.skipUnless(os.environ.get("TEST_DATABASE_URL"), "Set TEST_DATABASE_URL for PostgreSQL tests")
class PostgresForecastTests(unittest.TestCase):
    def setUp(self):
        self.dsn = os.environ["TEST_DATABASE_URL"]
        self.schema = f"forecast_test_{uuid4().hex}"
        with psycopg.connect(self.dsn) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema)))
        self.addCleanup(self.drop_schema)
        with self.connect() as conn:
            conn.execute(SCHEMA_FILE.read_text(encoding="utf-8"))

        self.enterContext(patch.object(db, "get_connection", side_effect=self.connect))
        self.enterContext(patch.dict(os.environ, {
            "EIA_API_KEY": "test-key",
            "VISUAL_CROSSING_API_KEY": "test-key",
        }))
        self.enterContext(patch("app.ingest_demand.requests.get", side_effect=self.eia_response))
        self.weather = self.enterContext(patch(
            "app.build_row_inference.urllib.request.urlopen", side_effect=self.weather_response
        ))

        self.demand = {
            ANCHOR_DATE - dt.timedelta(days=lag): 130_000.0 - lag * 10
            for lag in range(31)
        }
        for lag, estimate in ((365, 118_120.0), (366, 128_613.0)):
            db.upsert_demand(ANCHOR_DATE - dt.timedelta(days=lag), estimate, is_imputed=True)

    def connect(self):
        return psycopg.connect(self.dsn, options=f"-c search_path={self.schema}")

    def drop_schema(self):
        with psycopg.connect(self.dsn) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.schema)))

    def eia_response(self, url, *, params, timeout):
        start = dt.date.fromisoformat(params["start"])
        end = dt.date.fromisoformat(params["end"])
        records = [
            {"period": day.isoformat(), "value": str(value), "timezone": "Eastern"}
            for day, value in sorted(self.demand.items()) if start <= day <= end
        ]
        response = Mock()
        response.json.return_value = {"response": {"data": records}}
        return response

    def weather_response(self, url, *, timeout):
        if urlsplit(url).path.endswith(f"/{TARGET_DATE}"):
            days = [{
                "datetime": TARGET_DATE.isoformat(),
                "tempmin": 40.0,
                "tempmax": 50.0,
                "dew": 30.0,
                "humidity": 72.0,
                "precip": 0.2,
                "snow": 0.0,
                "snowdepth": None,
                "solarenergy": 6.0,
                "cloudcover": 25.0,
                "sunriseEpoch": 1_000_000_000,
                "sunsetEpoch": 1_000_034_200,
            }]
        else:
            days = [
                {"datetime": (ANCHOR_DATE - dt.timedelta(days=1)).isoformat(),
                 "tempmin": 35.0, "tempmax": 43.0},
                {"datetime": ANCHOR_DATE.isoformat(), "tempmin": 36.0, "tempmax": 48.0},
            ]
        return BytesIO(json.dumps({"days": days}).encode("utf-8"))

    def test_forecast_rerun_and_revised_actual_update_existing_rows(self):
        run_daily.run_daily(TARGET_DATE)
        run_daily.run_daily(TARGET_DATE)

        with self.connect() as conn:
            predictions = conn.execute(
                "SELECT target_date, predicted_demand, model_version, features FROM predictions"
            ).fetchall()
            weather_dates = conn.execute("SELECT target_date FROM weather_forecast").fetchall()

        self.assertEqual(len(predictions), 1)
        target_date, prediction, version, features = predictions[0]
        self.assertEqual(target_date, TARGET_DATE)
        self.assertTrue(math.isfinite(prediction))
        self.assertEqual(version, "lgbm_v1")
        self.assertEqual(set(features), set(load_metadata()["features"]))
        self.assertEqual(features["lag_0"], self.demand[ANCHOR_DATE])
        self.assertEqual(features["lag_365"], 118_120.0)
        self.assertEqual(features["lag_366"], 128_613.0)
        self.assertEqual(weather_dates, [(TARGET_DATE,)])

        evaluation_date = TARGET_DATE + dt.timedelta(days=1)
        db.upsert_demand(TARGET_DATE, 124_000.0, is_imputed=True)
        self.assertEqual(evaluate_predictions.run_evaluation(today=evaluation_date)["scored_count"], 0)

        for actual in (125_000.0, 125_500.0):
            with self.subTest(actual=actual):
                self.demand[TARGET_DATE] = actual
                result = evaluate_predictions.run_evaluation(today=evaluation_date)
                self.assertEqual(result["scored_count"], 1)
                with self.connect() as conn:
                    scores = conn.execute(
                        "SELECT target_date, actual_demand, error, model_version, is_imputed "
                        "FROM prediction_scores"
                    ).fetchall()
                self.assertEqual(scores, [(TARGET_DATE, actual, prediction - actual, version, False)])

    def test_unpublished_anchor_blocks_forecast_but_evaluation_scores_earlier_day(self):
        previous_target = TARGET_DATE - dt.timedelta(days=2)
        db.upsert_prediction(previous_target, 129_000.0, "lgbm_v1", {})
        del self.demand[ANCHOR_DATE]

        with patch("app.pipeline_tasks.default_target_date", return_value=TARGET_DATE):
            self.assertEqual(run_daily.main(), 75)
        self.weather.assert_not_called()
        self.assertEqual(evaluate_predictions.run_evaluation(today=TARGET_DATE)["scored_count"], 1)

        with self.connect() as conn:
            predictions = conn.execute("SELECT target_date FROM predictions").fetchall()
            scores = conn.execute("SELECT target_date FROM prediction_scores").fetchall()
        self.assertEqual(predictions, [(previous_target,)])
        self.assertEqual(scores, [(previous_target,)])
