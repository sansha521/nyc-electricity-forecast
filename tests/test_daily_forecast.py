import datetime as dt
import os
import unittest
from unittest.mock import call, patch

import pandas as pd

with patch.dict(os.environ, {
    "DATABASE_URL": "postgresql://test:test@localhost/test",
    "VISUAL_CROSSING_API_KEY": "test-key",
    "PYTHON_DOTENV_DISABLED": "1",
}):
    from app import build_row_inference, db, pipeline_tasks, run_daily
    from app.predict import load_metadata


TARGET_DATE = dt.date(2026, 11, 23)
ANCHOR_DATE = TARGET_DATE - dt.timedelta(days=1)


class BuildFeatureRowTests(unittest.TestCase):
    def test_uses_supplied_demand_and_preserves_model_feature_order(self):
        demand = pd.Series(
            range(100_000, 100_367),
            index=pd.date_range(end=ANCHOR_DATE, periods=367, freq="D"),
            dtype=float,
        )
        weather_forecast = pd.Series({
            "tmin_fc": 40.0,
            "tmax_fc": 50.0,
            "tavg_fc": 45.0,
            "dew_point_fc": 30.0,
            "rh_mean_fc": 65.0,
            "prcp_fc": 0.0,
            "snow_fc": 0.0,
            "snwd_fc": 0.0,
            "solar_rad_fc": 10.0,
            "cloud_cover_fc": 25.0,
            "hdd_fc": 5.0,
            "cdd_fc": 0.0,
            "daylight_hours_next": 10.0,
        })
        observed_weather = pd.DataFrame(
            {"tavg": [39.0, 42.0]},
            index=[TARGET_DATE - dt.timedelta(days=2), ANCHOR_DATE],
        )

        with (
            patch.object(pipeline_tasks, "load_demand_history", return_value=demand),
            patch.object(build_row_inference, "fetch_recent_demand", return_value=demand)
            as fetch_recent_demand,
            patch.object(build_row_inference, "fetch_target_forecast", return_value=weather_forecast),
            patch.object(build_row_inference, "_fetch_visual_crossing_days", return_value=observed_weather),
            patch.object(build_row_inference, "upsert_forecast"),
        ):
            features = pipeline_tasks.build_feature_row(TARGET_DATE)

        fetch_recent_demand.assert_not_called()
        self.assertEqual(list(features), load_metadata()["features"])
        self.assertEqual(features["lag_0"], 100_366.0)
        self.assertEqual(features["lag_365"], 100_001.0)
        self.assertEqual(features["lag_366"], 100_000.0)
        self.assertEqual(features["rolling_30"], 100_350.5)
        self.assertEqual(features["tavg_3day_fc"], 42.0)


class DemandValidationTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"ENABLE_NYISO_DEMAND_FALLBACK": "0"}))
        connect = self.enterContext(patch.object(db, "get_connection"))
        connection = connect.return_value.__enter__.return_value
        self.cursor = connection.cursor.return_value.__enter__.return_value
        self.history = [
            (ANCHOR_DATE - dt.timedelta(days=lag), 100_000.0 - lag, lag >= 365)
            for lag in (*range(31), 365, 366)
        ]

    def test_accepts_complete_history_with_imputed_annual_lags(self):
        self.cursor.fetchall.return_value = self.history
        with patch.object(pipeline_tasks, "fetch_missing_anchor") as refresh:
            self.assertTrue(pipeline_tasks.validate_data_availability(TARGET_DATE))
        refresh.assert_not_called()

    def test_waits_for_missing_or_imputed_anchor(self):
        for state, rows in (
            ("missing", self.history[1:]),
            ("imputed", [(ANCHOR_DATE, 100_000.0, True), *self.history[1:]]),
        ):
            with self.subTest(anchor=state):
                self.cursor.fetchall.return_value = rows
                with (
                    patch.object(pipeline_tasks, "fetch_missing_anchor", return_value=False) as refresh,
                    self.assertLogs("app.pipeline_tasks", level="INFO") as logs,
                ):
                    self.assertFalse(pipeline_tasks.validate_data_availability(TARGET_DATE))
                refresh.assert_called_once_with(TARGET_DATE)
                self.assertEqual([record.getMessage() for record in logs.records], [
                    f"Validating demand history: anchor_date={ANCHOR_DATE} nyiso_fallback_enabled=False",
                    f"Refreshing EIA demand: anchor_date={ANCHOR_DATE}",
                ])

    def test_rechecks_history_after_anchor_is_published(self):
        self.cursor.fetchall.side_effect = [self.history[1:], self.history]
        with patch.object(pipeline_tasks, "fetch_missing_anchor", return_value=True) as refresh:
            self.assertTrue(pipeline_tasks.validate_data_availability(TARGET_DATE))
        refresh.assert_called_once_with(TARGET_DATE)

    def test_missing_historical_lag_is_an_error_without_polling_eia(self):
        self.cursor.fetchall.return_value = self.history[:-1]
        with patch.object(pipeline_tasks, "fetch_missing_anchor", return_value=True) as refresh:
            with self.assertRaises(ValueError):
                pipeline_tasks.validate_data_availability(TARGET_DATE)
        refresh.assert_not_called()

    def test_rejects_null_and_infinite_historical_demand(self):
        annual_lag_date = self.history[-1][0]
        for invalid in (None, float("inf")):
            with self.subTest(demand=invalid):
                self.cursor.fetchall.return_value = [
                    *self.history[:-1],
                    (annual_lag_date, invalid, True),
                ]
                with self.assertRaises(ValueError):
                    pipeline_tasks.load_demand_history(TARGET_DATE)

    def test_invalid_historical_imputation_errors_without_fetching_nyiso(self):
        self.enterContext(patch.dict(os.environ, {"ENABLE_NYISO_DEMAND_FALLBACK": "1"}))
        annual_lag_date = self.history[-1][0]
        self.cursor.fetchall.return_value = [
            *self.history[:-1], (annual_lag_date, float("inf"), True),
        ]

        with patch("app.ingest_demand.fetch_nyiso_demand") as refresh:
            with self.assertRaisesRegex(ValueError, annual_lag_date.isoformat()):
                pipeline_tasks.validate_data_availability(TARGET_DATE)

        refresh.assert_not_called()

    def test_logs_nyiso_fallback_after_eia_anchor_refresh(self):
        self.cursor.fetchall.side_effect = [self.history[1:], self.history]
        with (
            patch.dict(os.environ, {"ENABLE_NYISO_DEMAND_FALLBACK": "1"}),
            patch.object(pipeline_tasks, "fetch_missing_anchor", return_value=False),
            patch("app.ingest_demand.fetch_nyiso_demand", return_value=99_000.0) as fetch_nyiso,
            patch.object(db, "upsert_demand"),
            self.assertLogs("app.pipeline_tasks", level="INFO") as logs,
        ):
            self.assertTrue(pipeline_tasks.validate_data_availability(TARGET_DATE))

        fetch_nyiso.assert_called_once_with(ANCHOR_DATE)
        self.assertEqual([record.getMessage() for record in logs.records], [
            f"Validating demand history: anchor_date={ANCHOR_DATE} nyiso_fallback_enabled=True",
            f"Refreshing EIA demand: anchor_date={ANCHOR_DATE}",
            f"Fetching NYISO fallback demand: date={ANCHOR_DATE}",
            f"Fetched NYISO Zone J demand: date={ANCHOR_DATE} demand_mwh=99000.0",
        ])


class RunDailyTests(unittest.TestCase):
    def setUp(self):
        self.tasks = self.enterContext(patch.object(run_daily, "tasks", autospec=True))
        self.enterContext(patch.object(run_daily, "setup_logging"))
        self.tasks.default_target_date.return_value = TARGET_DATE
        self.tasks.ingest_latest_eia.return_value = 33
        self.tasks.MODEL_VERSION = pipeline_tasks.MODEL_VERSION

    def test_success_runs_stages_in_order_and_returns_zero(self):
        features = {"lag_0": 100_000.0}
        self.tasks.validate_data_availability.return_value = True
        self.tasks.build_feature_row.return_value = features
        self.tasks.generate_and_store_forecast.return_value = 99_000.0

        with (
            patch.object(run_daily, "perf_counter", side_effect=[10.0, 12.5, 20.0, 23.0, 30.0, 31.0]),
            self.assertLogs("app.run_daily", level="INFO") as logs,
        ):
            self.assertEqual(run_daily.main(), 0)

        self.assertEqual([record.getMessage() for record in logs.records], [
            f"Starting daily inference for target_date={TARGET_DATE}",
            "Ingested EIA demand: rows=33 elapsed_s=2.50",
            f"Building inference features: target_date={TARGET_DATE}",
            f"Feature row built: target_date={TARGET_DATE} feature_count=1 elapsed_s=3.00",
            f"Generating forecast: target_date={TARGET_DATE} model_version={pipeline_tasks.MODEL_VERSION}",
            f"Stored forecast: target_date={TARGET_DATE} prediction=99000.00 unit=MWh "
            f"model_version={pipeline_tasks.MODEL_VERSION} elapsed_s=1.00",
        ])
        self.assertEqual(self.tasks.method_calls, [
            call.default_target_date(),
            call.ingest_latest_eia(TARGET_DATE),
            call.validate_data_availability(TARGET_DATE),
            call.build_feature_row(TARGET_DATE),
            call.generate_and_store_forecast(TARGET_DATE, features),
        ])

    def test_unpublished_anchor_returns_75_without_building_or_saving(self):
        self.tasks.validate_data_availability.return_value = False

        with self.assertLogs("app.run_daily", level="INFO") as logs:
            self.assertEqual(run_daily.main(), 75)

        self.assertEqual(len(logs.records), 3)
        self.assertEqual(logs.records[-1].levelname, "WARNING")
        self.assertIn("Daily inference delayed", logs.records[-1].getMessage())
        self.tasks.build_feature_row.assert_not_called()
        self.tasks.generate_and_store_forecast.assert_not_called()

    def test_forecast_write_failure_logs_error_without_storage_confirmation(self):
        self.tasks.validate_data_availability.return_value = True
        self.tasks.build_feature_row.return_value = dict.fromkeys(build_row_inference.FEATURES, 1.0)
        self.tasks.generate_and_store_forecast.side_effect = pipeline_tasks.generate_and_store_forecast
        error = RuntimeError("Postgres unavailable")
        with (
            patch("app.predict.load_model"),
            patch("app.predict.predict", return_value=99_000.0),
            patch.object(db, "upsert_prediction", side_effect=error) as store,
            self.assertLogs("app", level="INFO") as logs,
        ):
            self.assertEqual(run_daily.main(), 1)

        store.assert_called_once()
        messages = [record.getMessage() for record in logs.records]
        self.assertEqual(messages[-2:], [
            f"Saving forecast to Postgres: target_date={TARGET_DATE}",
            "Daily inference failed",
        ])
        self.assertFalse(any(message.startswith("Stored forecast") for message in messages))
        self.assertIs(logs.records[-1].exc_info[1], error)

    def test_historical_data_error_returns_one_without_building_or_saving(self):
        self.tasks.validate_data_availability.side_effect = ValueError("Missing annual lag")

        with self.assertLogs("app.run_daily", level="ERROR") as logs:
            self.assertEqual(run_daily.main(), 1)

        self.assertIsInstance(logs.records[0].exc_info[1], ValueError)
        self.tasks.build_feature_row.assert_not_called()
        self.tasks.generate_and_store_forecast.assert_not_called()


if __name__ == "__main__":
    unittest.main()
