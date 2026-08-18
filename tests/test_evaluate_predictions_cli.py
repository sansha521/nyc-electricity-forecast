import datetime as dt
import os
import unittest
from unittest.mock import patch

import pandas as pd

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")

from app.evaluate_predictions import run_evaluation


class EvaluatePredictionsCliTests(unittest.TestCase):
    def test_run_evaluation_refreshes_recent_demand_and_upserts_scores(self):
        today = dt.date(2026, 8, 17)
        joined = pd.DataFrame(
            [
                {
                    "target_date": dt.date(2026, 8, 15),
                    "predicted_demand": 100.0,
                    "actual_demand": 90.0,
                    "model_version": "lgbm_v1",
                    "is_imputed": False,
                },
                {
                    "target_date": dt.date(2026, 8, 16),
                    "predicted_demand": 120.0,
                    "actual_demand": 100.0,
                    "model_version": "lgbm_v1",
                    "is_imputed": True,
                },
            ]
        )

        with patch("app.evaluate_predictions.ingest_recent_demand") as ingest:
            with patch("app.evaluate_predictions.load_prediction_actuals", return_value=joined) as load:
                with patch("app.evaluate_predictions.upsert_prediction_score") as upsert:
                    result = run_evaluation(today=today, days_back=30)

        ingest.assert_called_once_with(
            today - dt.timedelta(days=30),
            today - dt.timedelta(days=1),
        )
        load.assert_called_once_with(
            today - dt.timedelta(days=30),
            today - dt.timedelta(days=1),
        )
        self.assertEqual(upsert.call_count, 1)
        self.assertEqual(result["scored_count"], 1)
        self.assertEqual(result["last_7"]["count"], 1)
        self.assertEqual(result["last_30"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
