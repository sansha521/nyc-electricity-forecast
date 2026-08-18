import datetime as dt
import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")

from app import db


class DbScoreTests(unittest.TestCase):
    def test_upsert_prediction_score_executes_expected_values(self):
        cursor = MagicMock()
        connection = MagicMock()
        connection.__enter__.return_value.cursor.return_value.__enter__.return_value = cursor

        with patch.object(db, "get_connection", return_value=connection):
            db.upsert_prediction_score(
                target_date=dt.date(2026, 8, 15),
                predicted_demand=100.0,
                actual_demand=90.0,
                error=10.0,
                abs_error=10.0,
                pct_error=11.111,
                abs_pct_error=11.111,
                model_version="lgbm_v1",
                is_imputed=False,
            )

        self.assertEqual(cursor.execute.call_count, 1)
        _, params = cursor.execute.call_args.args
        self.assertEqual(
            params,
            (
                dt.date(2026, 8, 15),
                100.0,
                90.0,
                10.0,
                10.0,
                11.111,
                11.111,
                "lgbm_v1",
                False,
            ),
        )

    def test_load_prediction_actuals_uses_recent_window(self):
        with patch.object(db, "get_connection") as get_connection:
            with patch.object(pd, "read_sql", return_value=pd.DataFrame()) as read_sql:
                result = db.load_prediction_actuals(
                    start=dt.date(2026, 8, 1),
                    end=dt.date(2026, 8, 31),
                )

        self.assertTrue(result.empty)
        self.assertEqual(get_connection.call_count, 1)
        params = read_sql.call_args.kwargs["params"]
        self.assertEqual(params, (dt.date(2026, 8, 1), dt.date(2026, 8, 31)))


if __name__ == "__main__":
    unittest.main()
