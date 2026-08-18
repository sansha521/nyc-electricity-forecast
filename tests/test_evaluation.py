import math
import unittest

import pandas as pd

from app.evaluation import compute_score_rows, summarize_scores


class EvaluationTests(unittest.TestCase):
    def test_compute_score_rows_excludes_imputed_actuals(self):
        joined = pd.DataFrame(
            [
                {
                    "target_date": pd.Timestamp("2026-08-15").date(),
                    "predicted_demand": 100.0,
                    "actual_demand": 90.0,
                    "model_version": "lgbm_v1",
                    "is_imputed": False,
                },
                {
                    "target_date": pd.Timestamp("2026-08-16").date(),
                    "predicted_demand": 120.0,
                    "actual_demand": 100.0,
                    "model_version": "lgbm_v1",
                    "is_imputed": True,
                },
            ]
        )

        scores = compute_score_rows(joined)

        self.assertEqual(len(scores), 1)
        row = scores.iloc[0]
        self.assertEqual(row["target_date"], pd.Timestamp("2026-08-15").date())
        self.assertEqual(row["predicted_demand"], 100.0)
        self.assertEqual(row["actual_demand"], 90.0)
        self.assertEqual(row["error"], 10.0)
        self.assertEqual(row["abs_error"], 10.0)
        self.assertAlmostEqual(row["pct_error"], 11.1111111111)
        self.assertAlmostEqual(row["abs_pct_error"], 11.1111111111)
        self.assertFalse(row["is_imputed"])

    def test_summarize_scores_computes_core_metrics(self):
        scores = pd.DataFrame(
            [
                {"error": 10.0, "abs_error": 10.0, "abs_pct_error": 10.0},
                {"error": -20.0, "abs_error": 20.0, "abs_pct_error": 20.0},
                {"error": 30.0, "abs_error": 30.0, "abs_pct_error": 30.0},
            ]
        )

        summary = summarize_scores(scores)

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["mae"], 20.0)
        self.assertAlmostEqual(summary["rmse"], math.sqrt((100 + 400 + 900) / 3))
        self.assertEqual(summary["mape"], 20.0)
        self.assertAlmostEqual(summary["bias"], 20.0 / 3)

    def test_summarize_scores_handles_empty_input(self):
        summary = summarize_scores(pd.DataFrame())

        self.assertEqual(summary["count"], 0)
        self.assertTrue(math.isnan(summary["mae"]))
        self.assertTrue(math.isnan(summary["rmse"]))
        self.assertTrue(math.isnan(summary["mape"]))
        self.assertTrue(math.isnan(summary["bias"]))


if __name__ == "__main__":
    unittest.main()
