import datetime as dt
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")

from app.ingest_demand import ingest_recent_demand


class IngestDemandTests(unittest.TestCase):
    def test_ingest_recent_demand_upserts_eastern_rows(self):
        response = MagicMock()
        response.json.return_value = {
            "response": {
                "data": [
                    {
                        "period": "2026-08-15",
                        "value": "100",
                        "timezone": "Eastern",
                    },
                    {
                        "period": "2026-08-15",
                        "value": "95",
                        "timezone": "Pacific",
                    },
                    {
                        "period": "2026-08-16",
                        "value": "110",
                        "timezone": "Eastern",
                    },
                ]
            }
        }

        with patch("app.ingest_demand.requests.get", return_value=response):
            with patch("app.ingest_demand.upsert_demand") as upsert_demand:
                df = ingest_recent_demand(
                    start=dt.date(2026, 8, 15),
                    end=dt.date(2026, 8, 16),
                    api_key="test-key",
                )

        self.assertEqual(len(df), 2)
        self.assertEqual(upsert_demand.call_count, 2)
        upsert_demand.assert_any_call(
            date=dt.date(2026, 8, 15),
            demand=100.0,
            is_imputed=False,
        )
        upsert_demand.assert_any_call(
            date=dt.date(2026, 8, 16),
            demand=110.0,
            is_imputed=False,
        )


if __name__ == "__main__":
    unittest.main()
