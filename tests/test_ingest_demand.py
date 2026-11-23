import datetime as dt
import os
import unittest
from unittest.mock import Mock, patch

from requests import Response
from requests.exceptions import HTTPError, Timeout

with patch.dict(os.environ, {
    "DATABASE_URL": "postgresql://test:test@localhost/test",
    "PYTHON_DOTENV_DISABLED": "1",
}):
    from app.ingest_demand import ingest_recent_demand


START = dt.date(2026, 8, 15)
END = dt.date(2026, 8, 16)
API_KEY = "test-key"


class IngestDemandTests(unittest.TestCase):
    def setUp(self):
        self.get = self.enterContext(patch("app.ingest_demand.requests.get"))
        self.upsert_demand = self.enterContext(patch("app.ingest_demand.upsert_demand"))

    def test_ingest_recent_demand_upserts_eastern_rows(self):
        response = Mock(spec=Response)
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

        self.get.return_value = response

        df = ingest_recent_demand(START, END, api_key=API_KEY)

        self.get.assert_called_once()
        request = self.get.call_args
        self.assertEqual(
            request.args[0],
            "https://api.eia.gov/v2/electricity/rto/daily-region-sub-ba-data/data/",
        )
        params = request.kwargs["params"]
        self.assertEqual(params["facets[subba][]"], "ZONJ")
        self.assertEqual(params["frequency"], "daily")
        self.assertEqual(params["data[0]"], "value")
        self.assertEqual(params["start"], START.isoformat())
        self.assertEqual(params["end"], END.isoformat())
        response.raise_for_status.assert_called_once_with()
        self.assertEqual(len(df), 2)
        self.assertEqual(self.upsert_demand.call_count, 2)
        self.upsert_demand.assert_any_call(
            date=START,
            demand=100.0,
            is_imputed=False,
        )
        self.upsert_demand.assert_any_call(
            date=END,
            demand=110.0,
            is_imputed=False,
        )

    def test_empty_eia_response_has_typed_columns_and_writes_no_rows(self):
        response = Mock(spec=Response)
        response.json.return_value = {"response": {"data": []}}
        self.get.return_value = response

        df = ingest_recent_demand(START, END, api_key=API_KEY)

        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), ["date", "demand"])
        self.assertEqual(str(df["date"].dtype), "datetime64[ns]")
        self.assertEqual(str(df["demand"].dtype), "float64")
        self.upsert_demand.assert_not_called()

    def test_eia_rate_limit_raises_http_error_without_writing_demand(self):
        response = Response()
        response.status_code = 429
        response.reason = "Too Many Requests"
        response.url = "https://api.eia.gov/"
        self.get.return_value = response

        with self.assertRaises(HTTPError) as error:
            ingest_recent_demand(START, END, api_key=API_KEY)

        self.assertEqual(error.exception.response.status_code, 429)
        self.upsert_demand.assert_not_called()

    def test_eia_timeout_propagates_without_writing_demand(self):
        self.get.side_effect = Timeout("EIA request timed out")

        with self.assertRaises(Timeout):
            ingest_recent_demand(START, END, api_key=API_KEY, timeout=5)

        self.get.assert_called_once()
        self.assertEqual(self.get.call_args.kwargs["timeout"], 5)
        self.upsert_demand.assert_not_called()


if __name__ == "__main__":
    unittest.main()
