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
    from app.ingest_demand import fetch_nyiso_demand, ingest_recent_demand


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


class NyisoDemandTests(unittest.TestCase):
    def setUp(self):
        self.get = self.enterContext(patch("app.ingest_demand.requests.get"))
        self.response = Mock(spec=Response, status_code=200)
        self.get.return_value = self.response

    def set_report(self, date_code, hours, loads):
        columns = [
            "Product_Name", "Posting_Type", "Balancing_Authority_Code", "Data_Type",
            "Data_Code", "Data_Date", "UTC Offset", *(f"HR{i}" for i in range(1, 26)),
        ]
        rows = [
            columns,
            ["EIA-930", "Daily", "NYIS", "UTCD", "", date_code, "5", *hours],
            ["EIA-930", "Daily", "NYIS", "D", "ZONJ", date_code, "5", *loads],
            ["EIA-930", "Daily", "NYIS", "D", "ZONI", date_code, "5", *("999" for _ in hours)],
        ]
        self.response.content = "\n".join(",".join(row + [""] * (len(columns) - len(row))) for row in rows).encode()

    def ending_hours(self, first, count):
        timestamp = dt.datetime.fromisoformat(first)
        return [(timestamp + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S.000Z") for i in range(count)]

    def test_november_23_matches_eia_daily_total(self):
        # Zone J hourly values from NYISO's November 24, 2026 EIA-930 posting.
        loads = (
            "4334 4174 4060 4002 4040 4247 4671 5118 5337 5428 5477 5544 "
            "5618 5651 5710 5759 5915 5928 5709 5526 5356 5132 4847 4530"
        ).split()
        self.set_report("3272026", self.ending_hours("2026-11-23T06:00:00Z", 24), loads)

        self.assertEqual(fetch_nyiso_demand(dt.date(2026, 11, 23)), 122_113.0)
        self.get.assert_called_once_with(
            "https://mis.nyiso.com/public/csv/eiadaily/NYISdaily3282026.csv", timeout=15,
        )

    def test_accepts_clock_change_days_and_uses_next_days_year_in_url(self):
        cases = (
            (dt.date(2026, 3, 8), "0672026", "2026-03-08T06:00:00Z", 23, "0682026"),
            (dt.date(2026, 11, 1), "3052026", "2026-11-01T05:00:00Z", 25, "3062026"),
            (dt.date(2026, 12, 31), "3652026", "2026-12-31T06:00:00Z", 24, "0012027"),
        )
        for anchor, date_code, first, count, publication_code in cases:
            with self.subTest(anchor=anchor):
                self.set_report(date_code, self.ending_hours(first, count), ["100"] * count)
                self.assertEqual(fetch_nyiso_demand(anchor), 100 * count)
                self.assertTrue(self.get.call_args.args[0].endswith(f"NYISdaily{publication_code}.csv"))

    def test_missing_or_incomplete_report_is_pending(self):
        anchor = dt.date(2026, 11, 23)
        self.response.status_code = 404
        self.assertIsNone(fetch_nyiso_demand(anchor))

        self.response.status_code = 200
        hours = self.ending_hours("2026-11-23T06:00:00Z", 24)
        self.set_report("3262026", hours, ["100"] * 24)
        self.assertIsNone(fetch_nyiso_demand(anchor))

        incomplete_loads = ["100"] * 24
        incomplete_loads[10] = ""
        self.set_report("3272026", hours, incomplete_loads)
        self.assertIsNone(fetch_nyiso_demand(anchor))

    def test_rejects_invalid_loads_and_wrong_hour_boundaries(self):
        anchor = dt.date(2026, 11, 23)
        hours = self.ending_hours("2026-11-23T06:00:00Z", 24)
        for bad_value in ("NaN", "inf", "invalid"):
            with self.subTest(bad_value=bad_value):
                self.set_report("3272026", hours, [bad_value, *(["100"] * 23)])
                with self.assertRaises(ValueError):
                    fetch_nyiso_demand(anchor)

        hours[7] = hours[6]
        self.set_report("3272026", hours, ["100"] * 24)
        with self.assertRaisesRegex(ValueError, "Invalid NYISO hourly data"):
            fetch_nyiso_demand(anchor)


if __name__ == "__main__":
    unittest.main()
