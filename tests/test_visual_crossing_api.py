import datetime as dt
from io import BytesIO
import json
import os
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

with patch.dict(os.environ, {"PYTHON_DOTENV_DISABLED": "1"}):
    from app import build_row_inference


TARGET_DATE = dt.date(2026, 11, 23)


class VisualCrossingApiTests(unittest.TestCase):
    def setUp(self):
        self.urlopen = self.enterContext(patch("app.build_row_inference.urllib.request.urlopen"))
        self.enterContext(patch.dict(os.environ, {"VISUAL_CROSSING_API_KEY": "test-key"}))
        self.forecast_day = {
            "datetime": TARGET_DATE.isoformat(),
            "tempmin": 40.0,
            "tempmax": 50.0,
            "dew": 30.0,
            "humidity": 72.0,
            "precip": 0.2,
            "snow": 0.1,
            "snowdepth": 1.2,
            "solarenergy": 6.0,
            "cloudcover": 25.0,
            "sunriseEpoch": 1_000_000_000,
            "sunsetEpoch": 1_000_034_200,
        }

    def set_weather_response(self, days):
        self.urlopen.return_value = BytesIO(json.dumps({"days": days}).encode("utf-8"))

    def test_requests_target_day_and_maps_forecast_features(self):
        previous_day = dict(
            self.forecast_day,
            datetime=(TARGET_DATE - dt.timedelta(days=1)).isoformat(),
            tempmax=80.0,
        )
        self.set_weather_response([previous_day, self.forecast_day])

        features = build_row_inference.fetch_target_forecast(TARGET_DATE)

        self.urlopen.assert_called_once()
        request = self.urlopen.call_args
        url = urlsplit(request.args[0])
        params = parse_qs(url.query)
        self.assertEqual(url.netloc, "weather.visualcrossing.com")
        self.assertEqual(url.path.rsplit("/", 1)[-1], TARGET_DATE.isoformat())
        self.assertEqual(params["unitGroup"], ["us"])
        self.assertEqual(params["include"], ["days"])
        self.assertEqual(params["key"], ["test-key"])
        self.assertEqual(request.kwargs["timeout"], 60)
        self.assertEqual(features.to_dict(), {
            "tmin_fc": 40.0,
            "tmax_fc": 50.0,
            "tavg_fc": 45.0,
            "dew_point_fc": 30.0,
            "rh_mean_fc": 72.0,
            "prcp_fc": 0.2,
            "snow_fc": 0.1,
            "snwd_fc": 1.2,
            "solar_rad_fc": 6.0,
            "cloud_cover_fc": 25.0,
            "hdd_fc": 5.0,
            "cdd_fc": 0.0,
            "daylight_hours_next": 9.5,
        })

    def test_null_snow_depth_defaults_to_zero(self):
        self.set_weather_response([dict(self.forecast_day, snowdepth=None)])

        features = build_row_inference.fetch_target_forecast(TARGET_DATE)

        self.assertEqual(features["snwd_fc"], 0.0)

    def test_response_missing_target_date_raises_an_error(self):
        another_day = dict(
            self.forecast_day, datetime=(TARGET_DATE + dt.timedelta(days=1)).isoformat()
        )
        self.set_weather_response([another_day])

        with self.assertRaises(KeyError) as error:
            build_row_inference.fetch_target_forecast(TARGET_DATE)

        self.assertEqual(error.exception.args[0], TARGET_DATE)

    def test_response_missing_required_weather_field_raises_an_error(self):
        del self.forecast_day["dew"]
        self.set_weather_response([self.forecast_day])

        with self.assertRaisesRegex(KeyError, "dew"):
            build_row_inference.fetch_target_forecast(TARGET_DATE)

    def test_requests_two_observed_days_for_three_day_features(self):
        start = TARGET_DATE - dt.timedelta(days=2)
        end = TARGET_DATE - dt.timedelta(days=1)
        self.set_weather_response([
            {"datetime": start.isoformat(), "tempmin": 35.0, "tempmax": 43.0},
            {"datetime": end.isoformat(), "tempmin": 36.0, "tempmax": 48.0},
        ])

        features = build_row_inference.build_3day_forecast_features(TARGET_DATE, 45.0)

        self.urlopen.assert_called_once()
        url = urlsplit(self.urlopen.call_args.args[0])
        self.assertEqual(url.path.rsplit("/", 2)[-2:], [start.isoformat(), end.isoformat()])
        self.assertEqual(parse_qs(url.query)["key"], ["test-key"])
        self.assertEqual(features, {
            "tavg_3day_fc": 42.0,
            "cdd_3day_fc": 0.0,
            "hdd_3day_fc": 8.0,
        })


if __name__ == "__main__":
    unittest.main()
