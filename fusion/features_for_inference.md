# Inference Feature Set

The deployed LightGBM model uses 30 features. The exact column order is stored in
`fusion/models/lgbm_v1.meta.json`, and `app/predict.py` requires live inference
rows to match that order exactly.

One inference row predicts demand for a target date `D`. In the live pipeline,
`D` is today's New York date. Because EIA daily demand is published with a lag,
the newest usable observed demand is expected to be `D - 1`, called the anchor
date below.

```text
target date D
anchor date d = D - 1

lag_0       = demand(d)
lag_1       = demand(d - 1)
rolling_7   = mean demand over d-7 through d-1
*_fc        = weather forecast for D
*_next      = calendar / daylight features for D
prediction  = demand(D)
```

Demand is NYISO Zone J daily demand in MWh. Weather units follow the training
data conventions documented in `all_data.md`.

## Demand History

These 12 features describe recent and seasonal demand before the target date.

| Feature | Meaning |
|---|---|
| `lag_0` | Demand on the anchor date `D - 1`; the most recent observed daily demand. |
| `lag_1` | Demand two days before the target date, `D - 2`. |
| `lag_2` | Demand three days before the target date, `D - 3`. |
| `lag_3` | Demand four days before the target date, `D - 4`. |
| `lag_7` | Demand eight days before the target date, roughly the same weekday one week earlier. |
| `lag_14` | Demand fifteen days before the target date, two weekly cycles back. |
| `lag_21` | Demand twenty-two days before the target date, three weekly cycles back. |
| `lag_28` | Demand twenty-nine days before the target date, four weekly cycles back. |
| `lag_365` | Demand 366 days before the target date, carrying annual seasonality. |
| `lag_366` | Demand 367 days before the target date, paired with `lag_365` to handle leap-year alignment. |
| `rolling_7` | Mean demand over the seven days before the anchor date, `D - 8` through `D - 2`; excludes `lag_0`. |
| `rolling_30` | Mean demand over the thirty days before the anchor date, `D - 31` through `D - 2`; a slower recent level signal. |

## Target-Day Weather Forecast

These 10 features describe weather forecast values for the target date `D`.
In offline training, the `_fc` columns were built from shifted observed weather,
so validation is optimistic compared with true live forecasts.

| Feature | Unit | Meaning |
|---|---|---|
| `tmin_fc` | deg F | Forecast minimum temperature for `D`. |
| `tmax_fc` | deg F | Forecast maximum temperature for `D`. |
| `tavg_fc` | deg F | Average target-day temperature, `(tmin_fc + tmax_fc) / 2`. |
| `dew_point_fc` | deg F | Forecast dew point; a direct moisture signal for air-conditioning load. |
| `rh_mean_fc` | percent | Forecast mean relative humidity. |
| `prcp_fc` | inches | Forecast total precipitation. |
| `snow_fc` | inches | Forecast snowfall during the target day. |
| `snwd_fc` | inches | Forecast snow depth on the ground; missing values are treated as `0.0`. |
| `solar_rad_fc` | MJ/m2 | Forecast solar energy reaching a horizontal surface. |
| `cloud_cover_fc` | percent | Forecast mean cloud cover. |

## Thermal Load Features

These 5 features transform target-day temperature into heating/cooling signals
and short-term thermal inertia.

| Feature | Unit | Meaning |
|---|---|---|
| `hdd_fc` | deg F-days | Heating degree days for `D`: `max(50 - tavg_fc, 0)`. The 50 F base was fitted for this series. |
| `cdd_fc` | deg F-days | Cooling degree days for `D`: `max(tavg_fc - 65, 0)`. |
| `tavg_3day_fc` | deg F | Mean temperature over `D - 2`, `D - 1`, and forecast `D`. |
| `cdd_3day_fc` | deg F-days | Cooling degree days computed from `tavg_3day_fc`; captures accumulated heat. |
| `hdd_3day_fc` | deg F-days | Heating degree days computed from `tavg_3day_fc`; captures accumulated cold. |

The degree-day features are included because demand has a V-shaped relationship
with temperature: it rises in cold weather for heating and in hot weather for
cooling.

## Calendar And Daylight

These 3 features are known exactly for the target date.

| Feature | Unit | Meaning |
|---|---|---|
| `is_weekend_next` | 0/1 | `1` if the target date `D` is Saturday or Sunday. |
| `is_holiday_next` | 0/1 | `1` if `D` is a US holiday according to the `holidays` package. |
| `daylight_hours_next` | hours | Sunrise-to-sunset duration for `D`, computed from Visual Crossing `sunriseEpoch` and `sunsetEpoch`. |

## Exact Model Column Order

```text
lag_0, lag_1, lag_2, lag_3, lag_7, lag_14, lag_21, lag_28,
lag_365, lag_366, rolling_7, rolling_30,
tmin_fc, tmax_fc, tavg_fc, dew_point_fc, rh_mean_fc,
prcp_fc, snow_fc, snwd_fc, solar_rad_fc, cloud_cover_fc,
hdd_fc, cdd_fc, tavg_3day_fc, cdd_3day_fc, hdd_3day_fc,
is_weekend_next, is_holiday_next, daylight_hours_next
```
