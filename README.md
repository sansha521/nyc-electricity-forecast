# NYC Electricity Demand Forecasting

Daily forecasting for New York City electricity demand, focused on NYISO Zone J.
The project combines demand history, weather, calendar signals, and a LightGBM
model to produce one demand prediction per target date and write it to Postgres.

The repository has two halves:

- An offline modelling pipeline that builds historical training data, engineers
  features, compares models, and saves a LightGBM artifact.
- A live inference pipeline that runs from GitHub Actions, fetches current EIA
  demand and Visual Crossing weather, builds the exact 30-feature row expected by
  the model, predicts, and upserts the result.

## Current Status

| Item | Value |
|---|---|
| Target | Daily NYC electricity demand, EIA sub-BA `ZONJ` |
| Deployed model | `fusion/models/lgbm_v1.txt` |
| Model type | LightGBM regression booster |
| Feature count | 30 |
| Validation result | MAE 3,157, RMSE 4,644, MAPE 2.16%, bias -1,944 MWh |
| Live schedule | GitHub Actions daily at `22:30 UTC`, with retries |
| Runtime storage | Postgres, typically Neon |
| Python | 3.13 |

## Forecasting Semantics

The live job runs on target day `D` and predicts demand for that same date.
Because EIA publishes daily demand with a lag, the newest usable observed demand
is expected to be `D - 1`.

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

This naming comes from the offline day-ahead modelling setup, where `_fc` and
`_next` meant "next day." In production they describe the target day being
predicted. Keep this distinction in mind when changing `app/build_row_inference.py`.

## Data Sources

| Source | Used For | Code |
|---|---|---|
| EIA API v2, `electricity/rto/daily-region-sub-ba-data`, sub-BA `ZONJ` | Historical and live NYC demand | `data_loading/nyc_get_request.py`, `app/ingest_demand.py` |
| EIA API v2 regional demand | Imputing missing NYC demand days from broader NY demand | `data_loading/add_missing_eia_to_all.py` |
| NOAA / Central Park weather files | Historical observed weather | `data_loading/central_park_weather.py`, CSV files under `data_loading/` |
| Open-Meteo archive | Historical humidity, dew point, cloud cover, solar radiation, daylight-related data | `data_loading/humidity_data.py` |
| Visual Crossing Timeline API | Live weather forecast and recent observed weather for inference | `app/vc.py`, `app/build_row_inference.py` |
| `holidays` Python package | US holiday indicators | `app/build_row_inference.py`, `data_loading/weekday_data.py` |

Visual Crossing sunrise and sunset arithmetic should use `sunriseEpoch` and
`sunsetEpoch`, not the formatted `sunrise` and `sunset` strings.

## Repository Architecture

```text
.
|-- app/
|   |-- run_daily.py              Live entry point: build row, predict, upsert
|   |-- build_row_inference.py    Live feature row builder used by run_daily.py
|   |-- predict.py                Loads LightGBM model and checks feature order
|   |-- evaluation.py             Pure scoring and summary metric functions
|   |-- evaluate_predictions.py   Refreshes actuals and scores stored predictions
|   |-- ingest_demand.py          EIA live demand fetch helper
|   |-- vc.py                     Visual Crossing helper
|   |-- eia.py                    EIA helper
|   |-- db.py                     Postgres connections and upserts
|   |-- bootstrap.py              Historical DB bootstrap helpers
|   `-- db/schema.sql             Postgres table definitions
|
|-- data_loading/
|   |-- *.py                      Historical demand/weather build scripts
|   |-- *.csv                     Intermediate historical data files
|   `-- README.md                 Data loading details
|
|-- time_series/
|   |-- feature_engineering.py     Demand-only lag/rolling feature generation
|   |-- naive.py, ridge.py, rf.py, xgb.py
|   |-- splits/                   Demand-only train/validation/test splits
|   `-- metrics.csv               Baseline metrics
|
|-- fusion/
|   |-- forecast_feature_engineering.py
|   |-- feature_inference.py       Produces the 30 live-servable features
|   |-- train_lgbm.py              Trains and saves LightGBM artifacts
|   |-- metrics.csv                Model comparison metrics
|   |-- models/
|   |   |-- lgbm_v1.txt
|   |   |-- lgbm_v1.meta.json
|   |   `-- lgbm_v1.feature_importance.csv
|   |-- inference_feature_splits/  Final train/validation/test feature sets
|   `-- *_splits/                 Earlier/generated experiment split sets
|
|-- .github/workflows/
|   `-- daily_forecast.yml         Scheduled GitHub Actions job
|-- all_data_recompute.csv         Historical modelling table
|-- all_features_and_target.csv    Modelling table plus next-day target
|-- all_data.md                    Data dictionary / audit notes
|-- pyproject.toml                 Project dependency ranges
`-- requirements.txt              Pinned dependencies used by CI
```

The `app/` folder is the production package. The other top-level folders are
mostly script collections and modelling artifacts. Some scripts use relative paths
and are easiest to run from their own directory.

## Machine Learning Pipeline

### 1. Historical Data Build

The historical data pipeline starts from daily demand and joins weather,
humidity/solar/cloud signals, and calendar features by date. The key outputs are:

- `all_data_recompute.csv`: main feature table before target creation.
- `all_features_and_target.csv`: table with `target`, the next-day demand value.
- `all_data.md`: documentation and audit notes for the assembled data.

Important data handling choices:

- Missing ZONJ demand days are imputed before lag and rolling features are built.
  This avoids rolling windows silently spanning gaps.
- Historical weather comes from multiple providers. Live inference uses Visual
  Crossing, so there is some provider shift between training and production.
- Demand is treated as the calendar backbone: days without demand are not useful
  training rows.

### 2. Feature Engineering

The deployed model uses 30 live-servable features.

Demand history:

```text
lag_0, lag_1, lag_2, lag_3, lag_7, lag_14, lag_21, lag_28,
lag_365, lag_366, rolling_7, rolling_30
```

Weather forecast for the target day:

```text
tmin_fc, tmax_fc, tavg_fc, dew_point_fc, rh_mean_fc, prcp_fc,
snow_fc, snwd_fc, solar_rad_fc, cloud_cover_fc,
hdd_fc, cdd_fc, tavg_3day_fc, cdd_3day_fc, hdd_3day_fc
```

Calendar / astronomy:

```text
is_weekend_next, is_holiday_next, daylight_hours_next
```

The core weather transforms are:

```python
tavg = (tmin + tmax) / 2
hdd = max(50 - tavg, 0)
cdd = max(tavg - 65, 0)
daylight_hours_next = round((sunsetEpoch - sunriseEpoch) / 3600.0, 2)
```

The 3-day weather features combine two observed/recent days with the target-day
forecast. Snow depth missing values are treated as `0.0`.

### 3. Model Training

Model comparison lives under `fusion/`:

- `ridge.py`
- `rf.py`
- `xgb.py`
- `light_gbm.py`
- `train_lgbm.py`
- `evaluate.py`

The saved model is `fusion/models/lgbm_v1.txt`, with metadata in
`fusion/models/lgbm_v1.meta.json`. The metadata stores:

- ordered feature list
- LightGBM version
- validation metrics
- training cutoff information
- best hyperparameters

The live predictor asserts that the inference DataFrame columns match the metadata
feature order exactly. This is important because LightGBM predicts by column
position.

Current LightGBM hyperparameters:

```json
{
  "n_estimators": 800,
  "learning_rate": 0.03,
  "num_leaves": 31,
  "max_depth": 4,
  "min_child_samples": 5,
  "subsample": 0.7,
  "subsample_freq": 1,
  "colsample_bytree": 0.8,
  "reg_alpha": 0.5,
  "reg_lambda": 0.1
}
```

## Live Inference Pipeline

The live entry point is:

```bash
python -m app.run_daily
```

The job performs:

1. Determine the target date in `America/New_York`.
2. Fetch recent EIA demand.
3. Build the 30-feature inference row in `app/build_row_inference.py`.
4. Fetch target-day Visual Crossing forecast.
5. Compute lags, rolling means, weather transforms, holiday/weekend flags, and
   daylight hours.
6. Upsert the weather forecast snapshot to `weather_forecast`.
7. Load `fusion/models/lgbm_v1.txt`.
8. Predict one float demand value.
9. Upsert the result and feature JSON to `predictions`.

If EIA has not yet published the required anchor demand date, the code raises
`DemandDataUnavailable`. `app/run_daily.py` exits with status `75` for that case,
which lets the GitHub workflow retry instead of producing a misleading prediction
from stale data.

## Evaluation Pipeline

The evaluation entry point is:

```bash
python -m app.evaluate_predictions
```

It scores live predictions once the actual EIA demand for the target date is
available. Each run:

1. Refreshes the last 30 days of EIA demand into `demand_daily`.
2. Joins `predictions.target_date` to `demand_daily.date`.
3. Excludes imputed actual demand rows from scoring.
4. Computes error, absolute error, percent error, and absolute percent error.
5. Upserts one permanent row per scored target date into `prediction_scores`.
6. Logs the latest scored prediction plus last-7 and last-30 summary metrics.

Scores are refreshed, not frozen. This is intentional because EIA can revise
recently published demand values. Re-running evaluation for the same date updates
`prediction_scores` with the current actual demand and recomputed metrics.

The metric definitions are:

```text
error         = predicted_demand - actual_demand
abs_error     = abs(error)
pct_error     = error / actual_demand * 100
abs_pct_error = abs(pct_error)
MAE           = mean(abs_error)
RMSE          = sqrt(mean(error ** 2))
MAPE          = mean(abs_pct_error)
bias          = mean(error)
```

Example scored result from the first live evaluation:

```text
target_date = 2026-08-14
predicted   = 172,982.71
actual      = 176,757.00
error       = -3,774.29
APE         = 2.14%
```

## GitHub Actions

The workflow is `.github/workflows/daily_forecast.yml`.

It runs:

```text
30 22 * * *
```

That is `22:30 UTC`, which is usually `18:30` Eastern during daylight saving time
and `17:30` Eastern during standard time.

The workflow also supports manual runs through `workflow_dispatch`.

Runtime secrets required in GitHub:

```text
DATABASE_URL
EIA_API_KEY
VISUAL_CROSSING_API_KEY
```

The job installs `requirements.txt`, runs `python -m app.run_daily`, and retries
up to four times with 15 minutes between attempts before one final run. This is
mainly to handle late EIA publication. After a successful forecast run, the
workflow runs `python -m app.evaluate_predictions` as a separate step to score
any recent predictions whose actual demand is now available.

## Database

Schema file:

```bash
app/db/schema.sql
```

Tables:

| Table | Purpose |
|---|---|
| `demand_daily` | Observed demand and imputation flag |
| `weather_daily` | Historical/observed weather |
| `weather_forecast` | Latest forecast snapshot for each target date |
| `predictions` | Prediction, model version, and exact feature JSON |
| `prediction_scores` | Permanent per-date scores joining predictions to actual demand |

The most important table for live serving is `predictions`. The `features` JSONB
column makes each prediction auditable and allows future scoring against actual
demand without reconstructing the feature row.

`prediction_scores` is maintained by `app.evaluate_predictions`. It is upserted
over a rolling 30-day window so revised EIA actuals update recent scores.

Current implementation note: `app/build_row_inference.py` still fetches demand
directly from EIA when building the row, even though `app/ingest_demand.py` now
persists recent demand into `demand_daily`. Consolidating live inference around the
database demand table is a good cleanup target.

## Setup

Create and install a local environment:

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\pip install -r requirements.txt

# macOS / Linux
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Create `.env` in the repository root:

```text
EIA_API_KEY=...
VISUAL_CROSSING_API_KEY=...
DATABASE_URL=postgresql://user:pass@host/db?sslmode=require
```

Initialize Postgres:

```bash
psql "$DATABASE_URL" -f app/db/schema.sql
```

On Windows, use your Postgres client equivalent, for example:

```powershell
psql $env:DATABASE_URL -f app\db\schema.sql
```

Run the live job:

```bash
python -m app.run_daily
```

Run the evaluation pipeline:

```bash
python -m app.evaluate_predictions
```

Run a prediction row only:

```bash
python -m app.build_row_inference
```

Run syntax checks:

```bash
python -m py_compile app/build_row_inference.py app/run_daily.py app/predict.py app/evaluate_predictions.py
```

Run tests:

```bash
python -m unittest discover
```

## Bootstrap and Backfills

`app/bootstrap.py` contains helper functions to populate:

- historical demand from `all_features_and_target.csv`
- recent EIA demand after the historical file ends
- historical observed weather
- recent observed weather from Visual Crossing

At the moment, `main()` prints messages but has all bootstrap calls commented out.
To use it, uncomment the specific loaders you need or call the functions directly.
Also note that `app/bootstrap.py` imports `db`, `eia`, and `vc` as sibling modules,
so it is currently easiest to run from inside `app/` unless the imports are changed
to `app.db`, `app.eia`, and `app.vc`.

## Retraining

A typical retraining flow is:

```bash
cd fusion
python forecast_feature_engineering.py
python feature_inference.py
python train_lgbm.py --save models/lgbm_v2
```

Before deploying a new model:

1. Confirm the validation metrics improve or the tradeoff is intentional.
2. Inspect `models/lgbm_v2.meta.json`.
3. Confirm the feature list exactly matches what live inference can build.
4. Update `app/predict.py` if the deployed model filename changes.
5. Run `python -m app.run_daily` against a safe database or staging environment.

## Development Notes

- `app/run_daily.py` imports `build_row` from `app.build_row_inference`.
- `requirements.txt` is what GitHub Actions installs. Keep it in sync with any
  dependency changes.
- The workflow runs on Ubuntu, while local development here may happen on Windows.
  Avoid shell-specific assumptions in Python code.
- GitHub Actions cron uses UTC and is not daylight-saving aware.
- Re-running the same target date is safe because forecast and prediction writes
  use upserts.
- Re-running evaluation is safe because `prediction_scores` uses upserts and
  intentionally refreshes recent dates after EIA revisions.
- The project intentionally fails fast on missing features, NaNs, and feature-order
  mismatches. Silent prediction drift is worse than a failed run.

## Known Gaps

- Training `_fc` columns are based on shifted observed weather, not archived
  forecasts as issued. That makes validation an optimistic estimate of live
  performance.
- The inference builder still fetches its own EIA demand window instead of reading
  the demand rows just upserted by `app/ingest_demand.py`.
- `weather_forecast` is keyed by `target_date`, so it stores the latest forecast
  for a day rather than a full `(issued_at, target_date)` forecast archive.
- `app/bootstrap.py` needs import cleanup and explicit command-line switches before
  it is a reliable one-command bootstrap tool.

## Quick Commands

```bash
# Live daily run
python -m app.run_daily

# Refresh actuals and score recent predictions
python -m app.evaluate_predictions

# Build and print today's inference row
python -m app.build_row_inference

# Train a new LightGBM model from fusion/
cd fusion
python train_lgbm.py --save models/lgbm_v2

# Check currently deployed model metadata
python -m json.tool fusion/models/lgbm_v1.meta.json
```
