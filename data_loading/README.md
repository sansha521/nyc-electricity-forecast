# Data Loading

This directory contains the historical data rebuild scripts and CSV inputs used
to create the modeling table. The downstream modeling table is documented in
`../all_data.md`.

Most scripts use relative file paths, so run them from this directory:

```bash
cd data_loading
python concatenate_data.py
```

## Pipeline

```text
nyc_get_request.py
  -> daily_nyc_demand.csv
  -> daily_nyc_demand_cleaned.csv

get_demand.py
  -> daily_eastern_demand.csv
  -> daily_demand_fill_missing.csv

add_missing_eia_to_all.py
  daily_nyc_demand_cleaned.csv + daily_demand_fill_missing.csv
  -> daily_nyc_demand_filled.csv

central_park_weather.py
  NYC_Central_Park_weather_1869-2022.csv
  -> weather_2019-2022.csv

central_park_all_weathers.py
  weather_2019-2022.csv + central_park_2022-2026.csv
  -> all_weathers.csv

humidity_data.py
  -> humidity.csv

weekday_data.py
  -> weekday_holiday_data.csv

concatenate_data.py
  daily_nyc_demand_filled.csv
  + all_weathers.csv
  + humidity.csv
  + weekday_holiday_data.csv
  -> ../all_data_recompute.csv
```

## Final Output

| File | Description |
|---|---|
| `all_data.csv` | Historical modeling table kept in this directory for reference. |
| `../all_data_recompute.csv` | Current output written by `concatenate_data.py`. |

`all_data.csv` and `../all_data_recompute.csv` are duplicate copies today. If
the historical rebuild is rerun, `all_data.csv` can become stale unless it is
refreshed too.

## Raw Inputs

These are source or near-source files. Keep them if the public repo should allow
historical data rebuilds without re-downloading all inputs.

| File | Source | Used by |
|---|---|---|
| `daily_nyc_demand.csv` | EIA Zone J demand API response. | `nyc_get_request.py` writes it. |
| `daily_eastern_data.csv` | EIA statewide NY demand API response, all timezones. | Reference download from `get_demand.py`. |
| `daily_eastern_demand.csv` | EIA statewide NY demand filtered to Eastern rows. | `get_demand.py`. |
| `central_park_2022-2026.csv` | NOAA Central Park recent weather. | `central_park_all_weathers.py`. |
| `NYC_Central_Park_weather_1869-2022.csv` | NOAA Central Park historical weather archive. | `central_park_weather.py`. |

Statewide NY demand is not interchangeable with NYC demand. It is used only as a
scaling reference to impute missing Zone J days.

## Intermediate Files

These are regenerable from the raw inputs and scripts.

| File | Produced by | Used by |
|---|---|---|
| `daily_nyc_demand_cleaned.csv` | `nyc_get_request.py` | `add_missing_eia_to_all.py` |
| `daily_demand_fill_missing.csv` | `get_demand.py` | `add_missing_eia_to_all.py` |
| `daily_nyc_demand_filled.csv` | `add_missing_eia_to_all.py` | `concatenate_data.py` |
| `weather_2019-2022.csv` | `central_park_weather.py` | `central_park_all_weathers.py` |
| `all_weathers.csv` | `central_park_all_weathers.py` | `concatenate_data.py` |
| `humidity.csv` | `humidity_data.py` | `concatenate_data.py` |
| `weekday_holiday_data.csv` | `weekday_data.py` | `concatenate_data.py` |

## Duplicate Kept For Compatibility

| File | Why it remains |
|---|---|
| `daily_demand.csv` | Duplicate of `daily_demand_fill_missing.csv`, but `weekday_data.py` reads it. Keep it unless that script is updated to read `daily_demand_fill_missing.csv` directly. |

## Scripts

| Script | Role |
|---|---|
| `nyc_get_request.py` | Fetch EIA Zone J demand. |
| `get_demand.py` | Fetch EIA statewide NY demand. |
| `humidity_data.py` | Fetch Open-Meteo humidity, dew point, solar, cloud, and daylight data. |
| `central_park_weather.py` | Slice 2019+ rows from the NOAA historical archive. |
| `central_park_all_weathers.py` | Concatenate and de-duplicate the two NOAA weather sources. |
| `add_missing_eia_to_all.py` | Impute missing Zone J demand days from statewide demand. |
| `weekday_data.py` | Build weekend and holiday flags. |
| `concatenate_data.py` | Merge demand, weather, humidity, and calendar data into the modeling table. |

## Notes

- `nyc_get_request.py` fetches NYC Zone J demand. `get_demand.py` fetches
  statewide NY demand.
- Missing Zone J demand days are imputed and marked with `is_imputed = 1` in
  `daily_nyc_demand_filled.csv` and `all_data.csv`.
- Weather features mix NOAA station observations and Open-Meteo reanalysis data.
- Superseded 2019-only weather files and the older hourly Open-Meteo export were
  removed from the public repo to keep the data directory readable.
