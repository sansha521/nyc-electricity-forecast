# `all_data_recompute.csv` — NYC Daily Electricity Demand Dataset

Daily electricity demand for New York City paired with same-day weather and
calendar features, for modelling / forecasting `value`.

| | |
|---|---|
| **Grain** | one row per calendar day |
| **Range** | 2019-01-01 → 2026-07-22 |
| **Rows** | 2760 — complete, no date gaps (52 imputed, see below) |
| **Columns** | 26 |
| **Target** | `value` (daily electricity demand, MWh) |
| **Built by** | `data_loading/add_missing_eia_to_all.py` → `data_loading/concatenate_data.py` |

`data_loading/all_data.csv` and root-level `all_data_recompute.csv` currently
have the same shape, columns, date range, and imputation count. The current
rebuild artifact used by the root-level modeling scripts is
`all_data_recompute.csv`.

The downstream training table is `all_features_and_target.csv`. It has the same
26 columns plus `target`, where `target = value.shift(-1)` is the next day's
demand. The final row has a null target because no next-day value exists inside
the historical range.

---

## Pipeline

```
nyc_get_request.py    EIA API      -> daily_nyc_demand_cleaned.csv   (NYC demand, gappy)
get_demand.py         EIA API      -> daily_demand_fill_missing.csv  (statewide demand)
                                        |
                        add_missing_eia_to_all.py
                                        |
                                 daily_nyc_demand_filled.csv         (demand, gap-filled)
central_park_all_weathers.py       -> all_weathers.csv               (NOAA weather)
humidity_data.py      Open-Meteo   -> humidity.csv                   (humidity + solar)
weekday_data.py       holidays pkg -> weekday_holiday_data.csv       (calendar)
                                        |
                        concatenate_data.py
                                        |
                             all_data_recompute.csv
```

The three feature inputs are merged onto the demand table with a **left join on
date**, so the demand series determines which days survive. Because that series
is now gap-filled first, every calendar day in the range survives — which also
means the rolling and lag features are computed over a continuous series.

### Sources

**Demand** — EIA API v2, `electricity/rto/daily-region-sub-ba-data`, sub-balancing
authority `ZONJ` (NYISO Zone J = New York City), Eastern timezone rows only.
Units are megawatthours. `nyc_get_request.py` reads `EIA_API_KEY` from the
environment.

**Demand, statewide** — EIA API v2, `electricity/rto/daily-region-data`,
respondent `NY`, type `D`, Eastern rows. This is **all of New York State**, so
it is roughly 3x the NYC series and is *not* interchangeable with it — it is
used only as a scaling reference to impute the 52 days ZONJ is missing.
Pulled by `get_demand.py`.

**Weather** — NOAA NCEI GHCN-Daily, station `USW00094728` (NY City Central Park).
History through 2022 came from `NYC_Central_Park_weather_1869-2022.csv`; 2022
onward from the NCEI Access Data Service. The two overlap through 2022 and are
de-duplicated keeping the newer NCEI rows.

**Humidity / solar** — Open-Meteo Archive API at Central Park's coordinates
(40.7794, -73.9692). NOAA's daily summaries don't carry humidity or radiation,
so this is a second provider; values are model reanalysis, not station
observations.

**Calendar** — Python `holidays` package, US federal holidays.

---

## Features

All correlations and summary statistics quoted below are computed on the **2708
observed rows only** (`is_imputed == 0`), so imputed values can't prop up a
relationship they were derived from. Including them moves every figure by
≤ 0.003.

### Target

| Column | Units | Definition |
|---|---|---|
| `value` | MWh | Total electricity demand for NYISO Zone J over the day |

### Identifier

| Column | Units | Definition |
|---|---|---|
| `period` | date | Calendar date. Sorted ascending (oldest first) |

### Provenance

| Column | Units | Definition |
|---|---|---|
| `is_imputed` | 0/1 | 1 if `value` was estimated from statewide demand rather than reported by EIA. 52 rows flagged |

### Weather — observed (NOAA Central Park)

| Column | Units | Definition |
|---|---|---|
| `tmin` | °F | Minimum temperature |
| `tmax` | °F | Maximum temperature |
| `prcp` | inches | Total liquid precipitation (melted, so snow counts here too) |
| `snow` | inches | **Snowfall** — new snow that fell during the day. A *flow* |
| `snwd` | inches | **Snow depth** — snow lying on the ground at observation time. A *stock* |

`snow` and `snwd` are not interchangeable. `snow > 0` means it actually snowed
that day; `snwd > 0` can persist for days afterward and can be nonzero on a
clear sunny day. For a "did it snow" flag, use `snow`.

`snwd < snow` is normal, not an error — snowfall is measured on a cleared board
every few hours and summed, while depth is a single instantaneous reading that
reflects compaction and melt. Urban pavement makes this especially pronounced
in Central Park.

### Weather — modelled (Open-Meteo)

| Column | Units | Definition |
|---|---|---|
| `rh_mean` | % | Daily mean relative humidity |
| `rh_max` | % | Daily maximum relative humidity |
| `dew_point` | °F | Daily mean dew point — an *absolute* moisture measure |
| `solar_rad` | MJ/m² | Total shortwave solar energy reaching a horizontal surface over the day |
| `cloud_cover` | % | Daily mean sky cloud coverage |
| `daylight_hours` | hours | Sunrise to sunset. Purely astronomical — depends only on date and latitude, unaffected by weather. Range 9.24–15.10 in NYC |
| `sunshine_hours` | hours | Portion of daylight when direct sun actually reached the ground. Always ≤ `daylight_hours` |

`dew_point` is far more useful than `rh_mean` for load modelling. Relative
humidity is relative *to temperature*, so a muggy 90°F day and a damp 45°F day
can both read 70%. Dew point measures absolute moisture, which is what drives
the latent-heat load on air conditioning.

`solar_rad` is the summary measure of the four — it folds in day length, solar
angle (winter sun is weak even at noon) and cloud blocking.

### Calendar

| Column | Units | Definition |
|---|---|---|
| `is_weekend` | 0/1 | 1 if Saturday or Sunday |
| `is_holiday` | 0/1 | 1 if a US federal holiday. 91 days flagged |
| `day_of_week` | 0–6 | Monday = 0 … Sunday = 6 |
| `day_name` | string | Weekday name, for reading the CSV. Drop before training |

`day_of_week` must be treated as **categorical** (one-hot, or a tree model). As
a raw integer it implies Sunday is 6× Monday, which is meaningless.

Holidays carry real signal — restricted to weekdays, holidays average 131,171
MWh against 139,814 for ordinary weekdays, a 6% drop.

### Derived

| Column | Units | Formula |
|---|---|---|
| `tavg` | °F | `(tmin + tmax) / 2` |
| `hdd` | °F-days | `max(0, 50 - tavg)` — note the **50°F** base, not the conventional 65. See below |
| `cdd` | °F-days | `max(0, tavg - 65)` |
| `tavg_3day` | °F | `tavg.rolling(3).mean()` |
| `cdd_3day` | °F-days | `cdd.rolling(3).mean()` |
| `hdd_3day` | °F-days | `hdd.rolling(3).mean()` |
| `tavg_lag1` | °F | `tavg.shift(1)` — previous day's mean temperature |

All seven are rounded to 1 decimal place. Rolling windows are computed **after**
sorting ascending by `period`, otherwise they would look forward in time.

---

## Formulae explained

### Degree days

```
tavg = (tmin + tmax) / 2
hdd  = max(0, 50 - tavg)     heating degree days   <- 50F base, fitted
cdd  = max(0, tavg - 65)     cooling degree days   <- 65F base, fitted
```

Demand versus temperature is **V-shaped**: it rises when it's cold (heating) and
again when it's hot (air conditioning). A linear model given raw temperature
cannot represent a V. Splitting into two one-sided variables linearises each arm
separately, and raw `tavg` only reaches +0.477 against `value` precisely because
it averages the two opposing arms together.

**The bases here are fitted to this series, not taken from convention.**

#### Why the heating base is 50°F

The textbook base for both arms is 65°F, and that is what this dataset used
until it was checked. Binning observed days by `tavg` shows where demand
actually turns:

| `tavg` bin | n | mean `value` | change |
|---|---|---|---|
| (40, 45] | 249 | 126,149 | −3,961 |
| (45, 50] | 238 | 120,924 | −5,224 |
| (50, 55] | 251 | 116,312 | −4,612 |
| **(55, 60]** | 230 | **116,141** | −171 |
| (60, 65] | 221 | 120,268 | +4,127 |
| (65, 70] | 259 | 129,021 | +8,753 |
| (70, 75] | 295 | 147,009 | +17,987 |
| (75, 80] | 286 | 166,300 | +19,291 |

Demand is flat across roughly 50–65°F — 719 days averaging 117,541 MWh — and
climbs away from that floor in both directions. A 65°F heating base puts
**positive `hdd` on all 719 of those days**, so the model is asked to fit a
heating slope through a region where there is no heating response. That both
wastes the coefficient and biases it.

Scanning candidate bases confirms it. Fitting
`value ~ hdd + cdd + is_weekend + is_holiday` by OLS on the 2708 observed rows,
holding the cooling base at 65:

| heating base | R² |
|---|---|
| 45°F | 0.8791 |
| 49°F | 0.8833 |
| **51°F** | **0.8835** |
| 55°F | 0.8804 |
| 60°F | 0.8710 |
| 65°F | 0.8654 |

The optimum is broad and flat between 49 and 53, so **50°F** is chosen as a
round number inside it rather than as a precise estimate. The same scan run over
the cooling base returns 65°F as the peak, which is why `cdd` is unchanged —
the conventional base happens to be right for the cooling arm and wrong for the
heating arm.

Fitted slopes at 50/65: **+1,103 MWh per heating °F-day, +3,941 MWh per cooling
°F-day.** Cooling response is ~3.6× steeper, but heating is real and worth
modelling.

#### What the change buys

| | R² (in-sample) | holdout MAPE | holdout MAE |
|---|---|---|---|
| 65 / 65 | 0.8654 | 5.13% | 7,299 |
| **50 / 65** | **0.8835** | **4.36%** | **6,229** |

Holdout trains on 2019–2024 and tests on 2025 onward — a time-based split, not
random. A 15% relative cut in forecast error from changing one constant.

#### Don't judge a one-sided variable by its overall correlation

The overall correlation of `hdd` with `value` is a **misleading diagnostic** and
earlier versions of this document read it wrong. The figures:

| | corr. with `value`, all days | corr. on heating days (`tavg < 50`, n = 983) |
|---|---|---|
| `hdd` base 65 | −0.246 | +0.740 |
| `hdd` base 50 | −0.007 | +0.740 |

The negative whole-series number does **not** mean cold days suppress demand.
It is an artifact of the V: `hdd` is large in winter, demand peaks in summer, so
a single linear correlation across all seasons comes out negative even though
the cold-side response is strongly positive. Restricted to days that are
actually in the heating regime, `hdd` correlates **+0.740** with demand, and
demand climbs from the 116,141 floor to 162,944 in the coldest bin.

The previous claim here — that the negative sign showed NYC heats with gas and
steam so cold days run below average — was an over-reading of that artifact. The
cold-side response is genuine; it is simply smaller than the cooling response
and hidden by aggregation. Cooling still dominates the series.

Note also that the correlation is **identical at both bases** on heating days
(+0.740), because within the heating regime the two differ only by a constant.
The base change does not improve correlation and was never going to — it
improves *fit*, by not asserting a heating effect on the 719 mild days where
none exists.

### Rolling temperature (thermal inertia)

Buildings store heat, so day 3 of a heat wave draws more power than day 1 at the
same temperature — the structure is already saturated and AC never catches up.
On hot weekdays (`cdd > 8`, n = 409):

| feature | corr. with `value` |
|---|---|
| `cdd_3day` | **+0.848** |
| `cdd` | +0.805 |

Days building on prior heat average 177,355 MWh versus 174,234 for days
following milder weather — roughly 3,100 MWh attributable to heat accumulated
*before* the day began.

Across all 2708 observed days the two tie at 0.848, because the seasonal swing dominates
and both capture it equally. `cdd_3day` earns its place specifically on hot
days, which is where forecast error is expensive.

---

## Known issues

**52 imputed days.** EIA has no ZONJ rows for these 52 dates. The gap is in the
**source itself**, not the pipeline — they are absent from the raw download
before any timezone filtering or merging, and re-querying the sub-BA endpoint
still returns nothing for them, so this is not a stale-download problem. There
are no duplicate dates.

| range | days |
|---|---|
| 2024-10-09 | 1 |
| 2025-01-15 | 1 |
| **2025-11-06 → 2025-12-10** | **35** |
| **2025-12-21 → 2025-12-31** | **11** |
| 2026-02-09 | 1 |
| 2026-04-19 | 1 |
| 2026-05-12 | 1 |
| 2026-05-18 | 1 |

46 of the 52 fall in a two-month window at the end of 2025 — November and
December 2025 have only 15 of 61 days reported. The six isolated single days
look like ordinary EIA reporting dropouts; the long run looks like a genuine
reporting outage for Zone J.

*How they're filled.* `add_missing_eia_to_all.py` estimates NYC demand as
`(local NYC/statewide ratio) x (statewide NY demand)`. The two series correlate
at **0.95**, and the ratio is seasonally stable — 0.309 ± 0.006 in December,
0.345 ± 0.012 in July. The ratio is anchored on the **nearest 14 observed days**
rather than a day-of-year average, so it stays on the correct side of any
year-over-year level shift (COVID, the post-2020 trend).

*How accurate.* Backtested by punching synthetic 35-day holes through the
observed series and predicting the held-out days:

| | error |
|---|---|
| MAPE | **2.44%** |
| median | 2.04% |
| p90 | 5.11% |

A day-of-year-window ratio was also tried and did worse (2.73% MAPE), which is
what motivated the local anchor. Note the Nov 6 → Dec 10 block is the hardest
case: its anchor days sit in late October and mid-December, where the ratio is
mid-seasonal-drift, so expect the worse end of that range there.

*Use the flag.* Filter on `is_imputed == 0` for anything where fabricated
targets would mislead — evaluating a model, reporting actuals, fitting residual
diagnostics. **50 of the 52 fall in 2025–2026**, so ignoring the flag biases the
most recent period specifically, which is usually the holdout.

*Side effect: the rolling features are now correct.* They previously spanned the
gaps invisibly — `tavg_lag1` on 2025-12-11 pulled from 2025-11-05, 36 days
earlier, and `cdd_3day` averaged days spread across five weeks. Those rows were
quietly **wrong** rather than missing. Filling before the merge makes the series
continuous, which corrected `tavg_3day` on 16 rows, `hdd_3day` on 14,
`tavg_lag1` on 8, and `cdd_3day` on 4. All of `value` on observed days is
unchanged.

*It still leans on one winter.* The imputed stretch covers heating season plus
Christmas and New Year, exactly where holiday and cold-weather effects live. The
fill makes those rows usable as *features*; it does not make them evidence about
winter demand.

**Leading NaNs.** `tavg_3day`, `cdd_3day`, `hdd_3day` are null for the first 2
rows and `tavg_lag1` for the first row — no history exists at the start of the
series. Drop those rows or handle explicitly; don't let them become zeros.

**`snwd` has 4 nulls.** NOAA didn't report depth on those days.

**Mixed provenance.** NOAA columns are station observations; Open-Meteo columns
are model reanalysis on a grid cell that resolves to roughly 40.81, −74.02 —
several km from the Central Park station. They will not agree perfectly.

**`sunshine_hours` is quantized.** It's derived by counting hourly model steps
above a radiation threshold, so it clusters on whole numbers (14.0 appears 228
times, 12.0 appears 104). Least trustworthy of the solar group.

---

## Modelling notes

**Collinearity.** `cdd`/`cdd_3day`/`tavg`/`dew_point`/`tmax` all move together —
hot days are humid days in NYC. In plain OLS they will fight over coefficients
and produce unstable, uninterpretable signs. Use a tree-based model or
regularisation, or select one from each group.

**COVID is unhandled and matters.** Weekday mean demand by year, at essentially
identical cooling load:

| year | weekday mean | mean `cdd` |
|---|---|---|
| 2019 | 146,385 | 3.3 |
| 2020 | **134,367** | 3.6 |
| 2021 | 137,550 | 3.7 |
| 2023 | 135,588 | 3.4 |
| 2026 | 143,492 | 3.4 |

2020 fell 8% below 2019 on slightly *hotter* weather, and demand still hasn't
returned to 2019 levels — remote work permanently moved load off the commercial
grid. No column in this dataset can express that, so a model will distort its
weather coefficients trying to explain it. Consider a `covid` dummy
(≈2020-03-15 to 2021-06-30) plus a linear time trend.

**Weak features.** Measured against `value`, all days / cooling days only:

| feature | all | cooling |
|---|---|---|
| `daylight_hours` | +0.351 | +0.293 |
| `solar_rad` | +0.295 | +0.106 |
| `sunshine_hours` | +0.259 | +0.127 |
| `cloud_cover` | −0.035 | +0.020 |
| `rh_mean` | +0.073 | +0.133 |

`cloud_cover` and `rh_mean` are near-flat. `solar_rad` largely collapses on
cooling days, meaning its overall 0.295 is mostly just "summer", which `cdd`
already encodes. `daylight_hours` holds up best and is free (deterministic from
the date).

**Lagged demand is not included in this base table.** `value.shift(1)` and
`value.shift(7)` would be the strongest predictors available, and are
appropriate for day-ahead forecasting. They are deliberately absent here because
they dominate and obscure the weather relationships if the goal is explaining
*what drives* demand. Downstream modeling scripts add lag and rolling demand
features separately from `time_series/time_series_data.csv`.

---

## Not yet included

- COVID / regime dummy and long-run time trend
- Day-after-holiday and bridge days (the Friday after Thanksgiving behaves like
  a holiday but `is_holiday` says otherwise)
- Fourier seasonality terms — `sin`/`cos` of day-of-year, for school calendars
  and vacation patterns that temperature misses
- Wind speed (`wind_speed_10m_max`, available from the same Open-Meteo call)
