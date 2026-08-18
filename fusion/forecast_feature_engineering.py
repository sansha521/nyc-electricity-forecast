"""Day-ahead demand forecasting: predict tomorrow's demand from
   (a) demand history through today, and (b) tomorrow's weather forecast.

Information set = everything knowable at the END OF TODAY (day t).
Target          = demand on day t+1.

*** PERFECT-FORECAST CAVEAT ***
all_features_and_target.csv contains weather OBSERVATIONS, not forecasts. The
"_fc" columns below are tomorrow's *observed* weather, which a real day-ahead
system would not have -- it would have an NWS/OpenWeather forecast carrying its
own error. So these features are an ORACLE: results are an UPPER BOUND on what a
deployable model achieves. Use them to decide whether wiring up a real forecast
feed is worth it, and do not report them as production numbers.

Columns marked EXACT below are genuinely knowable today and carry no such caveat.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("../")
TIME_SERIES = Path("../time_series")
SAVE_PATH = Path("forecast_feature_splits")

# Split boundaries match feature_engineering.py so metrics stay comparable
TRAIN_END = "2025-01-01"
VAL_END = "2026-01-01"

# ---------------------------------------------------------------- load & merge
df = pd.read_csv(ROOT / "all_features_and_target.csv", parse_dates=["period"])
ts = pd.read_csv(TIME_SERIES / "time_series_data.csv", parse_dates=["period"])

# shift()/merge order matters -- sort first (same reasoning as time_series/feature_engineering.py)
df = df.sort_values("period").reset_index(drop=True)

LAG_COLS = [c for c in ts.columns if c.startswith(("lag_", "rolling_"))]
df = df.merge(ts[["period"] + LAG_COLS], on="period", how="left")

# lag_0 = today's demand. Known at end of today, so it is a legal feature.
df.insert(df.columns.get_loc("lag_1"), "lag_0", df["value"])

# snow depth is NaN on 4 days; missing here means "no snow on the ground"
df["snwd"] = df["snwd"].fillna(0.0)

# ------------------------------------------------------- tomorrow's information
# Weather that a day-ahead forecast would supply (ORACLE -- see caveat above)
FORECAST_WX = [
    "tmin", "tmax", "tavg", "dew_point", "rh_mean", "rh_max",
    "prcp", "snow", "snwd", "solar_rad", "cloud_cover", "sunshine_hours",
    "hdd", "cdd",                          # nonlinear temp transforms: the V-shape
    "tavg_3day", "cdd_3day", "hdd_3day",   # 2 observed days + 1 forecast day
]

# EXACT: knowable today with zero uncertainty (calendar + astronomy)
EXACT_NEXT = ["is_weekend", "is_holiday", "day_of_week", "daylight_hours"]

# Join day t+1's row onto day t by shifting the key back one day. Date-based
# rather than positional so a missing date can never silently misalign.
nxt = df[["period"] + FORECAST_WX + EXACT_NEXT].copy()
nxt["period"] = nxt["period"] - pd.Timedelta(days=1)
nxt = nxt.rename(columns={c: f"{c}_fc" for c in FORECAST_WX}
                 | {c: f"{c}_next" for c in EXACT_NEXT})
df = df.merge(nxt, on="period", how="left")

# ------------------------------------------------------------ derived features
# Today's weather is deliberately excluded -- the weather signal comes only from
# tomorrow's forecast, and today's conditions reach the model through lag_0
# (today's demand is itself a thermometer reading).
#
# Today-vs-tomorrow deltas are excluded for the same reason: d_tavg = tavg_fc -
# tavg_today, so with tavg_fc present a linear model recovers today's weather
# exactly. Adding them back would reintroduce what this cut is meant to remove.

# Tomorrow's weekday, one-hot. A single 0-6 integer is meaningless to a linear
# model -- it can only fit one slope across Mon..Sun. (Measured: ~720 MW of MAE.)
# dow = pd.get_dummies(df["day_of_week_next"].astype("Int64"), prefix="dow")
# dow = dow.reindex(columns=[f"dow_{i}" for i in range(7)], fill_value=0).astype(float)
# df = pd.concat([df, dow], axis=1)

# # Annual cycle, smooth. Complements lag_365/366 without their single-day noise.
# doy = (df["period"] + pd.Timedelta(days=1)).dt.dayofyear
# for k in (1, 2):
#     df[f"doy_sin_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
#     df[f"doy_cos_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)

# ------------------------------------------------------------ assemble features
HISTORY = ["lag_0"] + [c for c in LAG_COLS if c != "lag_0"]
FC = [f"{c}_fc" for c in FORECAST_WX]
CALENDAR = ["is_weekend_next", "is_holiday_next", "daylight_hours_next"]
# SEASONAL = [f"dow_{i}" for i in range(7)] + [f"doy_{t}_{k}" for k in (1, 2) for t in ("sin", "cos")]

FEATURES = HISTORY + FC + CALENDAR

df = df.dropna(subset=["target"] + FEATURES).reset_index(drop=True)

# ------------------------------------------------------------------- integrity
# target[t] must be demand on t+1, and tomorrow's features must come from t+1.
chk = df.set_index("period")
nxt_day = chk.index + pd.Timedelta(days=1)
present = nxt_day.isin(chk.index)
assert np.allclose(chk.loc[present, "target"], chk["value"].reindex(nxt_day[present])), \
    "target is not next-day demand"
assert np.allclose(chk.loc[present, "tavg_fc"], chk["tavg"].reindex(nxt_day[present])), \
    "_fc columns are not next-day weather"
assert np.allclose(chk.loc[present, "is_holiday_next"], chk["is_holiday"].reindex(nxt_day[present])), \
    "_next columns are not next-day calendar"
assert not df[FEATURES].isna().any().any(), "NaNs left in features"
print(f"alignment checks passed ({present.sum()} consecutive-day pairs verified)\n")

# ----------------------------------------------------------------------- splits
train = df[df["period"] < TRAIN_END]
val = df[(df["period"] >= TRAIN_END) & (df["period"] < VAL_END)]
test = df[df["period"] >= VAL_END]

SAVE_PATH.mkdir(exist_ok=True)
for name, part in {"train": train, "val": val, "test": test}.items():
    part[FEATURES].to_csv(SAVE_PATH / f"X_{name}.csv", index=False)
    part[["target"]].to_csv(SAVE_PATH / f"y_{name}.csv", index=False)
    imp = int(part["is_imputed"].sum())
    print(f"{name:5s} {part['period'].min().date()} -> {part['period'].max().date()}  "
          f"n={len(part):5d}  imputed_demand_rows={imp}")

print(f"\n{len(FEATURES)} features -> {SAVE_PATH}/")
for label, block in [("history", HISTORY), ("forecast wx (ORACLE)", FC),
                     ("calendar (exact)", CALENDAR)]:
    print(f"  {label:22s} {len(block):2d}  {', '.join(block)}")
