"""
    12 demand history   lag_0..lag_366, rolling_7, rolling_30
     4 weather forecast hdd_fc, cdd_fc, dew_point_fc, solar_rad_fc
    --
    16 features         (down from 43)

*** PERFECT-FORECAST CAVEAT (inherited) ***
The "_fc" columns are tomorrow's *observed* weather, not a real forecast, so
they are an ORACLE and results are an UPPER BOUND. Measured cost of a realistic
NWS day-ahead forecast on the full 43-feature set: 2.08% -> ~2.7% MAPE. Do not
report these as production numbers.

NOTE: this set carries NO calendar or seasonal columns.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("../")
TIME_SERIES = Path("../time_series")
SAVE_PATH = Path("simple_feature_splits")

TRAIN_END = "2025-01-01"
VAL_END = "2026-01-01"

df = pd.read_csv(ROOT / "all_features_and_target.csv", parse_dates=["period"])
ts = pd.read_csv(TIME_SERIES / "time_series_data.csv", parse_dates=["period"])

# shift()/merge order matters -- sort first (same reasoning as time_series/feature_engineering.py)
df = df.sort_values("period").reset_index(drop=True)

LAG_COLS = [c for c in ts.columns if c.startswith(("lag_", "rolling_"))]
df = df.merge(ts[["period"] + LAG_COLS], on="period", how="left")

# lag_0 = today's demand. Known at end of today, so it is a legal feature.
df.insert(df.columns.get_loc("lag_1"), "lag_0", df["value"])

# tomorrow's information
# Weather that a day-ahead forecast would supply (ORACLE -- see caveat above)
FORECAST_WX = ["hdd", "cdd", "dew_point", "solar_rad"]

# Join day t+1's row onto day t by shifting the key back one day. 
nxt = df[["period"] + FORECAST_WX].copy()
nxt["period"] = nxt["period"] - pd.Timedelta(days=1)
nxt = nxt.rename(columns={c: f"{c}_fc" for c in FORECAST_WX})
df = df.merge(nxt, on="period", how="left")

# assemble features
# Today's weather is deliberately excluded --  today's conditions reach the model through lag_0
HISTORY = ["lag_0"] + [c for c in LAG_COLS if c != "lag_0"]
FC = [f"{c}_fc" for c in FORECAST_WX]

FEATURES = HISTORY + FC

df = df.dropna(subset=["target"] + FEATURES).reset_index(drop=True)

# splits
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
for label, block in [("history", HISTORY), ("forecast wx (ORACLE)", FC)]:
    print(f"  {label:22s} {len(block):2d}  {', '.join(block)}")
