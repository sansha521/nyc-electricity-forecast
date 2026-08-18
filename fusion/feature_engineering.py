import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("../")
TIME_SERIES_DATA_PATH = Path("../time_series")

df = pd.read_csv(DATA_PATH / "all_features_and_target.csv", parse_dates=["period"])
df_time_series = pd.read_csv(TIME_SERIES_DATA_PATH / "time_series_data.csv", parse_dates=["period"])

LAG_COLS = [c for c in df_time_series.columns if c.startswith(("lag_", "rolling_"))]
df = df.merge(df_time_series[["period"] + LAG_COLS], on="period", how="left")

# df.dropna(subset=["target"] + LAG_COLS, inplace=True)
# print(df.head())
# print(df.shape)

# nan_rows = df[df.isna().any(axis=1)]
# print(nan_rows)
# snwd == NaN

df.insert(df.columns.get_loc("lag_1"), "lag_0", df["value"])

df.dropna(inplace=True)
# print(df.head)
# print(df.shape)

# print(df.columns)

df[["rolling_7", "rolling_30"]] = df[["rolling_7", "rolling_30"]].round(1)

train = df[df["period"] < "2025-01-01"]

val = df[
    (df["period"] >= "2025-01-01") &
    (df["period"] < "2026-01-01")
]

test = df[df["period"] >= "2026-01-01"]

# features = [
#     "lag_1",
#     "lag_7",
#     "CDD",
#     "HDD",
#     "TMAX",
#     "is_weekend",
#     "is_holiday"
# ]

X_train = train.drop(columns=["period", "value", "target", "is_imputed", "day_name"])
y_train = train["target"]

X_val = val.drop(columns=["period", "value", "target", "is_imputed", "day_name"])
y_val = val["target"]

X_test = test.drop(columns=["period", "value", "target", "is_imputed", "day_name"])
y_test = test["target"]

# print(X_train.head)
# print(y_train.head)

SPLITS = Path("splits")
SPLITS.mkdir(exist_ok=True)

for name, obj in {
    "X_train": X_train, "y_train": y_train,
    "X_val":   X_val,   "y_val":   y_val,
    "X_test":  X_test,  "y_test":  y_test,
}.items():
    obj.to_csv(SPLITS / f"{name}.csv", index=False)
    rows, cols = (obj.shape if obj.ndim == 2 else (obj.shape[0], 1))
    print(f"{name:8s} {rows:5d} x {cols:2d}  -> splits/{name}.csv")

for name, part in {"train": train, "val": val, "test": test}.items():
    print(f"{name:5s} {part['period'].min()} -> {part['period'].max()}  n={len(part)}")

# model.fit(X_train, y_train)
