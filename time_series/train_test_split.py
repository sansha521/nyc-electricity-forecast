# Train-test split based on time-series
from pathlib import Path

import pandas as pd

df = pd.read_csv("time_series_data.csv")
df.insert(0, "lag_0", df["value"])

train = df[df["period"] < "2024-09-01"]

val = df[
    (df["period"] >= "2024-09-01") &
    (df["period"] < "2025-09-01")
]

test = df[df["period"] >= "2025-09-01"]

X_train = train.drop(columns=["period", "value", "target"])
y_train = train["target"]

X_val = val.drop(columns=["period", "value", "target"])
y_val = val["target"]

X_test = test.drop(columns=["period", "value", "target"])
y_test = test["target"]


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