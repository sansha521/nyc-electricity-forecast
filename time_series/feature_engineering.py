# Try time-series forecasting

import pandas as pd

df = pd.read_csv("all_features_and_target.csv", parse_dates=["period"])
df = df[["period", "value", "target"]]
df = df.iloc[:-1] # drop last row with NaN target

# shift() is positional, not date-aware -- sort first or the lags silently
# pick up whatever row happens to sit above
df = df.sort_values("period").reset_index(drop=True)

# recent days (1-3), same weekday history (7/14/21/28), same day last year
# (365 + 366 to straddle leap years)
LAGS = [1, 2, 3, 7, 14, 21, 28, 365, 366]
for lag in LAGS:
    df[f"lag_{lag}"] = df["value"].shift(lag)

print(df.head())
print(df.shape)

# rolling
for window in [7, 30]:
    df[f"rolling_{window}"] = (
        df["value"]
        .shift(1)
        .rolling(window)
        .mean()
    )
print(df.head())
print(df.shape)

df = df.dropna()
print(df.head())
print(df.shape)

df.to_csv("time_series_data.csv", index=False)
print("csv created")