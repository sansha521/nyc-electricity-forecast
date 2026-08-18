import pandas as pd

# Gap-filled by add_missing_eia_to_all.py -- carries is_imputed alongside value
demand_df = pd.read_csv("daily_nyc_demand_filled.csv")
weather_df = pd.read_csv("all_weathers.csv")
holiday_df = pd.read_csv("weekday_holiday_data.csv")
humidity_df = pd.read_csv("humidity.csv")

"""
- SNOW — snowfall: new snow that fell during that day. A flow.
- SNWD — snow depth: how much snow was on the ground at observation
  time. A stock.

SNOW > 0 means it actually snowed that day. SNWD > 0 just means snow was
lying around — it stays nonzero for days after a storm, and it can
be nonzero on a clear sunny day.
"""

# Convert dates to datetime
demand_df["period"] = pd.to_datetime(demand_df["period"])
weather_df["date"] = pd.to_datetime(weather_df["date"])
holiday_df["period"] = pd.to_datetime(holiday_df["period"])
humidity_df["date"] = pd.to_datetime(humidity_df["date"])

# Merge demand + weather
df = pd.merge(
    demand_df,
    weather_df,
    left_on="period",
    right_on="date",
    how="left"
)

# Drop duplicate date column
df = df.drop(columns=["date", "DATE"], errors="ignore")

# Merge humidity (Open-Meteo -- NOAA daily summaries don't carry it)
df = pd.merge(
    df,
    humidity_df,
    left_on="period",
    right_on="date",
    how="left"
)
df = df.drop(columns=["date"], errors="ignore")

# Merge holiday features
df = pd.merge(
    df,
    holiday_df[["period", "is_weekend", "is_holiday"]],
    on="period",
    how="left"
)

# Day of week: 0 = Monday ... 6 = Sunday
df["day_of_week"] = df["period"].dt.dayofweek
df["day_name"] = df["period"].dt.day_name()

# Degree days split the V-shaped load/temperature curve into a heating side
# and a cooling side. The two bases are fitted, not conventional -- scanning
# candidate bases against demand puts the cooling kink at the usual 65F but
# the heating kink at 50F. See "Degree days" in all_data.md.
HDD_BASE, CDD_BASE = 50, 65
df["tavg"] = (df["tmin"] + df["tmax"]) / 2
df["hdd"] = (HDD_BASE - df["tavg"]).clip(lower=0)
df["cdd"] = (df["tavg"] - CDD_BASE).clip(lower=0)

# Rolling windows have to run oldest -> newest, otherwise they look forward
df = df.sort_values("period").reset_index(drop=True)

# Thermal inertia: buildings hold heat, so day 3 of a heat wave draws more
# than day 1 at the same temperature
df["tavg_3day"] = df["tavg"].rolling(3).mean()
df["cdd_3day"] = df["cdd"].rolling(3).mean()
df["hdd_3day"] = df["hdd"].rolling(3).mean()
df["tavg_lag1"] = df["tavg"].shift(1)

# Rolling means come out as repeating decimals -- 0.1F is well past the
# precision the underlying observations actually have
DERIVED = ["tavg", "hdd", "cdd", "tavg_3day", "cdd_3day", "hdd_3day", "tavg_lag1"]
df[DERIVED] = df[DERIVED].round(1)

print(df.head(10))
print(df.tail)
print(df.columns)
# df.to_csv("all_data.csv", index=False)
df.to_csv("all_data_recompute.csv", index=False)