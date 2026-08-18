import pandas as pd

COLUMNS = ["date", "prcp", "snow", "snwd", "tmin", "tmax"]

df_2019 = pd.read_csv("weather_2019-2022.csv")
df_2022 = pd.read_csv("central_park_2022-2026.csv")

# Both files use NOAA's uppercase names -- normalize so they line up
df_2019.columns = df_2019.columns.str.lower()
df_2022.columns = df_2022.columns.str.lower()

# weather_2019-2022.csv carries both DATE and date, which collide once lowercased
df_2019 = df_2019.loc[:, ~df_2019.columns.duplicated()]

df_2019["date"] = pd.to_datetime(df_2019["date"])
df_2022["date"] = pd.to_datetime(df_2022["date"])

df = pd.concat([df_2019[COLUMNS], df_2022[COLUMNS]], ignore_index=True)

# The two files overlap through 2022 -- keep the newer NOAA rows
df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
df = df.reset_index(drop=True)

print(df.head)
print(len(df), "rows from", df["date"].min().date(), "to", df["date"].max().date())
df.to_csv("all_weathers.csv", index=False)
