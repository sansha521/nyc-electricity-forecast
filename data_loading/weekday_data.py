import pandas as pd
import holidays

df = pd.read_csv("daily_demand.csv")

# Convert string dates to pandas datetime objects
df['date_parsed'] = pd.to_datetime(df['period'])

# 1 if Saturday or Sunday, else 0
df['is_weekend'] = (df['date_parsed'].dt.dayofweek >= 5).astype(int)

# Load US holidays from 2019 to 2026
years = list(range(2019, 2027))
us_holidays = holidays.country_holidays('US', years=years)

# 1 if holiday, else 0. holidays is keyed by datetime.date, so compare dates --
# matching Timestamps against it silently returns all zeros
df['is_holiday'] = df['date_parsed'].dt.date.isin(us_holidays).astype(int)

# Optional: Drop the temporary parsed column
df = df.drop(columns=['date_parsed'])

print(df.head(10))
df.to_csv("weekday_holiday_data.csv", index=False)
print("CSV file created.")