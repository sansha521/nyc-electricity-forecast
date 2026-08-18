import pandas as pd

# Load CSV
df = pd.read_csv("NYC_Central_Park_weather_1869-2022.csv")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["DATE"])

# Extract everything under 2019-01-01 (excluding the row itself)
extracted_df = df[df["date"] >= "2019-01-01"]

print(extracted_df.head)
extracted_df.to_csv("weather_2019-2022.csv", index=False)