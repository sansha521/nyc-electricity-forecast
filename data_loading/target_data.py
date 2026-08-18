"""Create target column of the next day's demand for machine learning"""
# today information → tomorrow demand

import pandas as pd

df = pd.read_csv("all_data_recompute.csv")

df["target"] = df["value"].shift(-1)

print(df.head)

df.to_csv("all_features_and_target.csv", index=False)
print("csv created")