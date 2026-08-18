import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("./splits")

X_train = pd.read_csv(DATA_PATH / "X_train.csv")
X_val = pd.read_csv(DATA_PATH / "X_val.csv")
X_test = pd.read_csv(DATA_PATH / "X_test.csv")

y_train = pd.read_csv(DATA_PATH / "y_train.csv")
y_val = pd.read_csv(DATA_PATH / "y_val.csv")
y_test = pd.read_csv(DATA_PATH / "y_test.csv")

# drop is_holiday, hdd, hdd_3day, is_weekend, snwd, snow
# based on rf feaure importance
X_train.drop(columns=["snwd", "snow"], inplace=True)
X_val.drop(columns=["snwd", "snow"], inplace=True)
X_test.drop(columns=["snwd", "snow"], inplace=True)

# print(X_train.head)
# print(X_val.head)
# print(X_test.head)

# Save the new splits
SAVE_PATH = Path("select_feature_splits")
SAVE_PATH.mkdir(exist_ok=True)

for name, df in {
    "X_train": X_train,
    "X_val": X_val,
    "X_test": X_test,
    "y_train": y_train,
    "y_val": y_val,
    "y_test": y_test,
}.items():
    file = SAVE_PATH / f"{name}.csv"
    df.to_csv(file, index=False)
    print(f"Saved {file} ({df.shape[0]} rows × {df.shape[1]} columns)")


"""
           feature  importance
22           lag_0    0.533133
3             tmin    0.244207
17             cdd    0.046256
14     day_of_week    0.036123
15            tavg    0.034809
32       rolling_7    0.008244
33      rolling_30    0.006573
26           lag_7    0.006300
7        dew_point    0.006206
10  daylight_hours    0.005537
23           lag_1    0.005436
18       tavg_3day    0.005044
30         lag_365    0.004998
25           lag_3    0.004985
27          lag_14    0.004825
31         lag_366    0.004685
11  sunshine_hours    0.004197
19        cdd_3day    0.004175
4             tmax    0.004026
28          lag_21    0.003928
8        solar_rad    0.003891
29          lag_28    0.003844
21       tavg_lag1    0.003318
24           lag_2    0.003157
9      cloud_cover    0.002880
5          rh_mean    0.002477
0             prcp    0.002269
6           rh_max    0.002176
13      is_holiday    0.000840
16             hdd    0.000532
20        hdd_3day    0.000446
12      is_weekend    0.000413
2             snwd    0.000056
1             snow    0.000014
"""