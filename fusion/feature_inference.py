import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("./forecast_feature_splits")

X_train = pd.read_csv(DATA_PATH / "X_train.csv")
X_val = pd.read_csv(DATA_PATH / "X_val.csv")
X_test = pd.read_csv(DATA_PATH / "X_test.csv")

y_train = pd.read_csv(DATA_PATH / "y_train.csv")
y_val = pd.read_csv(DATA_PATH / "y_val.csv")
y_test = pd.read_csv(DATA_PATH / "y_test.csv")

# drop 
# prcp_fc,snow_fc,snwd_fc,solar_rad_fc,cloud_cover_fc,sunshine_hours_fc,hdd_fc,cdd_fc,tavg_3day_fc,cdd_3day_fc,hdd_3day_fc,is_weekend_next,is_holiday_next,daylight_hours_next,dow_0,dow_1,dow_2,dow_3,dow_4,dow_5,dow_6,doy_sin_1,doy_cos_1,doy_sin_2,doy_cos_2
"""

Weather forecast data for tomorrow:
tmin_fc,tmax_fc,tavg_fc,dew_point_fc,rh_mean_fc,
prcp_fc,snow_fc,snwd_fc,solar_rad_fc,cloud_cover_fc,
(OMIT rh_max_fc, sunshine_hours_fc)

Weather data based on historical weather:
hdd_fc,cdd_fc,tavg_3day_fc,cdd_3day_fc,hdd_3day_fc,

Calendar data:
is_weekend_next,is_holiday_next,daylight_hours_next,

Other:
dow_0,dow_1,dow_2,dow_3,dow_4,dow_5,dow_6,
doy_sin_1,doy_cos_1,doy_sin_2,doy_cos_2
"""
DROP_COLUMNS = [
    "rh_max_fc", "sunshine_hours_fc", 
    "dow_0", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6",
    "doy_sin_1", "doy_cos_1", "doy_sin_2", "doy_cos_2"
]

X_train.drop(columns=DROP_COLUMNS, inplace=True)
X_val.drop(columns=DROP_COLUMNS, inplace=True)
X_test.drop(columns=DROP_COLUMNS, inplace=True)

# The rolling means carry float noise from the division. Round to 1dp to match
# the precision of every other derived column (all_data.md). The raw lags are
# reported demand and stay untouched.
ROUND_COLUMNS = ["rolling_7", "rolling_30"]

X_train[ROUND_COLUMNS] = X_train[ROUND_COLUMNS].round(1)
X_val[ROUND_COLUMNS] = X_val[ROUND_COLUMNS].round(1)
X_test[ROUND_COLUMNS] = X_test[ROUND_COLUMNS].round(1)

# print(X_train.head)
# print(X_val.head)
# print(X_test.head)
# print(X_train.columns)

# Save the new splits
SAVE_PATH = Path("inference_feature_splits")
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
