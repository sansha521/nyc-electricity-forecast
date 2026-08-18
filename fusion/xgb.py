# Radomize search CV XGBoost

from pathlib import Path
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from evaluate import metrics

DATA_PATH = Path("inference_feature_splits")

X_train = pd.read_csv(DATA_PATH / "X_train.csv")
y_train = pd.read_csv(DATA_PATH / "y_train.csv")["target"]
X_val = pd.read_csv(DATA_PATH / "X_val.csv")
y_val = pd.read_csv(DATA_PATH / "y_val.csv")["target"]

csv_file = "metrics.csv"

tscv = TimeSeriesSplit(n_splits=5)

param_dist = {
    "n_estimators": [300, 500, 800],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.3],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=75,                    # try 50-100
    scoring="neg_mean_absolute_error",
    cv=tscv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

y_pred = best_model.predict(X_val)

xgb_metrics = { "model": f"XGB randomized_search_cv ({best_model})", **metrics(y_val, y_pred) }
# Convert dictionary to DataFrame
df = pd.DataFrame([xgb_metrics])

# Append to CSV. Disable header if file already exists. Drop default indices.
df.to_csv(csv_file, mode="a", index=False, header=not Path(csv_file).exists())
# skill = 100 * (1 - model_mape / base)     # % error reduction vs naive
print(xgb_metrics)


importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": best_model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print(importance)

"""
inference_feature_splits
                feature  importance
25          cdd_3day_fc    0.288381
23               cdd_fc    0.177307
24         tavg_3day_fc    0.150753
12              tmin_fc    0.122064
14              tavg_fc    0.118576
0                 lag_0    0.045015
27      is_weekend_next    0.025388
15         dew_point_fc    0.016942
10            rolling_7    0.015348
22               hdd_fc    0.012158
13              tmax_fc    0.004930
28      is_holiday_next    0.004044
1                 lag_1    0.002981
11           rolling_30    0.001753
26          hdd_3day_fc    0.001752
17              prcp_fc    0.001380
3                 lag_3    0.001338
16           rh_mean_fc    0.001201
29  daylight_hours_next    0.001169
2                 lag_2    0.001055
6                lag_21    0.001034
4                 lag_7    0.000866
20         solar_rad_fc    0.000851
8               lag_365    0.000737
7                lag_28    0.000665
5                lag_14    0.000600
21       cloud_cover_fc    0.000547
9               lag_366    0.000524
19              snwd_fc    0.000372
18              snow_fc    0.000267
"""