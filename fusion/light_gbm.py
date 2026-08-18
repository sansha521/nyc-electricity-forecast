# Randomized search CV LightGBM

from pathlib import Path
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from lightgbm import LGBMRegressor
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
    "num_leaves": [15, 31, 63, 127],       # LightGBM's main capacity knob
    "max_depth": [-1, 4, 6, 8],            # -1 = no limit; pairs with num_leaves
    "min_child_samples": [5, 10, 20, 30],  # ~ min_child_weight analogue
    "subsample": [0.7, 0.8, 1.0],          # needs subsample_freq > 0 to activate
    "subsample_freq": [1],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 0.5],            # L1
    "reg_lambda": [0, 0.1, 0.5],           # L2
}

lgbm = LGBMRegressor(
    objective="regression",     # "regression_l1" optimizes MAE directly
    random_state=42,
    n_jobs=1,                   # data is tiny; threading overhead >> work. Parallelize in the search instead.
    verbose=-1,                 # silence per-iteration logging
)

search = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=param_dist,
    n_iter=75,                    # try 50-100
    scoring="neg_mean_absolute_error",
    cv=tscv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

y_pred = best_model.predict(X_val)

lgbm_metrics = {"model": f"LGBM randomized_search_cv ({best_model})", **metrics(y_val, y_pred)}
df = pd.DataFrame([lgbm_metrics])

# Append to CSV. Disable header if file already exists. Drop default indices.
df.to_csv(csv_file, mode="a", index=False, header=not Path(csv_file).exists())
print(lgbm_metrics)


# importance = pd.DataFrame({
#     "feature": X_train.columns,
#     "importance": best_model.feature_importances_,
# }).sort_values(
#     "importance",
#     ascending=False,
# )

import numpy as np
booster = best_model.booster_
gain = booster.feature_importance(importance_type="gain")
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": gain,
}).sort_values("importance", ascending=False)

print(importance)

"""
inference_feature_splits
                feature    importance
14              tavg_fc  2.536717e+12
25          cdd_3day_fc  2.354150e+12
24         tavg_3day_fc  1.692999e+12
12              tmin_fc  1.420833e+12
0                 lag_0  1.396348e+12
27      is_weekend_next  4.650586e+11
15         dew_point_fc  4.078801e+11
10            rolling_7  2.473618e+11
23               cdd_fc  1.641410e+11
13              tmax_fc  8.685668e+10
11           rolling_30  6.338281e+10
3                 lag_3  3.130455e+10
1                 lag_1  3.092756e+10
28      is_holiday_next  2.628182e+10
29  daylight_hours_next  2.520144e+10
22               hdd_fc  1.928775e+10
6                lag_21  1.830334e+10
2                 lag_2  1.746168e+10
8               lag_365  1.698788e+10
4                 lag_7  1.508048e+10
20         solar_rad_fc  1.501424e+10
17              prcp_fc  1.415114e+10
16           rh_mean_fc  1.256719e+10
7                lag_28  1.247955e+10
5                lag_14  9.868772e+09
9               lag_366  9.475828e+09
21       cloud_cover_fc  7.031397e+09
26          hdd_3day_fc  6.172827e+09
19              snwd_fc  3.498492e+08
18              snow_fc  1.008797e+08
"""