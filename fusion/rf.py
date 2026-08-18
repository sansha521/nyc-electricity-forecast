from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from evaluate import metrics


DATA_PATH = Path("inference_feature_splits")
X_train = pd.read_csv(DATA_PATH / "X_train.csv")
y_train = pd.read_csv(DATA_PATH / "y_train.csv")["target"]
X_val = pd.read_csv(DATA_PATH / "X_val.csv")
y_val = pd.read_csv(DATA_PATH / "y_val.csv")["target"]

csv_file = "metrics.csv"

tscv = TimeSeriesSplit(n_splits=5)

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5, 10, 20, None],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", 0.5, 1.0]
}

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_rf = grid.best_estimator_

y_pred = best_rf.predict(X_val)

rf_metrics = { "model": f"RF ({best_rf})", **metrics(y_val, y_pred) }
# Convert dictionary to DataFrame
df = pd.DataFrame([rf_metrics])

# Append to CSV. Disable header if file already exists. Drop default indices.
df.to_csv(csv_file, mode="a", index=False, header=not Path(csv_file).exists())
# skill = 100 * (1 - model_mape / base)     # % error reduction vs naive
print(rf_metrics)


importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": best_rf.feature_importances_
}).sort_values("importance", ascending=False)

print(importance)

"""
{'model': 'RF (RandomForestRegressor(max_features=0.5, n_estimators=500, n_jobs=-1,\n                      random_state=42))', 'MAE': 3625, 'RMSE': 5183, 'MAPE': 2.47, 'bias': -2223.4}
                feature  importance
26         tavg_3day_fc    0.276952
27          cdd_3day_fc    0.218788
12              tmin_fc    0.106791
0                 lag_0    0.099811
14              tavg_fc    0.085073
25               cdd_fc    0.073101
29      is_weekend_next    0.028846
10            rolling_7    0.028602
15         dew_point_fc    0.012069
13              tmax_fc    0.010791
1                 lag_1    0.006768
24               hdd_fc    0.005051
4                 lag_7    0.004623
37                dow_5    0.004016
11           rolling_30    0.003810
32                dow_0    0.002790
3                 lag_3    0.002750
2                 lag_2    0.002214
28          hdd_3day_fc    0.002170
38                dow_6    0.001914
5                lag_14    0.001855
6                lag_21    0.001725
8               lag_365    0.001632
9               lag_366    0.001500
42            doy_cos_2    0.001430
17            rh_max_fc    0.001412
21         solar_rad_fc    0.001403
40            doy_cos_1    0.001335
16           rh_mean_fc    0.001312
31  daylight_hours_next    0.001308
7                lag_28    0.001246
41            doy_sin_2    0.001211
23    sunshine_hours_fc    0.001195
22       cloud_cover_fc    0.001170
39            doy_sin_1    0.001151
18              prcp_fc    0.001042
30      is_holiday_next    0.000657
36                dow_4    0.000146
35                dow_3    0.000112
33                dow_1    0.000100
34                dow_2    0.000077
19              snow_fc    0.000026
20              snwd_fc    0.000025


inference_feature_set
{'model': 'RF (RandomForestRegressor(max_features=0.5, n_estimators=500, n_jobs=-1,\n                      random_state=42))', 'MAE': 3663, 'RMSE': 5211, 'MAPE': 2.5, 'bias': -2242.3}
                feature  importance
24         tavg_3day_fc    0.289483
25          cdd_3day_fc    0.195410
12              tmin_fc    0.112979
14              tavg_fc    0.102953
0                 lag_0    0.094400
23               cdd_fc    0.073350
27      is_weekend_next    0.034913
10            rolling_7    0.024733
15         dew_point_fc    0.012843
1                 lag_1    0.007827
13              tmax_fc    0.007334
22               hdd_fc    0.005600
4                 lag_7    0.005108
11           rolling_30    0.004794
3                 lag_3    0.003220
2                 lag_2    0.002994
26          hdd_3day_fc    0.002815
5                lag_14    0.002355
29  daylight_hours_next    0.002304
6                lag_21    0.002044
8               lag_365    0.001993
20         solar_rad_fc    0.001903
9               lag_366    0.001821
16           rh_mean_fc    0.001713
7                lag_28    0.001619
21       cloud_cover_fc    0.001454
17              prcp_fc    0.001237
28      is_holiday_next    0.000729
18              snow_fc    0.000036
19              snwd_fc    0.000035
"""