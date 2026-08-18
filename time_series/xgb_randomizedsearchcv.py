from pathlib import Path
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from evaluate import metrics

DATA_PATH = Path("splits")
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

# y_pred = best_model.predict(X_val)

# xgb_metrics = { "model": f"XGB randomized_search_cv ({best_model})", **metrics(y_val, y_pred) }
# # Convert dictionary to DataFrame
# df = pd.DataFrame([xgb_metrics])

# # Append to CSV. Disable header if file already exists. Drop default indices.
# df.to_csv(csv_file, mode="a", index=False, header=not Path(csv_file).exists())
# # skill = 100 * (1 - model_mape / base)     # % error reduction vs naive
# print(xgb_metrics)


importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": best_model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print(importance)

"""
       feature  importance
0        lag_0    0.676242
10   rolling_7    0.064659
11  rolling_30    0.031253
1        lag_1    0.030988
6       lag_21    0.027859
7       lag_28    0.026894
9      lag_366    0.026270
3        lag_3    0.024534
8      lag_365    0.023931
4        lag_7    0.023864
2        lag_2    0.021943
5       lag_14    0.021564
"""