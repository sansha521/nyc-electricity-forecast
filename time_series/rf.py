from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from evaluate import metrics

DATA_PATH = Path("splits")
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
       feature  importance
0        lag_0    0.890456
1        lag_1    0.019743
10   rolling_7    0.017509
7       lag_28    0.009667
4        lag_7    0.009505
3        lag_3    0.008592
11  rolling_30    0.008435
9      lag_366    0.008009
8      lag_365    0.007879
6       lag_21    0.007101
2        lag_2    0.007094
5       lag_14    0.006010
"""