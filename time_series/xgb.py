from pathlib import Path
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from evaluate import metrics

DATA_PATH = Path("splits")
X_train = pd.read_csv(DATA_PATH / "X_train.csv")
y_train = pd.read_csv(DATA_PATH / "y_train.csv")["target"]
X_val = pd.read_csv(DATA_PATH / "X_val.csv")
y_val = pd.read_csv(DATA_PATH / "y_val.csv")["target"]

csv_file = "metrics.csv"

# model = XGBRegressor(
#     n_estimators=500,
#     learning_rate=0.05,
#     max_depth=5,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective="reg:squarederror",
#     random_state=42
# )
# model.fit(X_train, y_train)

# With Early Stopping
# model = XGBRegressor(
#     n_estimators=1000,
#     learning_rate=0.05,
#     max_depth=5,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective="reg:squarederror",
#     random_state=42,
#     early_stopping_rounds=30
# )

# model.fit(
#     X_train,
#     y_train,
#     eval_set=[(X_val, y_val)],
#     verbose=False
# )

# y_pred = model.predict(X_val)

# xgb_metrics = { "model": f"XGBoost", **metrics(y_val, y_pred) }
# # Convert dictionary to DataFrame
# df = pd.DataFrame([xgb_metrics])

# # Append to CSV. Disable header if file already exists. Drop default indices.
# df.to_csv(csv_file, mode="a", index=False, header=not Path(csv_file).exists())
# # skill = 100 * (1 - model_mape / base)     # % error reduction vs naive
# print(xgb_metrics)


# Hyperparameter tuning with GridSearchCV
tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    "n_estimators": [200, 500],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_val)

xgb_metrics = { "model": f"XGB with GridSearchCSV ({best_model})", **metrics(y_val, y_pred) }
# Convert dictionary to DataFrame
df = pd.DataFrame([xgb_metrics])

# Append to CSV. Disable header if file already exists. Drop default indices.
df.to_csv(csv_file, mode="a", index=False, header=not Path(csv_file).exists())
# skill = 100 * (1 - model_mape / base)     # % error reduction vs naive
print(xgb_metrics)





"""
"RF (XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=1.0, device=None, early_stopping_rounds=None,
             enable_categorical=True, eval_metric=None, feature_types=None,
             feature_weights=None, gamma=None, grow_policy=None,
             importance_type=None, interaction_constraints=None,
             learning_rate=0.05, max_bin=None, max_cat_threshold=None,
             max_cat_to_onehot=None, max_delta_step=None, max_depth=3,
             max_leaves=None, min_child_weight=None, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=500,
             n_jobs=None, num_parallel_tree=None, ...))",6440,9116,4.56,-351.3

"""