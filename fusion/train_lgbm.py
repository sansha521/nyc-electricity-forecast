# Randomized search CV LightGBM

from pathlib import Path
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import lightgbm as lgb
from lightgbm import LGBMRegressor
from evaluate import metrics


# command input
parser = argparse.ArgumentParser()
parser.add_argument(
    "--save",
    type=Path,
    help="Base path for saving the model,e.g. models/lgbm_v1"
)
args = parser.parse_args()
# Example run: python your_script.py --save models/lgbm_v1

DATA_PATH = Path("inference_feature_splits")

X_train = pd.read_csv(DATA_PATH / "X_train.csv")
y_train = pd.read_csv(DATA_PATH / "y_train.csv")["target"]
X_val = pd.read_csv(DATA_PATH / "X_val.csv")
y_val = pd.read_csv(DATA_PATH / "y_val.csv")["target"]

csv_file = "metrics.csv"

# Time-series CV
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

# Hyperparameter search on TRAIN only
search.fit(X_train, y_train)

best_model = search.best_estimator_
print("Best parameters:")
print(search.best_params_)

# Evaluate on held-out validation set
y_pred = best_model.predict(X_val)

val_metrics = metrics(y_val, y_pred)

lgbm_metrics = {
    "model": f"LGBM randomized_search_cv ({best_model})",
    **val_metrics,
}
print("Validation metrics:")
print(lgbm_metrics)

# lgbm_metrics = {"model": f"LGBM randomized_search_cv ({best_model})", **metrics(y_val, y_pred)}
df = pd.DataFrame([lgbm_metrics])

# Append to CSV. Disable header if file already exists. Drop default indices.
df.to_csv(csv_file, mode="a", index=False, header=not Path(csv_file).exists())


# Retrain winning model on TRAIN + VALIDATION
X_all = pd.concat([X_train, X_val], ignore_index=True)
y_all = pd.concat([y_train, y_val], ignore_index=True)

final_model = LGBMRegressor(
    objective="regression",
    random_state=42,
    n_jobs=1,
    verbose=-1,
    **search.best_params_,
)

final_model.fit(X_all, y_all)

# Feature importance from FINAL deployed model
booster = final_model.booster_

gain = booster.feature_importance(importance_type="gain")

importance = pd.DataFrame({
    "feature": X_all.columns,
    "gain_importance": gain,
}).sort_values("gain_importance", ascending=False)

print("Feature importance:")
print(importance)


# Save model + metadata if --save was provided.
if args.save:
    model_path = args.save.with_suffix(".txt")
    meta_path = args.save.with_suffix(".meta.json")
    importance_path = args.save.with_suffix(".feature_importance.csv")

    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the underlying LightGBM booster.
    final_model.booster_.save_model(str(model_path))

    metadata = {
        "features": list(X_all.columns),
        "lightgbm_version": lgb.__version__,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "val_metrics": val_metrics,
        "train_end": "2025-01-01",
        "val_end": "2026-01-01",
        "best_params": search.best_params_,
        "final_fit_rows": len(X_all),
    }

    meta_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Save feature importance
    importance.to_csv(
        importance_path,
        index=False,
    )

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {meta_path}")
    print(f"Saved feature importance: {importance_path}")

print(lgbm_metrics)


