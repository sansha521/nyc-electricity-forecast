from sklearn.linear_model import Ridge
from pathlib import Path
import pandas as pd
import numpy as np

from evaluate import metrics

DATA_PATH = Path("splits")
X_train = pd.read_csv(DATA_PATH / "X_train.csv")
y_train = pd.read_csv(DATA_PATH / "y_train.csv")["target"]
X_val = pd.read_csv(DATA_PATH / "X_val.csv")
y_val = pd.read_csv(DATA_PATH / "y_val.csv")["target"]

csv_file = "metrics.csv"

# model = Ridge(alpha=0.1)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_val)

# ridge_metrics = { "model": "ridge (alpha=0)", **metrics(y_val, y_pred) }
# # Convert dictionary to DataFrame
# df = pd.DataFrame([ridge_metrics])

# # Append to CSV. Disable header if file already exists. Drop default indices.
# df.to_csv(csv_file, mode="a", index=False, header=not Path(csv_file).exists())
# # skill = 100 * (1 - model_mape / base)     # % error reduction vs naive
# print(ridge_metrics)


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

# Standardize the training and test features using StandardScaler to ensure all variables are on the same scale, which improves regression model performance.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train Ridge Regression with Cross-Validation
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

ridge_cv = RidgeCV(
    alphas = np.logspace(-4, 4, 50),
    cv=tscv
)
ridge_cv.fit(X_train_scaled, y_train)

# Make predictions and evaluate model
y_pred = ridge_cv.predict(X_val_scaled)

ridge_metrics = { "model": f"Ridge (alpha={ round(float(ridge_cv.alpha_), 2) })", **metrics(y_val, y_pred) }
# Convert dictionary to DataFrame
df = pd.DataFrame([ridge_metrics])

# Append to CSV. Disable header if file already exists. Drop default indices.
df.to_csv(csv_file, mode="a", index=False, header=not Path(csv_file).exists())
# skill = 100 * (1 - model_mape / base)     # % error reduction vs naive
print(ridge_metrics)