from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH = Path("splits")
X_train = pd.read_csv(DATA_PATH / "X_train.csv")
y_train = pd.read_csv(DATA_PATH / "y_train.csv")
X_val = pd.read_csv(DATA_PATH / "X_val.csv")
y_val = pd.read_csv(DATA_PATH / "y_val.csv")["target"]

csv_file = "metrics.csv"

def metrics(y, p):
    err = p - y
    return {
        "MAE":  round(float(np.mean(np.abs(err)))),              # "On average, how many MW is my prediction off?"
        "RMSE": round(float(np.sqrt(np.mean(err**2)))),          # Unlike MAE, RMSE penalizes large mistakes much more.
        "MAPE": round(float(np.mean(np.abs(err / y)) * 100), 2),    # MAPE = 2.8% means "The model is off by 2.8% on average."
        "bias": round(float(np.mean(err)), 1),                      # MWh — signed, catches drift
    }

# naive = X_val["lag_0"]      # naive prediction: tomorrow's demand is today's demand
# base_metrics = { "model": "persistence (lag_0)", **metrics(y_val, naive) }
# # Convert dictionary to DataFrame
# df = pd.DataFrame([base_metrics])

# # Append to CSV. Disable header if file already exists. Drop default indices.
# df.to_csv(csv_file, mode="a", index=False, header=not Path(csv_file).exists())
# # skill = 100 * (1 - model_mape / base)     # % error reduction vs naive
# print(base_metrics)

# naive = X_val["lag_7"]
# base_metrics = { "model": "persistence (lag_7)", **metrics(y_val, naive) }
# # Convert dictionary to DataFrame
# df = pd.DataFrame([base_metrics])

# # Append to CSV. Disable header if file already exists. Drop default indices.
# df.to_csv(csv_file, mode="a", index=False, header=not Path(csv_file).exists())
# # skill = 100 * (1 - model_mape / base)     # % error reduction vs naive
# print(base_metrics)

naive = X_val["lag_28"]
base_metrics = { "model": "persistence (lag_28)", **metrics(y_val, naive) }
# Convert dictionary to DataFrame
df = pd.DataFrame([base_metrics])

# Append to CSV. Disable header if file already exists. Drop default indices.
df.to_csv(csv_file, mode="a", index=False, header=not Path(csv_file).exists())
# skill = 100 * (1 - model_mape / base)     # % error reduction vs naive
print(base_metrics)
