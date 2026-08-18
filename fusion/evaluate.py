import numpy as np

def metrics(y, p):
    err = p - y
    return {
        "MAE":  round(float(np.mean(np.abs(err)))),              # "On average, how many MW is my prediction off?"
        "RMSE": round(float(np.sqrt(np.mean(err**2)))),          # Unlike MAE, RMSE penalizes large mistakes much more.
        "MAPE": round(float(np.mean(np.abs(err / y)) * 100), 2),    # MAPE = 2.8% means "The model is off by 2.8% on average."
        "bias": round(float(np.mean(err)), 1),                      # MWh — signed, catches drift
    }