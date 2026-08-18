from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score

from evaluate import metrics


DATA_PATH = Path("inference_feature_splits")
X_train = pd.read_csv(DATA_PATH / "X_train.csv")
y_train = pd.read_csv(DATA_PATH / "y_train.csv")["target"]
X_val = pd.read_csv(DATA_PATH / "X_val.csv")
y_val = pd.read_csv(DATA_PATH / "y_val.csv")["target"]

csv_file = "metrics.csv"

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

# Feature importance from Ridge coefficients
feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": ridge_cv.coef_,
    "importance": np.abs(ridge_cv.coef_)
})

feature_importance = feature_importance.sort_values(
    "importance", ascending=False
)

print("\nRidge Feature Importance:")
print(feature_importance.to_string(index=False))

# Make predictions and evaluate model
y_pred = ridge_cv.predict(X_val_scaled)

ridge_metrics = { "model": f"Ridge (alpha={ round(float(ridge_cv.alpha_), 2) })", **metrics(y_val, y_pred) }
# Convert dictionary to DataFrame
df = pd.DataFrame([ridge_metrics])

# Append to CSV. Disable header if file already exists. Drop default indices.
df.to_csv(csv_file, mode="a", index=False, header=not Path(csv_file).exists())
# skill = 100 * (1 - model_mape / base)     # % error reduction vs naive
print(ridge_metrics)


"""
Ridge Feature Importance:
       feature  coefficient   importance
         lag_0 16120.494933 16120.494933
         lag_1 -8428.795292  8428.795292
     rolling_7  7068.387318  7068.387318
           cdd  5114.562109  5114.562109
         lag_3 -3944.703258  3944.703258
         lag_2  2764.827503  2764.827503
      cdd_3day  2507.738220  2507.738220
     dew_point  2507.359498  2507.359498
    is_weekend  2359.932247  2359.932247
    rolling_30  1923.474977  1923.474977
   day_of_week -1727.613510  1727.613510
           hdd  1646.429241  1646.429241
     tavg_3day -1242.599925  1242.599925
        lag_28 -1216.679745  1216.679745
         lag_7 -1049.447934  1049.447934
        lag_14   891.729193   891.729193
          prcp  -888.084046   888.084046
          tmin  -757.742829   757.742829
    is_holiday   755.772706   755.772706
          tavg  -664.428793   664.428793
          tmax  -563.963870   563.963870
       lag_366   480.159570   480.159570
       lag_365  -266.246416   266.246416
daylight_hours  -261.974531   261.974531
       rh_mean   259.375483   259.375483
          snow   226.746357   226.746357
     solar_rad   222.897960   222.897960
sunshine_hours   211.909455   211.909455
     tavg_lag1   159.970229   159.970229
          snwd  -138.285388   138.285388
        lag_21  -132.709405   132.709405
        rh_max   -92.834069    92.834069
   cloud_cover   -69.021794    69.021794
      hdd_3day    57.879102    57.879102

      
inference_feature_splits
Ridge Feature Importance:
            feature  coefficient   importance
             cdd_fc 12879.362105 12879.362105
              lag_0  9168.034600  9168.034600
    is_weekend_next -4775.677356  4775.677356
             hdd_fc  4595.990411  4595.990411
         rolling_30  4247.502457  4247.502457
          rolling_7  3851.595997  3851.595997
       dew_point_fc  3521.957052  3521.957052
              lag_1 -2126.305625  2126.305625
       tavg_3day_fc -1780.643694  1780.643694
              lag_7 -1643.688983  1643.688983
            tmin_fc -1293.035496  1293.035496
    is_holiday_next -1227.791687  1227.791687
             lag_28 -1216.026374  1216.026374
        cdd_3day_fc  1180.209710  1180.209710
              lag_3 -1127.491372  1127.491372
             lag_21 -1097.686990  1097.686990
daylight_hours_next -1067.486185  1067.486185
            lag_365  -754.455865   754.455865
              lag_2   623.783901   623.783901
            tavg_fc  -580.826689   580.826689
        hdd_3day_fc  -510.364765   510.364765
       solar_rad_fc  -499.567121   499.567121
             lag_14  -448.420975   448.420975
            prcp_fc   302.177237   302.177237
         rh_mean_fc  -276.827244   276.827244
            snwd_fc  -224.293248   224.293248
            snow_fc   -96.554179    96.554179
            tmax_fc    85.056594    85.056594
            lag_366   -71.311096    71.311096
     cloud_cover_fc   -70.961807    70.961807
{'model': 'Ridge (alpha=5.43)', 'MAE': 3229, 'RMSE': 4264, 'MAPE': 2.28, 'bias': -1400.5}
"""