from __future__ import annotations

import math

import pandas as pd


SCORE_COLUMNS = [
    "target_date",
    "predicted_demand",
    "actual_demand",
    "error",
    "abs_error",
    "pct_error",
    "abs_pct_error",
    "model_version",
    "is_imputed",
]


def compute_score_rows(joined: pd.DataFrame) -> pd.DataFrame:
    """Compute per-day prediction scores from prediction/actual rows.

    The input is expected to contain prediction rows already joined to actual
    demand. Imputed actual demand is deliberately excluded from scoring.
    """
    if joined.empty:
        return pd.DataFrame(columns=SCORE_COLUMNS)

    required = {
        "target_date",
        "predicted_demand",
        "actual_demand",
        "model_version",
        "is_imputed",
    }
    missing = sorted(required - set(joined.columns))
    if missing:
        raise ValueError(f"missing score input columns: {missing}")

    scores = joined.loc[~joined["is_imputed"].astype(bool)].copy()
    if scores.empty:
        return pd.DataFrame(columns=SCORE_COLUMNS)

    scores["error"] = scores["predicted_demand"] - scores["actual_demand"]
    scores["abs_error"] = scores["error"].abs()
    scores["pct_error"] = scores["error"] / scores["actual_demand"] * 100.0
    scores["abs_pct_error"] = scores["pct_error"].abs()

    return scores[SCORE_COLUMNS].sort_values("target_date").reset_index(drop=True)


def summarize_scores(scores: pd.DataFrame) -> dict[str, float | int]:
    """Return count, MAE, RMSE, MAPE, and bias for scored prediction rows."""
    if scores.empty:
        return {
            "count": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "mape": math.nan,
            "bias": math.nan,
        }

    required = {"error", "abs_error", "abs_pct_error"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"missing summary input columns: {missing}")

    squared_error = scores["error"] ** 2

    return {
        "count": int(len(scores)),
        "mae": float(scores["abs_error"].mean()),
        "rmse": float(math.sqrt(squared_error.mean())),
        "mape": float(scores["abs_pct_error"].mean()),
        "bias": float(scores["error"].mean()),
    }
