import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "fusion" / "models" / "lgbm_v1.txt"
META_PATH = PROJECT_ROOT / "fusion" / "models" / "lgbm_v1.meta.json"


def load_model():
    return lgb.Booster(model_file=str(MODEL_PATH))


def load_metadata() -> dict:
    with open(META_PATH) as f:
        return json.load(f)


def predict(model, row: pd.DataFrame) -> float:
    meta = load_metadata()
    features = meta["features"]

    # Make absolutely sure inference order matches training.
    assert list(row.columns) == features, (
        f"Feature mismatch.\n"
        f"Expected: {features}\n"
        f"Got: {list(row.columns)}"
    )

    prediction = model.predict(row)

    return float(prediction[0])
