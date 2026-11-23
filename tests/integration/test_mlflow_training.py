from importlib.util import find_spec
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


TRAINER = Path(__file__).resolve().parents[2] / "fusion" / "train_lgbm.py"
MODEL_NAME = "nyc-electricity-lgbm"


@unittest.skipUnless(find_spec("mlflow"), "Install MLflow for trainer integration tests")
class MlflowTrainingTests(unittest.TestCase):
    def test_trainer_logs_metrics_and_registers_a_reloadable_lightgbm_model(self):
        import mlflow.lightgbm
        from mlflow import MlflowClient

        work_dir = Path(self.enterContext(TemporaryDirectory()))
        splits = work_dir / "inference_feature_splits"
        splits.mkdir()
        features = pd.DataFrame({"lag_0": range(100, 148), "tavg_fc": range(48)})
        for split, rows in (("train", features.iloc[:40]), ("val", features.iloc[40:])):
            rows.to_csv(splits / f"X_{split}.csv", index=False)
            (rows["lag_0"] * 2).to_frame(name="target").to_csv(
                splits / f"y_{split}.csv", index=False
            )

        uri = f"sqlite:///{work_dir / 'mlflow.db'}"
        tracking_env = {"MLFLOW_TRACKING_URI": uri, "MLFLOW_REGISTRY_URI": uri}
        env = {key: value for key, value in os.environ.items() if not key.startswith("MLFLOW_")}
        env.update(tracking_env, LOKY_MAX_CPU_COUNT="2", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        completed = subprocess.run(
            [sys.executable, "-B", str(TRAINER)],
            cwd=work_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

        client = MlflowClient(tracking_uri=uri, registry_uri=uri)
        experiment = client.get_experiment_by_name("nyc-electricity")
        self.assertIsNotNone(experiment)
        runs = client.search_runs([experiment.experiment_id])
        self.assertEqual(len(runs), 1)
        run = runs[0]
        self.assertEqual(run.info.status, "FINISHED")
        self.assertIn("n_estimators", run.data.params)
        self.assertTrue({"cv_mae", "val_MAE", "val_RMSE"} <= run.data.metrics.keys())
        self.assertIn("feature_importance.json", {a.path for a in client.list_artifacts(run.info.run_id)})

        version = client.get_model_version(MODEL_NAME, "1")
        self.assertEqual(version.run_id, run.info.run_id)
        self.assertEqual(version.status, "READY")
        logged_models = client.search_logged_models(experiment_ids=[experiment.experiment_id])
        self.assertEqual(len(logged_models), 1)

        with patch.dict(os.environ, tracking_env):
            original = mlflow.lightgbm.load_model(f"models:/{logged_models[0].model_id}")
            registered = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}/1")
        sample = features.iloc[-3:]
        predictions = registered.predict(sample, num_threads=1)
        self.assertTrue(np.isfinite(predictions).all())
        np.testing.assert_allclose(predictions, original.predict(sample, num_threads=1))
